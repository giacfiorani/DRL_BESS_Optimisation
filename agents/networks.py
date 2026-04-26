import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions.normal import Normal


class DeepQNetwork(nn.Module):
    """Two-hidden-layer MLP Q-network with Huber loss and Adam optimiser."""

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """Initialise the network.

        Args:
            lr: Learning rate for the Adam optimiser.
            input_dims: Dimensionality of the input observation vector.
            fc1_dims: Width of the first hidden layer.
            fc2_dims: Width of the second hidden layer.
            n_actions: Number of discrete output Q-values.
        """
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        # Huber loss clips large TD errors, preventing gradient explosion on
        # energy-price spikes that produce occasional reward outliers.
        self.loss = nn.SmoothL1Loss()

        # Device selection: MPS (Apple Silicon), CUDA, or CPU.
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        """Compute Q-values for all actions.

        Args:
            state: Float32 tensor of shape (batch, input_dims).

        Returns:
            Q-value tensor of shape (batch, n_actions).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, file_name='networks.pth'):
        """Save network weights to the ``./models`` directory.

        Args:
            file_name: Output filename.
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        T.save(self.state_dict(), os.path.join(model_folder_path, file_name))


class DuelingDQN(nn.Module):
    """Dueling network architecture with shared feature layer, value stream, and advantage stream."""

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """Initialise the dueling network.

        Args:
            lr: Learning rate for the Adam optimiser.
            input_dims: Dimensionality of the input observation vector.
            fc1_dims: Width of the shared feature layer.
            fc2_dims: Width of the value and advantage stream hidden layers.
            n_actions: Number of discrete actions.
        """
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Linear(input_dims, fc1_dims)

        self.value_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        """Compute Q-values using the dueling aggregation module.

        Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

        Args:
            state: Float32 tensor of shape (batch, input_dims).

        Returns:
            Q-value tensor of shape (batch, n_actions).
        """
        features   = F.relu(self.feature_layer(state))
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def save(self, file_name='networks.pth'):
        """Save network weights to the ``./models`` directory.

        Args:
            file_name: Output filename.
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        T.save(self.state_dict(), os.path.join(model_folder_path, file_name))


class SACCriticNetwork(nn.Module):
    """Twin-critic Q-network for SAC. Input is the concatenation of state and action."""

    def __init__(self, lr, input_dims=103, n_actions=3, fc1_dims=512, fc2_dims=512):
        """Initialise the critic network.

        Args:
            lr: Learning rate for the Adam optimiser.
            input_dims: Dimensionality of the observation vector.
            n_actions: Number of continuous action dimensions.
            fc1_dims: Width of the first hidden layer.
            fc2_dims: Width of the second hidden layer.
        """
        super(SACCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.q   = nn.Linear(fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        """Compute a scalar Q-value estimate.

        Args:
            state: Float32 tensor of shape (batch, input_dims).
            action: Float32 tensor of shape (batch, n_actions).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        q_value = T.cat([state, action], dim=1)
        q_value = F.relu(self.ln1(self.fc1(q_value)))
        q_value = F.relu(self.ln2(self.fc2(q_value)))
        return self.q(q_value)

    def save(self, file_name='sac_critic.pth'):
        """Save network weights to the ``./models`` directory.

        Args:
            file_name: Output filename.
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        T.save(self.state_dict(), os.path.join(model_folder_path, file_name))


class SAC2CriticNetwork(nn.Module):
    """Variant critic network (architecture identical to SACCriticNetwork)."""

    def __init__(self, lr, input_dims=103, n_actions=3, fc1_dims=512, fc2_dims=512):
        super(SACCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.q   = nn.Linear(fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        """Compute a scalar Q-value estimate.

        Args:
            state: Float32 tensor of shape (batch, input_dims).
            action: Float32 tensor of shape (batch, n_actions).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        q_value = T.cat([state, action], dim=1)
        q_value = F.relu(self.ln1(self.fc1(q_value)))
        q_value = F.relu(self.ln2(self.fc2(q_value)))
        return self.q(q_value)

    def save(self, file_name='sac_critic.pth'):
        """Save network weights to the ``./models`` directory.

        Args:
            file_name: Output filename.
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        T.save(self.state_dict(), os.path.join(model_folder_path, file_name))


class SACActorNetwork(nn.Module):
    """Gaussian policy network for SAC with tanh squashing and numerically stable log-prob."""

    def __init__(self, lr, input_dims=103, n_actions=3, fc1_dims=512, fc2_dims=512):
        """Initialise the actor network.

        Args:
            lr: Learning rate for the Adam optimiser.
            input_dims: Dimensionality of the observation vector.
            n_actions: Number of continuous action dimensions.
            fc1_dims: Width of the first hidden layer.
            fc2_dims: Width of the second hidden layer.
        """
        super(SACActorNetwork, self).__init__()

        self.reparam_noise = 1e-6

        self.fc1     = nn.Linear(input_dims, fc1_dims)
        self.ln1     = nn.LayerNorm(fc1_dims)
        self.fc2     = nn.Linear(fc1_dims, fc2_dims)
        self.ln2     = nn.LayerNorm(fc2_dims)

        # Separate output heads for Gaussian mean and log standard deviation.
        self.mu      = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        """Compute Gaussian policy parameters.

        Args:
            state: Float32 tensor of shape (batch, input_dims).

        Returns:
            Tuple of (mu, std) tensors, each of shape (batch, n_actions).
        """
        prob = F.relu(self.ln1(self.fc1(state)))
        prob = F.relu(self.ln2(self.fc2(prob)))

        mu      = self.mu(prob)
        log_std = self.log_std(prob)

        # Clamp log_std to a finite range to prevent numerical overflow.
        log_std = T.clamp(log_std, min=-20, max=2)
        std     = T.exp(log_std)

        return mu, std

    def sample_normal(self, state, reparameterize=True):
        """Sample a tanh-squashed action and compute its log-probability.

        Args:
            state: Float32 tensor of shape (batch, input_dims).
            reparameterize: If True, use the reparameterisation trick
                (rsample) to allow gradients to flow through the sample.

        Returns:
            Tuple of (action, log_prob) where action has shape
            (batch, n_actions) and log_prob has shape (batch, 1).
        """
        mu, std = self.forward(state)
        probabilities = Normal(mu, std)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)

        # Numerically stable log-prob with tanh Jacobian correction.
        # Avoids T.log(1 - tanh²) which kills gradients when saturated.
        # Identity: log(1 - tanh²(x)) = 2(log 2 - x - softplus(-2x))
        log_probs  = probabilities.log_prob(actions)
        log_probs -= 2 * (math.log(2) - actions - F.softplus(-2 * actions))
        log_probs  = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save(self, file_name='sac_actor.pth'):
        """Save network weights to the ``./models`` directory.

        Args:
            file_name: Output filename.
        """
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        T.save(self.state_dict(), os.path.join(model_folder_path, file_name))
