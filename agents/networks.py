import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions.normal import Normal

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        # Neural Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        # Optimizer 
        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        # Loss function
        # (Huber Loss) because it prevents crazy gradients if the agent makes a huge mistake early on.
        self.loss = nn.SmoothL1Loss()

        # Device mapping (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    # Handling forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values 

    def save(self, file_name='networks.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)

class DuelingDQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingDQN, self).__init__()
        
        # 1. Shared Feature Layer
        self.feature_layer = nn.Linear(input_dims, fc1_dims)
        
        # 2. The Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        # 3. The Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        # 4. Boilerplate (Optimizer, Loss, Device)
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        
        # Mac M4 Pro friendly device setup!
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
            
        self.to(self.device)

    def forward(self, state):
        features = F.relu(self.feature_layer(state))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def save(self, file_name='networks.pth'):
        model_folder_path = './models' 
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)

class SACCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims=103, n_actions=49, fc1_dims=256, fc2_dims=256):
        super(SACCriticNetwork, self).__init__()
        
        # 1. Neural Network Layers (State + Action concatenated)
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # 2. Boilerplate (Optimizer, Loss, Device)
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss() # SAC usually uses MSE for the critic
        
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
            
        self.to(self.device)

    def forward(self, state, action):
        q_value = T.cat([state, action], dim=1)
        q_value = F.relu(self.fc1(q_value))
        q_value = F.relu(self.fc2(q_value))
        q_value = self.q(q_value)
        return q_value

    def save(self, file_name='sac_critic.pth'):
        model_folder_path = './models' 
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)

class SACActorNetwork(nn.Module):
    def __init__(self, lr, input_dims=103, n_actions=49, fc1_dims=256, fc2_dims=256):
        super(SACActorNetwork, self).__init__()

        self.reparam_noise = 1e-6
        
        # 1. Neural Network Layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        # Two heads: Mean and Standard Deviation
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        # 2. Boilerplate
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
            
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        log_std = self.log_std(prob)
        
        # Clamp log_std to prevent infinite explosion
        log_std = T.clamp(log_std, min=-20, max=2)
        std = T.exp(log_std)

        return mu, std

    def sample_normal(self, state, reparameterize=True):
        mu, std = self.forward(state)
        probabilities = Normal(mu, std)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save(self, file_name='sac_actor.pth'):
        model_folder_path = './models' 
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)
