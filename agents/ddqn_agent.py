import torch as T
import numpy as np
from agents.networks import DeepQNetwork
from agents.replay_buffer import ReplayBuffer
import copy


class DDQNAgent:
    """Double Deep Q-Network agent.

    Decouples action selection from action evaluation to eliminate the
    maximisation bias present in standard DQN (van Hasselt et al., 2016).
    The online network selects the greedy next action; the target network
    evaluates it.
    """

    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=100000,
        eps_min=0.01,
        eps_dec=1e-5,
        replace_target_cnt=5000,
    ):
        """Initialise the DDQN agent.

        Args:
            gamma: Discount factor.
            epsilon: Initial exploration probability.
            lr: Learning rate for the Adam optimiser.
            input_dims: Dimensionality of the observation vector.
            batch_size: Number of transitions sampled per learning step.
            n_actions: Number of discrete actions.
            max_mem_size: Maximum replay buffer capacity.
            eps_min: Minimum exploration probability.
            eps_dec: Linear epsilon decrement per learning step.
            replace_target_cnt: Number of learning steps between hard target
                network updates.
        """
        self.gamma    = gamma
        self.epsilon  = epsilon
        self.lr       = lr
        self.n_actions  = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_dec    = eps_dec
        self.eps_min    = eps_min

        self.Q_eval   = DeepQNetwork(self.lr, n_actions=self.n_actions,
                                     input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_target = copy.deepcopy(self.Q_eval)
        self.learn_step_counter      = 0
        self.target_update_frequency = replace_target_cnt

        self.memory = ReplayBuffer(max_size=max_mem_size, obs_dim=input_dims)

    def store_transition(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Scalar reward received.
            next_state: Subsequent observation.
            done: Episode termination flag.
        """
        self.memory.add_experience(state, action, reward, next_state, done)

    def choose_action(self, observation):
        """Select an action using an ε-greedy policy.

        Args:
            observation: Current environment observation.

        Returns:
            Integer action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with T.no_grad():
            # Zero-copy: as_tensor shares memory when the array is float32 C-contiguous.
            state    = T.from_numpy(np.asarray(observation, dtype=np.float32)).to(self.Q_eval.device).unsqueeze(0)
            q_values = self.Q_eval.forward(state)
            return T.argmax(q_values, dim=1).item()

    def learn(self):
        """Sample a minibatch and perform one DDQN gradient update.

        Returns:
            Tuple of (loss, grad_norm, q_mean), or (None, None, None) if the
            replay buffer contains fewer transitions than ``batch_size``.
        """
        if len(self.memory) < self.batch_size:
            return None, None, None

        self.Q_eval.optimiser.zero_grad()

        states, actions, rewards, states_, dones = self.memory.sample_batch(self.batch_size)

        # as_tensor is zero-copy when the array dtype matches and is C-contiguous.
        device  = self.Q_eval.device
        states  = T.as_tensor(states,  dtype=T.float32).to(device)
        actions = T.as_tensor(actions, dtype=T.int64).to(device)
        rewards = T.as_tensor(rewards, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_, dtype=T.float32).to(device)
        dones   = T.as_tensor(dones,   dtype=T.bool).to(device)

        # Online network selects greedy next action; gradient flows through q_pred only.
        q_eval = self.Q_eval.forward(states)                              # [B, n_actions]
        q_pred = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)        # [B]

        # Target network evaluates the selected next action — no gradient required.
        with T.no_grad():
            best_actions    = self.Q_eval.forward(states_).argmax(dim=1)               # [B]
            q_next          = self.Q_target.forward(states_)                            # [B, n_actions]
            q_target_values = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)   # [B]

        # Bellman target — all tensors are [B], preventing inadvertent broadcasting.
        expected_q_values = rewards + self.gamma * q_target_values * (~dones).float()  # [B]

        assert q_pred.shape == expected_q_values.shape, (
            f"Shape mismatch before loss: q_pred={q_pred.shape}, "
            f"expected={expected_q_values.shape}"
        )

        loss = self.Q_eval.loss(q_pred, expected_q_values)
        loss.backward()

        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.Q_eval.parameters() if p.grad is not None
        ) ** 0.5

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=10.0)
        self.Q_eval.optimiser.step()

        with T.no_grad():
            q_mean = q_eval.max(dim=1)[0].mean().item()

        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        return loss.item(), total_norm, q_mean
