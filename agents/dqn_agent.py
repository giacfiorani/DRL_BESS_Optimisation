import torch as T
import numpy as np
from agents.networks import DeepQNetwork
from agents.replay_buffer import ReplayBuffer
import copy


class DQNAgent():

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                        max_mem_size = 100000, eps_min =0.01, eps_dec=1e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.Q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_target = copy.deepcopy(self.Q_eval)
        self.learn_step_counter = 0
        self.target_update_frequency = 5000 #update target every 5000 steps.

        self.memory = ReplayBuffer(max_size=max_mem_size, obs_dim=input_dims)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add_experience(state, action, reward, next_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            # EXPLORE: Pick any of the 5,808 actions at random
            action = np.random.randint(self.n_actions)
        else:
            # EXPLOIT: Use the brain
            with T.no_grad():
                # Zero-copy: from_numpy shares memory when obs is already float32 C-contiguous
                state = T.from_numpy(np.asarray(observation, dtype=np.float32)).to(self.Q_eval.device).unsqueeze(0)
                q_values = self.Q_eval.forward(state)
                action = T.argmax(q_values, dim=1).item()
        return action

    def learn(self):
        # 1. Only start learning if we have enough memory
        if len(self.memory) < self.batch_size:
            return None, None, None
        
        # 2. Reset the otpimiser gradients to zero - should we?
        self.Q_eval.optimiser.zero_grad()

        # 3. Sample a batch from ReplayBuffer
        states, actions, rewards, states_, dones =  self.memory.sample_batch(self.batch_size)
        
        # as_tensor is zero-copy when the numpy array dtype already matches (float32/int64/bool)
        # and the array is C-contiguous (guaranteed by the ring buffer's fancy-index slices)
        device  = self.Q_eval.device
        states  = T.as_tensor(states,  dtype=T.float32).to(device)
        actions = T.as_tensor(actions, dtype=T.int64).to(device)
        rewards = T.as_tensor(rewards, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_, dtype=T.float32).to(device)
        dones   = T.as_tensor(dones,   dtype=T.bool).to(device)
        
        # Predicted Q-values for the actions we actually took
        q_eval = self.Q_eval.forward(states)
        q_pred = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (The Bellman Equation)
        q_next = self.Q_target.forward(states_).detach()
        max_q_next = T.max(q_next, dim=1)[0]

        # Mask the target if the episode is done
        expected_q_values = rewards + self.gamma * max_q_next * (~dones).float()

        assert q_pred.shape == expected_q_values.shape, \
            f"Shape mismatch before loss: q_pred={q_pred.shape}, expected={expected_q_values.shape}"
        #compute Loss Function
        loss = self.Q_eval.loss(q_pred, expected_q_values)

        # optimise the model
        loss.backward()

        # Capture pre-clip grad norm (tells you when gradients were exploding)
        total_norm = 0.0
        for p in self.Q_eval.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=10.0)
        self.Q_eval.optimiser.step()

        # Mean max Q-valuye over the batch (proxy for value estimate health)
        with T.no_grad():
            q_mean = q_eval.max(dim=1)[0].mean().item()

        # Epsilon decay logic
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

        # Target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        return loss.item(), grad_norm, q_mean