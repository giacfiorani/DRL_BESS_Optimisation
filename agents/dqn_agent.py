import torch as T
import numpy as np
import random
from collections import deque

from torch.nn.functional import smooth_l1_loss
from agents.networks import DeepQNetwork
from agents.replay_buffer import ReplayBuffer
import copy


class DQNAgent():

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, lambda_ci,
                        max_mem_size = 100000, eps_min =0.01, eps_dec=1e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.lambda_ci = lambda_ci # set in env_config.py - carbon penalty weight
        self.action_space =  [i for i in range(self.n_actions)] # TO BE LOOKED AT MORE CAREFULLY
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.mem_cntr = 0 # memory counter to keep track of the first available memory point
        

        self.Q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_target = copy.deepcopy(self.Q_eval)
        self.learn_step_counter = 0
        self.target_update_frequency = 100 #update target every 100 steps.


        #call the replay buffer 
        self.memory = ReplayBuffer(max_size=max_mem_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add_experience(state, action, reward, next_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            # EXPLORE: Pick any of the 5,808 actions at random
            action = np.random.randint(self.n_actions)
        else:
            # EXPLOIT: Use the brain
            # 1. Convert to tensor, move to GPU/CPU, add batch dimension
            state = T.tensor(observation, dtype=T.float32).to(self.Q_eval.device).unsqueeze(0)
            
            # 2. Get the Q-values (predictions)
            q_values = self.Q_eval.forward(state)
            
            # 3. Find the index of the highest prediction
            action = T.argmax(q_values, dim=1).item() # we could remove the dim=1

        return action

    def learn(self):
        # 1. Only start learning if we have enough memory
        if len(self.memory) < self.batch_size:
            return
        
        # 2. Reset the otpimiser gradients to zero - should we?
        self.Q_eval.optimiser.zero_grad()

        # 3. Sample a batch from ReplayBuffer
        states, actions, rewards, states_, dones =  self.memory.sample_batch(self.batch_size)
        
        # 4. Convert NumPy arrays to Tensors and move to device
        # Tensors
        states = T.tensor(states, dtype=T.float32).to(self.Q_eval.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.Q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.Q_eval.device)
        states_ = T.tensor(states_, dtype=T.float32).to(self.Q_eval.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.Q_eval.device)
        
        # Predicted Q-values for the actions we actually took
        q_eval = self.Q_eval.forward(states)
        q_pred = q_eval.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Target Q-values (The Bellman Equation)
        q_next = self.Q_target.forward(states_).detach()
        max_q_next = T.max(q_next, dim=1)[0]

        # Mask the target if the episode is done
        q_target = rewards + self.gamma * max_q_next * (~dones).float()

        #compute Loss Function
        loss = self.Q_eval.loss(q_pred, q_target)

        # optimise the model
        loss.backward()
        self.Q_eval.optimiser.step()

        # Epsilon decay logic
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

        # Target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())





        
