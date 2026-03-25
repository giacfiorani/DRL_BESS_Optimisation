import os
import torch as T
import torch.nn.functional as F
import numpy as np
from agents.continuous_replay_buffer import ContinuousReplayBuffer
from agents.networks import SACActorNetwork, SACCriticNetwork

class SACAgent:
    def __init__(self, input_dims=103, n_actions=49, max_size=200000, 
                 lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, reward_scale=1.0):
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        # 1. The Memory
        self.memory = ContinuousReplayBuffer(max_size, input_dims, n_actions)

        # 2. The Actor (The Trader)
        self.actor = SACActorNetwork(lr, input_dims, n_actions=n_actions)
        
        # 3. The Twin Critics 
        self.critic_1 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)
        self.critic_2 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)
        
        # 4. The Target Critics (For stable learning)
        self.target_critic_1 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)
        self.target_critic_2 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)

        # Copy initial weights to target networks
        self.update_network_parameters(tau=1)

        # 5. Auto-Entropy Tuning (The dynamic exploration dial)
        # Target entropy is typically set to -dim(action_space)
        self.target_entropy = -T.tensor(n_actions, dtype=T.float32).to(self.actor.device)
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=lr)

    def choose_action(self, observation, warmup=False):
        # During the warmup phase, we just take completely random physical actions 
        # to populate the memory buffer before the brain starts learning.
        if warmup:
            action = np.random.uniform(-1, 1, self.n_actions)
            return action

        # Convert observation to tensor
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        
        # Ask the Actor for an action
        # We don't need gradients when just picking an action to execute in the env
        with T.no_grad():
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
            
        return actions.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.add_experience(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        # If no tau is specified, use the agent's default (usually 0.005)
        if tau is None:
            tau = self.tau

        # "Soft Update" - Slowly bleed the weights from the live critics into the target critics
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        # Don't learn until we have enough memories to fill a batch
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None

        # 1. Sample from Memory
        state, action, reward, new_state, done = self.memory.sample_batch(self.batch_size)

        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)

        # Scale reward (optional, helps stabilize Q-values)
        reward = reward * self.reward_scale

        # =========================================================
        # TRAIN THE CRITICS 
        # =========================================================
        with T.no_grad():
            # What does the Actor think it will do next?
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            
            # What do the Target Critics think those future actions are worth?
            q1_next = self.target_critic_1.forward(state_, next_actions).view(-1)
            q2_next = self.target_critic_2.forward(state_, next_actions).view(-1)
            
            # TWIN CRITIC TRICK: Always trust the more pessimistic CFO to prevent overconfidence
            q_next = T.min(q1_next, q2_next)
            
            # The Bellman Equation (Target Q) including the Entropy Bonus
            target_q = reward + (1 - done.int()) * self.gamma * (q_next - self.log_alpha.exp() * next_log_probs.view(-1))

        # What did our live Critics predict for the action we ACTUALLY took in the past?
        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)

        # Update both critics using Mean Squared Error
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # =========================================================
        # TRAIN THE ACTOR (The Trader)
        # =========================================================
        # We must reparameterize here so the calculus gradients can flow back through the random sample
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        # The Actor asks the live Critics: "If I do this, how much money will we make?"
        q1_new = self.critic_1.forward(state, actions).view(-1)
        q2_new = self.critic_2.forward(state, actions).view(-1)
        q_new = T.min(q1_new, q2_new)

        # The Actor's goal is to maximize profit (q_new) while maximizing entropy (log_probs)
        # PyTorch only minimizes, so we minimize the negative of that goal
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()

        self.actor.optimiser.zero_grad()
        actor_loss.backward()
        self.actor.optimiser.step()

        # =========================================================
        # AUTO-TUNE ENTROPY (The Exploration Dial)
        # =========================================================
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # =========================================================
        # SOFT UPDATE TARGET NETWORKS
        # =========================================================
        self.update_network_parameters()

        return critic_loss.item(), actor_loss.item(), self.log_alpha.exp().item()

    def save_models(self, run_dir):
        self.actor.save(os.path.join(run_dir, 'sac_actor.pth'))
        self.critic_1.save(os.path.join(run_dir, 'sac_critic_1.pth'))
        self.critic_2.save(os.path.join(run_dir, 'sac_critic_2.pth'))

    def load_models(self, run_dir):
        self.actor.load_state_dict(T.load(os.path.join(run_dir, 'sac_actor.pth')))
        self.critic_1.load_state_dict(T.load(os.path.join(run_dir, 'sac_critic_1.pth')))
        self.critic_2.load_state_dict(T.load(os.path.join(run_dir, 'sac_critic_2.pth')))