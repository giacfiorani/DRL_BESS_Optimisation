import os
import torch as T
import torch.nn.functional as F
import numpy as np
from agents.continuous_replay_buffer import ContinuousReplayBuffer
from agents.networks import SACActorNetwork, SACCriticNetwork


class SACAgent:
    """Soft Actor-Critic agent with automatic entropy tuning.

    Implements SAC (Haarnoja et al., 2018) with twin critics, Polyak-averaged
    target networks, and adaptive temperature α tuned to match a target entropy
    of -dim(A) nats per step.
    """

    def __init__(
        self,
        input_dims=103,
        n_actions=3,
        max_size=200000,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        reward_scale=1.0,
        alpha_lr=1e-4,
    ):
        """Initialise the SAC agent.

        Args:
            input_dims: Dimensionality of the observation vector.
            n_actions: Number of continuous action dimensions.
            max_size: Maximum replay buffer capacity.
            lr: Learning rate for actor and critic optimisers.
            gamma: Discount factor.
            tau: Polyak averaging coefficient for soft target updates.
            batch_size: Number of transitions sampled per learning step.
            reward_scale: Multiplicative scaling applied to rewards before the
                Bellman update, used to control the effective entropy weight.
            alpha_lr: Learning rate for the entropy temperature optimiser.
        """
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.n_actions    = n_actions
        self.reward_scale = reward_scale

        self.memory = ContinuousReplayBuffer(max_size, input_dims, n_actions)

        self.actor     = SACActorNetwork(lr, input_dims, n_actions=n_actions)
        self.critic_1  = SACCriticNetwork(lr, input_dims, n_actions=n_actions)
        self.critic_2  = SACCriticNetwork(lr, input_dims, n_actions=n_actions)

        self.target_critic_1 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)
        self.target_critic_2 = SACCriticNetwork(lr, input_dims, n_actions=n_actions)

        # Initialise target networks as exact copies of the live critics.
        self.update_network_parameters(tau=1)

        # Auto-entropy tuning: target_entropy = -dim(A).
        # For a 3-D action space the expected entropy at convergence is ~2.1 nats/step.
        self.target_entropy = -T.tensor(float(n_actions), dtype=T.float32).to(self.actor.device)
        # Initialise log_alpha = 0 (α = 1.0).
        self.log_alpha  = T.tensor([0.0], requires_grad=True, device=self.actor.device)
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=alpha_lr)

    def choose_action(self, observation, warmup=False):
        """Select an action from the current stochastic policy.

        Args:
            observation: Current environment observation.
            warmup: If True, return a uniformly random action to populate the
                replay buffer before training begins.

        Returns:
            Float32 array of shape (n_actions,).
        """
        if warmup:
            return np.random.uniform(-1, 1, self.n_actions)

        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        with T.no_grad():
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def choose_action_deterministic(self, observation):
        """Return the policy mean action (no sampling noise).

        Used for stable validation and test evaluation.

        Args:
            observation: Current environment observation.

        Returns:
            Float32 array of shape (n_actions,).
        """
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        with T.no_grad():
            mu, _ = self.actor.forward(state)
            action = T.tanh(mu)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        """Add a transition to the replay buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Scalar reward received.
            state_: Subsequent observation.
            done: Episode termination flag.
        """
        self.memory.add_experience(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        """Perform a Polyak soft update of the target critic networks.

        Args:
            tau: Averaging coefficient. If None, uses ``self.tau``.
                Pass ``tau=1`` to perform a hard copy.
        """
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        """Sample a minibatch and perform one SAC update step.

        Updates critics, actor, and entropy temperature in sequence, followed
        by a soft update of the target networks.

        Returns:
            Tuple of (critic_loss, actor_loss, alpha), or (None, None, None)
            if the replay buffer contains fewer transitions than ``batch_size``.
        """
        if self.memory.size < self.batch_size:
            return None, None, None

        state, action, reward, new_state, done = self.memory.sample_batch(self.batch_size)

        reward  = T.tensor(reward,     dtype=T.float32).to(self.actor.device)
        done    = T.tensor(done,       dtype=T.bool).to(self.actor.device)
        state   = T.tensor(state,      dtype=T.float32).to(self.actor.device)
        state_  = T.tensor(new_state,  dtype=T.float32).to(self.actor.device)
        action  = T.tensor(action,     dtype=T.float32).to(self.actor.device)

        reward = reward * self.reward_scale
        alpha  = self.log_alpha.exp().detach()

        # Critic update
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            next_log_probs = next_log_probs.view(-1)

            q1_next = self.target_critic_1.forward(state_, next_actions).view(-1)
            q2_next = self.target_critic_2.forward(state_, next_actions).view(-1)

            # Clipped double-Q trick: use the pessimistic target to prevent over-estimation.
            q_next   = T.min(q1_next, q2_next)
            target_q = reward + (1 - done.int()) * self.gamma * (q_next - alpha * next_log_probs)

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)

        # Huber loss (smooth L1) instead of MSE: energy-price spikes create reward outliers
        # whose squared error dominates MSE gradients. Huber loss clips at delta=1.
        critic_1_loss = F.smooth_l1_loss(q1, target_q)
        critic_2_loss = F.smooth_l1_loss(q2, target_q)
        critic_loss   = critic_1_loss + critic_2_loss

        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # Actor update
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1_new = self.critic_1.forward(state, actions).view(-1)
        q2_new = self.critic_2.forward(state, actions).view(-1)
        q_new  = T.min(q1_new, q2_new)

        # Maximise Q while maintaining target entropy: min α log π - Q.
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor.optimiser.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimiser.step()

        # Entropy temperature update:
        #   α_loss > 0 when policy entropy < target → increase α to encourage exploration.
        #   α_loss < 0 when policy entropy > target → decrease α to sharpen the policy.
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.update_network_parameters()

        return critic_loss.item(), actor_loss.item(), self.log_alpha.exp().item()

    def save_models(self, run_dir):
        """Persist actor and critic weights to disk.

        Args:
            run_dir: Directory path in which to save model files.
        """
        self.actor.save(os.path.join(run_dir, 'sac_actor.pth'))
        self.critic_1.save(os.path.join(run_dir, 'sac_critic_1.pth'))
        self.critic_2.save(os.path.join(run_dir, 'sac_critic_2.pth'))

    def load_models(self, run_dir):
        """Load actor and critic weights from disk.

        Args:
            run_dir: Directory path from which to load model files.
        """
        self.actor.load_state_dict(T.load(os.path.join(run_dir, 'sac_actor.pth')))
        self.critic_1.load_state_dict(T.load(os.path.join(run_dir, 'sac_critic_1.pth')))
        self.critic_2.load_state_dict(T.load(os.path.join(run_dir, 'sac_critic_2.pth')))
