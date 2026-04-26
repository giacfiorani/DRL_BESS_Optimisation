import numpy as np


class ContinuousReplayBuffer:
    """Uniform experience replay buffer for continuous action spaces.

    Actions are stored as float32 vectors rather than integer indices.
    """

    def __init__(self, max_size, obs_dim=103, act_dim=3):
        """Initialise the replay buffer.

        Args:
            max_size: Maximum buffer capacity.
            obs_dim: Dimensionality of the observation vector.
            act_dim: Dimensionality of the continuous action vector.
        """
        self.max_size = max_size
        self.ptr  = 0  # write pointer
        self.size = 0  # current fill level

        self.states      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards     = np.zeros(max_size,            dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(max_size,            dtype=np.bool_)

    def add_experience(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state: Current observation.
            action: Continuous action vector.
            reward: Scalar reward received.
            next_state: Subsequent observation.
            done: Episode termination flag.
        """
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """Sample a uniformly random minibatch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size
