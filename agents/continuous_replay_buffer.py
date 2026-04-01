import numpy as np

class ContinuousReplayBuffer:
    """
    Uniform experience replay buffer backed by pre-allocated NumPy arrays.
    Adapted for Continuous Action Spaces (Float32).
    """

    def __init__(self, max_size, obs_dim=103, act_dim=3):
        self.max_size = max_size
        self.ptr = 0       # write pointer
        self.size = 0      # current fill level

        self.states      = np.zeros((max_size, obs_dim), dtype=np.float32)
        
        #  Action array 3D and Float32
        self.actions     = np.zeros((max_size, act_dim), dtype=np.float32)
        
        self.rewards     = np.zeros(max_size,            dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(max_size,            dtype=np.bool_)

    def add_experience(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
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