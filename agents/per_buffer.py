import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2*capacity -1)
    
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def get(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return (tree_idx, self.tree[tree_idx], data_idx)

    def _retrieve(self, tree_idx, s):
        left = 2 * tree_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:

    def __init__(self, max_size, obs_dim=103, alpha=0.6):
        self.max_size = max_size
        self.ptr = 0       # write pointer
        self.size = 0      # current fill level

        #PER Parameters 
        self.alpha = alpha
        self.tree = SumTree(max_size)
        self.max_priority = 1.0 # Track the highest priority ever seen

        self.states      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(max_size,            dtype=np.int64)
        self.rewards     = np.zeros(max_size,            dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(max_size,            dtype=np.bool_)


    def add_experience(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done
        
        tree_idx = self.ptr + self.tree.capacity - 1
        self.tree.update(tree_idx, self.max_priority)
        
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, beta=0.4):
        batch_idxs = np.zeros(batch_size, dtype=np.int32)
        tree_idxs = np.zeros(batch_size, dtype=np.int32)
        IS_weights = np.zeros(batch_size, dtype=np.float32)

        segment = self.tree.total() / batch_size
        
        # To calculate IS weights, we need the minimum probability in the tree
        # We handle the case where size is not yet max_size to avoid dividing by 0
        start = self.tree.capacity - 1
        min_prob = np.min(self.tree.tree[start : start + self.size]) / self.tree.total()
        max_weight = (min_prob * self.size) ** (-beta)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a,b)

            tree_idx, priority, data_idx = self.tree.get(s)

            batch_idxs[i] = data_idx
            tree_idxs[i] = tree_idx

            # Calculate Importance Sampling weight
            prob = priority / self.tree.total()
            weight = (prob * self.size) ** (-beta)
            IS_weights[i] = weight / max_weight  # Normalize by max weight for stability

        return (
                self.states[batch_idxs],
                self.actions[batch_idxs],
                self.rewards[batch_idxs],
                self.next_states[batch_idxs],
                self.dones[batch_idxs],
                tree_idxs,
                IS_weights

            )

    def update_priorities(self, tree_idxs, td_errors):
        # Add tiny epsilon to avoid 0 priority
        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha

        for tree_idx, p in zip(tree_idxs, priorities):
            self.tree.update(tree_idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size

