import numpy as np
import random


class SumTree:
    """Binary segment tree for O(log n) priority-weighted sampling.

    The tree stores summed priorities in internal nodes; leaf nodes hold
    individual transition priorities. This enables stratified sampling
    proportional to priority in O(log n) per sample.
    """

    def __init__(self, capacity):
        """Initialise the sum tree.

        Args:
            capacity: Number of leaf nodes (must equal the buffer capacity).
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def update(self, tree_idx, p):
        """Update the priority of one leaf and propagate the change upward.

        Args:
            tree_idx: Index of the leaf node in the flat tree array.
            p: New priority value.
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        """Return the leaf whose cumulative priority interval contains s.

        Args:
            s: Scalar in [0, total_priority].

        Returns:
            Tuple of (tree_idx, priority, data_idx).
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return (tree_idx, self.tree[tree_idx], data_idx)

    def _retrieve(self, tree_idx, s):
        left  = 2 * tree_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return the sum of all priorities (root node value)."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer (Schaul et al., 2016).

    Transitions are sampled with probability proportional to their TD-error
    priority raised to exponent α. Importance-sampling weights correct for the
    non-uniform sampling distribution during gradient updates.
    """

    def __init__(self, max_size, obs_dim=103, alpha=0.6):
        """Initialise the prioritized replay buffer.

        Args:
            max_size: Maximum buffer capacity.
            obs_dim: Dimensionality of the observation vector.
            alpha: Priority exponent controlling how much prioritisation is
                applied. α = 0 gives uniform sampling; α = 1 is fully
                proportional to priority.
        """
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0

        self.alpha        = alpha
        self.tree         = SumTree(max_size)
        self.max_priority = 1.0

        self.states      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(max_size,            dtype=np.int64)
        self.rewards     = np.zeros(max_size,            dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(max_size,            dtype=np.bool_)

    def add_experience(self, state, action, reward, next_state, done):
        """Add a transition with maximum current priority.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Scalar reward received.
            next_state: Subsequent observation.
            done: Episode termination flag.
        """
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
        """Sample a prioritised minibatch with importance-sampling weights.

        Args:
            batch_size: Number of transitions to sample.
            beta: Importance-sampling exponent that anneals bias correction
                from 0 (no correction) toward 1 (full correction).

        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
            tree_idxs, IS_weights).
        """
        batch_idxs = np.zeros(batch_size, dtype=np.int32)
        tree_idxs  = np.zeros(batch_size, dtype=np.int32)
        IS_weights = np.zeros(batch_size, dtype=np.float32)

        segment = self.tree.total() / batch_size

        # Minimum priority among stored transitions for IS normalisation.
        start    = self.tree.capacity - 1
        min_prob = np.min(self.tree.tree[start : start + self.size]) / self.tree.total()
        max_weight = (min_prob * self.size) ** (-beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.get(s)

            batch_idxs[i] = data_idx
            tree_idxs[i]  = tree_idx

            prob          = priority / self.tree.total()
            weight        = (prob * self.size) ** (-beta)
            IS_weights[i] = weight / max_weight  # normalise by max weight

        return (
            self.states[batch_idxs],
            self.actions[batch_idxs],
            self.rewards[batch_idxs],
            self.next_states[batch_idxs],
            self.dones[batch_idxs],
            tree_idxs,
            IS_weights,
        )

    def update_priorities(self, tree_idxs, td_errors):
        """Update transition priorities after a gradient step.

        Args:
            tree_idxs: Array of tree indices returned by ``sample_batch``.
            td_errors: Array of absolute TD errors for the sampled transitions.
        """
        # Add a small constant to prevent zero priority.
        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha

        for tree_idx, p in zip(tree_idxs, priorities):
            self.tree.update(tree_idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size
