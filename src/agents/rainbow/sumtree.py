import numpy as np


class SumTree:
    """
    SumTree for Prioritized Experience Replay.

    Tree structure:
        parent = (i - 1) // 2
        left   = 2 * i + 1
        right  = 2 * i + 2

    Leaves contain priorities.
    """

    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity

        # tree nodes (internal + leaves)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

        # data pointer
        self.write = 0
        self.size = 0

    def total(self) -> float:
        """Total priority (root node)."""
        return float(self.tree[0])

    def add(self, priority: float):
        """Add priority and return leaf index."""
        leaf = self.write + self.capacity - 1
        self.update(leaf, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return leaf

    def update(self, leaf_idx: int, priority: float):
        """Update priority and propagate change upward."""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        parent = (leaf_idx - 1) // 2

        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s: float):
        """
        Retrieve leaf index for prefix sum value s.

        Returns:
            leaf_idx
            priority
            data_index
        """
        idx = 0

        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                leaf = idx
                break

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_index = leaf - (self.capacity - 1)

        return leaf, float(self.tree[leaf]), data_index

    def batch_get(self, values):
        """
        Vectorized sampling helper (optional but useful).
        """
        results = [self.get(v) for v in values]
        idxs, ps, data_idxs = zip(*results)
        return np.array(idxs), np.array(ps), np.array(data_idxs)