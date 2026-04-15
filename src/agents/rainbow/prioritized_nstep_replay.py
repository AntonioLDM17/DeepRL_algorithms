import numpy as np
import torch
from collections import deque
from src.agents.rainbow.sumtree import SumTree


def _next_power_of_two(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


class PrioritizedNStepReplay:
    """
    Prioritized Experience Replay with N-step returns.

    Stores:
        (imgs, action, reward, next_imgs, done)

    imgs are uint8 [H,W,C] stacked frames.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 2_000_000,
        eps: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
        img_shape=(84, 84, 12),
    ):
        self.capacity = _next_power_of_two(capacity)
        self.batch_size = batch_size
        self.device = device

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.frame = 1

        self.n_step = n_step
        self.gamma = gamma
        self.nstep_buffer = deque(maxlen=n_step)

        self.tree = SumTree(self.capacity)

        # replay storage
        self.imgs = np.zeros((self.capacity, *img_shape), dtype=np.uint8)
        self.next_imgs = np.zeros((self.capacity, *img_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

        self.max_priority = 1.0
        self.size = 0

    def beta_by_frame(self):
        return min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * (self.frame / self.beta_frames),
        )

    def _get_nstep_transition(self):
        """
        Build n-step transition:
        (s0, a0, R_n, s_n, done_n)
        """
        imgs0, a0, _, _, _ = self.nstep_buffer[0]

        R = 0.0
        done_n = False
        next_imgs_n = None

        for i, (_, _, r, next_imgs, done) in enumerate(self.nstep_buffer):
            R += (self.gamma ** i) * float(r)
            next_imgs_n = next_imgs

            if done:
                done_n = True
                break

        return imgs0, a0, R, next_imgs_n, done_n

    def add(self, imgs, action, reward, next_imgs, done):
        """
        Add transition to n-step buffer.
        """
        self.nstep_buffer.append((imgs, action, reward, next_imgs, done))

        if len(self.nstep_buffer) < self.n_step and not done:
            return

        imgs0, a0, Rn, next_imgs_n, done_n = self._get_nstep_transition()
        self._add_to_main(imgs0, a0, Rn, next_imgs_n, done_n)

        if done:
            while len(self.nstep_buffer) > 1:
                self.nstep_buffer.popleft()

                imgs0, a0, Rn, next_imgs_n, done_n = self._get_nstep_transition()
                self._add_to_main(imgs0, a0, Rn, next_imgs_n, done_n)

            self.nstep_buffer.clear()
        else:
            self.nstep_buffer.popleft()

    def _add_to_main(self, imgs, action, reward, next_imgs, done):

        ptr = self.tree.write

        self.imgs[ptr] = imgs
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.next_imgs[ptr] = next_imgs
        self.dones[ptr] = done

        priority = (self.max_priority + self.eps) ** self.alpha

        self.tree.add(priority)

        self.size = self.tree.size

    def sample(self):
        assert self.size >= self.batch_size

        total = self.tree.total()
        beta = self.beta_by_frame()
        self.frame += 1

        idxs = []
        priorities = []
        data_idxs = []

        segment = total / self.batch_size

        for i in range(self.batch_size):

            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)

            leaf_idx, p, data_idx = self.tree.get(s)

            idxs.append(leaf_idx)
            priorities.append(p)

            data_idx = np.clip(data_idx, 0, self.size - 1)
            data_idxs.append(data_idx)

        priorities = np.array(priorities, dtype=np.float32)

        probs = priorities / (total + 1e-8)
        probs = np.maximum(probs, 1e-8)

        weights = (self.size * probs) ** (-beta)
        weights /= weights.max() + 1e-8

        data_idxs = np.array(data_idxs)

        # faster tensor conversion
        b_imgs = torch.from_numpy(self.imgs[data_idxs]).to(self.device).float().div_(255.0)
        b_next_imgs = torch.from_numpy(self.next_imgs[data_idxs]).to(self.device).float().div_(255.0)

        b_actions = torch.from_numpy(self.actions[data_idxs]).to(self.device)
        b_rewards = torch.from_numpy(self.rewards[data_idxs]).to(self.device)
        b_dones = torch.from_numpy(self.dones[data_idxs].astype(np.float32)).to(self.device)

        b_weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        # NHWC → NCHW
        b_imgs = b_imgs.permute(0, 3, 1, 2)
        b_next_imgs = b_next_imgs.permute(0, 3, 1, 2)

        return b_imgs, b_actions, b_rewards, b_next_imgs, b_dones, b_weights, idxs

    def update_priorities(self, idxs, priorities):

        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, 1e-6)

        self.max_priority = max(self.max_priority, float(priorities.max()))

        for idx, p in zip(idxs, priorities):

            new_p = (p + self.eps) ** self.alpha
            self.tree.update(idx, new_p)