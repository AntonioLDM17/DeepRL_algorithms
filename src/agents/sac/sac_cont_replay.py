# sac_cont_replay.py
"""
Uniform replay buffer for continuous SAC.

The only difference from the discrete version (sac_replay.py) is that
actions are stored as float32 vectors [action_dim] instead of int64 scalars.
"""

from __future__ import annotations

import numpy as np
import torch


class ContinuousReplayBuffer:
    """
    Circular uniform replay buffer for continuous action spaces.

    Args:
        capacity   : maximum transitions
        batch_size : transitions returned per sample() call
        device     : torch device for sampled tensors
        img_shape  : HWC shape of a stacked observation, e.g. (84, 84, 12)
        action_dim : dimensionality of the continuous action vector
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device: torch.device,
        img_shape: tuple,
        action_dim: int,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        H, W, C = img_shape

        self.imgs = np.zeros((capacity, H, W, C), dtype=np.uint8)
        self.next_imgs = np.zeros((capacity, H, W, C), dtype=np.uint8)
        self.actions = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )  # ← float, not int
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(
        self,
        img: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_img: np.ndarray,
        done: bool,
    ):
        self.imgs[self._ptr] = img
        self.next_imgs[self._ptr] = next_img
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self):
        """
        Returns:
            imgs       [B, H, W, C]   float32  [0,1]  — NOTE: HWC for encoder
            actions    [B, action_dim] float32
            rewards    [B]             float32
            next_imgs  [B, H, W, C]   float32  [0,1]
            dones      [B]             float32
        """
        idxs = np.random.randint(0, self._size, size=self.batch_size)

        imgs = torch.tensor(self.imgs[idxs], dtype=torch.float32, device=self.device)
        next_imgs = torch.tensor(
            self.next_imgs[idxs], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            self.actions[idxs], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            self.rewards[idxs], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device)

        return imgs, actions, rewards, next_imgs, dones

    @property
    def size(self) -> int:
        return self._size
