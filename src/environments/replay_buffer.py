import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, size, imgs_shape, obs_shape, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Store images as uint8 to save memory
        self.imgs = np.zeros((size, *imgs_shape), dtype=np.uint8)
        self.next_imgs = np.zeros((size, *imgs_shape), dtype=np.uint8)
        self.obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
    
    def add(self, imgs, obs, action, reward, next_imgs, next_obs, done):
        self.imgs[self.ptr] = imgs
        self.next_imgs[self.ptr] = next_imgs
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0
    
    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)
        
        imgs = torch.tensor(self.imgs[idx], device=self.device).float()
        next_imgs = torch.tensor(self.next_imgs[idx], device=self.device).float()
        obs = torch.tensor(self.obs[idx], device=self.device).float()
        next_obs = torch.tensor(self.next_obs[idx], device=self.device).float()
        
        return (
            imgs,
            obs,
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device),
            next_imgs,
            next_obs,
            torch.tensor(self.dones[idx], device=self.device),
        )