import torch
import torch.nn.functional as F
import numpy as np
from src.agents.dqn.network import DQN
from src.config import *

class DQNAgent:
    def __init__(self, num_actions, input_channels=3):
        self.device = DEVICE
        self.num_actions = num_actions
        
        self.q_net = DQN(num_actions, input_channels).to(self.device)
        self.target_net = DQN(num_actions, input_channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.steps = 0
    
    def epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * \
               np.exp(-1. * self.steps / EPS_DECAY)
    
    def act(self, imgs):
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.num_actions)
        
        # Preprocess observation (image only)
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=0)  # Add batch dimension
        imgs_tensor = torch.tensor(imgs, device=self.device).float() # [B, H, W, C]
        imgs_tensor = imgs_tensor.permute(0, 3, 1, 2) / 255.0  # (B, H, W, C) -> (B, C, H, W)
        
        with torch.no_grad():
            q_values = self.q_net(imgs_tensor)
        return q_values.argmax().item()
    
    def update(self, buffer):
        imgs, obs, actions, rewards, next_imgs, next_obs, dones = buffer.sample(BATCH_SIZE)
        
        # Normalize images
        imgs = imgs.permute(0, 3, 1, 2) / 255.0
        next_imgs = next_imgs.permute(0, 3, 1, 2) / 255.0
        
        # Current Q values
        q_values = self.q_net(imgs).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            if AlgorithmConfig.USE_DOUBLE_DQN:
                # Double DQN: use online network to select actions
                next_actions = self.q_net(next_imgs).argmax(1)
                next_q = self.target_net(next_imgs).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q = self.target_net(next_imgs).max(1)[0]
            
            target = rewards + GAMMA * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target)

        # debug prints occasionally
        if self.steps % 5000 == 0:
            try:
                print(f"[AGENT DEBUG] step={self.steps} loss={loss.item():.6f} q_mean={q_values.mean().item():.6f} target_mean={target.mean().item():.6f}")
            except Exception:
                pass
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())