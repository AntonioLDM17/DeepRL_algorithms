# humanoid_reward_wrapper.py
import gymnasium as gym
import numpy as np

from src.environments.humanoid_wrapper import DiscreteHumanoidWrapper


class HumanoidRewardWrapper(gym.Wrapper):
    """
    Reward shaping ligero para Humanoid:

        reward = orig_reward + vel_coeff * x_velocity

    Nota: en Humanoid, la posición x/y se expone en info["x_position"], info["y_position"]
    (aunque estén excluidas de la observación por defecto)
    """

    def __init__(self, env: DiscreteHumanoidWrapper, vel_coeff: float = 0.1):
        super().__init__(env)
        self.vel_coeff = float(vel_coeff)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        pelvis_id = self.env.unwrapped.model.body("pelvis").id
        height = float(self.env.unwrapped.data.xpos[pelvis_id][2])
        x_velocity = float(info.get("x_velocity", 0.0))
 
        # 1. forward speed
        reward: float = min(max(x_velocity, -2.0), 2.0)

        # 2. stay upright (height bonus)

        height_bonus: float =  height - 0.6
        reward += 2 * height_bonus
    
        # 3. survival bonus
        if not terminated:
            reward += 0.3
        else:
            # 4. death penalty
            reward -= 5.0
        info["original_reward"] = orig_reward
        return obs, reward, terminated, truncated, info    
    
        
        