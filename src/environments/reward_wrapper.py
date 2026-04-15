import gymnasium as gym
import numpy as np

from src.environments.walker_wrapper import DiscreteWalkerWrapper


class WalkerRewardWrapper(gym.Wrapper):
    """
    Reward shaping para Walker2d:

    reward = forward_progress + healthy_bonus - ctrl_cost

    - forward_progress: velocidad en x
    - healthy_bonus: bonus por estar vivo (no terminado)
    - ctrl_cost: penalizaciÃ³n por acciones grandes
    """

    def __init__(self, env: DiscreteWalkerWrapper):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        reward = orig_reward + 0.1 * self.env.unwrapped.data.qvel[0]

        return obs, reward, terminated, truncated, info
