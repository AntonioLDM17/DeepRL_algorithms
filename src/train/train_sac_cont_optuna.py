import optuna
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.train.train_sac_cont import make_env, set_seed
from src.agents.sac.sac_cont_agent import SACAgentCont
from src.agents.sac.sac_cont_replay import ContinuousReplayBuffer
from src.environments.image_wrapper import obs_to_uint8_hwc_stacked
from src.config import *


def run_trial(trial):

    writer = SummaryWriter(
        log_dir=f"runs/optuna_sac/trial_{trial.number}", flush_secs=5
    )

    # ---- Hiperparámetros ----
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.005, 0.02)
    alpha_init = trial.suggest_float("alpha_init", 1e-3, 0.2, log=True)

    AlgorithmConfig.LR = lr
    AlgorithmConfig.GAMMA = gamma
    AlgorithmConfig.TAU = tau
    AlgorithmConfig.SAC_ALPHA_INIT = alpha_init

    seed = 42
    set_seed(seed)
    device = DEVICE

    env, _ = make_env(task="walker", record_video=False)
    obs, _ = env.reset(seed=seed)
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

    action_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.reshape(-1)
    action_high = env.action_space.high.reshape(-1)

    replay = ContinuousReplayBuffer(
        capacity=50000,
        batch_size=64,
        device=device,
        img_shape=imgs.shape,
        action_dim=action_dim,
    )

    agent = SACAgentCont(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        in_channels=imgs.shape[2],
        cfg=AlgorithmConfig,
        device=device,
    )

    total_steps = 1000000
    returns = []
    ep_return = 0

    for step in range(1, total_steps + 1):

        if step < 1000:
            action = env.action_space.sample()
        else:
            action = agent.act(imgs)

        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        next_imgs = obs_to_uint8_hwc_stacked(
            next_obs, image_size=NetworkConfig.IMAGE_SIZE
        )

        replay.add(imgs, action, reward, next_imgs, done)

        imgs = next_imgs
        ep_return += reward

        writer.add_scalar("reward/step", reward, step)

        if done:
            returns.append(ep_return)

            writer.add_scalar("reward/episode", ep_return, len(returns))

            writer.flush()

            ep_return = 0
            obs, _ = env.reset()
            imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

        if step > 1000 and replay.size >= 64:
            metrics = agent.learn(replay)

            writer.add_scalar("loss/critic", metrics["critic_loss"], step)
            writer.add_scalar("loss/actor", metrics["actor_loss"], step)
            writer.add_scalar("sac/alpha", metrics["alpha"], step)

    env.close()

    writer.flush()
    writer.close()

    if len(returns) < 5:
        return -1e6

    return np.mean(returns[-5:])


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=10)

    print("\nBest trial:")
    print(study.best_trial.params)
