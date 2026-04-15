import optuna
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

import src.config as cfg
from src.train.train_ppo_cont import (
    make_env,
    set_seed,
    set_active_env_from_task,
    reset_env,
    make_buffer,
)
from src.agents.ppo.ppo_cont import PPOAgentCont
from src.environments.image_wrapper import obs_to_uint8_hwc_stacked


def run_trial(trial):

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=f"runs/optuna_ppo_trial_{trial.number}")

    # ---- Hiperparámetros ----
    lr = trial.suggest_loguniform("lr", 1e-5, 3e-4)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.98)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    entropy_coef = trial.suggest_loguniform("entropy_coef", 1e-4, 0.05)

    # ---- Setup ----
    seed = 42
    set_seed(seed)
    device = cfg.DEVICE

    set_active_env_from_task("walker", None)

    env = make_env(task="walker", shaping=False, record_video=False)
    obs = reset_env(env, seed=seed)

    action_low = env.action_space.low.reshape(-1)
    action_high = env.action_space.high.reshape(-1)
    action_dim = int(np.prod(env.action_space.shape))

    agent = PPOAgentCont(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        input_channels=obs.shape[-1],
        input_size=cfg.NetworkConfig.IMAGE_SIZE,
        device=device,
        lr=lr,
        gamma=gamma,
        lam=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        init_log_std=-0.5,
    )

    # ---- Loop corto ----
    total_steps = 1000000
    rollout_steps = 512

    buffer = make_buffer()

    returns = []
    ep_return = 0

    rollout_last_obs = obs
    rollout_last_terminated = False

    for step in range(1, total_steps + 1):

        env_action, policy_action, logprob, value, action_std = agent.act(obs)

        next_obs, reward, term, trunc, _ = env.step(env_action)
        done = term or trunc

        next_imgs = obs_to_uint8_hwc_stacked(
            next_obs, image_size=cfg.NetworkConfig.IMAGE_SIZE
        )

        buffer["obs"].append(obs)
        buffer["policy_actions"].append(policy_action)
        buffer["env_actions"].append(env_action)
        buffer["logprobs"].append(logprob)
        buffer["rewards"].append(float(reward))
        buffer["dones"].append(done)
        buffer["terminals"].append(term)
        buffer["values"].append(float(value))

        rollout_last_obs = next_imgs
        rollout_last_terminated = term

        obs = next_imgs
        ep_return += reward

        # ---- Logging ----
        writer.add_scalar("Step/Reward", reward, step)
        writer.add_scalar("Policy/ActionStd", action_std, step)

        if done:
            returns.append(ep_return)
            writer.add_scalar("Episode/Return", ep_return, len(returns))
            ep_return = 0
            obs = reset_env(env)

        # ---- Update PPO ----
        if step % rollout_steps == 0:
            last_value = (
                0.0 if rollout_last_terminated else agent.get_value(rollout_last_obs)
            )

            metrics = agent.update(
                buffer,
                last_value=last_value,
                epochs=3,
                batch_size=64,
            )

            writer.add_scalar("Loss/Total", metrics["loss"], step)
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], step)
            writer.add_scalar("Loss/Critic", metrics["critic_loss"], step)
            writer.add_scalar("Stats/Entropy", metrics["entropy"], step)
            writer.add_scalar("Stats/KL", metrics["approx_kl"], step)
            writer.add_scalar("Stats/ClipFraction", metrics["clip_fraction"], step)
            writer.add_scalar(
                "Stats/ExplainedVariance", metrics["explained_variance"], step
            )

            buffer = make_buffer()

    env.close()
    writer.close()

    # ---- Métrica Optuna ----
    if len(returns) < 5:
        return -1e6

    return np.mean(returns[-5:])


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=10)

    print("\nBest trial:")
    print(study.best_trial.params)
