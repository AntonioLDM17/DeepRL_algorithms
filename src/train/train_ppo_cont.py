from __future__ import annotations

import argparse
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

import src.config as cfg
from src.agents.ppo.ppo_cont import PPOAgentCont
from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked
from src.environments.reward_wrapper import WalkerRewardWrapper
from src.utils import RewardLogger, save_model

ACTIVE_ENV = {}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["walker", "humanoid"], default="walker")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--use_shaping", action="store_true")

    p.add_argument("--total_steps", type=int, default=cfg.TOTAL_STEPS)
    p.add_argument("--rollout_steps", type=int, default=cfg.PPOConfig.ROLLOUT_STEPS)
    p.add_argument("--ppo_epochs", type=int, default=cfg.PPOConfig.EPOCHS)
    p.add_argument("--batch_size", type=int, default=cfg.PPOConfig.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.PPOConfig.LEARNING_RATE)
    p.add_argument("--gamma", type=float, default=cfg.PPOConfig.GAMMA)
    p.add_argument("--gae_lambda", type=float, default=cfg.PPOConfig.GAE_LAMBDA)
    p.add_argument("--clip_eps", type=float, default=cfg.PPOConfig.CLIP_EPS)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--value_coef", type=float, default=cfg.PPOConfig.VALUE_COEF)
    p.add_argument("--max_grad_norm", type=float, default=cfg.PPOConfig.MAX_GRAD_NORM)
    p.add_argument("--target_kl", type=float, default=cfg.PPOConfig.TARGET_KL)
    p.add_argument("--init_log_std", type=float, default=cfg.PPOConfig.INIT_LOG_STD)
    p.add_argument("--anneal_lr", action="store_true", default=True)
    p.add_argument("--no_anneal_lr", action="store_false", dest="anneal_lr")
    p.add_argument("--seed", type=int, default=cfg.AlgorithmConfig.SEED)
    return p.parse_args()


def set_active_env_from_task(task: str, env_id_override: str | None):
    global ACTIVE_ENV
    ACTIVE_ENV = cfg.EnvConfig.WALKER2D if task == "walker" else cfg.EnvConfig.HUMANOID
    if env_id_override is not None:
        ACTIVE_ENV = dict(ACTIVE_ENV)
        ACTIVE_ENV["env_id"] = env_id_override

    cfg.ENV_ID = ACTIVE_ENV["env_id"]
    cfg.ENV_NAME = ACTIVE_ENV["name"]
    cfg.NUM_DISCRETE_ACTIONS = ACTIVE_ENV["num_discrete_actions"]
    cfg.ACTION_SCALE = ACTIVE_ENV["action_scale"]
    cfg.MAX_EPISODE_STEPS = ACTIVE_ENV["max_episode_steps"]


def make_env(record_video: bool = False, task: str = "walker", shaping: bool = False):
    env_id = ACTIVE_ENV["env_id"]
    env_name = ACTIVE_ENV["name"]

    env = gym.make(env_id, render_mode="rgb_array")

    if shaping:
        if task == "walker":
            env = WalkerRewardWrapper(env)
        else:
            try:
                from src.environments.humanoid_reward_wrapper import (
                    HumanoidRewardWrapper,
                )

                env = HumanoidRewardWrapper(env)
            except ImportError:
                print(
                    "[WARN] HumanoidRewardWrapper not found. Continuing without reward shaping."
                )

    if record_video:
        video_path = cfg.get_video_path(env_name, "train")
        env = RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep_id: ep_id % cfg.VIDEO_FREQ_TRAIN == 0,
            name_prefix=f"{env_name}_train_ppo_continuous",
        )

    env = make_pixel_env(env, frame_stack=cfg.NetworkConfig.FRAME_STACK)
    return env


def make_buffer():
    return {
        "obs": [],
        "policy_actions": [],
        "env_actions": [],
        "logprobs": [],
        "rewards": [],
        "dones": [],
        "terminals": [],
        "values": [],
    }


def reset_env(env, seed=None):
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return obs_to_uint8_hwc_stacked(obs, image_size=cfg.NetworkConfig.IMAGE_SIZE)


def print_run_config(args):
    print("=" * 80)
    print("RUN CONFIGURATION - PPO CONTINUOUS")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Env ID: {ACTIVE_ENV['env_id']}")
    print(f"Use reward shaping: {args.use_shaping}")
    print(f"Total steps: {args.total_steps}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"PPO epochs: {args.ppo_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Anneal LR: {args.anneal_lr}")
    print(f"Gamma: {args.gamma}")
    print(f"GAE lambda: {args.gae_lambda}")
    print(f"Clip eps: {args.clip_eps}")
    print(f"Entropy coef: {args.entropy_coef}")
    print(f"Value coef: {args.value_coef}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Target KL: {args.target_kl}")
    print(f"Init log std: {args.init_log_std}")
    print(f"Seed: {args.seed}")
    print("=" * 80)


def main():
    args = parse_args()
    set_seed(args.seed)

    set_active_env_from_task(args.task, args.env_id)
    cfg.AlgorithmConfig.MODE = "ppo"
    cfg.print_config()
    print_run_config(args)

    env_name = ACTIVE_ENV["name"]
    log_dir = f"{env_name}_PPO_CONT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")

    env = make_env(
        task=args.task,
        shaping=args.use_shaping,
        record_video=cfg.LoggingConfig.RECORD_VIDEO_TRAIN,
    )

    if not isinstance(env.action_space, Box):
        raise TypeError(
            f"PPO continuo necesita action_space Box, pero recibi� {type(env.action_space).__name__}. "
            "Aseg�rate de no envolver el entorno con un wrapper discreto."
        )

    obs = reset_env(env, seed=args.seed)
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action_dim = int(np.prod(env.action_space.shape))

    agent = PPOAgentCont(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        input_channels=obs.shape[-1],
        input_size=cfg.NetworkConfig.IMAGE_SIZE,
        device=cfg.DEVICE,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        init_log_std=args.init_log_std,
    )

    logger = RewardLogger(window=cfg.LoggingConfig.REWARD_WINDOW)
    buffer = make_buffer()

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    rollout_last_obs = obs
    rollout_last_terminated = False

    print("[INFO] Training continuous PPO...")
    print("=" * 80)

    for step in range(1, args.total_steps + 1):
        if args.anneal_lr:
            frac = 1.0 - (step - 1.0) / float(args.total_steps)
            current_lr = frac * args.lr
            agent.set_learning_rate(current_lr)
        else:
            current_lr = args.lr

        env_action, policy_action, logprob, value, action_std = agent.act(obs)

        env_next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = bool(terminated or truncated)
        terminal = bool(terminated)
        processed_next_obs = obs_to_uint8_hwc_stacked(
            env_next_obs, image_size=cfg.NetworkConfig.IMAGE_SIZE
        )

        buffer["obs"].append(obs)
        buffer["policy_actions"].append(policy_action.copy())
        buffer["env_actions"].append(env_action.copy())
        buffer["logprobs"].append(logprob)
        buffer["rewards"].append(float(reward))
        buffer["dones"].append(done)
        buffer["terminals"].append(terminal)
        buffer["values"].append(float(value))

        rollout_last_obs = processed_next_obs
        rollout_last_terminated = terminal

        episode_reward += float(reward)
        episode_length += 1
        obs = processed_next_obs

        if done:
            episode_count += 1
            logger.log_reward(episode_reward)
            writer.add_scalar("Reward/Episode", episode_reward, episode_count)
            writer.add_scalar("Reward/Mean", logger.mean_reward(), episode_count)
            writer.add_scalar("Episode/Length", episode_length, episode_count)
            writer.add_scalar("Episode/EndStep", step, episode_count)
            writer.add_scalar("Stats/ActionStdStep", action_std, step)

            obs = reset_env(env)
            episode_reward = 0.0
            episode_length = 0

        if step % args.rollout_steps == 0:
            last_value = (
                0.0 if rollout_last_terminated else agent.get_value(rollout_last_obs)
            )

            metrics = agent.update(
                buffer,
                last_value=last_value,
                epochs=args.ppo_epochs,
                batch_size=args.batch_size,
            )

            logger.log_loss(metrics["loss"])
            writer.add_scalar("Loss/PPO", metrics["loss"], step)
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], step)
            writer.add_scalar("Loss/Critic", metrics["critic_loss"], step)
            writer.add_scalar("Stats/Entropy", metrics["entropy"], step)
            writer.add_scalar("Stats/ApproxKL", metrics["approx_kl"], step)
            writer.add_scalar("Stats/ClipFraction", metrics["clip_fraction"], step)
            writer.add_scalar(
                "Stats/ExplainedVariance", metrics["explained_variance"], step
            )
            writer.add_scalar("Stats/ActionStd", metrics["action_std"], step)
            writer.add_scalar("Stats/EarlyStop", metrics["early_stop"], step)
            writer.add_scalar("Stats/LearningRate", current_lr, step)
            writer.add_scalar("Loss/Mean", logger.mean_loss(), step)

            buffer = make_buffer()

            print(
                f"[UPDATE] step={step} "
                f"lr={current_lr:.6e} "
                f"loss={metrics['loss']:.4f} "
                f"actor={metrics['actor_loss']:.4f} "
                f"critic={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f} "
                f"kl={metrics['approx_kl']:.6f} "
                f"clipfrac={metrics['clip_fraction']:.4f} "
                f"expl_var={metrics['explained_variance']:.4f} "
                f"std={metrics['action_std']:.4f} "
                f"early_stop={int(metrics['early_stop'])}"
            )

        if step % cfg.SAVE_FREQ == 0:
            save_model(
                agent,
                path=cfg.get_checkpoint_path(env_name),
                step=step,
                prefix=env_name,
            )
            print(f"[CHECKPOINT] Model saved at step {step}")

    if len(buffer["obs"]) > 0:
        last_value = (
            0.0 if rollout_last_terminated else agent.get_value(rollout_last_obs)
        )
        metrics = agent.update(
            buffer,
            last_value=last_value,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        writer.add_scalar("Loss/PPO", metrics["loss"], args.total_steps)
        writer.add_scalar(
            "Stats/ExplainedVariance", metrics["explained_variance"], args.total_steps
        )
        writer.add_scalar("Stats/ActionStd", metrics["action_std"], args.total_steps)

    save_model(
        agent,
        path=cfg.get_checkpoint_path(env_name),
        step=args.total_steps,
        prefix=env_name,
    )
    print(f"[CHECKPOINT] Final model saved at step {args.total_steps}")

    env.close()
    writer.close()
    print("\n[INFO] Training finished!")
    print(f"[INFO] TensorBoard logs: runs/{log_dir}")
    print(f"[INFO] Checkpoints: {cfg.get_checkpoint_path(env_name)}")
    print(f"[INFO] Video recordings: {cfg.get_video_path(env_name, 'train')}")


if __name__ == "__main__":
    main()
