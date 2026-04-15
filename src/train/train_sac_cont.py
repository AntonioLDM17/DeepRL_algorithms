# train_sac_cont.py
"""
Training entry-point for continuous SAC on pixel-based MuJoCo environments.
No action discretisation — actions are raw continuous vectors.

Usage
-----
    python train_sac_cont.py --task walker
    python train_sac_cont.py --task humanoid --no_auto_entropy --alpha_init 0.1
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from src.config import *

from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.humanoid_reward_wrapper import HumanoidRewardWrapper
from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked

from src.agents.sac.sac_cont_replay import ContinuousReplayBuffer
from src.agents.sac.sac_cont_agent import SACAgentCont

from src.utils import RewardLogger, save_model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Train continuous SAC on a pixel-based MuJoCo task."
    )
    p.add_argument("--task", choices=["walker", "humanoid"], default="walker")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--use_shaping", action="store_true")
    p.add_argument("--no_auto_entropy", action="store_true")
    p.add_argument("--alpha_init", type=float, default=None)
    return p.parse_args()

def _defaults_for_task(task: str) -> dict:
    if task == "walker":
        return EnvConfig.WALKER2D
    if task == "humanoid":
        return EnvConfig.HUMANOID
    raise ValueError(f"Invalid task: {task}")


def make_env(
    task: str = "walker",
    env_id_override: str | None = None,
    use_shaping: bool = False,
    record_video: bool = False,
    run_suffix: str = "",
):
    active = _defaults_for_task(task)
    env_id = active["env_id"] if env_id_override is None else env_id_override

    make_kwargs = {"render_mode": "rgb_array"}
    if task == "walker" and "Walker2d" in env_id:
        make_kwargs["healthy_angle_range"] = (-0.4, 0.4)

    env = gym.make(env_id, **make_kwargs)

    if use_shaping:
        if task == "walker":
            env = WalkerRewardWrapper(env)
        else:
            env = HumanoidRewardWrapper(env)

    if record_video:
        video_path = get_video_path(task, f"train_sac_cont_{task}{run_suffix}")
        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda ep_id: ep_id % VIDEO_FREQ_TRAIN == 0,
            name_prefix=f"{task}_train_sac_cont{run_suffix}",
        )

    # Pixel wrapper — NO discrete action wrapper
    env = make_pixel_env(env, frame_stack=NetworkConfig.FRAME_STACK)

    if not isinstance(env.action_space, Box):
        raise TypeError(
            f"Expected continuous Box action space, got {type(env.action_space).__name__}. "
            "Do not apply a discrete wrapper for continuous SAC."
        )

    return env, env_id


def main():
    AlgorithmConfig.MODE = "sac_cont"

    args = parse_args()

    if args.no_auto_entropy:
        AlgorithmConfig.SAC_AUTO_ENTROPY = False
    if args.alpha_init is not None:
        AlgorithmConfig.SAC_ALPHA_INIT = args.alpha_init

    print_config()

    seed = getattr(AlgorithmConfig, "SEED", 42)
    set_seed(seed)
    device = DEVICE

    log_dir = f"{args.task}_SAC_CONT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")

    # ── Environment ───────────────────────────────────────────────────────
    env, used_env_id = make_env(
        task=args.task,
        env_id_override=args.env_id,
        use_shaping=args.use_shaping,
        record_video=LoggingConfig.RECORD_VIDEO_TRAIN,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)

    obs, _ = env.reset()
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

    action_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    img_shape = imgs.shape  # (H, W, C)
    in_channels = img_shape[2]

    print("\n[SAC-Cont Setup]")
    print(f"TASK            : {args.task}")
    print(f"ENV_ID          : {used_env_id}")
    print(f"action_dim      : {action_dim}")
    print(f"action_low      : {action_low}")
    print(f"action_high     : {action_high}")
    print(f"img_shape       : {img_shape}")
    print(f"in_channels     : {in_channels}")
    print(f"DEVICE          : {device}")
    print(f"AUTO_ENTROPY    : {getattr(AlgorithmConfig, 'SAC_AUTO_ENTROPY', True)}")
    print()

    
    replay = ContinuousReplayBuffer(
        capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        img_shape=img_shape,
        action_dim=action_dim,
    )

    agent = SACAgentCont(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        in_channels=in_channels,
        cfg=AlgorithmConfig,
        device=device,
    )

    logger = RewardLogger(window=LoggingConfig.REWARD_WINDOW)
    loss_buffer = deque(maxlen=100)

    ep_return = 0.0
    ep = 0

    for step in range(1, TOTAL_STEPS + 1):

        if step <= LEARNING_STARTS:
            action = env.action_space.sample()
        else:
            action = agent.act(imgs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_imgs = obs_to_uint8_hwc_stacked(
            next_obs, image_size=NetworkConfig.IMAGE_SIZE
        )

        replay.add(imgs, action, reward, next_imgs, done)

        imgs = next_imgs
        ep_return += float(reward)

        if done:
            ep += 1
            logger.log_reward(ep_return)
            writer.add_scalar("Reward/Episode", ep_return, ep)
            writer.add_scalar("Reward/MeanWindow", logger.mean_reward(), ep)

            if ep % LoggingConfig.LOG_FREQ == 0:
                print(
                    f"Episode {ep} | Step {step} | "
                    f"Return {ep_return:.2f} | "
                    f"Mean{LoggingConfig.REWARD_WINDOW} {logger.mean_reward():.2f} | "
                    f"α {agent.alpha:.4f}"
                )

            obs, _ = env.reset()
            imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
            ep_return = 0.0

        if (
            step > LEARNING_STARTS
            and replay.size >= BATCH_SIZE
            and step % TRAIN_FREQ == 0
        ):
            metrics = agent.learn(replay)
            loss_buffer.append(metrics["critic_loss"])

            writer.add_scalar("Loss/Critic", metrics["critic_loss"], step)
            writer.add_scalar("Loss/QF1", metrics["qf1_loss"], step)
            writer.add_scalar("Loss/QF2", metrics["qf2_loss"], step)
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], step)
            writer.add_scalar("Loss/AlphaLoss", metrics["alpha_loss"], step)
            writer.add_scalar("SAC/Alpha", metrics["alpha"], step)

            if (
                step % (TRAIN_FREQ * 100) == 0
                and len(loss_buffer) == loss_buffer.maxlen
            ):
                print(
                    f"[TRAIN] step={step} "
                    f"avg_critic={np.mean(loss_buffer):.4f} "
                    f"α={agent.alpha:.4f}"
                )

        if step % SAVE_FREQ == 0:
            save_model(
                agent,
                path=get_checkpoint_path(args.task),
                step=step,
                prefix=f"{args.task}_sac_cont",
            )

    env.close()

    save_model(
        agent,
        path=get_checkpoint_path(args.task),
        step=TOTAL_STEPS,
        prefix=f"{args.task}_sac_cont",
    )

    results_path = RESULTS_DIR / args.task / get_algo_name()
    results_path.mkdir(parents=True, exist_ok=True)
    logger.plot(save_path=str(results_path / f"{args.task}_sac_cont_training.png"))

    writer.close()

    print("\n" + "=" * 80)
    print("[INFO] Continuous SAC training completed")
    print(f"[INFO] Checkpoints : {get_checkpoint_path(args.task)}")
    print(f"[INFO] TensorBoard : runs/{log_dir}")
    print(f"[INFO] Plots       : {results_path}")


if __name__ == "__main__":
    main()
