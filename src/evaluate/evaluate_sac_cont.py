from __future__ import annotations


import argparse
import json
import os
from pathlib import Path


import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo


from src.config import *
from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked
from src.agents.sac.sac_cont_agent import SACAgentCont
from src.utils import load_model, make_dir

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained continuous SAC agent.")
    p.add_argument("--task", choices=["walker"], default="walker")
    p.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Override the environment ID from config.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pth checkpoint. "
        "If omitted, the latest one in the default dir is used.",
    )
    p.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (default: LoggingConfig.NUM_EVAL_EPISODES).",
    )
    p.add_argument("--no_video", action="store_true", help="Disable video recording.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _defaults_for_task(task: str) -> dict:
    if task == "walker":
        return EnvConfig.WALKER2D
    raise ValueError(f"Invalid task: {task}")


def find_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """Return the path to the checkpoint with the highest step number, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    if not candidates:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    candidates.sort(key=_step, reverse=True)
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_eval_env(
    task: str,
    env_id: str,
    video_folder: str | Path | None,
    record_video: bool = True,
    run_suffix: str = "",
):
    """
    Build the pixel-wrapped evaluation environment.
    One video is recorded per episode (episode_trigger always True).
    """
    make_kwargs = {"render_mode": "rgb_array"}
    if task == "walker" and "Walker2d" in env_id:
        make_kwargs["healthy_angle_range"] = (-0.4, 0.4)

    env = gym.make(env_id, **make_kwargs)

    # Mirror the reward shaping used during training so evaluation scores are comparable.
    if task == "walker":
        env = WalkerRewardWrapper(env)

    if record_video and video_folder is not None:
        make_dir(str(video_folder))
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda ep_id: True,  # record every episode
            name_prefix=f"eval_{task}_sac_cont{run_suffix}",
        )

    # Pixel wrapper — same settings as training, NO discrete wrapper.
    env = make_pixel_env(env, frame_stack=NetworkConfig.FRAME_STACK)
    return env


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(agent: SACAgentCont, env, num_episodes: int) -> dict:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

        ep_return = 0.0
        step_count = 0
        done = False

        while not done:
            # Deterministic policy: tanh(mean), no noise
            action = agent.act(imgs, deterministic=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            imgs = obs_to_uint8_hwc_stacked(
                next_obs, image_size=NetworkConfig.IMAGE_SIZE
            )
            ep_return += float(reward)
            step_count += 1

        episode_rewards.append(ep_return)
        episode_lengths.append(step_count)
        print(
            f"Episode {ep + 1}/{num_episodes} | "
            f"Return: {ep_return:.2f} | "
            f"Steps: {step_count}"
        )

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS — Continuous SAC")
    print(f"{'=' * 60}")
    print(f"Mean Reward : {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min  Reward : {float(np.min(episode_rewards)):.2f}")
    print(f"Max  Reward : {float(np.max(episode_rewards)):.2f}")
    print(f"Mean Length : {mean_length:.1f} steps")
    print(f"{'=' * 60}\n")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": mean_length,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(results: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Evaluation results saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    num_episodes = args.num_episodes or LoggingConfig.NUM_EVAL_EPISODES

    active = _defaults_for_task(args.task)
    env_id = active["env_id"] if args.env_id is None else args.env_id

    device = DEVICE

    print("=" * 80)
    print("CONTINUOUS SAC AGENT EVALUATION")
    print("=" * 80)
    print(f"Task          : {args.task}")
    print(f"Env ID        : {env_id}")
    print(f"Frame Stack   : {NetworkConfig.FRAME_STACK}")
    print(f"Image Size    : {NetworkConfig.IMAGE_SIZE}x{NetworkConfig.IMAGE_SIZE}")
    print(f"Device        : {device}")
    print(f"Num Episodes  : {num_episodes}")
    print("=" * 80)

    # ── Environment ───────────────────────────────────────────────────────
    video_folder = (
        None
        if args.no_video
        else get_video_path(args.task, f"eval_sac_cont_{args.task}")
    )

    env = make_eval_env(
        task=args.task,
        env_id=env_id,
        video_folder=video_folder,
        record_video=not args.no_video,
    )

    obs, _ = env.reset()
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

    action_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    img_shape = imgs.shape  # (H, W, C)
    in_channels = img_shape[2]

    print(f"\nimg_shape   : {img_shape}")
    print(f"in_channels : {in_channels}")
    print(f"action_dim  : {action_dim}\n")

    # ── Agent ─────────────────────────────────────────────────────────────
    agent = SACAgentCont(
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        in_channels=in_channels,
        cfg=AlgorithmConfig,
        device=device,
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = get_checkpoint_path(args.task)
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    model_loaded = False
    if checkpoint_path:
        try:
            load_model(agent, checkpoint_path)
            print(f"✓ Loaded checkpoint: {checkpoint_path}\n")
            model_loaded = True
        except Exception as exc:
            print(f"✗ Failed to load checkpoint: {exc}\n")

    if not model_loaded:
        print("⚠  WARNING: No trained checkpoint found or loaded.")
        print("   Evaluating with RANDOMLY INITIALISED weights.")
        print("   Run 'python train_sac_cont.py' first to train the agent.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"Starting evaluation ({num_episodes} episodes) …")
    print("=" * 60)

    results = evaluate(agent, env, num_episodes=num_episodes)

    env.close()

    # Attach metadata
    results["checkpoint"] = str(checkpoint_path) if checkpoint_path else None
    results["task"] = args.task
    results["env_id"] = env_id

    # ── Save results ──────────────────────────────────────────────────────
    results_dir = RESULTS_DIR / args.task / get_algo_name()
    save_results(results, results_dir / f"{args.task}_sac_cont_evaluation.json")

    if not args.no_video and video_folder:
        print(f"✓ Videos saved  : {video_folder}")
    print(f"✓ Results saved : {results_dir}")


if __name__ == "__main__":
    main()
