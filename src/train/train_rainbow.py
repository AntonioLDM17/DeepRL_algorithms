import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from datetime import datetime
from gymnasium.wrappers import RecordVideo
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from src.config import *

from src.environments.walker_wrapper import DiscreteWalkerWrapper
from src.environments.humanoid_wrapper import DiscreteHumanoidWrapper

from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.humanoid_reward_wrapper import HumanoidRewardWrapper

from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked

from src.agents.rainbow.prioritized_nstep_replay import PrioritizedNStepReplay
from src.agents.rainbow.rainbow_agent import RainbowAgent

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
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["walker", "humanoid"], default="walker")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--num_bins", type=int, default=3)
    p.add_argument("--action_scale", type=float, default=None)
    p.add_argument(
        "--no_per",
        action="store_true",
        help="Disable Prioritized Experience Replay (overrides AlgorithmConfig.USE_PER).",
    )
    p.add_argument(
        "--no_distributional",
        action="store_true",
        help="Disable distributional (C51) update (overrides AlgorithmConfig.USE_DISTRIBUTIONAL).",
    )

    return p.parse_args()


def _defaults_for_task(task: str):

    if task == "walker":
        return EnvConfig.WALKER2D

    if task == "humanoid":
        return EnvConfig.HUMANOID

    raise ValueError(f"Invalid task: {task}")


def make_env(
    record_video: bool = False,
    task: str = "walker",
    env_id_override: str | None = None,
    num_bins: int = 3,
    action_scale: float | None = None,
    run_suffix: str = "",
):

    active = _defaults_for_task(task)

    env_id = active["env_id"] if env_id_override is None else env_id_override
    default_scale = float(active["action_scale"])

    if action_scale is None:
        action_scale = default_scale

    make_kwargs = {"render_mode": "rgb_array"}

    if task == "walker" and "Walker2d" in env_id:
        make_kwargs["healthy_angle_range"] = (-0.4, 0.4)

    env = gym.make(env_id, **make_kwargs)

    if task == "walker":
        env = DiscreteWalkerWrapper(env, num_bins=num_bins, scale=float(action_scale), include_zero=True)
        env = WalkerRewardWrapper(env)
    else:
        env = DiscreteHumanoidWrapper(env, num_bins=num_bins, scale=float(action_scale), include_zero=True)
        env = HumanoidRewardWrapper(env)

    if record_video:
        video_path = get_video_path(task, f"train_rainbow_{task}{run_suffix}")

        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda episode_id: episode_id % VIDEO_FREQ_TRAIN == 0,
            name_prefix=f"{task}_train_rainbow{run_suffix}",
        )

    env = make_pixel_env(env, frame_stack=NetworkConfig.FRAME_STACK)

    return env, env_id, float(action_scale)


def _variant_suffix(use_per: bool, use_distributional: bool) -> str:
    parts = []
    if not use_per:
        parts.append("noPER")
    if not use_distributional:
        parts.append("noDIST")
    return f"_{'_'.join(parts)}" if parts else ""


def main():

    AlgorithmConfig.MODE = "rainbow"

    args = parse_args()

    if args.no_per:
        AlgorithmConfig.USE_PER = False

    if args.no_distributional:
        AlgorithmConfig.USE_DISTRIBUTIONAL = False

    print_config()

    seed = getattr(AlgorithmConfig, "SEED", 42)
    set_seed(seed)

    device = DEVICE

    run_suffix = _variant_suffix(AlgorithmConfig.USE_PER, AlgorithmConfig.USE_DISTRIBUTIONAL)
    log_dir = f"{args.task}_RAINBOW{run_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")

    env, used_env_id, used_action_scale = make_env(
        record_video=LoggingConfig.RECORD_VIDEO_TRAIN,
        task=args.task,
        env_id_override=args.env_id,
        num_bins=args.num_bins,
        action_scale=args.action_scale,
        run_suffix=run_suffix,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)

    obs, _ = env.reset()

    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

    num_actions = env.action_space.n
    img_shape = imgs.shape

    print("\n[Rainbow Setup]")
    print(f"TASK: {args.task}")
    print(f"ENV_ID: {used_env_id}")
    print(f"num_bins: {args.num_bins}")
    print(f"action_scale: {used_action_scale}")
    print(f"num_actions: {num_actions}")
    print(f"img_shape: {img_shape}")
    print(f"DEVICE: {device}")
    print(f"USE_PER: {AlgorithmConfig.USE_PER}")
    print(f"USE_DISTRIBUTIONAL: {AlgorithmConfig.USE_DISTRIBUTIONAL}")
    print()

    per_alpha = AlgorithmConfig.PER_ALPHA if AlgorithmConfig.USE_PER else 0.0
    per_beta_start = AlgorithmConfig.PER_BETA_START if AlgorithmConfig.USE_PER else 0.0
    per_beta_frames = AlgorithmConfig.PER_BETA_FRAMES if AlgorithmConfig.USE_PER else 1

    replay = PrioritizedNStepReplay(
        capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        alpha=per_alpha,
        beta_start=per_beta_start,
        beta_frames=per_beta_frames,
        eps=AlgorithmConfig.PER_EPS,
        n_step=AlgorithmConfig.N_STEP if AlgorithmConfig.USE_N_STEP else 1,
        gamma=AlgorithmConfig.N_STEP_GAMMA if AlgorithmConfig.USE_N_STEP else GAMMA,
        img_shape=img_shape,
    )

    agent = RainbowAgent(num_actions=num_actions, cfg=AlgorithmConfig, device=device)

    logger = RewardLogger(window=LoggingConfig.REWARD_WINDOW)

    loss_buffer = deque(maxlen=100)

    ep_return = 0.0
    ep = 0

    for step in range(1, TOTAL_STEPS + 1):

        action = agent.act(imgs)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_imgs = obs_to_uint8_hwc_stacked(next_obs, image_size=NetworkConfig.IMAGE_SIZE)

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
                    f"Mean{LoggingConfig.REWARD_WINDOW} {logger.mean_reward():.2f}"
                )

            obs, _ = env.reset()

            imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)

            ep_return = 0.0

        if step > LEARNING_STARTS and replay.size >= BATCH_SIZE and step % TRAIN_FREQ == 0:

            loss, mean_sample_loss = agent.learn(replay)

            loss_buffer.append(loss)

            writer.add_scalar("Loss/DistributionalLoss", loss, step)
            writer.add_scalar("Loss/MeanSampleLoss", mean_sample_loss, step)

            if step % (TRAIN_FREQ * 100) == 0 and len(loss_buffer) == loss_buffer.maxlen:

                print(f"[TRAIN] step={step} avg_loss={np.mean(loss_buffer):.6f}")

        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if step % SAVE_FREQ == 0:

            save_model(
                agent,
                path=get_checkpoint_path(args.task),
                step=step,
                prefix=f"{args.task}{run_suffix}",
            )

    env.close()

    save_model(
        agent,
        path=get_checkpoint_path(args.task),
        step=TOTAL_STEPS,
        prefix=f"{args.task}{run_suffix}",
    )

    results_path = RESULTS_DIR / args.task / get_algo_name()
    results_path.mkdir(parents=True, exist_ok=True)

    logger.plot(save_path=str(results_path / f"{args.task}_rainbow_training{run_suffix}.png"))

    writer.close()

    print("\n" + "=" * 80)
    print("[INFO] Rainbow training completed")
    print(f"[INFO] Checkpoints: {get_checkpoint_path(args.task)}")
    print(f"[INFO] TensorBoard logs: runs/{log_dir}")
    print(f"[INFO] Plots: {results_path}")


if __name__ == "__main__":
    main()
