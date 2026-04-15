import argparse
import random
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from gymnasium.wrappers import RecordVideo
from collections import deque
from torch.utils.tensorboard import SummaryWriter
 
from src.agents.dqn.agent import DQNAgent
from src.environments.replay_buffer import ReplayBuffer
from src.environments.walker_wrapper import DiscreteWalkerWrapper
from src.environments.humanoid_wrapper import DiscreteHumanoidWrapper
# For reward shaping
from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.humanoid_reward_wrapper import HumanoidRewardWrapper
 
from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked
 
from src.utils import RewardLogger, save_model, log_obs
from src.config import *
 
 
print_config()
 
 
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["walker", "humanoid"],
        default="walker",
    )
    p.add_argument(
        "--env_id",
        type=str,
        default=None,
    )
 
    # Por defecto shaping=True. Para desactivarlo explícitamente: --no-shaping
    p.add_argument(
        "--no-shaping",
        action="store_true",
        help="Desactiva reward shaping (por defecto está activado).",
    )
 
    p.add_argument(
        "--num_bins",
        type=int,
        default=NUM_DISCRETE_ACTIONS,
    )
    p.add_argument(
        "--action_scale",
        type=float,
        default=None,
    )
    return p.parse_args()
 
 
# ============================================================
# IMPORTANTE:
# Tu config.py decide rutas con EnvConfig.ACTIVE (ENV_NAME/ENV_ID).
# Si ACTIVE está en HUMANOID, aunque ejecutes --task walker,
# get_checkpoint_path/get_video_path/get_results_path seguirán usando humanoid.
#
# Solución:
#   En runtime, sobreescribimos ENV_NAME/ENV_ID/NUM_DISCRETE_ACTIONS/ACTION_SCALE
#   según el --task y flags.
# ============================================================
def set_active_env_from_task(task: str, env_id_override: str | None, num_bins: int, action_scale: float | None):
    global ENV_ID, ENV_NAME, NUM_DISCRETE_ACTIONS, ACTION_SCALE, MAX_EPISODE_STEPS
 
    if task == "walker":
        active = EnvConfig.WALKER2D
    else:
        active = EnvConfig.HUMANOID
 
    # Base (de dict)
    ENV_ID = active["env_id"]
    ENV_NAME = active["name"]
    NUM_DISCRETE_ACTIONS = active["num_discrete_actions"]
    ACTION_SCALE = active["action_scale"]
    MAX_EPISODE_STEPS = active["max_episode_steps"]
 
    # Overrides por CLI
    if env_id_override is not None:
        ENV_ID = env_id_override
 
    # num_bins: si lo pasas por CLI, manda
    if num_bins is not None:
        NUM_DISCRETE_ACTIONS = int(num_bins)
 
    # action_scale: si lo pasas por CLI, manda
    if action_scale is not None:
        ACTION_SCALE = float(action_scale)
 
 
def make_env(record_video=False, task="walker", env_id_override=None, shaping=True, num_bins=3, action_scale=None):
    env_id = env_id_override if env_id_override is not None else (ENV_ID if task == "walker" else "Humanoid-v5")
    make_kwargs = {"render_mode": "rgb_array"}
 
    if task == "walker" and "Walker2d" in env_id:
        make_kwargs["healthy_angle_range"] = (-0.4, 0.4)
 
    env = gym.make(env_id, **make_kwargs)
 
    if task == "walker":
        scale = ACTION_SCALE if action_scale is None else float(action_scale)
        env = DiscreteWalkerWrapper(env, num_bins=num_bins, scale=scale, include_zero=True)
        if shaping:
            env = WalkerRewardWrapper(env)
    else:
        scale = 0.4 if action_scale is None else float(action_scale)
        env = DiscreteHumanoidWrapper(env, num_bins=num_bins, scale=scale, include_zero=True)
        if shaping:
            env = HumanoidRewardWrapper(env)
 
    if record_video:
        video_path = get_video_path(task, f"train_new_{task}")
        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda episode_id: episode_id % VIDEO_FREQ_TRAIN == 0,
            name_prefix=f"{ENV_NAME}_train_dqn_new_{task}",
        )
 
    env = make_pixel_env(env, frame_stack=NetworkConfig.FRAME_STACK)
 
    return env, env_id
 
 
def main():
    args = parse_args()
    set_seed(getattr(AlgorithmConfig, "SEED", 42))
 
    # shaping por defecto ON
    shaping = not args.no_shaping
 
    # ------------------------------------------------------------
    # CLAVE: setear el "ACTIVE" efectivo según --task
    # Esto arregla:
    #  - rutas de checkpoints
    #  - rutas de vídeos
    #  - rutas de results
    #  - nombres de env en logs
    # ------------------------------------------------------------
    set_active_env_from_task(
        task=args.task,
        env_id_override=args.env_id,
        num_bins=args.num_bins,
        action_scale=args.action_scale,
    )
 
    # Opción: si quieres ver el resumen ya “bien” según el task:
    print_config()
 
    # TensorBoard: incluye env para no mezclar walker/humanoid
    log_dir = f"{ENV_NAME}_DQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")
    # =======================
 
    env, used_env_id = make_env(
        record_video=LoggingConfig.RECORD_VIDEO_TRAIN,
        task=args.task,
        env_id_override=args.env_id,
        shaping=shaping,
        num_bins=args.num_bins,
        action_scale=args.action_scale,
    )
 
    obs, info = env.reset()
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)  # uint8 (H,W,3*stack)
 
    imgs_shape = imgs.shape
    num_actions = env.action_space.n
    input_channels = imgs.shape[-1]
 
    print(f"\n[Environment Setup - NEW]")
    print(f"TASK: {args.task}")
    print(f"ENV_ID: {used_env_id}")
    print(f"ENV_NAME(active): {ENV_NAME}")
    print(f"num_bins: {args.num_bins}")
    print(f"action_scale: {args.action_scale}")
    print(f"shaping: {shaping}")
    print(f"Images shape (H,W,C*stack): {imgs_shape}")
    print(f"Input channels: {input_channels}")
    print(f"Number of actions (NEW): {num_actions}")
    print()
 
    obs_dummy = np.zeros((1,), dtype=np.float32)
 
    agent = DQNAgent(num_actions, input_channels)
    buffer = ReplayBuffer(BUFFER_SIZE, imgs_shape, obs_dummy.shape, DEVICE)
    logger = RewardLogger(window=LoggingConfig.REWARD_WINDOW)
    loss_buffer = deque(maxlen=100)
 
    episode_reward = 0.0
    episode_count = 0
 
    print("[INFO] Iniciando entrenamiento DQN (NEW wrappers)...")
    print("=" * 80)
 
    for step in range(TOTAL_STEPS):
 
        action = agent.act(imgs)
 
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
 
        next_imgs = obs_to_uint8_hwc_stacked(next_obs, image_size=NetworkConfig.IMAGE_SIZE)
 
        buffer.add(imgs, obs_dummy, action, reward, next_imgs, obs_dummy, done)
        imgs = next_imgs
        episode_reward += float(reward)
 
        if done:
            episode_count += 1
            logger.log_reward(episode_reward)
 
            # ====== TENSORBOARD LOG REWARD ======
            writer.add_scalar("Reward/Episode", episode_reward, episode_count)
            writer.add_scalar("Reward/MeanWindow", logger.mean_reward(), episode_count)
            writer.add_scalar("Policy/Epsilon", agent.epsilon(), episode_count)
            # ====================================
 
            if episode_count % LoggingConfig.LOG_FREQ == 0:
                print(
                    f"Episode: {episode_count} | Step: {step} | "
                    f"Reward: {episode_reward:.2f} | Mean{LoggingConfig.REWARD_WINDOW}: {logger.mean_reward():.2f} | "
                    f"Epsilon: {agent.epsilon():.3f}"
                )
 
            obs, info = env.reset()
            imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
            episode_reward = 0.0
 
        can_sample = buffer.full or buffer.ptr >= BATCH_SIZE
        if step > LEARNING_STARTS and step % TRAIN_FREQ == 0 and can_sample:
            loss = agent.update(buffer)
            loss_buffer.append(loss)
            logger.log_loss(loss)
 
            # ====== TENSORBOARD LOG LOSS ======
            writer.add_scalar("Loss/TD_Loss", loss, step)
            # ==================================
 
            if len(loss_buffer) == loss_buffer.maxlen and step % (TRAIN_FREQ * 100) == 0:
                print(f"[TRAIN] step={step} avg_loss={np.mean(loss_buffer):.6f}")
 
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
 
        agent.steps += 1
 
        if step > 0 and step % SAVE_FREQ == 0:
            save_model(agent, path=get_checkpoint_path(args.task), step=step, prefix=f"{args.task}")
 
    env.close()
 
    save_model(agent, path=get_checkpoint_path(args.task), step=TOTAL_STEPS, prefix=f"{ENV_NAME}_{args.task}")
 
    results_path = get_results_path()
    logger.plot(save_path=str(results_path / f"{ENV_NAME}_{args.task}_dqn_training_new.png"))
 
    writer.close()  # <-- CIERRE TENSORBOARD
 
    print("\n" + "=" * 80)
    print("[INFO] Entrenamiento completado (NEW)!")
    print(f"[INFO] Checkpoints guardados en: {get_checkpoint_path(args.task)}")
    print(f"[INFO] TensorBoard logs en: runs/{log_dir}")
    print(f"[INFO] Gráficas guardadas en: {results_path}")
 
 
if __name__ == "__main__":
    main()