# resume_checkpoint.py
# Ejemplos:
#   python -m src.train.resume_checkpoint --checkpoint checkpoints/walker/dqn/dqn_walker_20500000.pth --extra-steps 200000 --task walker --algo dqn
#   python -m src.train.resume_checkpoint --checkpoint checkpoints/humanoid/dqn/dqn_humanoid_5000000.pth --extra-steps 200000 --task humanoid --algo dqn
#
# Por defecto: shaping ACTIVADO.
# Para desactivarlo explícitamente: añade --no-shaping
 
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
from src.utils import RewardLogger, save_model, load_model
 
# DQN
from src.agents.dqn.agent import DQNAgent
from src.environments.replay_buffer import ReplayBuffer
 
# Rainbow
from src.agents.rainbow.prioritized_nstep_replay import PrioritizedNStepReplay
from src.agents.rainbow.rainbow_agent import RainbowAgent
 
# Wrappers
from src.environments.walker_wrapper import DiscreteWalkerWrapper
from src.environments.humanoid_wrapper import DiscreteHumanoidWrapper
 
# Reward shaping
from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.humanoid_reward_wrapper import HumanoidRewardWrapper
 
from src.environments.image_wrapper import make_pixel_env, obs_to_uint8_hwc_stacked
 
 
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def detect_algo_from_checkpoint(ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    algo = ckpt.get("algo", None)
    if algo in ("dqn", "rainbow"):
        return algo
    if "online_state_dict" in ckpt and "target_state_dict" in ckpt:
        return "rainbow"
    if "q_net_state_dict" in ckpt and "target_net_state_dict" in ckpt:
        return "dqn"
    raise ValueError(f"No puedo detectar el tipo de checkpoint en {ckpt_path}")
 
 
def _defaults_for_task(task: str):
    if task == "walker":
        return EnvConfig.WALKER2D
    if task == "humanoid":
        return EnvConfig.HUMANOID
    raise ValueError(f"Task inválido: {task}")
 
 
def make_env(
    record_video: bool,
    video_tag: str,
    task: str,
    env_id_override: str | None,
    num_bins: int | None,
    action_scale: float | None,
    shaping: bool,
):
    """
    Env compatible Walker/Humanoid, SIN depender de ENV_ID/ENV_NAME globales.
    """
    active = _defaults_for_task(task)
 
    # Defaults por task
    env_id = active["env_id"]
    default_bins = int(active["num_discrete_actions"])
    default_scale = float(active["action_scale"])
 
    # Overrides por CLI
    if env_id_override is not None:
        env_id = env_id_override
    if num_bins is None:
        num_bins = default_bins
    if action_scale is None:
        action_scale = default_scale
 
    make_kwargs = {"render_mode": "rgb_array"}
 
    if task == "walker" and "Walker2d" in env_id:
        make_kwargs["healthy_angle_range"] = (-0.4, 0.4)
 
    env = gym.make(env_id, **make_kwargs)
 
    # Discretización + shaping
    if task == "walker":
        env = DiscreteWalkerWrapper(env, num_bins=int(num_bins), scale=float(action_scale), include_zero=True)
        if shaping:
            env = WalkerRewardWrapper(env)
    else:
        env = DiscreteHumanoidWrapper(env, num_bins=int(num_bins), scale=float(action_scale), include_zero=True)
        if shaping:
            print("hey")
            env = HumanoidRewardWrapper(env)
 
    # Vídeo opcional (IMPORTANTE: ruta por task)
    if record_video:
        video_path = get_video_path(task, video_tag)
        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda episode_id: episode_id % VIDEO_FREQ_TRAIN == 0,
            name_prefix=f"{task}_{video_tag}",
        )
 
    env = make_pixel_env(env, frame_stack=NetworkConfig.FRAME_STACK)
    return env, env_id, int(num_bins), float(action_scale)
 
 
def resume_dqn(env, task: str, ckpt_path: str, extra_steps: int, writer: SummaryWriter, run_prefix: str):
    AlgorithmConfig.MODE = "dqn"
 
    obs, info = env.reset()
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
    img_shape = imgs.shape
    num_actions = env.action_space.n
    input_channels = imgs.shape[-1]
 
    # Dummy propriocepción para compatibilidad con ReplayBuffer viejo
    obs_dummy = np.zeros((1,), dtype=np.float32)
 
    agent = DQNAgent(num_actions, input_channels)
 
    # --- Carga checkpoint (si mismatch -> es que env/task no coincide) ---
    load_model(agent, ckpt_path)
 
    start_step = int(getattr(agent, "steps", 0))
    end_step = start_step + int(extra_steps)
 
    buffer = ReplayBuffer(BUFFER_SIZE, img_shape, obs_dummy.shape, DEVICE)
    logger = RewardLogger(window=LoggingConfig.REWARD_WINDOW)
    loss_buffer = deque(maxlen=100)
 
    episode_reward = 0.0
    episode_count = 0
 
    print(f"[RESUME DQN] checkpoint={ckpt_path}")
    print(f"[RESUME DQN] start_step={start_step} -> end_step={end_step}")
    print(f"[RESUME DQN] task={task} num_actions={num_actions} img_shape={img_shape}")
 
    step = start_step
    while step < end_step:
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
 
            # TensorBoard (igual que train_dqn)
            writer.add_scalar("Reward/Episode", episode_reward, episode_count)
            writer.add_scalar("Reward/MeanWindow", logger.mean_reward(), episode_count)
            writer.add_scalar("Policy/Epsilon", agent.epsilon(), episode_count)
 
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
        if step > LEARNING_STARTS and (step % TRAIN_FREQ == 0) and can_sample:
            loss = agent.update(buffer)
            loss_buffer.append(loss)
            logger.log_loss(loss)
 
            writer.add_scalar("Loss/TD_Loss", loss, step)
 
            if len(loss_buffer) == loss_buffer.maxlen and step % (TRAIN_FREQ * 100) == 0:
                print(f"[TRAIN] step={step} avg_loss={np.mean(loss_buffer):.6f}")
 
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
 
        agent.steps += 1
        step += 1
 
        if step > 0 and step % SAVE_FREQ == 0:
            save_model(agent, path=get_checkpoint_path(task), step=step, prefix=run_prefix)
 
    save_model(agent, path=get_checkpoint_path(task), step=end_step, prefix=run_prefix)
 
    results_path = get_results_path()  # si quieres también desacoplar esto, lo hacemos igual que checkpoint/video
    logger.plot(save_path=str(results_path / f"{run_prefix}_dqn_resume.png"))
    print(f"[RESUME DQN] Done. Saved to {get_checkpoint_path(task)}")
 
 
def resume_rainbow(env, task: str, ckpt_path: str, extra_steps: int, writer: SummaryWriter, run_prefix: str):
    AlgorithmConfig.MODE = "rainbow"
 
    obs, info = env.reset()
    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
 
    num_actions = env.action_space.n
    img_shape = imgs.shape
 
    replay = PrioritizedNStepReplay(
        capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=str(DEVICE),
        alpha=AlgorithmConfig.PER_ALPHA,
        beta_start=AlgorithmConfig.PER_BETA_START,
        beta_frames=AlgorithmConfig.PER_BETA_FRAMES,
        eps=AlgorithmConfig.PER_EPS,
        n_step=AlgorithmConfig.N_STEP if getattr(AlgorithmConfig, "USE_N_STEP", True) else 1,
        gamma=AlgorithmConfig.N_STEP_GAMMA if getattr(AlgorithmConfig, "USE_N_STEP", True) else GAMMA,
        img_shape=img_shape,
    )
 
    agent = RainbowAgent(num_actions=num_actions, cfg=AlgorithmConfig, device=DEVICE)
    load_model(agent, ckpt_path)
 
    start_step = int(getattr(agent, "train_steps", 0))
    end_step = start_step + int(extra_steps)
 
    logger = RewardLogger(window=LoggingConfig.REWARD_WINDOW)
    loss_buffer = deque(maxlen=100)
 
    ep_return = 0.0
    ep = 0
 
    print(f"[RESUME RAINBOW] checkpoint={ckpt_path}")
    print(f"[RESUME RAINBOW] start_step={start_step} -> end_step={end_step}")
    print(f"[RESUME RAINBOW] task={task} num_actions={num_actions} img_shape={img_shape}")
 
    step = start_step
    while step < end_step:
        action = agent.act(imgs, epsilon=None)
 
        next_obs, reward, terminated, truncated, info = env.step(action)
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
                    f"Episode: {ep} | Step: {step} | "
                    f"Return: {ep_return:.2f} | Mean{LoggingConfig.REWARD_WINDOW}: {logger.mean_reward():.2f}"
                )
 
            obs, info = env.reset()
            imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
            ep_return = 0.0
 
        if step > LEARNING_STARTS and replay.size >= BATCH_SIZE and (step % TRAIN_FREQ == 0):
            loss, mean_sample_loss = agent.learn(replay)
            loss_buffer.append(loss)
            logger.log_loss(loss)
 
            writer.add_scalar("Loss/TD_Loss", loss, step)
 
            if step % (TRAIN_FREQ * 100) == 0 and len(loss_buffer) == loss_buffer.maxlen:
                print(f"[TRAIN] step={step} avg_loss={np.mean(loss_buffer):.6f}")
 
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
 
        agent.train_steps += 1
        step += 1
 
        if step > 0 and step % SAVE_FREQ == 0:
            save_model(agent, path=get_checkpoint_path(task), step=step, prefix=run_prefix)
 
    save_model(agent, path=get_checkpoint_path(task), step=end_step, prefix=run_prefix)
 
    results_path = get_results_path()
    logger.plot(save_path=str(results_path / f"{run_prefix}_rainbow_resume.png"))
    print(f"[RESUME RAINBOW] Done. Saved to {get_checkpoint_path(task)}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al .pth a reanudar")
    parser.add_argument("--extra-steps", type=int, default=200_000, help="Pasos extra a entrenar desde el checkpoint")
    parser.add_argument("--algo", type=str, default="auto", choices=["auto", "dqn", "rainbow"], help="Tipo de agente")
    parser.add_argument("--task", type=str, default="walker", choices=["walker", "humanoid"], help="Entorno a usar")
    parser.add_argument("--env_id", type=str, default=None, help="Override ENV_ID")
    parser.add_argument("--num_bins", type=int, default=None, help="Bins para discretización (None usa default por task).")
    parser.add_argument("--action_scale", type=float, default=None, help="Escala de torque (None usa default por task).")
    parser.add_argument(
        "--no-shaping",
        action="store_true",
        help="Desactiva reward shaping (por defecto está activado).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Grabar vídeos durante el resume")
    args = parser.parse_args()
 
    set_seed(args.seed)
 
    algo = args.algo
    if algo == "auto":
        algo = detect_algo_from_checkpoint(args.checkpoint)
 
    shaping = not args.no_shaping
 
    AlgorithmConfig.MODE = algo
 
    # TensorBoard: NO uses ENV_NAME global, usa task
    run_prefix = f"{args.task}_{algo}"
    log_dir = f"{run_prefix}_RESUME_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")
 
    env, used_env_id, used_bins, used_scale = make_env(
        record_video=args.video,
        video_tag=f"resume_{args.task}_{algo}",
        task=args.task,
        env_id_override=args.env_id,
        num_bins=args.num_bins,
        action_scale=args.action_scale,
        shaping=shaping,
    )
 
    print("\n[RESUME SETUP]")
    print(f"TASK: {args.task}")
    print(f"ENV_ID: {used_env_id}")
    print(f"algo: {algo}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"extra_steps: {args.extra_steps}")
    print(f"num_bins: {used_bins}")
    print(f"action_scale: {used_scale}")
    print(f"shaping: {shaping}")
    print(f"tensorboard: runs/{log_dir}")
    print("=" * 80)
 
    try:
        if algo == "dqn":
            resume_dqn(env, args.task, args.checkpoint, args.extra_steps, writer, run_prefix=run_prefix)
        elif algo == "rainbow":
            resume_rainbow(env, args.task, args.checkpoint, args.extra_steps, writer, run_prefix=run_prefix)
        else:
            raise ValueError(f"Algo inválido: {algo}")
    finally:
        env.close()
        writer.close()
 
    print("\n" + "=" * 80)
    print("[INFO] Resume completado!")
    print(f"[INFO] TensorBoard logs en: runs/{log_dir}")
    print(f"[INFO] Checkpoints en: {get_checkpoint_path(args.task)}")
 
 
if __name__ == "__main__":
    main()