import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import *
from src.environments.image_wrapper import obs_to_uint8_hwc_stacked


# --------------------------
# Directorios de checkpoints
# --------------------------
def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------
# Detección de tipo de agente
# --------------------------
def _is_rainbow(agent) -> bool:
    return (
        hasattr(agent, "online")
        and hasattr(agent, "target")
        and hasattr(agent, "optim")
    )


def _is_dqn(agent) -> bool:
    return (
        hasattr(agent, "q_net")
        and hasattr(agent, "target_net")
        and hasattr(agent, "optimizer")
    )


def _is_ppo(agent) -> bool:
    return (
        hasattr(agent, "model")
        and hasattr(agent, "optimizer")
        and hasattr(agent, "act")
        and hasattr(agent, "update")
    )


def _is_sac(agent) -> bool:
    """Discrete SAC — has num_actions."""
    return (
        hasattr(agent, "actor")
        and hasattr(agent, "critic")
        and hasattr(agent, "critic_target")
        and hasattr(agent, "actor_optim")
        and hasattr(agent, "critic_optim")
        and hasattr(agent, "num_actions")
    )


def _is_sac_cont(agent) -> bool:
    """Continuous SAC — has action_dim instead of num_actions."""
    return (
        hasattr(agent, "actor")
        and hasattr(agent, "critic")
        and hasattr(agent, "critic_target")
        and hasattr(agent, "actor_optim")
        and hasattr(agent, "critic_optim")
        and hasattr(agent, "action_dim")
    )


def save_model(agent, path, step, prefix=None):
    """Guarda checkpoints para DQN, Rainbow, PPO o SAC."""
    make_dir(path)

    if _is_rainbow(agent):
        name = prefix or "walker"
        filename = os.path.join(path, f"rainbow_{name}_{step}.pth")
        torch.save(
            {
                "algo": "rainbow",
                "online_state_dict": agent.online.state_dict(),
                "target_state_dict": agent.target.state_dict(),
                "optimizer_state_dict": agent.optim.state_dict(),
                "train_steps": getattr(agent, "train_steps", 0),
                "num_actions": getattr(agent, "num_actions", None),
                "num_atoms": getattr(agent, "num_atoms", None),
                "v_min": getattr(agent, "v_min", None),
                "v_max": getattr(agent, "v_max", None),
            },
            filename,
        )
        print(f"[INFO] Modelo Rainbow guardado en {filename}")
        return filename

    if _is_dqn(agent):
        name = prefix or "walker"
        filename = os.path.join(path, f"dqn_{name}_{step}.pth")
        torch.save(
            {
                "algo": "dqn",
                "q_net_state_dict": agent.q_net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps": getattr(agent, "steps", 0),
            },
            filename,
        )
        print(f"[INFO] Modelo DQN guardado en {filename}")
        return filename

    if _is_ppo(agent):
        name = prefix or "walker"
        filename = os.path.join(path, f"ppo_{name}_{step}.pth")
        torch.save(
            {
                "algo": "ppo",
                "model_state_dict": agent.model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "gamma": getattr(agent, "gamma", None),
                "lam": getattr(agent, "lam", None),
                "clip_eps": getattr(agent, "clip_eps", None),
                "entropy_coef": getattr(agent, "entropy_coef", None),
                "value_coef": getattr(agent, "value_coef", None),
                "max_grad_norm": getattr(agent, "max_grad_norm", None),
            },
            filename,
        )
        print(f"[INFO] Modelo PPO guardado en {filename}")
        return filename

    if _is_sac(agent):
        name = prefix or "walker"
        filename = os.path.join(path, f"sac_{name}_{step}.pth")
        torch.save(
            {
                "algo": "sac",
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "critic_target_state_dict": agent.critic_target.state_dict(),
                "actor_optim_state_dict": agent.actor_optim.state_dict(),
                "critic_optim_state_dict": agent.critic_optim.state_dict(),
                "alpha_optim_state_dict": agent.alpha_optim.state_dict(),
                "log_alpha": agent.log_alpha.detach().cpu(),
                "train_steps": getattr(agent, "train_steps", 0),
                "num_actions": getattr(agent, "num_actions", None),
            },
            filename,
        )
        print(f"[INFO] Modelo SAC guardado en {filename}")
        return filename

    if _is_sac_cont(agent):
        name = prefix or "walker"
        filename = os.path.join(path, f"sac_cont_{name}_{step}.pth")
        torch.save(
            {
                "algo": "sac_cont",
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "critic_target_state_dict": agent.critic_target.state_dict(),
                "actor_optim_state_dict": agent.actor_optim.state_dict(),
                "critic_optim_state_dict": agent.critic_optim.state_dict(),
                "alpha_optim_state_dict": agent.alpha_optim.state_dict(),
                "log_alpha": agent.log_alpha.detach().cpu(),
                "train_steps": getattr(agent, "train_steps", 0),
                "action_dim": getattr(agent, "action_dim", None),
            },
            filename,
        )
        print(f"[INFO] Modelo SAC-Cont guardado en {filename}")
        return filename

    raise AttributeError(
        "save_model: agente no reconocido. "
        "Esperaba DQNAgent, RainbowAgent, PPOAgent o SACAgent."
    )


def load_model(agent, checkpoint_path):
    """Carga checkpoints para DQN, Rainbow, PPO o SAC."""
    dev = getattr(agent, "device", "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    algo = checkpoint.get("algo", None)

    if algo == "rainbow" or _is_rainbow(agent):
        agent.online.load_state_dict(checkpoint["online_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.train_steps = checkpoint.get("train_steps", 0)
        print(f"[INFO] Modelo Rainbow cargado desde {checkpoint_path}")
        return

    if algo == "dqn" or _is_dqn(agent):
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps = checkpoint.get("steps", 0)
        print(f"[INFO] Modelo DQN cargado desde {checkpoint_path}")
        return

    if algo == "ppo" or _is_ppo(agent):
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[INFO] Modelo PPO cargado desde {checkpoint_path}")
        return

    if algo == "sac" or _is_sac(agent):
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        if "actor_optim_state_dict" in checkpoint:
            agent.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        if "critic_optim_state_dict" in checkpoint:
            agent.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
        if "alpha_optim_state_dict" in checkpoint:
            agent.alpha_optim.load_state_dict(checkpoint["alpha_optim_state_dict"])
        if "log_alpha" in checkpoint:
            with torch.no_grad():
                agent.log_alpha.copy_(checkpoint["log_alpha"].to(agent.device))
            agent.alpha = agent.log_alpha.exp().item()
        agent.train_steps = checkpoint.get("train_steps", 0)
        print(f"[INFO] Modelo SAC cargado desde {checkpoint_path}")
        return

    if algo == "sac_cont" or _is_sac_cont(agent):
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        if "actor_optim_state_dict" in checkpoint:
            agent.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        if "critic_optim_state_dict" in checkpoint:
            agent.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
        if "alpha_optim_state_dict" in checkpoint:
            agent.alpha_optim.load_state_dict(checkpoint["alpha_optim_state_dict"])
        if "log_alpha" in checkpoint:
            with torch.no_grad():
                agent.log_alpha.copy_(checkpoint["log_alpha"].to(agent.device))
            agent.alpha = agent.log_alpha.exp().item()
        agent.train_steps = checkpoint.get("train_steps", 0)
        print(f"[INFO] Modelo SAC-Cont cargado desde {checkpoint_path}")
        return

    raise AttributeError("load_model: checkpoint o agente no reconocido.")


# --------------------------
# Recompensas y métricas
# --------------------------
class RewardLogger:
    def __init__(self, window=50):
        self.rewards = []
        self.losses = []
        self.reward_window = deque(maxlen=window)
        self.loss_window = deque(maxlen=window)

    def log_reward(self, reward):
        self.rewards.append(reward)
        self.reward_window.append(reward)

    def log_loss(self, loss):
        self.losses.append(loss)
        self.loss_window.append(loss)

    def mean_reward(self):
        return np.mean(self.reward_window) if self.reward_window else 0.0

    def mean_loss(self):
        return np.mean(self.loss_window) if self.loss_window else 0.0

    def plot(self, save_path=None):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards, label="Reward")
        plt.plot(
            [
                np.mean(self.rewards[max(0, i - 50) : i + 1])
                for i in range(len(self.rewards))
            ],
            label="MA50",
        )
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Recompensa por episodio")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.losses, label="Loss")
        plt.plot(
            [
                np.mean(self.losses[max(0, i - 50) : i + 1])
                for i in range(len(self.losses))
            ],
            label="MA50",
        )
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.title("Pérdida por actualización")
        plt.legend()

        plt.tight_layout()
        if save_path:
            make_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            print(f"[INFO] Plot guardado en {save_path}")
        else:
            plt.show()
        plt.close()


# --------------------------
# Normalización de imágenes
# --------------------------
def preprocess_obs(obs):
    return np.asarray(obs, dtype=np.float32) / 255.0


def log_obs(obs, writer):
    print(
        "\n[DEBUG] obs raw shape (FrameStackObservation output):",
        obs.shape,
        "dtype:",
        obs.dtype,
    )

    if obs.ndim == 4:
        raw_last = obs[-1]
    elif obs.ndim == 3:
        raw_last = obs
    else:
        raise ValueError(f"[DEBUG] Forma de obs inesperada: {obs.shape}")

    raw_last_vis = np.clip(raw_last, 0, 255).astype(np.uint8)

    imgs = obs_to_uint8_hwc_stacked(obs, image_size=NetworkConfig.IMAGE_SIZE)
    print("[DEBUG] imgs processed shape:", imgs.shape, "dtype:", imgs.dtype)

    if imgs.shape[-1] < 3:
        raise ValueError(f"[DEBUG] imgs tiene menos de 3 canales: {imgs.shape}")

    proc_last_vis = imgs[:, :, -3:]

    raw_chw = torch.from_numpy(raw_last_vis).permute(2, 0, 1).float() / 255.0
    proc_chw = torch.from_numpy(proc_last_vis).permute(2, 0, 1).float() / 255.0

    writer.add_image("Debug/raw_last_frame", raw_chw, global_step=0)
    writer.add_image("Debug/processed_last_frame", proc_chw, global_step=0)

    try:
        plt.figure()
        plt.title("RAW last frame (before crop/resize)")
        plt.imshow(raw_last_vis)
        plt.axis("off")

        plt.figure()
        plt.title("PROCESSED last frame (after crop/resize)")
        plt.imshow(proc_last_vis)
        plt.axis("off")

        plt.show()
    except Exception as e:
        print(
            f"[DEBUG] No se pudo mostrar con matplotlib (probable entorno sin GUI): {e}"
        )
        print("[DEBUG] Revisa TensorBoard: runs/<tu_run>/ -> pestaña Images")

    print(
        "\n[DEBUG] Inspecciona las imágenes en TensorBoard (Images) o en las ventanas si han salido."
    )
    print(
        "[DEBUG] Si el humanoide sale cortado (pies/torso), hay que ajustar crop ratios en obs_to_uint8_hwc_stacked."
    )
