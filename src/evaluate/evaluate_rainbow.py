# evaluate_rainbow.py
import os
import json
import glob
import numpy as np
import torch
import gymnasium as gym

from src.config import AlgorithmConfig
from src.environments.walker_wrapper import DiscreteWalkerWrapper
from src.environments.reward_wrapper import WalkerRewardWrapper
from src.environments.image_wrapper import ImageObservationWrapper
from src.agents.rainbow.rainbow_network import RainbowDQN

def find_latest_checkpoint(ckpt_dir: str):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    return ckpts[-1] if ckpts else None

@torch.no_grad()
def main():
    cfg = AlgorithmConfig()
    cfg.ALGO = "rainbow"
    device = torch.device(cfg.DEVICE)

    run_dir = os.environ.get("RAINBOW_RUN_DIR", None)
    if run_dir is None:
        print("Set env var RAINBOW_RUN_DIR to your runs/<...> folder to evaluate.")
        return

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is None:
        print(f"No checkpoint found in {ckpt_dir}")
        return

    env = gym.make(cfg.ENV_ID, render_mode="rgb_array")
    env = DiscreteWalkerWrapper(env, num_actions=cfg.NUM_DISCRETE_ACTIONS, scale=cfg.ACTION_SCALE)
    env = WalkerRewardWrapper(env)
    env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(run_dir, "videos_eval"),
                                  episode_trigger=lambda ep: True, name_prefix="eval_rainbow")
    env = ImageObservationWrapper(env, img_size=cfg.IMAGE_SIZE, num_stack=cfg.FRAME_STACK)

    num_actions = env.action_space.n

    net = RainbowDQN(
        num_actions=num_actions,
        num_atoms=cfg.NUM_ATOMS,
        use_dueling=cfg.USE_DUELING,
        use_noisy=False,  # eval deterministic
        noisy_std_init=cfg.NOISY_STD_INIT
    ).to(device)

    net.load_state_dict(torch.load(latest, map_location=device))
    net.eval()

    support = torch.linspace(cfg.V_MIN, cfg.V_MAX, cfg.NUM_ATOMS, device=device)

    returns = []
    for ep in range(cfg.NUM_EVAL_EPISODES):
        imgs, obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            x = torch.tensor(imgs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            x = x.permute(0, 3, 1, 2)
            q = net.q_values(x, support)
            action = int(torch.argmax(q, dim=1).item())

            (next_imgs, next_obs), reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            imgs = next_imgs

        returns.append(ep_ret)
        print(f"Episode {ep+1}/{cfg.NUM_EVAL_EPISODES}: return={ep_ret:.2f}")

    env.close()

    out = {
        "checkpoint": latest,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
    }

    out_path = os.path.join(run_dir, "results", "eval_rainbow.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()