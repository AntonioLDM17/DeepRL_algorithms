import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import json
from pathlib import Path

from src.agents.dqn.agent import DQNAgent
from src.utils import load_model, make_dir
from src.config import *
from src.environments.walker_wrapper import DiscreteWalkerWrapper
from src.environments.image_wrapper import ImageObservationWrapper
from src.environments.reward_wrapper import WalkerRewardWrapper

def make_eval_env(video_folder="videos/walker/dqn/eval"):
    """Create environment for evaluation with video recording"""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteWalkerWrapper(env, num_actions=NUM_DISCRETE_ACTIONS, scale=ACTION_SCALE)
    
    # Apply same reward shaping as training
    env = WalkerRewardWrapper(env, ctrl_cost_coeff=1e-3)
    
    make_dir(video_folder)
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,  # Record all episodes
        name_prefix=f"eval_{ENV_NAME}_dqn",
        disable_logger=False
    )
    
    env = ImageObservationWrapper(env, image_size=IMAGE_SIZE, num_stack=FRAME_STACK)
    return env

def evaluate(agent, env, num_episodes=10):
    """Evaluate agent and record videos"""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        img, obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < MAX_EPISODE_STEPS:
            # Greedy action (no exploration)
            obs_tensor = torch.tensor(obs, device=agent.device).float().unsqueeze(0)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0
            
            with torch.no_grad():
                q_values = agent.q_net(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Steps: {step_count}")
    
    # Statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:    {np.min(episode_rewards):.2f}")
    print(f"Max Reward:    {np.max(episode_rewards):.2f}")
    print(f"Mean Length:   {mean_length:.1f} steps")
    print(f"{'='*60}\n")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': mean_length
    }

def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        return None
    
    # Sort by step number (extract from filename)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def save_evaluation_results(results, save_path):
    """Save evaluation results to JSON"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    results_json = {
        'mean_reward': float(results['mean_reward']),
        'std_reward': float(results['std_reward']),
        'min_reward': float(results['min_reward']),
        'max_reward': float(results['max_reward']),
        'mean_length': float(results['mean_length']),
        'episode_rewards': [float(r) for r in results['episode_rewards']],
        'episode_lengths': [int(l) for l in results['episode_lengths']],
    }
    
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"[INFO] Evaluation results saved to {save_path}")

if __name__ == "__main__":
    print("=" * 80)
    print("DQN AGENT EVALUATION")
    print("=" * 80)
    print(f"Environment: {ENV_NAME}")
    print(f"Frame Stack: {FRAME_STACK}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Discrete Actions: {NUM_DISCRETE_ACTIONS}")
    print("=" * 80)
    
    # Create environment
    video_folder = get_video_path(task, 'eval')
    env = make_eval_env(video_folder=str(video_folder))
    obs, _ = env.reset()
    
    obs_shape = obs.shape
    num_actions = env.action_space.n
    input_channels = obs.shape[-1]
    
    print(f"\nObservation shape: {obs_shape}")
    print(f"Input channels: {input_channels}")
    print(f"Number of actions: {num_actions}\n")
    
    # Create agent
    agent = DQNAgent(num_actions, input_channels)
    
    # Try to load the most recent checkpoint
    checkpoint_dir = get_checkpoint_path()
    model_path = find_latest_checkpoint(checkpoint_dir)
    model_loaded = False
    
    if model_path:
        try:
            load_model(agent, model_path)
            print(f"✓ Loaded model from {model_path}\n")
            model_loaded = True
        except Exception as e:
            print(f"✗ Error loading model: {e}\n")
    
    if not model_loaded:
        print("⚠ WARNING: No trained model found!")
        print("  Evaluating with UNTRAINED agent (random initialization).")
        print("  Run 'python train_dqn.py' first to train the agent.\n")
    
    # Evaluate
    num_eval_episodes = LoggingConfig.NUM_EVAL_EPISODES
    print(f"Starting evaluation with {num_eval_episodes} episodes...")
    print("=" * 60)
    
    results = evaluate(agent, env, num_episodes=num_eval_episodes)
    
    env.close()
    
    # Save results
    results_path = get_results_path()
    save_evaluation_results(
        results, 
        save_path=results_path / f"{ENV_NAME}_dqn_evaluation.json"
    )
    
    print(f"\n✓ Videos saved in {video_folder}")
    print(f"✓ Results saved in {results_path}")