import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from environments.env_factory import make_env
from environments.safe_reward_wrapper import SafeRewardWrapper
from evaluation.metrics import calculate_metrics, save_metrics
from configs.config import SAFE_CONFIG
import numpy as np
import argparse

def action_to_int(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)

def evaluate_model(model_path, num_episodes=100, save_name="baseline", safety_lambda=None):
    # Setup config
    current_safe_config = SAFE_CONFIG.copy()
    if safety_lambda is not None:
        current_safe_config["safety_lambda"] = safety_lambda

    # Create environment and inject config
    base_env = make_env(safe=False)
    env = SafeRewardWrapper(base_env)
    env.cfg = current_safe_config
    
    model = PPO.load(model_path)
    
    episode_data = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        total_original_reward = 0
        steps = 0
        speeds = []
        lane_changes = 0
        component_totals = {
            "progress_reward": 0.0,
            "collision_penalty": 0.0,
            "tailgating_penalty": 0.0,
            "lane_change_penalty": 0.0,
            "speed_penalty": 0.0,
        }
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            if action_to_int(action) in [0, 2]:
                lane_changes += 1
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            total_original_reward += info.get("original_reward", reward)
            steps += 1
            speeds.append(env.unwrapped.vehicle.speed)
            components = info.get("reward_components", {})
            for key in component_totals:
                component_totals[key] += components.get(key, 0.0)
            
        # Collect episode stats
        ep_stats = {
            "episode": ep,
            "reward": total_reward,
            "original_reward": total_original_reward,
            "steps": steps,
            "crashed": info.get("crashed", False),
            "avg_speed": np.mean(speeds),
            "lane_changes": lane_changes,
            "collision_count": info.get("collision_count", 0),
            "tailgating_count": info.get("tailgating_count", 0),
            "unsafe_lane_change_count": info.get("unsafe_lane_change_count", 0),
            "overspeed_count": info.get("overspeed_count", 0),
            "progress_reward": component_totals["progress_reward"],
            "collision_penalty": component_totals["collision_penalty"],
            "tailgating_penalty": component_totals["tailgating_penalty"],
            "lane_change_penalty": component_totals["lane_change_penalty"],
            "speed_penalty": component_totals["speed_penalty"],
        }
        episode_data.append(ep_stats)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} finished.")

    env.close()
    
    summary, detailed_df = calculate_metrics(episode_data)
    save_metrics(summary, detailed_df, save_name)
    
    print(f"\n--- Evaluation Summary: {save_name} ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--safety_lambda", type=float, default=None)
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes, args.name, args.safety_lambda)
