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
import pandas as pd
import numpy as np
import argparse

def evaluate_density(model_path, density_list=[20, 50, 100], episodes=50):
    results = []
    model = PPO.load(model_path)
    
    for density in density_list:
        print(f"\nEvaluating Traffic Density: {density} vehicles")
        # Use lambda=0.1 for the evaluation wrapper
        # The model was trained with lambda=0.1
        
        base_env = make_env(vehicles_count=density, safe=False)
        env = SafeRewardWrapper(base_env)
        # We don't need to inject lambda here if we just want original metrics, 
        # but the wrapper needs it for the 'reward' calculation (though we focus on counters)
        
        episode_data = []
        for ep in range(episodes):
            obs, info = env.reset()
            done = truncated = False
            total_original_reward = 0
            steps = 0
            speeds = []
            
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_original_reward += info.get("original_reward", reward)
                steps += 1
                speeds.append(env.unwrapped.vehicle.speed)
                
            ep_stats = {
                "episode": ep,
                "reward": total_original_reward,  # calculate_metrics expects 'reward'
                "original_reward": total_original_reward,
                "steps": steps,
                "crashed": info.get("crashed", False),
                "avg_speed": np.mean(speeds),
                "collision_count": info.get("collision_count", 0),
                "tailgating_count": info.get("tailgating_count", 0),
                "unsafe_lane_change_count": info.get("unsafe_lane_change_count", 0),
                "overspeed_count": info.get("overspeed_count", 0),
            }
            episode_data.append(ep_stats)
            if (ep + 1) % 10 == 0:
                print(f"Density {density} | Episode {ep+1}/{episodes} finished.")
        
        env.close()
        summary, _ = calculate_metrics(episode_data)
        summary['density'] = density
        results.append(summary)
        
    final_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    final_df.to_csv("results/traffic_density_results.csv", index=False)
    print("\nExperiment 2 Complete. Results saved to results/traffic_density_results.csv")
    print(final_df[['density', 'collision_rate', 'tailgating_rate', 'avg_speed']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/safeppo_agent.zip")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    evaluate_density(args.model, episodes=args.episodes)
