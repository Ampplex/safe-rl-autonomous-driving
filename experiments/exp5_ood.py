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

def run_ood_stress_test(densities=[150, 200, 250, 300, 350, 400, 450, 500], episodes=35):
    model_path = "models/safeppo_agent.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = PPO.load(model_path)

    results_file = "results/ood_results.csv"
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict('records')
        existing_densities = [r['density'] for r in results]
    else:
        results = []
        existing_densities = []

    print(f"\n🚀 Starting Out-of-Distribution (OOD) Stress Test")
    print(f"Full target densities: {densities}")

    for density in densities:
        if density in existing_densities:
            print(f"Skipping Density {density} (already completed).")
            continue

        # Use fewer episodes for extreme density to save time
        current_episodes = 10 if density >= 400 else episodes
        
        print(f"\n--- Testing Density: {density} vehicles ({current_episodes} eps) ---", flush=True)

        
        base_env = make_env(vehicles_count=density, safe=False)
        env = SafeRewardWrapper(base_env)
        
        episode_data = []
        for ep in range(current_episodes):
            obs, info = env.reset()
            done = truncated = False
            steps = 0
            total_original_reward = 0
            speeds = []
            
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_original_reward += info.get("original_reward", reward)
                steps += 1
                speeds.append(env.unwrapped.vehicle.speed)
                
            ep_stats = {
                "episode": ep,
                "reward": total_original_reward,
                "original_reward": total_original_reward,
                "steps": steps,
                "crashed": info.get("crashed", False),
                "avg_speed": np.mean(speeds) if speeds else 20.0,
                "collision_count": info.get("collision_count", 0),
                "tailgating_count": info.get("tailgating_count", 0),
                "unsafe_lane_change_count": info.get("unsafe_lane_change_count", 0),
                "overspeed_count": info.get("overspeed_count", 0),
            }
            episode_data.append(ep_stats)
            if (ep + 1) % 10 == 0:
                print(f"Density {density} | Episode {ep+1}/{current_episodes} finished.", flush=True)
        
        env.close()
        summary, _ = calculate_metrics(episode_data)
        summary["density"] = density
        results.append(summary)
        
        # Incremental save
        final_df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        final_df.to_csv("results/ood_results.csv", index=False)
        
        print(f"Results for Density {density}: Success Rate = {summary['success_rate']:.2%}, Collisions = {summary['collision_rate']:.2%}", flush=True)
    
    print("\n✅ OOD Stress Test Complete.", flush=True)
    print("Final OOD Results:", flush=True)
    print(final_df[['density', 'success_rate', 'collision_rate', 'avg_speed', 'tailgating_rate']], flush=True)
    print("\nResults saved to results/ood_results.csv", flush=True)

if __name__ == "__main__":
    # Reduced episodes to 30 per density for speed, can be increased if needed
    run_ood_stress_test(densities=[150, 200, 250, 300, 350, 400, 450, 500], episodes=20)
