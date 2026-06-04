import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from environments.env_factory import make_env
from environments.safe_reward_wrapper import SafeRewardWrapper

def action_to_int(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)

def evaluate_model_multi_seed(model_path, seeds=[42, 123, 456, 789, 2025], episodes_per_seed=20, density=50):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return None
        
    model = PPO.load(model_path)
    all_results = []
    
    print(f"Evaluating {model_path} across {len(seeds)} seeds...")
    
    for seed in seeds:
        print(f"  Seed {seed}...", flush=True)
        env = make_env(vehicles_count=density, safe=False)
        env = SafeRewardWrapper(env)
        
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=seed + ep) # Varied seed per episode within the group
            done = truncated = False
            ep_reward = 0
            ep_collisions = 0
            ep_tailgating = 0
            ep_lane_changes = 0
            ep_unsafe_lane_changes = 0
            steps = 0
            speeds = []
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                action_id = action_to_int(action)
                obs, reward, done, truncated, info = env.step(action)
                
                # Behavioral Analysis: Lane Change Detection
                # Action 0: Lane Left, 2: Lane Right (in HighwayEnv)
                if action_id in [0, 2]:
                    ep_lane_changes += 1
                
                ep_reward += reward
                if info.get('collision', False) or info.get('crashed', False):
                    ep_collisions = 1
                if info.get('tailgating', False):
                    ep_tailgating = 1
                if info.get('unsafe_lane_change', False):
                    ep_unsafe_lane_changes = 1
                speeds.append(info.get('speed', env.unwrapped.vehicle.speed))
                
                steps += 1
                
            all_results.append({
                'seed': seed,
                'episode': ep,
                'reward': ep_reward,
                'collision': ep_collisions,
                'tailgating': ep_tailgating,
                'unsafe_lane_change': ep_unsafe_lane_changes,
                'lane_changes': ep_lane_changes,
                'survival_time': steps,
                'speed': np.mean(speeds) if speeds else 0,
                'tailgating_rate': info.get("tailgating_count", 0) / max(steps, 1),
                'unsafe_lane_change_rate': info.get("unsafe_lane_change_count", 0) / max(steps, 1),
                'safety_violation_score': ep_collisions * 10
                    + info.get("tailgating_count", 0) / max(steps, 1)
                    + info.get("unsafe_lane_change_count", 0) / max(steps, 1),
            })
        env.close()
            
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    os.makedirs("results/multi_seed", exist_ok=True)
    
    # Baseline
    baseline_df = evaluate_model_multi_seed("models/ppo_baseline.zip")
    if baseline_df is not None:
        baseline_df.to_csv("results/multi_seed/baseline_detailed.csv", index=False)
        summary = baseline_df.groupby('seed').mean()
        summary.to_csv("results/multi_seed/baseline_summary.csv")
        
    # Safe PPO (λ=0.1)
    safe_df = evaluate_model_multi_seed("models/safeppo_agent.zip")
    if safe_df is not None:
        safe_df.to_csv("results/multi_seed/safeppo_detailed.csv", index=False)
        summary = safe_df.groupby('seed').mean()
        summary.to_csv("results/multi_seed/safeppo_summary.csv")
        
    print("Multi-seed evaluation complete.")
