import pandas as pd
import numpy as np
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

def evaluate_behavior(model_name, model_path, episodes=50, density=50):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return None
        
    model = PPO.load(model_path)
    env = make_env(vehicles_count=density, safe=False)
    env = SafeRewardWrapper(env)
    
    results = []
    
    print(f"Evaluating behavior for {model_name}...", flush=True)
    for ep in range(episodes):
        obs, info = env.reset(seed=42+ep)
        done = truncated = False
        
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
            
            if action_id in [0, 2]: # Lane Left or Right
                ep_lane_changes += 1
                
            if info.get('collision', False) or info.get('crashed', False):
                ep_collisions = 1
            if info.get('tailgating', False):
                ep_tailgating = 1
            if info.get('unsafe_lane_change', False):
                ep_unsafe_lane_changes = 1
            speeds.append(info.get('speed', env.unwrapped.vehicle.speed))
            
            steps += 1
            
        results.append({
            'collision': ep_collisions,
            'tailgating': ep_tailgating,
            'unsafe_lane_change': ep_unsafe_lane_changes,
            'lane_changes': ep_lane_changes,
            'speed': np.mean(speeds) if speeds else 0,
            'steps': steps,
            'tailgating_rate': info.get("tailgating_count", 0) / max(steps, 1),
            'unsafe_lane_change_rate': info.get("unsafe_lane_change_count", 0) / max(steps, 1),
            'safety_violation_score': ep_collisions * 10
                + info.get("tailgating_count", 0) / max(steps, 1)
                + info.get("unsafe_lane_change_count", 0) / max(steps, 1),
        })
        if (ep + 1) % 10 == 0:
            print(f"  {model_name}: episode {ep + 1}/{episodes}", flush=True)
        
    env.close()
    df = pd.DataFrame(results)
    summary = df.mean().to_dict()
    summary['model'] = model_name
    summary['success_rate'] = 1 - summary['collision']
    return summary

if __name__ == "__main__":
    models = {
        'Baseline (λ=0.0)': 'models/ppo_baseline.zip',
        'λ=0.1': 'models/safeppo_agent.zip',
        'λ=0.5': 'models/safeppo_agent_lam0.5.zip',
        'λ=1.0': 'models/safeppo_agent_lam1.0.zip',
        'λ=2.0': 'models/safeppo_agent_lam2.0.zip'
    }
    
    output_path = "results/lambda_behavior_analysis.csv"
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        all_summaries = existing.to_dict("records")
        completed_models = set(existing["model"].astype(str))
        print(f"Resuming behavior analysis with {len(all_summaries)} completed rows.", flush=True)
    else:
        all_summaries = []
        completed_models = set()

    for name, path in models.items():
        if name in completed_models:
            print(f"Skipping {name}: already complete.", flush=True)
            continue

        summary = evaluate_behavior(name, path)
        if summary:
            all_summaries.append(summary)
            completed_models.add(name)
            pd.DataFrame(all_summaries).to_csv(output_path, index=False)
            print("Saved partial behavior analysis.", flush=True)
            
    pd.DataFrame(all_summaries).to_csv(output_path, index=False)
    print(f"Behavior analysis complete. Saved to {output_path}", flush=True)
