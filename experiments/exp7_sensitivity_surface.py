import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from configs.config import SAFE_CONFIG
from environments.env_factory import make_env
from environments.safe_reward_wrapper import SafeRewardWrapper
from evaluation.lambda_multi_seed_eval import action_to_int
from training.train_safeppo import train_safe_ppo


def _model_path(collision_penalty, tailgating_penalty):
    return f"models/safeppo_cp{collision_penalty}_tp{tailgating_penalty}.zip"


def evaluate_surface_point(model_path, collision_penalty, tailgating_penalty, episodes=50, density=50):
    model = PPO.load(model_path)
    safe_config = SAFE_CONFIG.copy()
    safe_config["collision_penalty"] = collision_penalty
    safe_config["tailgating_penalty"] = tailgating_penalty

    env = make_env(vehicles_count=density, safe=False)
    env = SafeRewardWrapper(env)
    env.cfg = safe_config
    rows = []

    for episode in range(episodes):
        obs, info = env.reset(seed=20_000 + episode)
        done = truncated = False
        reward_sum = 0.0
        steps = 0
        lane_changes = 0
        speeds = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if action_to_int(action) in [0, 2]:
                lane_changes += 1
            obs, reward, done, truncated, info = env.step(action)
            reward_sum += reward
            speeds.append(info.get("speed", env.unwrapped.vehicle.speed))
            steps += 1

        collision = int(bool(info.get("crashed", False) or info.get("collision_count", 0) > 0))
        tailgating_rate = info.get("tailgating_count", 0) / max(steps, 1)
        unsafe_lane_change_rate = info.get("unsafe_lane_change_count", 0) / max(steps, 1)
        rows.append({
            "collision": collision,
            "success": 1 - collision,
            "tailgating_rate": tailgating_rate,
            "unsafe_lane_change_rate": unsafe_lane_change_rate,
            "lane_changes": lane_changes,
            "speed": np.mean(speeds) if speeds else 0.0,
            "survival_time": steps,
            "reward": reward_sum,
            "safety_score": 1 - min(1.0, collision + tailgating_rate + unsafe_lane_change_rate),
            "safety_violation_score": collision * 10 + tailgating_rate + unsafe_lane_change_rate,
        })

    env.close()
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def run_sensitivity_surface(
    collision_penalties=None,
    tailgating_penalties=None,
    timesteps=30000,
    episodes=50,
    density=50,
    skip_training=False,
    output_path="results/sensitivity_surface.csv",
):
    collision_penalties = collision_penalties or [25, 50, 75]
    tailgating_penalties = tailgating_penalties or [0, 5, 10]
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for collision_penalty in collision_penalties:
        for tailgating_penalty in tailgating_penalties:
            model_path = _model_path(collision_penalty, tailgating_penalty)
            if not skip_training:
                train_safe_ppo(
                    safety_lambda=0.1,
                    timesteps=timesteps,
                    collision_penalty=collision_penalty,
                    tailgating_penalty=tailgating_penalty,
                    model_path=model_path.replace(".zip", ""),
                )
            if not os.path.exists(model_path):
                print(f"Skipping missing model: {model_path}")
                continue

            summary = evaluate_surface_point(
                model_path=model_path,
                collision_penalty=collision_penalty,
                tailgating_penalty=tailgating_penalty,
                episodes=episodes,
                density=density,
            )
            summary["collision_penalty"] = collision_penalty
            summary["tailgating_penalty"] = tailgating_penalty
            results.append(summary)
            pd.DataFrame(results).to_csv(output_path, index=False)

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision_penalties", type=float, nargs="+", default=[25, 50, 75])
    parser.add_argument("--tailgating_penalties", type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--density", type=int, default=50)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    run_sensitivity_surface(
        collision_penalties=args.collision_penalties,
        tailgating_penalties=args.tailgating_penalties,
        timesteps=args.timesteps,
        episodes=args.episodes,
        density=args.density,
        skip_training=args.skip_training,
    )
