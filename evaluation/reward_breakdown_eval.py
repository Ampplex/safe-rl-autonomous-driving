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
from evaluation.lambda_multi_seed_eval import DEFAULT_MODELS, action_to_int


def evaluate_reward_breakdown(
    models=None,
    episodes=50,
    density=50,
    output_path="results/reward_component_breakdown.csv",
):
    models = models or DEFAULT_MODELS
    detailed_path = output_path.replace(".csv", "_detailed.csv")
    if os.path.exists(detailed_path):
        existing = pd.read_csv(detailed_path).drop_duplicates(
            subset=["lambda", "episode"],
            keep="last",
        )
        rows = existing.to_dict("records")
        completed = {
            (float(row["lambda"]), int(row["episode"]))
            for _, row in existing.iterrows()
        }
        print(f"Resuming reward breakdown from {detailed_path} with {len(rows)} rows.", flush=True)
    else:
        rows = []
        completed = set()

    for safety_lambda, (model_name, model_path) in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping lambda={safety_lambda}: {model_path} not found.")
            continue

        missing_episodes = [
            episode
            for episode in range(episodes)
            if (float(safety_lambda), int(episode)) not in completed
        ]
        if not missing_episodes:
            print(f"Skipping lambda={safety_lambda}: already complete.", flush=True)
            continue

        model = PPO.load(model_path)
        safe_config = SAFE_CONFIG.copy()
        safe_config["safety_lambda"] = safety_lambda
        env = make_env(vehicles_count=density, safe=False)
        env = SafeRewardWrapper(env)
        env.cfg = safe_config

        print(
            f"Evaluating reward components for {model_name} lambda={safety_lambda} "
            f"({len(missing_episodes)} missing episodes)...",
            flush=True,
        )
        for episode in missing_episodes:
            obs, info = env.reset(seed=1000 + episode)
            done = truncated = False
            steps = 0
            lane_changes = 0
            totals = {
                "progress_reward": 0.0,
                "collision_penalty": 0.0,
                "tailgating_penalty": 0.0,
                "lane_change_penalty": 0.0,
                "speed_penalty": 0.0,
                "net_reward": 0.0,
            }

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                if action_to_int(action) in [0, 2]:
                    lane_changes += 1
                obs, reward, done, truncated, info = env.step(action)
                components = info.get("reward_components", {})
                totals["progress_reward"] += components.get("progress_reward", info.get("original_reward", 0.0))
                totals["collision_penalty"] += components.get("collision_penalty", 0.0)
                totals["tailgating_penalty"] += components.get("tailgating_penalty", 0.0)
                totals["lane_change_penalty"] += components.get("lane_change_penalty", 0.0)
                totals["speed_penalty"] += components.get("speed_penalty", 0.0)
                totals["net_reward"] += reward
                steps += 1

            rows.append({
                "model": model_name,
                "lambda": safety_lambda,
                "episode": episode,
                "steps": steps,
                "lane_changes": lane_changes,
                "collision": int(bool(info.get("crashed", False) or info.get("collision_count", 0) > 0)),
                "tailgating_rate": info.get("tailgating_count", 0) / max(steps, 1),
                "unsafe_lane_change_rate": info.get("unsafe_lane_change_count", 0) / max(steps, 1),
                **totals,
            })
            completed.add((float(safety_lambda), int(episode)))
            completed_count = len([
                ep for ep in range(episodes)
                if (float(safety_lambda), int(ep)) in completed
            ])
            if completed_count % 10 == 0:
                print(
                    f"  {model_name} lambda={safety_lambda}: {completed_count}/{episodes} episodes complete",
                    flush=True,
                )

        env.close()
        if rows:
            pd.DataFrame(rows).to_csv(detailed_path, index=False)
            print("  saved partial reward breakdown rows.", flush=True)

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        print("No reward breakdown rows were generated.", flush=True)
        return detailed

    summary = detailed.groupby(["model", "lambda"])[[
        "progress_reward",
        "collision_penalty",
        "tailgating_penalty",
        "lane_change_penalty",
        "speed_penalty",
        "net_reward",
        "lane_changes",
        "collision",
        "tailgating_rate",
        "unsafe_lane_change_rate",
    ]].mean().reset_index()
    summary["total_penalty"] = (
        summary["collision_penalty"]
        + summary["tailgating_penalty"]
        + summary["lane_change_penalty"]
        + summary["speed_penalty"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    print(f"Saved {output_path}", flush=True)
    print(f"Saved {detailed_path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--density", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/reward_component_breakdown.csv")
    args = parser.parse_args()

    evaluate_reward_breakdown(episodes=args.episodes, density=args.density, output_path=args.output)
