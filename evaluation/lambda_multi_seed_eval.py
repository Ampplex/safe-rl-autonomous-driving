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


DEFAULT_MODELS = {
    0.0: ("Baseline PPO", "models/ppo_baseline.zip"),
    0.1: ("Safe PPO", "models/safeppo_agent.zip"),
    0.5: ("Safe PPO", "models/safeppo_agent_lam0.5.zip"),
    1.0: ("Safe PPO", "models/safeppo_agent_lam1.0.zip"),
    2.0: ("Safe PPO", "models/safeppo_agent_lam2.0.zip"),
}


def action_to_int(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)


def evaluate_lambda_multi_seed(
    seeds=None,
    episodes_per_seed=20,
    density=50,
    models=None,
    output_prefix="results/lambda_multi_seed",
):
    seeds = seeds or [42, 123, 456, 789, 2025]
    models = models or DEFAULT_MODELS
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    detailed_path = f"{output_prefix}_detailed.csv"
    summary_path = f"{output_prefix}_summary.csv"
    if os.path.exists(detailed_path):
        existing = pd.read_csv(detailed_path).drop_duplicates(
            subset=["lambda", "seed", "episode"],
            keep="last",
        )
        all_rows = existing.to_dict("records")
        completed = {
            (float(row["lambda"]), int(row["seed"]), int(row["episode"]))
            for _, row in existing.iterrows()
        }
        print(f"Resuming from {detailed_path} with {len(all_rows)} completed rows.", flush=True)
    else:
        all_rows = []
        completed = set()

    for safety_lambda, (model_name, model_path) in models.items():
        if not os.path.exists(model_path):
            print(f"Skipping lambda={safety_lambda}: {model_path} not found.")
            continue

        model = PPO.load(model_path)
        safe_config = SAFE_CONFIG.copy()
        safe_config["safety_lambda"] = safety_lambda
        print(f"Evaluating {model_name} lambda={safety_lambda} across {len(seeds)} seeds...", flush=True)

        for seed in seeds:
            missing_episodes = [
                episode
                for episode in range(episodes_per_seed)
                if (float(safety_lambda), int(seed), int(episode)) not in completed
            ]
            if not missing_episodes:
                print(f"  seed={seed}: already complete", flush=True)
                continue

            print(
                f"  seed={seed}: {len(missing_episodes)} missing episodes",
                flush=True,
            )
            env = make_env(vehicles_count=density, safe=False)
            env = SafeRewardWrapper(env)
            env.cfg = safe_config

            for episode in missing_episodes:
                obs, info = env.reset(seed=seed + episode)
                done = truncated = False
                reward_sum = 0.0
                original_reward_sum = 0.0
                lane_changes = 0
                steps = 0
                speeds = []

                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    if action_to_int(action) in [0, 2]:
                        lane_changes += 1
                    obs, reward, done, truncated, info = env.step(action)
                    reward_sum += reward
                    original_reward_sum += info.get("original_reward", reward)
                    speeds.append(info.get("speed", env.unwrapped.vehicle.speed))
                    steps += 1

                collision = int(bool(info.get("crashed", False) or info.get("collision_count", 0) > 0))
                tailgating_count = info.get("tailgating_count", 0)
                unsafe_lane_change_count = info.get("unsafe_lane_change_count", 0)
                tailgating_rate = tailgating_count / max(steps, 1)
                unsafe_lane_change_rate = unsafe_lane_change_count / max(steps, 1)

                all_rows.append({
                    "model": model_name,
                    "lambda": safety_lambda,
                    "seed": seed,
                    "episode": episode,
                    "reward": reward_sum,
                    "original_reward": original_reward_sum,
                    "collision": collision,
                    "success": 1 - collision,
                    "tailgating_count": tailgating_count,
                    "tailgating_rate": tailgating_rate,
                    "unsafe_lane_change_count": unsafe_lane_change_count,
                    "unsafe_lane_change_rate": unsafe_lane_change_rate,
                    "lane_changes": lane_changes,
                    "speed": np.mean(speeds) if speeds else 0.0,
                    "survival_time": steps,
                    "safety_violation_score": collision * 10 + tailgating_rate + unsafe_lane_change_rate,
                })
                completed.add((float(safety_lambda), int(seed), int(episode)))
                completed_count = episodes_per_seed - len([
                    ep for ep in range(episodes_per_seed)
                    if (float(safety_lambda), int(seed), int(ep)) not in completed
                ])
                if completed_count % 5 == 0:
                    pd.DataFrame(all_rows).to_csv(detailed_path, index=False)
                    print(
                        f"    seed={seed}: {completed_count}/{episodes_per_seed} episodes complete "
                        f"(rows={len(all_rows)})",
                        flush=True,
                    )

            env.close()
            if all_rows:
                pd.DataFrame(all_rows).to_csv(detailed_path, index=False)
                print(
                    f"  saved partial rows={len(all_rows)} to {detailed_path}",
                    flush=True,
                )

    detailed = pd.DataFrame(all_rows)
    if detailed.empty:
        print("No lambda multi-seed results were generated.", flush=True)
        return detailed, pd.DataFrame()

    detailed.to_csv(detailed_path, index=False)

    metrics = [
        "collision",
        "success",
        "tailgating_rate",
        "unsafe_lane_change_rate",
        "lane_changes",
        "speed",
        "survival_time",
        "safety_violation_score",
        "reward",
        "original_reward",
    ]
    grouped = detailed.groupby(["model", "lambda"])[metrics]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    for metric in metrics:
        count_col = f"{metric}_count"
        std_col = f"{metric}_std"
        sem_col = f"{metric}_sem"
        ci_col = f"{metric}_ci95"
        summary[sem_col] = summary[std_col] / np.sqrt(summary[count_col])
        summary[ci_col] = 1.96 * summary[sem_col]

    summary.to_csv(summary_path, index=False)
    print(f"Saved {detailed_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    return detailed, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_per_seed", type=int, default=20)
    parser.add_argument("--density", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2025])
    args = parser.parse_args()

    evaluate_lambda_multi_seed(
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        density=args.density,
    )
