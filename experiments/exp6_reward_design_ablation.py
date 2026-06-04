import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from evaluation.evaluate import evaluate_model
from training.train_safeppo import train_safe_ppo


def _load_summary(path, label):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Model"] = label
    return df


def run_reward_design_ablation(
    timesteps=None,
    episodes=100,
    skip_training=False,
):
    collision_only_model = "models/safeppo_collision_only"
    if not skip_training:
        train_safe_ppo(
            safety_lambda=0.1,
            timesteps=timesteps,
            collision_only=True,
            model_path=collision_only_model,
        )

    eval_specs = [
        ("models/ppo_baseline.zip", "reward_ablation_baseline", "Baseline PPO"),
        ("models/safeppo_agent.zip", "reward_ablation_safeppo", "Safe PPO"),
        (f"{collision_only_model}.zip", "reward_ablation_collision_only", "PPO + Collision Only"),
    ]

    summaries = []
    for model_path, save_name, label in eval_specs:
        if not os.path.exists(model_path):
            print(f"Skipping {label}: {model_path} not found.")
            continue
        evaluate_model(
            model_path=model_path,
            num_episodes=episodes,
            save_name=save_name,
            safety_lambda=0.1,
        )
        summary = _load_summary(f"results/{save_name}_summary.csv", label)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("No reward design ablation summaries were generated.")
        return pd.DataFrame()

    comparison = pd.concat(summaries, ignore_index=True)
    cols = ["Model"] + [c for c in comparison.columns if c != "Model"]
    comparison = comparison[cols]
    comparison.to_csv("results/reward_design_ablation.csv", index=False)
    print("Saved results/reward_design_ablation.csv")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    run_reward_design_ablation(
        timesteps=args.timesteps,
        episodes=args.episodes,
        skip_training=args.skip_training,
    )
