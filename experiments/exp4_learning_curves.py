import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from configs.config import PPO_CONFIG
from training.train_ppo import train_ppo
from training.train_safeppo import train_safe_ppo


def combine_curve_files(
    baseline_curve="results/learning_curves/ppo_baseline_curve.csv",
    safe_curve="results/learning_curves/safeppo_curve.csv",
    output_path="results/learning_curves.csv",
):
    frames = []
    if os.path.exists(baseline_curve):
        baseline = pd.read_csv(baseline_curve)
        baseline["model"] = "PPO"
        frames.append(baseline)
    if os.path.exists(safe_curve):
        safe = pd.read_csv(safe_curve)
        safe["model"] = "Safe PPO"
        frames.append(safe)

    if not frames:
        print("No curve files found to combine.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return combined


def run_learning_curve_experiment(
    timesteps=None,
    eval_freq=5000,
    eval_episodes=30,
    eval_density=50,
    eval_seed_start=20_000,
    skip_training=False,
):
    timesteps = timesteps if timesteps is not None else PPO_CONFIG["total_timesteps"]
    baseline_curve = "results/learning_curves/ppo_baseline_curve.csv"
    safe_curve = "results/learning_curves/safeppo_curve.csv"

    if not skip_training:
        train_ppo(
            timesteps=timesteps,
            record_curves=True,
            curve_file=baseline_curve,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            eval_density=eval_density,
            eval_seed_start=eval_seed_start,
            model_path="models/ppo_baseline_curve_run",
            run_name=f"ppo_learning_curve_d{eval_density}_{timesteps}",
        )
        train_safe_ppo(
            timesteps=timesteps,
            record_curves=True,
            curve_file=safe_curve,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            eval_density=eval_density,
            eval_seed_start=eval_seed_start,
            model_path="models/safeppo_curve_run",
            run_name=f"safeppo_learning_curve_d{eval_density}_{timesteps}",
        )

    return combine_curve_files(baseline_curve, safe_curve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--eval_density", type=int, default=50)
    parser.add_argument("--eval_seed_start", type=int, default=20_000)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    run_learning_curve_experiment(
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_density=args.eval_density,
        eval_seed_start=args.eval_seed_start,
        skip_training=args.skip_training,
    )
