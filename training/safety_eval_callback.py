import os
from math import sqrt

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


METRIC_COLUMNS = [
    "reward",
    "original_reward",
    "collision",
    "success",
    "tailgating_rate",
    "unsafe_lane_change_rate",
    "overspeed_rate",
    "lane_changes",
    "speed",
    "survival_time",
    "safety_violation_score",
]

METADATA_COLUMNS = {"step", "eval_density", "eval_seed_start", "eval_seed_end", "eval_episodes"}


class SafetyEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        csv_path,
        eval_freq=5000,
        n_eval_episodes=10,
        eval_seed_start=20_000,
        eval_density=None,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start
        self.eval_density = eval_density
        self.deterministic = deterministic
        self.records = []
        self._last_eval_step = None

    @staticmethod
    def _action_to_int(action):
        if isinstance(action, np.ndarray):
            return int(action.item())
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        return True

    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            self._evaluate_and_save()
        return True

    def _on_training_end(self):
        if self._last_eval_step != self.num_timesteps:
            self._evaluate_and_save()

    def _evaluate_and_save(self):
        episode_rows = []
        for episode in range(self.n_eval_episodes):
            seed = self.eval_seed_start + episode
            obs, info = self.eval_env.reset(seed=seed)
            done = truncated = False
            total_reward = 0.0
            total_original_reward = 0.0
            steps = 0
            lane_changes = 0
            speeds = []

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                if self._action_to_int(action) in [0, 2]:
                    lane_changes += 1
                obs, reward, done, truncated, info = self.eval_env.step(action)
                total_reward += reward
                total_original_reward += info.get("original_reward", reward)
                speeds.append(info.get("speed", self.eval_env.unwrapped.vehicle.speed))
                steps += 1

            collision = int(bool(info.get("crashed", False) or info.get("collision_count", 0) > 0))
            tailgating_rate = info.get("tailgating_count", 0) / max(steps, 1)
            unsafe_lane_change_rate = info.get("unsafe_lane_change_count", 0) / max(steps, 1)
            overspeed_rate = info.get("overspeed_count", 0) / max(steps, 1)
            episode_rows.append({
                "step": self.num_timesteps,
                "episode": episode,
                "seed": seed,
                "eval_density": self.eval_density,
                "reward": total_reward,
                "original_reward": total_original_reward,
                "collision": collision,
                "success": 1 - collision,
                "tailgating_rate": tailgating_rate,
                "unsafe_lane_change_rate": unsafe_lane_change_rate,
                "overspeed_rate": overspeed_rate,
                "lane_changes": lane_changes,
                "speed": np.mean(speeds) if speeds else 0.0,
                "survival_time": steps,
                "safety_violation_score": collision * 10 + tailgating_rate + unsafe_lane_change_rate,
            })

        episode_df = pd.DataFrame(episode_rows)
        row = {}
        for column in METRIC_COLUMNS:
            values = episode_df[column].astype(float)
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / sqrt(len(values)) if len(values) > 1 else 0.0
            row[column] = mean
            row[f"{column}_std"] = 0.0 if pd.isna(std) else std
            row[f"{column}_ci95"] = 0.0 if pd.isna(ci95) else ci95

        row["step"] = self.num_timesteps
        row["eval_density"] = self.eval_density
        row["eval_seed_start"] = self.eval_seed_start
        row["eval_seed_end"] = self.eval_seed_start + self.n_eval_episodes - 1
        row["eval_episodes"] = self.n_eval_episodes
        self.records.append(row)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)
        if mlflow.active_run():
            mlflow.log_metrics({
                f"eval/{key}": float(value)
                for key, value in row.items()
                if key not in METADATA_COLUMNS and np.isscalar(value)
            }, step=self.num_timesteps)
        self._last_eval_step = self.num_timesteps

        if self.verbose:
            print(
                f"Eval step={self.num_timesteps}: "
                f"reward={row['reward']:.3f}, "
                f"collision={row['collision']:.3f}, "
                f"tailgating={row['tailgating_rate']:.3f}, "
                f"success={row['success']:.3f}"
            )
