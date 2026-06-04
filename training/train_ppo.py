import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mlflow
from stable_baselines3 import PPO

from configs.config import PPO_CONFIG
from environments.env_factory import make_env
from environments.safe_reward_wrapper import SafeRewardWrapper
from tracking.mlflow_logger import setup_mlflow
from training.safety_eval_callback import SafetyEvalCallback


def train_ppo(
    timesteps=None,
    record_curves=False,
    curve_file="results/learning_curves/ppo_baseline_curve.csv",
    eval_freq=5000,
    eval_episodes=10,
    model_path="models/ppo_baseline",
):
    setup_mlflow()
    full_config = PPO_CONFIG.copy()
    if timesteps is not None:
        full_config["total_timesteps"] = timesteps
    full_config["record_curves"] = record_curves
    mlflow.log_params(full_config)

    env = make_env()
    callbacks = []
    eval_env = None
    if record_curves:
        eval_env = SafeRewardWrapper(make_env(safe=False))
        callbacks.append(
            SafetyEvalCallback(
                eval_env,
                csv_path=curve_file,
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                verbose=1,
            )
        )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/ppo",
        learning_rate=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
    )

    training_timesteps = timesteps if timesteps is not None else PPO_CONFIG["total_timesteps"]
    model.learn(
        total_timesteps=training_timesteps,
        callback=callbacks or None,
        tb_log_name="PPO",
    )

    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    artifact_path = f"{model_path}.zip"
    if os.path.exists(artifact_path):
        mlflow.log_artifact(artifact_path)
    if record_curves and os.path.exists(curve_file):
        mlflow.log_artifact(curve_file)

    env.close()
    if eval_env is not None:
        eval_env.close()
    mlflow.end_run()
    print("PPO training complete and logged to MLflow")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--record_curves", action="store_true")
    parser.add_argument("--curve_file", type=str, default="results/learning_curves/ppo_baseline_curve.csv")
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="models/ppo_baseline")
    args = parser.parse_args()

    train_ppo(
        timesteps=args.timesteps,
        record_curves=args.record_curves,
        curve_file=args.curve_file,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        model_path=args.model_path,
    )
