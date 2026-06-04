import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mlflow
from stable_baselines3 import PPO

from configs.config import PPO_CONFIG, SAFE_CONFIG
from environments.env_factory import make_env
from environments.safe_reward_wrapper import SafeRewardWrapper
from tracking.mlflow_logger import setup_mlflow
from training.safety_eval_callback import SafetyEvalCallback


def build_safe_config(
    safety_lambda=None,
    collision_only=False,
    collision_penalty=None,
    tailgating_penalty=None,
    lane_change_penalty=None,
    speed_penalty=None,
):
    safe_config = SAFE_CONFIG.copy()
    if safety_lambda is not None:
        safe_config["safety_lambda"] = safety_lambda
    if collision_penalty is not None:
        safe_config["collision_penalty"] = collision_penalty
    if tailgating_penalty is not None:
        safe_config["tailgating_penalty"] = tailgating_penalty
    if lane_change_penalty is not None:
        safe_config["lane_change_penalty"] = lane_change_penalty
    if speed_penalty is not None:
        safe_config["speed_penalty"] = speed_penalty
    if collision_only:
        safe_config["tailgating_penalty"] = 0
        safe_config["lane_change_penalty"] = 0
        safe_config["speed_penalty"] = 0
    return safe_config


def default_model_path(safety_lambda=None, collision_only=False, model_suffix=None):
    suffix = model_suffix
    if suffix is None:
        if collision_only:
            suffix = "_collision_only"
        elif safety_lambda is not None:
            suffix = f"_lam{safety_lambda}"
        else:
            suffix = ""
    return f"models/safeppo_agent{suffix}"


def train_safe_ppo(
    safety_lambda=None,
    timesteps=None,
    collision_only=False,
    collision_penalty=None,
    tailgating_penalty=None,
    lane_change_penalty=None,
    speed_penalty=None,
    model_suffix=None,
    model_path=None,
    record_curves=False,
    curve_file="results/learning_curves/safeppo_curve.csv",
    eval_freq=5000,
    eval_episodes=30,
    eval_density=50,
    eval_seed_start=20_000,
    run_name=None,
):
    setup_mlflow()
    if run_name is not None:
        mlflow.set_tag("mlflow.runName", run_name)
    mlflow.set_tag("model_type", "Safe PPO")
    mlflow.set_tag("curve_run", str(record_curves))

    safe_config = build_safe_config(
        safety_lambda=safety_lambda,
        collision_only=collision_only,
        collision_penalty=collision_penalty,
        tailgating_penalty=tailgating_penalty,
        lane_change_penalty=lane_change_penalty,
        speed_penalty=speed_penalty,
    )

    full_config = PPO_CONFIG.copy()
    full_config.update(safe_config)
    full_config["safe_ppo"] = True
    full_config["collision_only"] = collision_only
    full_config["record_curves"] = record_curves
    full_config["train_density"] = 30
    full_config["eval_density"] = eval_density
    full_config["eval_seed_start"] = eval_seed_start
    full_config["eval_seed_end"] = eval_seed_start + eval_episodes - 1
    full_config["eval_freq"] = eval_freq
    full_config["eval_episodes"] = eval_episodes
    full_config["eval_protocol"] = "heldout_fixed_seed_bank"
    if timesteps is not None:
        full_config["total_timesteps"] = timesteps
    mlflow.log_params(full_config)

    base_env = make_env(safe=False)
    env = SafeRewardWrapper(base_env)
    env.cfg = safe_config

    callbacks = []
    eval_env = None
    if record_curves:
        eval_env = SafeRewardWrapper(make_env(vehicles_count=eval_density, safe=False))
        eval_env.cfg = safe_config
        callbacks.append(
            SafetyEvalCallback(
                eval_env,
                csv_path=curve_file,
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                eval_seed_start=eval_seed_start,
                eval_density=eval_density,
                verbose=1,
            )
        )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/safeppo",
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
        tb_log_name="SafePPO",
    )

    os.makedirs("models", exist_ok=True)
    save_path = model_path or default_model_path(safety_lambda, collision_only, model_suffix)
    model.save(save_path)
    artifact_path = save_path if save_path.endswith(".zip") else f"{save_path}.zip"
    if os.path.exists(artifact_path):
        mlflow.log_artifact(artifact_path)
    if record_curves and os.path.exists(curve_file):
        mlflow.log_artifact(curve_file)

    env.close()
    if eval_env is not None:
        eval_env.close()
    mlflow.end_run()
    print(
        "Safe PPO training complete "
        f"(lambda={safe_config['safety_lambda']}, timesteps={training_timesteps}, "
        f"collision_only={collision_only})"
    )
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety_lambda", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--collision_only", action="store_true")
    parser.add_argument("--collision_penalty", type=float, default=None)
    parser.add_argument("--tailgating_penalty", type=float, default=None)
    parser.add_argument("--lane_change_penalty", type=float, default=None)
    parser.add_argument("--speed_penalty", type=float, default=None)
    parser.add_argument("--model_suffix", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--record_curves", action="store_true")
    parser.add_argument("--curve_file", type=str, default="results/learning_curves/safeppo_curve.csv")
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--eval_density", type=int, default=50)
    parser.add_argument("--eval_seed_start", type=int, default=20_000)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    train_safe_ppo(
        safety_lambda=args.safety_lambda,
        timesteps=args.timesteps,
        collision_only=args.collision_only,
        collision_penalty=args.collision_penalty,
        tailgating_penalty=args.tailgating_penalty,
        lane_change_penalty=args.lane_change_penalty,
        speed_penalty=args.speed_penalty,
        model_suffix=args.model_suffix,
        model_path=args.model_path,
        record_curves=args.record_curves,
        curve_file=args.curve_file,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_density=args.eval_density,
        eval_seed_start=args.eval_seed_start,
        run_name=args.run_name,
    )
