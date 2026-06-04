PPO_CONFIG = {

    "learning_rate": 3e-4,

    "gamma": 0.99,

    "gae_lambda": 0.95,

    "n_steps": 2048,

    "batch_size": 64,

    "clip_range": 0.2,

    "total_timesteps": 50_000
}

SAFE_CONFIG = {
    "target_speed": 25,
    "collision_penalty": 50,
    "tailgating_penalty": 5,
    "lane_change_penalty": 10,
    "speed_penalty": 2,
    "min_lane_change_distance": 15,
    "safe_time_gap": 1.5,
    "safety_lambda": 0.1,
}
