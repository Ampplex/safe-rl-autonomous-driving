import gymnasium as gym
import highway_env
from environments.safe_reward_wrapper import SafeRewardWrapper


def make_env(
    vehicles_count=30,
    duration=30,
    safe=False
):
    env = gym.make("highway-v0")

    env.unwrapped.configure({
        "vehicles_count": vehicles_count,
        "duration": duration,
        "lanes_count": 3,
        "policy_frequency": 1,
        "simulation_frequency": 10,
    })

    env.reset()

    if safe:
        env = SafeRewardWrapper(env)

    return env
