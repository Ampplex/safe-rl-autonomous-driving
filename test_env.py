from environments.env_factory import make_env

env = make_env()

obs, info = env.reset()

print("Observation shape:", obs.shape)

for _ in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Reward={reward:.3f}"
    )

env.close()
