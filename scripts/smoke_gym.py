import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
obs, info = env.reset(seed=0)
done = False
steps = 0
while not done and steps < 20:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
print("✅ Gymnasium OK; steps:", steps)
