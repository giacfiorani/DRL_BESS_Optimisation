from battery_env import BatteryEnv
import env_config
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt

env = BatteryEnv(env_config)

check_env(env, skip_render_check=True)

obs, info = env.reset()

soc_history = [obs[0]]
reward_history = []
action_idx_history = []
power_requested_history = []
power_applied_history = []

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    soc_history.append(obs[0])
    reward_history.append(reward)
    action_idx_history.append(action)
    power_requested_history.append(env.power_levels[action])  # MW
    power_applied_history.append(info["P_applied_MW"])

    # sanity check
    if not (env.SoC_min - 1e-6 <= obs[0] <= env.SoC_max + 1e-6):
        print("SoC out of bounds!", obs[0])
        break

    done = terminated or truncated

plt.figure(figsize=(12,5))
plt.plot(soc_history)
plt.title("State of Charge over time (Random Policy)")
plt.xlabel("Step")
plt.ylabel("SoC")
plt.grid(True)

plt.figure(figsize=(12,5))
plt.plot(reward_history)
plt.title("Reward over time (Random Policy)")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)

plt.figure(figsize=(12,5))
plt.plot(power_requested_history)
plt.title("Requested Power over time (MW)")
plt.xlabel("Step")
plt.ylabel("P_requested (MW)")
plt.grid(True)

plt.figure(figsize=(12,5))
plt.plot(power_applied_history)
plt.title("Applied Grid Power over time (MW)")
plt.xlabel("Step")
plt.ylabel("P_applied (MW)")
plt.grid(True)

plt.show()