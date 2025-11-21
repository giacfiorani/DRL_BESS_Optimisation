# BESS DRL – Grid-Scale Battery Optimization with DQN

Goal: Train a Deep Reinforcement Learning agent (DQN) to operate a lithium-ion
Battery Energy Storage System (BESS) connected to the UK grid, using Elexon
price data and carbon intensity, to:

- Maximise arbitrage profit
- Minimise lifecycle carbon emissions (via carbon-aware dispatch)

Core components:
- `envs/battery_env.py`: Gymnasium environment implementing:
  - State: [SoC, price, carbon_intensity, time_index]
  - Action: charge / idle / discharge (discrete, first version)
  - Reward: profit – λ × carbon_cost
  - SoC dynamics with efficiency and SoC bounds
- `config.py`: Battery technical parameters (P_max, E_max, SoC_min/max,
  efficiencies, dt, λ, etc.) and training hyperparameters.

Planned:
- Discrete → continuous actions (C-rate or power)
- Degradation-aware reward
- Baseline heuristics vs DQN performance comparison
