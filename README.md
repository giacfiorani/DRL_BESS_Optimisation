# Rank-Based Action Parameterisation for Multi-Objective Joint Day-Ahead and Intraday Battery Storage Dispatch via Deep Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![Gymnasium](https://img.shields.io/badge/env-Gymnasium-red)

## Overview

Battery energy storage systems (BESS) are being deployed at record scale across Great Britain, yet joint Day-Ahead (DA) commitment and real-time Intraday (ID) dispatch under price uncertainty, throughput degradation, and carbon costs remain an open optimisation challenge. This work proposes a three-dimensional action parameterisation that reduces joint DA–ID scheduling to dispatch power, commitment aggressiveness, and planning power, with a rank-based expansion algorithm mapping these to a state-of-charge-feasible 48-slot schedule. Applied with Soft Actor-Critic (SAC) and benchmarked against Dueling DQN (D3QN), DDQN, DQN, a P20/P80 heuristic, and a perfect-foresight LP oracle over a 219-day held-out test (five seeds), the parameterised formulation achieves £337k profit (42.5% LP efficiency) against £251k for D3QN (31.6%); P20/P80 loses money. A five-seed Pareto sweep identifies a non-monotonic optimum at λ_ci = 0.1 where profit and carbon displacement are jointly maximised, driven by the UK gas-peaker correlation between carbon intensity and DA price.

## Key Features

- **R-int equivalent circuit model** — battery physics solved via quadratic power–current relationship with OCV–SoC lookup, charge/discharge efficiencies, and a hard SoC safety shield
- **Dual-market settlement** — explicitly models Day-Ahead commitment at 12:00 D−1 and real-time Intraday execution with separate revenue streams and no information leakage
- **Rank-based DA expansion algorithm** — maps a scalar aggressiveness parameter α to a 48-slot SoC-feasible DA schedule by activating settlement periods in descending price-spread order, reducing the 49D joint DA–ID problem to three continuous variables
- **Controlled agent hierarchy** — DQN → DDQN → D3QN within an identical 5,808-action space isolates the contribution of overestimation correction and advantage–value factorisation
- **SAC with auto-entropy tuning** — twin-critic soft actor-critic with reparameterisation policy and learned temperature parameter α; operates over the continuous 3D parameterisation
- **Perfect-foresight LP oracle** — PuLP-based linear programme with full knowledge of future prices as an efficiency ceiling benchmark
- **Five-seed evaluation** — all test results reported as mean ± standard deviation over five random seeds on a chronologically held-out 219-day test split
- **Multi-objective Pareto sweep** — five λ_ci values sweeping the profit–carbon displacement frontier with five seeds each

## Architecture

The agent interacts with a custom Gymnasium environment (`envs/battery_env.py`) that models a 99.9 MWh / 49.95 MW LFP grid-scale BESS operating in the UK wholesale market. Each environment step corresponds to a 30-minute settlement period; episodes span a configurable number of delivery days.

**Observation space (103-dimensional):** seven scalar features (SoC, ID price, carbon intensity, settlement period index, previous applied power, DA availability flag, planned DA power) plus the 48-slot DA price curve and 48-slot carbon-intensity forecast for the next delivery day, all normalised via `tanh(x / S_95)` using 95th-percentile data-derived scales.

**Action parameterisation:** the discrete agents use a flattened 5,808-action space (11 dispatch levels × 11 planning levels × 48 DA slots), encoded and decoded via `utils/action_encoding.py`. SAC operates over a continuous 3D vector — (dispatch power fraction, commitment aggressiveness α, planning power fraction) — which the rank-based expansion algorithm in `envs/battery_env.py` converts to a full 48-slot SoC-feasible DA schedule at each step.

**Reward:** `r_t = R_total / scale_universal − λ_ci × (AEF × E_import − MEF × E_export) × UKA_price`, where `scale_universal = S_profit / E_max` is frozen from training-set 95th percentiles. The asymmetric carbon attribution uses the Average Emission Factor (AEF) on net imports and the Marginal Emission Factor (MEF) on net exports, priced at UK Allowance (UKA) rates.

**Degradation:** throughput degradation following Cortés-Arcos et al. (2020) Eq. 23: `C_deg = κ × |P_act_MW| × Δt`, with κ = £10/MWh derived from CATL cell cost and rated cycle life.

## Repository Structure

```
bess-rl/
├── agents/
│   ├── dqn_agent.py                # Standard DQN
│   ├── ddqn_agent.py               # Double DQN (overestimation correction)
│   ├── d3qn.py                     # Dueling D3QN (V/A factorisation)
│   ├── d3qn_per_agent.py           # D3QN with Prioritised Experience Replay
│   ├── sac_agent.py                # Soft Actor-Critic with auto-entropy
│   ├── networks.py                 # MLP, Dueling, SAC actor/critic architectures
│   ├── replay_buffer.py            # Uniform replay buffer (discrete agents)
│   ├── per_buffer.py               # Prioritised replay buffer (SumTree)
│   ├── continuous_replay_buffer.py # Float32 replay buffer (SAC)
│   └── hyperparams.py              # Optuna-tuned hyperparameter dictionaries
├── envs/
│   ├── battery_env.py              # Gymnasium BESS environment
│   ├── env_config.py               # Hardware and market configuration
│   ├── degradation.py              # Throughput degradation model
│   └── reward_scaling.py           # Data-driven reward normalisation
├── training/
│   ├── train.py                    # Discrete agent training (DQN / DDQN / D3QN)
│   ├── train_sac.py                # SAC training loop with early stopping
│   └── optuna_search.py            # Bayesian HPO (Optuna TPE, MedianPruner)
├── benchmarks/
│   └── lp_benchmark.py             # Perfect-foresight LP oracle (PuLP)
├── evaluation/
│   └── evaluate.py                 # Five-seed test-set evaluation
├── utils/
│   └── action_encoding.py          # Encode / decode 5,808-action indices
└── data/
    └── training_data.parquet       # UK DA / ID / CI dataset (2022–2026)
```

## Installation

```bash
git clone https://github.com/<your-username>/bess-rl.git
cd bess-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. The data file `data/training_data.parquet` must be present before running any training or evaluation script.

## Usage

### Train a discrete agent (DQN / DDQN / D3QN)

```bash
python training/train.py --agent d3qn --run-id 1
```

Optional flags: `--n-episodes 1000`, `--seeds 0 1 2 3 4`, `--val-interval 50`, `--patience 200`.

### Train SAC

```bash
python training/train_sac.py --run-id 1 --seeds 0 1 2 3 4
```

Optional flags: `--n-episodes 1000`, `--warmup-steps 5000`, `--lambda-ci 0.1`.

### Bayesian hyperparameter search

```bash
python training/optuna_search.py --agent d3qn --n-trials 30
python training/optuna_search.py --agent sac   --n-trials 30
```

Results are stored in `optuna_results/<agent>_rand_study.db`. Pass `--show-best` to print the best trial without running new trials.

### LP oracle benchmark

```bash
python benchmarks/lp_benchmark.py
```

### Evaluate on the held-out test split

```bash
python evaluation/evaluate.py --agent sac --model-dir models/<run_name>
```

### Monitor training

```bash
tensorboard --logdir runs/
```

## Results

Test-set performance over the 219-day held-out period (2025-05-26 – 2026-01-01), five seeds. LP efficiency is agent profit expressed as a percentage of the perfect-foresight LP oracle ceiling (£793.0k).

| Agent | Net Profit (£k) | LP Efficiency (%) | Net Carbon (tCO₂)¹ |
|---|---|---|---|
| **SAC (3D)** | **337.3 ± 48.4** | **42.5** | **−4,898** |
| D3QN | 250.9 ± 24.6 | 31.6 | — |
| DDQN | 118.1 ± 50.9 | 14.9 | — |
| DQN | 8.6 ± 27.5 | 1.1 | — |
| P20/P80 heuristic | −101.1 | −12.7 | — |
| LP oracle | 793.0 | 100.0 | — |

¹ Net carbon displacement (negative = emissions avoided) reported for SAC at λ_ci = 0.1 only; per-agent figures for discrete baselines were not recorded in the primary evaluation.

Key findings:
- SAC outperforms D3QN by 1.34× (£337k vs £251k) despite operating in a fundamentally different action space; the gap conflates action-space design, algorithm family, and power granularity.
- The DQN → DDQN → D3QN hierarchy (£8.6k → £118.1k → £250.9k) within an identical 5,808-action space provides clean evidence that overestimation correction and advantage–value factorisation are each necessary at this action-space scale.
- The Pareto sweep is non-monotonic: λ_ci = 0.1 simultaneously maximises profit and carbon displacement, explained by the UK grid's correlation between gas-peaker dispatch and DA price.

## Citation

```bibtex
@thesis{Fiorani2025BESSDispatch,
  author      = {Fiorani, Giacomo},
  title       = {Rank-Based Action Parameterisation for Multi-Objective Joint
                 Day-Ahead and Intraday Battery Storage Dispatch via Deep
                 Reinforcement Learning},
  school      = {University College London},
  year        = {2025},
  type        = {MEng Dissertation},
  department  = {Mechanical Engineering},
}
```
