# BESS DRL – UK Grid Battery Optimisation with DQN (Day-Ahead + Real-Time Overlap)

Goal: Train a Deep Reinforcement Learning agent (DQN) to operate a grid-scale
lithium-ion Battery Energy Storage System (BESS) connected to the UK grid using
Elexon-derived half-hourly data, to:

- Maximise arbitrage profit
- Reduce operational carbon impact via carbon-aware dispatch (weighted term)

## Core components

### Environment (`envs/battery_env.py`)
Gymnasium environment modelling a realistic UK market workflow with day-ahead (DA)
publication and overlapping “plan tomorrow + dispatch today” decisions.

Time and data:
- 1 step = 30 minutes (half-hour settlement period)
- Dataset provides:
  - trade_ts (decision / auction timeline)
  - trade_date (day D)
  - delivery_date (day D+1)
  - spot/settlement price proxy (used for reward at current step)
  - DA curve values per trade day (48-slot curve for delivery day D+1)
  - carbon intensity per slot

Observation (shape = 6 + 48):
- Main state (6):
  - SoC
  - spot price now
  - carbon intensity now
  - tau (1..48)
  - previous applied power
  - da_available flag (0/1)
- Tomorrow DA price curve (48):
  - visible only after DA publish time (e.g., 13:00 on the trade day)
  - before publish: curve is exactly zeros (no information leak)

Action space (MultiDiscrete):
- action = [dispatch_idx, plan_idx]
  - dispatch_idx: real-time dispatch request for the current slot (if not committed)
  - plan_idx: candidate DA commitment written into tomorrow_plan[tau] only if DA is published

Planning / execution logic (Design B):
- The environment maintains rolling buffers:
  - today_plan[48]: commitments for the current execution day
  - tomorrow_plan[48]: commitments being built for the next day
- “No plan” sentinel:
  - today_plan[tau] = -1 means no DA commitment for that slot → agent can dispatch freely
- Executed dispatch each step:
  - if today_plan[tau] >= 0: dispatch_idx_exec = today_plan[tau] (agent dispatch is ignored)
  - else: dispatch_idx_exec = dispatch_idx from the agent
- DA updates:
  - if da_available at this step: tomorrow_plan[tau] = plan_idx
  - else: tomorrow_plan is unchanged (no writes before publish)
- Day rollover:
  - at end of 48 slots: today_plan ← tomorrow_plan, tomorrow_plan reset to -1

Battery physics and constraints:
- R-int equivalent circuit model (OCV(SOC) + internal resistance)
- Power request → current solved from quadratic relationship
- SoC update includes charge/discharge efficiency
- SoC protection clamps power/current to enforce SoC_min / SoC_max

Reward:
- reward = profit − carbon_penalty
  - profit = E_MWh × price_now
  - carbon_penalty = λ × (E_import_kWh − E_export_kWh) × CI_now
- (λ is configurable for carbon-aware behaviour)

Episode definition:
- Episodes span a configurable number of trade days (e.g., 30), sampled from valid
  complete 48-slot days in the dataset.

### Configuration (`env_config.py`)
- Battery parameters: P_max_MW, SoC_min/max, efficiencies, timestep, OCV lookup table, R_sys, etc.
- Market / reward parameters: publish_hour, λ (carbon weight)
- RL hyperparameters (kept separate in your training scripts/configs)

## Training intent (DQN)

Primary objective:
- Learn a policy that:
  - uses DA information after publication to build tomorrow’s plan (tomorrow_plan)
  - respects DA commitments during execution day (today_plan)
  - dispatches freely when no commitment exists (planned_idx = -1)

Key modelling feature:
- “No cheating” guarantee:
  - before publish: DA curve is hidden (zeros) and planning writes are blocked
  - after publish: agent sees only the correct day’s DA curve (no future-of-future leak)

## Planned extensions
- Baselines:
  - heuristic policies (e.g., charge in low-price quantile, discharge in high-price quantile)
  - “DA-only optimiser” vs “reactive-only” comparison
- Reward refinement:
  - degradation proxy / cycle cost
  - terminal value / SoC target shaping
- Algorithm upgrades:
  - Double DQN / Dueling / Prioritized Replay (if needed)
  - multi-day planning architectures (sequence models) if DA planning dominates
