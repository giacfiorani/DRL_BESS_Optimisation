# BESS DRL – UK Grid Battery Optimisation  
## Day-Ahead Planning + Intraday Rebalancing with Deep Reinforcement Learning

This project develops a **research-grade reinforcement learning environment** for
optimising the operation of a **grid-scale lithium-ion Battery Energy Storage System (BESS)**
connected to the **UK electricity grid**.

The goal is to train a Deep Reinforcement Learning (DQN) agent that learns how to:

- maximise electricity market arbitrage revenue  
- incorporate **carbon-aware dispatch** using real carbon-intensity data  
- respect **physical battery constraints**  
- operate under **realistic UK market timelines**, including day-ahead publication and intraday rebalancing  

The environment explicitly models **day-ahead (DA) planning** and **intraday (ID/MID) execution**
as **distinct but interacting decision layers**.

---

## Project Structure
├── data/
│   ├── training_data.parquet      # Processed DA, ID, CI dataset
│   ├── data_loader.py
│   └── processed.py
│
├── envs/
│   ├── battery_env.py             # Gymnasium environment
│   ├── env_config.py              # Battery & market configuration
│   ├── reward_scaling.py          # Reward normalisation scales
│   └── env_test.ipynb             # Environment testing & rollouts
│
└── README.md

---

## Environment (`envs/battery_env.py`)

A custom **Gymnasium-compatible environment** modelling UK electricity market mechanics
with realistic publication timing, physical battery dynamics, and carbon-aware rewards.

---

## Time Structure & Market Timeline

- **1 environment step = 30-minute settlement period**
- **48 settlement periods per delivery day**
- The environment advances in **delivery time**, not trade time

The dataset provides two distinct timestamps:

- `delivery_ts`  
  → physical electricity delivery time (what the battery actually executes)

- `trade_ts`  
  → timestamp when market information (DA curves) becomes available

### Day-Ahead Publication Assumption
- DA prices for delivery day **D** are published at **12:00 on day D-1**
- Before publication: tomorrow’s DA curve is **not visible**
- After publication: the agent may observe and write DA commitments for tomorrow

This ensures **no information leakage**.

---

## Episode Definition

- Episodes are grouped by **delivery date**
- Each episode day consists of **exactly 48 settlement periods**
- Episodes span a configurable number of delivery days (e.g. 30)

Only **complete, strictly 30-minute-cadence days** are used.

---

## Observation Space (shape = 103)

The observation vector is composed of:

### Current delivery-time state (7)
- State of Charge (SoC)
- Intraday (MID) price for current settlement period
- Carbon intensity (gCO₂/kWh)
- Settlement period index τ ∈ {1,…,48}
- Previously applied power
- Day-ahead availability flag (0 or 1)
- Planned DA power for the current settlement period (if any)

### Tomorrow information (only after DA publication)
- Day-ahead price curve for the next delivery day (48 values)
- Carbon-intensity curve for the next delivery day (48 values)

Before DA publication, **all tomorrow curves are exactly zero**.

---

## Action Space

The action is a **MultiDiscrete vector**:
action = [dispatch_idx, plan_idx, plan_slot]

Where:

- `dispatch_idx`  
  → requested real-time charging/discharging power level  

- `plan_idx`  
  → power level written into tomorrow’s DA plan  

- `plan_slot ∈ {0,…,47}`  
  → settlement period of tomorrow’s DA plan to update  

### Planning vs Execution
- **Dispatch is always allowed** (subject to SoC protection)
- **Planning writes** are allowed **only after DA publication**
- The agent gradually builds tomorrow’s DA plan over time

---

## Rolling Day-Ahead Plans

The environment maintains two rolling buffers:

- `today_plan[48]`  
  → fixed DA commitments for the current delivery day  

- `tomorrow_plan[48]`  
  → DA commitments being constructed for the next day  

At day rollover:
today_plan ← tomorrow_plan
tomorrow_plan ← -1

A value of `-1` indicates **no DA commitment** for that slot.

---

## Battery Physics & Safety

- R-int equivalent circuit model  
- Power → current solved via quadratic relationship  
- Charge/discharge efficiencies applied  
- SoC updated every step  

### SoC Safety Shield
- Requested power is **clipped if physically infeasible**
- Guarantees:
  - no over-charge
  - no over-discharge
- The agent learns feasibility implicitly through reward feedback

**Important:**  
SoC protection applies **only to execution**, not to DA planning.

---

## Reward Function

### Raw components

**Day-Ahead revenue**
R_DA = E_plan × p_DA

**Intraday deviation revenue**
R_ID = (E_act − E_plan) × p_ID

**Carbon penalty**
C = λ × (E_import − E_export) × CI

Carbon intensity is expressed in **gCO₂/kWh**.  
Discharging during high-CI periods is rewarded as it offsets fossil generation.

---

## Reward Normalisation

To stabilise DRL training, rewards are **bounded and scaled**:

- Scaling constants are computed as **95th percentiles** over the full dataset:
  - DA profit
  - ID profit
  - carbon penalty

- Normalisation uses a **tanh activation**:
x_norm = tanh(x / S)

### Final reward

r_t = tanh(R_DA / S_DA) • tanh(R_ID / S_ID) − tanh(C / S_CI)

This yields a bounded reward approximately in **[−2, 2]**.

---

## Configuration (`env_config.py`)

Contains:
- Battery parameters (capacity, power limits, efficiencies)
- SoC limits
- OCV lookup table
- Internal resistance
- Carbon weight λ
- Timestep length

Reward scaling constants are stored separately in `reward_scaling.py`.

---

## Training Intent (DQN)

The agent learns a policy that:

- builds DA commitments **only after publication**
- executes real-time dispatch under physical constraints
- arbitrages DA and ID prices
- balances profit against carbon impact

The environment supports:
- DQN
- Double / Dueling DQN
- Future extensions to multi-day or sequence-based planning

---

## Planned Extensions

- Baseline heuristic policies
- Degradation or cycling cost proxy
- Terminal SoC shaping
- Deterministic optimisation benchmarks
- Hierarchical or multi-day planning architectures