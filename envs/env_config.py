import numpy as np
from pathlib import Path
import pandas as pd
import envs.reward_scaling
from envs.reward_scaling import compute_scales

# ====
# Load Whole dataset dataframe
# === 
ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "data.parquet"
df = pd.read_parquet(DATA_PATH).copy()


# ====
# Episode Parameters
# ====

episode_days = 30

# ========
# BESS HARDWARE SPECIFICATIONS (CATL EnerOne, Liquid-Cooled Rack)
# ========
# Single Cabinet Configuration (1P416S)
N_cells_per_cabinet = 416  # 1P416S: 416 cells in series
Q_cell = 280  # Ah (CATL 280Ah LFP prismatic cells)
V_nominal = 1331.2  # V (nominal pack voltage, invariant under parallel wiring)
E_nominal_per_cabinet = 372.7  # kWh (single cabinet nominal energy)
R_cell_mOhm = 0.4  # mΩ per cell (from Product Specification Sheet)

# Parallel Wiring Scaling (Utility-Scale Plant)
# Derivation: 99.7 MWh / 0.3727 MWh = 268 cabinets in parallel
N_cabinets = 268

# Scaled Plant Specifications
N_cells = N_cells_per_cabinet  # Total series depth per branch (unchanged)
Q_cell_C = Q_cell * N_cabinets * 3600  # C (Coulombs) — scaled pack charge: 280 Ah × 268 × 3600 s/h
E_nominal = E_nominal_per_cabinet * N_cabinets  # kWh (scaled nominal energy: 372.7 × 268)
E_max = E_nominal / 1000.0  # MWh (99,883.6 kWh → 99.9 MWh)


# ========
# OPERATIONAL PARAMETERS
# ========
SoC_min = 0.1  # Minimum SoC (fraction)
SoC_max = 0.9  # Maximum SoC (fraction)
SoC_initial = 0.5  # Initial SoC (fraction) - set to mid-range for safety

# ========
# EFFICIENCY PARAMETERS
# ========
eff_dis = 1.0  # Discharge efficiency (η_discharge = 1 per MDP)
eff_ch = 0.99  # Charge efficiency (η_charge = 0.99)
self_dis = 0.0  # Self-discharge rate per timestep (set to 0 for simplicity)

# ======
# DEGRADATION MODEL PARAMETERS — Cortés-Arcos et al. (2020) Eq. 23, Version 2 Cyclic
# κ = Cost_bat / Q_lifetime_MWh  [£/MWh]
#
# Updated derivation for 2024/2025 CATL EnerOne 280Ah LFP at 0.5C:
#   Cost_bat      = £120/kWh × 372.7 kWh = £44,724
#                   (BNEF LCOE 2024; BEIS Energy Storage Capital Cost Report 2024:
#                    £100–150/kWh installed for utility-scale LFP BESS)
#   N_cycles      = 8,000 (CATL EnerOne at 0.5C, DoD=80%, to 80% retained capacity;
#                    LFP cycle life at moderate C-rates: 6,000–10,000 cycles)
#   DoD           = 0.80  (SoC_min=0.1, SoC_max=0.9)
#   Q_lifetime    = 8000 × 2 × 0.80 × 372.7 kWh / 1000 = 4,770 MWh (bidirectional)
#   κ             = £44,724 / 4,770 = £9.38/MWh → 10.0 £/MWh
#
# Previous value (£35/MWh) used 2022 cell costs (£200/kWh) and 1C cycle life
# (3,500 cycles). This made cycling unprofitable below ~£70/MWh daily spreads,
# causing the agent to rationally idle in the 2025 test set.
# ======
deg_kappa = 10.0

scales = compute_scales(train_ratio=0.70, train_start=None)

# CARBON Thresholds

alpha_thresh = 100.0
monthly_budget = 20.0
scale_numeric = 5.0


# ========
# POWER AND TIMING PARAMETERS
# ========
C_rate = 0.5  # C-rate (0.5C = 2-hour discharge)
P_max_MW = 0.5 * E_max  # MW (C-rate 0.5 scales with energy: 0.5 × 0.0999 MWh = 0.04995 MW ≈ 49.9 MW)
dt = 0.5  # hours (timestep = 30 minutes)

# For discrete action space, we need to define the number of power levels
n_power_levels = 11 

#reward scaling values
S_profit = envs.reward_scaling.S_profit
S_carbon_gbp= envs.reward_scaling.S_carbon_gbp

# =======
# OBSERVATION SCALING
# =======

S_price = envs.reward_scaling.S_price
S_ci    = envs.reward_scaling.S_ci


# ========
# OCV LOOKUP TABLE (The DC OCV-SOC Curve from Spec Sheet @25°C)
# ========

def ocv_lookup_table():
    ocv_soc_points = np.array(
            [0.00, 0.05, 0.10, 0.15, 0.20,
             0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70,
             0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
            dtype=np.float32,
        )
    
    ocv_cell_volts = np.array(
            [2.893, 3.182, 3.205, 3.230, 3.250,
             3.264, 3.283, 3.288, 3.288, 3.289, 
             3.290, 3.293, 3.303, 3.327, 3.329,
             3.329, 3.330, 3.330, 3.331, 3.332, 3.386],
            dtype=np.float32,
        )
    return ocv_soc_points, ocv_cell_volts