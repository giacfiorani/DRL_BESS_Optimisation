import numpy as np
from pathlib import Path
import pandas as pd
import envs.reward_scaling
from envs.reward_scaling import compute_scales

ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "data.parquet"
df = pd.read_parquet(DATA_PATH).copy()

episode_days = 30

# BESS HARDWARE SPECIFICATIONS (CATL EnerOne, Liquid-Cooled Rack)
# Single Cabinet Configuration (1P416S)
N_cells_per_cabinet = 416         # 1P416S: 416 cells in series
Q_cell              = 280          # Ah (CATL 280Ah LFP prismatic cells)
V_nominal           = 1331.2       # V (nominal pack voltage, invariant under parallel wiring)
E_nominal_per_cabinet = 372.7      # kWh (single cabinet nominal energy)
R_cell_mOhm         = 0.4          # mΩ per cell (from Product Specification Sheet)

# Parallel Wiring Scaling (Utility-Scale Plant)
# Derivation: 99.7 MWh / 0.3727 MWh per cabinet = 268 cabinets in parallel
N_cabinets = 268

# Scaled Plant Specifications
N_cells  = N_cells_per_cabinet                    # Series depth per branch (unchanged by parallel wiring)
Q_cell_C = Q_cell * N_cabinets * 3600             # C — scaled pack charge: 280 Ah × 268 × 3600 s/h
E_nominal = E_nominal_per_cabinet * N_cabinets    # kWh — 372.7 × 268
E_max     = E_nominal / 1000.0                    # MWh (99,883.6 kWh ≈ 99.9 MWh)

SoC_min     = 0.1   # Minimum SoC (fraction)
SoC_max     = 0.9   # Maximum SoC (fraction)
SoC_initial = 0.5   # Initial SoC (fraction)

eff_dis  = 1.0   # Discharge efficiency (η_discharge)
eff_ch   = 0.99  # Charge efficiency (η_charge)
self_dis = 0.0   # Self-discharge rate per timestep (neglected)

# DEGRADATION MODEL PARAMETERS — Cortés-Arcos et al. (2020) Eq. 23, Version 2 Cyclic
# κ = Cost_bat / Q_lifetime_MWh  [£/MWh]
#
# Updated derivation for 2024/2025 CATL EnerOne 280Ah LFP at 0.5C:
#   Cost_bat   = £120/kWh × 372.7 kWh = £44,724
#                (BNEF LCOE 2024; BEIS Energy Storage Capital Cost Report 2024:
#                 £100–150/kWh installed for utility-scale LFP BESS)
#   N_cycles   = 8,000 (CATL EnerOne at 0.5C, DoD=80%, to 80% retained capacity;
#                LFP cycle life at moderate C-rates: 6,000–10,000 cycles)
#   DoD        = 0.80  (SoC_min=0.1, SoC_max=0.9)
#   Q_lifetime = 8000 × 2 × 0.80 × 372.7 kWh / 1000 = 4,770 MWh (bidirectional)
#   κ          = £44,724 / 4,770 ≈ £9.38/MWh  →  10.0 £/MWh (rounded)
#
# The previous value (£35/MWh) used 2022 cell costs (£200/kWh) and a 1C cycle
# life of 3,500 cycles, making cycling unprofitable below ~£70/MWh daily spreads.
deg_kappa = 10.0

scales = compute_scales(train_ratio=0.70, train_start=None)

alpha_thresh   = 100.0
monthly_budget = 20.0
scale_numeric  = 5.0

C_rate   = 0.5
P_max_MW = 0.5 * E_max  # MW (0.5C rate applied to scaled energy capacity)
dt       = 0.5           # hours

n_power_levels = 11

S_profit     = envs.reward_scaling.S_profit
S_carbon_gbp = envs.reward_scaling.S_carbon_gbp

S_price = envs.reward_scaling.S_price
S_ci    = envs.reward_scaling.S_ci


# OCV-SoC lookup table (DC OCV-SoC curve from CATL EnerOne spec sheet at 25°C)
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
