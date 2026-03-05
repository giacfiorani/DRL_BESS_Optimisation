import numpy as np
from pathlib import Path
import pandas as pd
from envs import reward_scaling

# ====
# Load Whole dataset dataframe
# === 
ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "training_data.parquet"
df = pd.read_parquet(DATA_PATH).copy()


# ====
# Episode Parameters
# ====

episode_days = 30

# ========
# BESS HARDWARE SPECIFICATIONS (CATL EnerOne, Liquid-Cooled Rack)
# ========
N_cells = 416  # 1P416S: 416 cells in series
Q_cell = 280  # Ah (CATL 280Ah LFP prismatic cells)
Q_cell_C = 280 * 3600  # C (Coulombs) for Coulomb counting: Q = 280 Ah × 3600 s/h
V_nominal = 1331.2  # V (nominal pack voltage)
E_nominal = 372.7  # kWh (nominal energy)
E_max = 0.3727  # MWh (same as E_nominal, converted to MWh)
R_cell_mOhm = 0.4 # mΩ per cell (from Product Specification Sheet)

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
# DEGRADATION MODEL PARAMETERS
# ======
deg_kappa = 25.0
deg_alpha = 4.0

# ========
# POWER AND TIMING PARAMETERS
# ========
C_rate = 0.5  # C-rate (0.5C = 2-hour discharge)
P_max_MW = C_rate * E_max  # MW (power step: P_step = 0.5C × E_nominal = 0.18635 MW)
dt = 0.5  # hours (timestep = 30 minutes)

# ========
# REWARD FUNCTION PARAMETERS
# ========
lambda_ci = 0.9  # Carbon penalty weight (λ) - tunable scalar

# For discrete action space, we need to define the number of power levels
n_power_levels = 11 

#reward scaling values
S_profit = reward_scaling.S_profit
S_carbon_gbp= reward_scaling.S_carbon_gbp

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