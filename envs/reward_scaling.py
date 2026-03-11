import numpy as np
from pathlib import Path
import pandas as pd
import envs.env_config

ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "data.parquet"
df = pd.read_parquet(DATA_PATH).copy()

# Battery step magnitude (worst-case)
E_max = 0.3727   # MWh
C_rate = 0.5
P_max_MW = C_rate * E_max
dt = 0.5
E_step_MWh = P_max_MW * dt
E_step_kWh = E_step_MWh * 1000.0

# ====
# REWARD SCALING
#=====

# --- Max Degradation Magnitude — Cortés-Arcos et al. (2020) Eq. 23 ---
# Read from env_config so kappa is defined in exactly one place
deg_kappa = 35
max_deg_cost = deg_kappa * E_step_MWh

# ---- PROFIT scale (DA+ID combined) ----
da_abs  = df["da_price_gbp_mwh"].astype(float).abs().to_numpy()
mid_abs = df["mid_price_gbp_mwh"].astype(float).abs().to_numpy()

# Add max_deg_cost to account for the worst-case net-negative cashflow
profit_series = ((da_abs + mid_abs) * E_step_MWh) + max_deg_cost   
S_profit = np.percentile(profit_series, 95)

# ---- CARBON CASHFLOW scale (monetised, £) ----
ci_abs  = df["ci_actual_gco2_kwh"].astype(float).abs().to_numpy()   # CI g/kWh
mef_abs = df["mef_gco2_kwh"].astype(float).abs().to_numpy()      # MEF g/kWh
uka_abs = df["uka_gbp_tco2"].astype(float).abs().to_numpy()      # £/tCO2

# Conservative magnitude: worst-case uses CI + MEF
tco2_series = (ci_abs + mef_abs) * E_step_kWh / 1e6              # tCO2 per step proxy
carbon_cashflow_series = uka_abs * tco2_series                    # £ per step proxy
S_carbon_gbp = np.percentile(carbon_cashflow_series, 95)

# =====
# OBSERVATION SCALING
# =====

# ---- Price scale (for observation normalisation) ----
price_series = np.abs(df["da_price_gbp_mwh"].to_numpy(dtype=float))
S_price = np.percentile(price_series, 95)   # ~£/MWh typical peak

# ---- Carbon Intensity scale (for observation normalisation) ----
ci_series = np.abs(df["ci_actual_gco2_kwh"].to_numpy(dtype=float))
S_ci = np.percentile(ci_series, 95)

# PRINTING SCALES

print(f"S_price: {S_price:.2f} £/MWh")
print(f"S_ci:    {S_ci:.2f} gCO2/kWh")
print("S_profit (DA+ID + Deg, £):", S_profit)
print("S_carbon (monetised, £):", S_carbon_gbp)