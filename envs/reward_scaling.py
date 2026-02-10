import numpy as np
from pathlib import Path
import pandas as pd

# ====
# Load Whole dataset dataframe
# === 
ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "training_data.parquet"
df = pd.read_parquet(DATA_PATH).copy()

E_max = 0.3727  # MWh (same as E_nominal, converted to MWh)
C_rate = 0.5  # C-rate (0.5C = 2-hour discharge)
P_max_MW = C_rate * E_max  # MW (power step: P_step = 0.5C × E_nominal = 0.18635 MW)
dt = 0.5  # hours (timestep = 30 minutes)

# -- Profit and Carbon Penalty Scaling using Percentiles over Distribution
mid_profit_series = df["mid_price_gbp_mwh"].astype(float).abs() * P_max_MW * dt
da_profit_series = df["da_price_gbp_mwh"].astype(float).abs() * P_max_MW * dt

carbon_cost_series = df["carbon_gco2_kwh"].astype(float).abs() * P_max_MW * dt #in kg/MWh

S_mid_profit = np.percentile(mid_profit_series, 95)
S_da_profit = np.percentile(da_profit_series,95)
S_carbon = np.percentile(carbon_cost_series,95)

print("ID Scale:", S_mid_profit)
print("DA Scale:", S_da_profit)
print("Carbon Scale:", S_carbon)
