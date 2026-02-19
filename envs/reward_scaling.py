import numpy as np
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "training_data.parquet"
df = pd.read_parquet(DATA_PATH).copy()

# Battery step magnitude (worst-case)
E_max = 0.3727   # MWh
C_rate = 0.5
P_max_MW = C_rate * E_max
dt = 0.5
E_step_MWh = P_max_MW * dt
E_step_kWh = E_step_MWh * 1000.0

# ---- PROFIT scale (DA+ID combined) ----
da_abs  = df["da_price_gbp_mwh"].astype(float).abs().to_numpy()
mid_abs = df["mid_price_gbp_mwh"].astype(float).abs().to_numpy()
profit_series = (da_abs + mid_abs) * E_step_MWh   # £ per step proxy
S_profit = np.percentile(profit_series, 95)

# ---- CARBON CASHFLOW scale (monetised, £) ----
ci_abs  = df["ci_actual_gco2_kwh"].astype(float).abs().to_numpy()   # CI g/kWh
mef_abs = df["mef_gco2_kwh"].astype(float).abs().to_numpy()      # MEF g/kWh
uka_abs = df["uka_gbp_tco2"].astype(float).abs().to_numpy()      # £/tCO2

# Conservative magnitude: worst-case uses CI + MEF
tco2_series = (ci_abs + mef_abs) * E_step_kWh / 1e6              # tCO2 per step proxy
carbon_cashflow_series = uka_abs * tco2_series                    # £ per step proxy
S_carbon_gbp = np.percentile(carbon_cashflow_series, 95)

print("S_profit (DA+ID, £):", S_profit)
print("S_carbon (monetised, £):", S_carbon_gbp)