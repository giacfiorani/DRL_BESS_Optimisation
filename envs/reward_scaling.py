import numpy as np
from pathlib import Path
import pandas as pd
# NOTE: Do NOT import env_config here to avoid circular import
# Instead, compute E_max directly from hardware specs below

ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "data.parquet"

# Battery step magnitude — COMPUTED from hardware config to avoid circular imports
# Match the exact calculation in env_config.py
N_cells_per_cabinet = 416
Q_cell = 280  # Ah
E_nominal_per_cabinet = 372.7  # kWh
N_cabinets = 268  # Parallel wiring for scaled system

# Scaled nominal energy (kWh) and capacity (MWh)
E_nominal = E_nominal_per_cabinet * N_cabinets
E_max = E_nominal / 1000.0  # MWh

# Scaled power and timestep
C_rate = 0.5
P_max_MW = C_rate * E_max
dt = 0.5  # hours

E_step_MWh = P_max_MW * dt
E_step_kWh = E_step_MWh * 1000.0

# --- Max Degradation Magnitude — Cortés-Arcos et al. (2020) Eq. 23 ---
# Must stay in sync with envs/env_config.py deg_kappa
deg_kappa = 10
max_deg_cost = deg_kappa * E_step_MWh


def compute_scales(
    train_ratio: float = 0.70,
    train_start: str | None = None,
) -> dict:
    """
    Compute 95th-percentile reward and observation scaling constants.

    Parameters
    ----------
    train_ratio : float
        Fraction of valid delivery days used for training (chronological).
    train_start : str or None
        If set (e.g. "2023-07-01"), discard all data before this date
        before computing scales. Use this to exclude the 2022 energy
        crisis and test the covariate-shift hypothesis.

    Returns
    -------
    dict with keys: S_profit, S_carbon_gbp, S_price, S_ci
    """
    df = pd.read_parquet(DATA_PATH).copy()

    # --- Chronological train-split filter ---
    df["_delivery_date"] = pd.to_datetime(df["delivery_ts"]).dt.floor("D")

    # Optional: trim start date (covariate-shift ablation)
    if train_start is not None:
        start_dt = pd.Timestamp(train_start)
        df = df[df["_delivery_date"] >= start_dt].copy()

    # Keep only the first train_ratio fraction of remaining days
    unique_days = np.sort(df["_delivery_date"].unique())
    n_train = int(len(unique_days) * train_ratio)
    train_cutoff = unique_days[n_train]
    df = df[df["_delivery_date"] < train_cutoff].copy()
    df = df.drop(columns=["_delivery_date"])

    # ---- PROFIT scale (DA+ID combined) ----
    da_abs  = df["da_price_gbp_mwh"].astype(float).abs().to_numpy()
    mid_abs = df["mid_price_gbp_mwh"].astype(float).abs().to_numpy()
    profit_series = ((da_abs + mid_abs) * E_step_MWh) + max_deg_cost
    S_profit = float(np.percentile(profit_series, 95))

    # ---- CARBON CASHFLOW scale (monetised, £) ----
    # Asymmetric formulation matching battery_env.py:
    # Import cost = E_step_kWh * ci_actual
    # Export credit = E_step_kWh * mef_actual
    ci_abs  = df["ci_actual_gco2_kwh"].astype(float).abs().to_numpy()
    mef_abs = df["mef_gco2_kwh"].astype(float).abs().to_numpy()
    uka_abs = df["uka_gbp_tco2"].astype(float).abs().to_numpy()
    
    # We take the maximum possible magnitude per step (worst case ci or mef)
    # to ensure the reward stays within a stable range [-1, 1].
    max_intensity = np.maximum(ci_abs, mef_abs)
    tco2_series = max_intensity * E_step_kWh / 1e6
    carbon_cashflow_series = uka_abs * tco2_series
    S_carbon_gbp = float(np.percentile(carbon_cashflow_series, 95))

    # ---- Price scale (observation normalisation) ----
    price_series = np.abs(df["da_price_gbp_mwh"].to_numpy(dtype=float))
    S_price = float(np.percentile(price_series, 95))

    # ---- Carbon Intensity scale (observation normalisation) ----
    ci_series = np.abs(df["ci_actual_gco2_kwh"].to_numpy(dtype=float))
    S_ci = float(np.percentile(ci_series, 95))

    return {
        "S_profit": S_profit,
        "S_carbon_gbp": S_carbon_gbp,
        "S_price": S_price,
        "S_ci": S_ci,
    }

def get_frozen_scales() -> dict:
    """
    Return hardcoded reward/observation scaling constants computed ONCE from
    the post-crisis window (2023-01-01 onward, first 70% of that range).

    These values were computed with:
        compute_scales(train_ratio=0.70, train_start="2023-01-01")

    Frozen so that train, val, and test environments all use identical
    normalisation regardless of which data slice they load.
    """
    return {
        "S_profit":     7487.199743,
        "S_carbon_gbp": 754.750447,
        "S_price":      145.000000,
        "S_ci":         247.000000,
    }


# ====
# DEFAULT MODULE-LEVEL CONSTANTS (backward compatible)
# Computed from full training set (2022-inclusive, first 70%)
# ====
_defaults = compute_scales(train_ratio=0.70, train_start=None)
S_profit     = _defaults["S_profit"]
S_carbon_gbp = _defaults["S_carbon_gbp"]
S_price      = _defaults["S_price"]
S_ci         = _defaults["S_ci"]

print(f"S_price: {S_price:.2f} £/MWh")
print(f"S_ci:    {S_ci:.2f} gCO2/kWh")
print(f"S_profit (DA+ID + Deg, £): {S_profit:.2f}")
print(f"S_carbon (monetised, £): {S_carbon_gbp:.2f}")
