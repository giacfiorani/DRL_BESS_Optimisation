import numpy as np
from pathlib import Path
import pandas as pd
# Do NOT import env_config here — compute E_max directly to avoid a circular import.

ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_PATH = ROOT_DIR / "data" / "data.parquet"

# Hardware constants — must stay in sync with env_config.py.
N_cells_per_cabinet   = 416
Q_cell                = 280    # Ah
E_nominal_per_cabinet = 372.7  # kWh
N_cabinets            = 268

E_nominal   = E_nominal_per_cabinet * N_cabinets
E_max       = E_nominal / 1000.0  # MWh

C_rate      = 0.5
P_max_MW    = C_rate * E_max
dt          = 0.5              # hours

E_step_MWh  = P_max_MW * dt
E_step_kWh  = E_step_MWh * 1000.0

# Must stay in sync with envs/env_config.py deg_kappa.
deg_kappa    = 10
max_deg_cost = deg_kappa * E_step_MWh


def compute_scales(
    train_ratio: float = 0.70,
    train_start: str | None = None,
) -> dict:
    """Compute 95th-percentile reward and observation scaling constants.

    Args:
        train_ratio: Fraction of valid delivery days used for training
            (chronological split).
        train_start: Optional ISO date string (e.g. ``"2023-01-01"``). Data
            before this date is discarded before computing scales, which can
            be used to exclude the 2022 energy-crisis period for a
            covariate-shift ablation.

    Returns:
        Dict with keys ``S_profit``, ``S_carbon_gbp``, ``S_price``, ``S_ci``.
    """
    df = pd.read_parquet(DATA_PATH).copy()

    df["_delivery_date"] = pd.to_datetime(df["delivery_ts"]).dt.floor("D")

    if train_start is not None:
        start_dt = pd.Timestamp(train_start)
        df = df[df["_delivery_date"] >= start_dt].copy()

    unique_days   = np.sort(df["_delivery_date"].unique())
    n_train       = int(len(unique_days) * train_ratio)
    train_cutoff  = unique_days[n_train]
    df = df[df["_delivery_date"] < train_cutoff].copy()
    df = df.drop(columns=["_delivery_date"])

    # Profit scale: 95th percentile of |DA price| + |ID price| per step plus deg cost.
    da_abs  = df["da_price_gbp_mwh"].astype(float).abs().to_numpy()
    mid_abs = df["mid_price_gbp_mwh"].astype(float).abs().to_numpy()
    profit_series = ((da_abs + mid_abs) * E_step_MWh) + max_deg_cost
    S_profit = float(np.percentile(profit_series, 95))

    # Carbon cashflow scale: asymmetric formulation matching battery_env.py.
    # Import cost uses AEF (ci_actual); export credit uses MEF.
    ci_abs  = df["ci_actual_gco2_kwh"].astype(float).abs().to_numpy()
    mef_abs = df["mef_gco2_kwh"].astype(float).abs().to_numpy()
    uka_abs = df["uka_gbp_tco2"].astype(float).abs().to_numpy()

    max_intensity         = np.maximum(ci_abs, mef_abs)
    tco2_series           = max_intensity * E_step_kWh / 1e6
    carbon_cashflow_series = uka_abs * tco2_series
    S_carbon_gbp = float(np.percentile(carbon_cashflow_series, 95))

    price_series = np.abs(df["da_price_gbp_mwh"].to_numpy(dtype=float))
    S_price = float(np.percentile(price_series, 95))

    ci_series = np.abs(df["ci_actual_gco2_kwh"].to_numpy(dtype=float))
    S_ci = float(np.percentile(ci_series, 95))

    return {
        "S_profit":     S_profit,
        "S_carbon_gbp": S_carbon_gbp,
        "S_price":      S_price,
        "S_ci":         S_ci,
    }


def get_frozen_scales() -> dict:
    """Return hardcoded scaling constants computed from the post-crisis window.

    These values were computed with:
        ``compute_scales(train_ratio=0.70, train_start="2023-01-01")``

    Frozen so that train, validation, and test environments all use identical
    normalisation regardless of which data slice they load.

    Returns:
        Dict with keys ``S_profit``, ``S_carbon_gbp``, ``S_price``, ``S_ci``.
    """
    return {
        "S_profit":     7487.199743,
        "S_carbon_gbp": 754.750447,
        "S_price":      145.000000,
        "S_ci":         247.000000,
    }


# Module-level constants for backward-compatible imports.
# Computed from the full training set (2022-inclusive, first 70%).
_defaults    = compute_scales(train_ratio=0.70, train_start=None)
S_profit     = _defaults["S_profit"]
S_carbon_gbp = _defaults["S_carbon_gbp"]
S_price      = _defaults["S_price"]
S_ci         = _defaults["S_ci"]
