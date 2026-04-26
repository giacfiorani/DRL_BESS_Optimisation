from __future__ import annotations

import pulp
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------------
# Default battery configuration — full 268-cabinet CATL EnerOne plant
# --------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "E_max":       99.7,    # MWh  — 268 × 372.7 kWh cabinets
    "P_max_MW":    49.95,   # MW   — 268 × 0.18635 MW (0.5C rate)
    "eta_ch":      0.99,    # charge efficiency
    "eta_dis":     1.0,     # discharge efficiency
    "SoC_min":     0.1,     # lower SoC bound
    "SoC_max":     0.9,     # upper SoC bound
    "SoC_initial": 0.5,     # initial SoC
    "kappa":       10.0,    # £/MWh — throughput degradation cost (Cortés-Arcos 2020)
    "dt":          0.5,     # hours — settlement period length
}


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_lp_benchmark(df: pd.DataFrame, cfg: dict | None = None, lambda_ci: float = 0.1, solver_msg: bool = False,) -> dict:
    """Perfect-foresight single-market physical oracle for a BESS.

    Every MWh of revenue is backed by a physical change in SoC.  The LP
    represents an omniscient dispatcher that knows all future prices and
    selects, for each slot, the better of the DA and ID price at which to
    settle the physical dispatch:

        p_best[t] = max(da_price[t], id_price[t])
                    (falls back to id_price[t] where da_price is NaN)

    There are no virtual-bidding or spread-trading variables.  This is the
    correct theoretical ceiling for an RL agent whose revenue is fully
    backed by SoC-constrained physical dispatch.
    """
    params: dict = {**DEFAULT_CFG, **(cfg or {})}

    E_max   = float(params["E_max"])
    P_max   = float(params["P_max_MW"])
    eta_ch  = float(params["eta_ch"])
    eta_dis = float(params["eta_dis"])
    soc_min = float(params["SoC_min"])
    soc_max = float(params["SoC_max"])
    soc_0   = float(params["SoC_initial"])
    kappa   = float(params["kappa"])
    dt      = float(params["dt"])

    T  = len(df)
    ts = range(T)

    # ── Price signals ─────────────────────────────────────────────────────────
    p_id  = df["mid_price_gbp_mwh"].to_numpy(dtype=np.float64)
    p_da  = pd.to_numeric(df["da_price_gbp_mwh"], errors="coerce").to_numpy(dtype=np.float64)
    ci    = df["ci_actual_gco2_kwh"].to_numpy(dtype=np.float64)
    mef   = df["mef_gco2_kwh"].to_numpy(dtype=np.float64)
    c_co2 = df["uka_gbp_tco2"].to_numpy(dtype=np.float64)

    # Best available price per slot: DA if present and higher, else ID.
    p_best = np.where(np.isnan(p_da), p_id, np.maximum(p_da, p_id))

    # ── Pre-computed objective coefficients ───────────────────────────────────
    deg_coeff  = kappa * dt
    carbon_ch  = lambda_ci * c_co2 * dt * ci  / 1000
    carbon_dis = lambda_ci * c_co2 * dt * mef / 1000

    # ── Build LP ──────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("BESS_PerfectForesight", pulp.LpMaximize)

    P_ch  = [pulp.LpVariable(f"Pch_{t}",  lowBound=0.0, upBound=P_max) for t in ts]
    P_dis = [pulp.LpVariable(f"Pdis_{t}", lowBound=0.0, upBound=P_max) for t in ts]
    SoC   = [pulp.LpVariable(f"SoC_{t}",  lowBound=soc_min, upBound=soc_max) for t in ts]

    # ── Objective ─────────────────────────────────────────────────────────────
    # P_dis coefficient: +p_best·dt - deg_coeff + carbon_dis[t]
    # P_ch  coefficient: -p_best·dt - deg_coeff - carbon_ch[t]
    prob += pulp.lpSum(
        P_dis[t] * ( p_best[t] * dt - deg_coeff + carbon_dis[t])
        + P_ch[t] * (-p_best[t] * dt - deg_coeff - carbon_ch[t])
        for t in ts
    )

    # ── SoC constraints ───────────────────────────────────────────────────────
    prob += (
        SoC[0] == soc_0 + (eta_ch * P_ch[0] - P_dis[0] / eta_dis) * dt / E_max,
        "SoC_init",
    )
    for t in range(1, T):
        prob += (
            SoC[t] == SoC[t-1] + (eta_ch * P_ch[t] - P_dis[t] / eta_dis) * dt / E_max,
            f"SoC_{t}",
        )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=int(solver_msg), timeLimit=300)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective)

    if status != "Optimal":
        return {
            "lp_status":      status,
            "objective_gbp":  None,
            "revenue_gbp":    None,
            "deg_cost_gbp":   None,
            "carbon_cost_gbp": None,
            "schedule":       None,
        }

    # ── Extract solution ──────────────────────────────────────────────────────
    pch_val  = np.array([pulp.value(P_ch[t])  for t in ts], dtype=np.float64)
    pdis_val = np.array([pulp.value(P_dis[t]) for t in ts], dtype=np.float64)
    soc_val  = np.array([pulp.value(SoC[t])   for t in ts], dtype=np.float64)

    p_net = pdis_val - pch_val

    revenue     = float(np.sum(p_net * p_best * dt))
    deg_cost    = float(np.sum(kappa * (pdis_val + pch_val) * dt))
    carbon_flow = float(np.sum(c_co2 * dt * (pch_val * ci - pdis_val * mef) / 1000.0))
    carbon_cost = lambda_ci * carbon_flow

    n_simultaneous = int(np.sum((pch_val > 1e-4) & (pdis_val > 1e-4)))
    if n_simultaneous > 0:
        print(
            f"  [warning] {n_simultaneous} periods with simultaneous charge/discharge "
            f"(LP artefact, magnitude negligible)."
        )

    schedule = pd.DataFrame(
        {
            "P_ch_MW":               pch_val,
            "P_dis_MW":              pdis_val,
            "P_net_MW":              p_net,
            "SoC":                   soc_val,
            "p_best_gbp_mwh":        p_best,
            "da_price_gbp_mwh":      p_da,
            "id_price_gbp_mwh":      p_id,
            "ci_gco2_kwh":           ci,
            "mef_gco2_kwh":          mef,
            "carbon_price_gbp_tco2": c_co2,
        },
        index=df.index,
    )

    return {
        "lp_status":       status,
        "objective_gbp":   float(obj),
        "revenue_gbp":     revenue,
        "deg_cost_gbp":    deg_cost,
        "carbon_cost_gbp": carbon_cost,
        "schedule":        schedule,
    }

# ---------------------------------------------------------------------------
# Pareto sweep over λ_ci
# ---------------------------------------------------------------------------
def run_pareto_sweep(
    df: pd.DataFrame,
    cfg: dict | None = None,
    lambdas: list[float] | None = None,
    solver_msg: bool = False,
) -> pd.DataFrame:
    """Solve the LP for each λ_ci and return a Pareto summary table.

    Parameters
    ----------
    df      : test-split DataFrame (same format as solve_lp_benchmark).
    cfg     : battery config dict (defaults to DEFAULT_CFG).
    lambdas : list of λ_ci values to sweep.
              Default: [0.0, 0.1, 0.3, 0.5, 1.0]
    solver_msg : bool
        If True, print CBC solver log for each solve.

    Returns
    -------
    pd.DataFrame with columns:
        lambda_ci, lp_status, objective_gbp, revenue_gbp,
        deg_cost_gbp, carbon_cost_gbp
    """
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.3, 0.6, 1.0]

    rows = []
    for lam in lambdas:
        print(f"  Solving λ_ci = {lam:.2f} …", end=" ", flush=True)
        result = solve_lp_benchmark(df, cfg=cfg, lambda_ci=lam, solver_msg=solver_msg)
        print(result["lp_status"])
        rows.append(
            {
                "lambda_ci":       lam,
                "lp_status":       result["lp_status"],
                "objective_gbp":   result["objective_gbp"],
                "revenue_gbp":     result["revenue_gbp"],
                "deg_cost_gbp":    result["deg_cost_gbp"],
                "carbon_cost_gbp": result["carbon_cost_gbp"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point — run Pareto sweep on the test split
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-start", type=str, default="2023-01-01",
        help="Exclude data before this date before computing the 70/15/15 split "
             "(default: 2023-01-01 to avoid Ukraine-war price spikes)",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 1.0],
        help="λ_ci values for Pareto sweep",
    )
    parser.add_argument("--solver-msg", action="store_true", help="Show CBC solver log")
    args = parser.parse_args()

    DATA_PATH = ROOT / "data" / "data.parquet"
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df_full = pd.read_parquet(DATA_PATH).copy()
    df_full["delivery_ts"] = pd.to_datetime(df_full["delivery_ts"])

    # Mirror BatteryEnv data windowing: discard rows before train_start so the
    # 70/15/15 split ratios apply only to the representative window.
    # See battery_env.py lines 138-143 for the identical logic.
    if args.train_start:
        start_dt = pd.Timestamp(args.train_start)
        df_full = df_full[df_full["delivery_ts"] >= start_dt].copy()
        print(f"Data windowed from {args.train_start} onwards "
              f"({df_full['delivery_ts'].dt.date.nunique()} days remaining)")

    # Reproduce the chronological split used in training (70/15/15)
    # env_config is a flat module — no BatteryConfig class exists; import attributes directly.
    import envs.env_config as ec

    n_total  = df_full["delivery_ts"].dt.date.nunique()
    n_train  = int(n_total * 0.70)
    n_val    = int(n_total * 0.15)

    all_dates = sorted(df_full["delivery_ts"].dt.date.unique())
    test_dates = set(all_dates[n_train + n_val :])
    df_test = df_full[df_full["delivery_ts"].dt.date.apply(lambda d: d in test_dates)].copy()

    print(f"Test split: {len(df_test)} rows, {len(test_dates)} days "
          f"(from {all_dates[n_train + n_val]} to {all_dates[-1]})")
    print(f"Running Pareto sweep …")

    summary = run_pareto_sweep(
        df_test,
        cfg={
            "E_max":       ec.E_max,
            "P_max_MW":    ec.P_max_MW,
            "eta_ch":      ec.eff_ch,
            "eta_dis":     ec.eff_dis,
            "SoC_min":     ec.SoC_min,
            "SoC_max":     ec.SoC_max,
            "SoC_initial": ec.SoC_initial,
            "kappa":       ec.deg_kappa,
            "dt":          ec.dt,
        },
        lambdas=args.lambdas,
        solver_msg=args.solver_msg,
    )

    print("\n── Pareto Sweep Results ─────────────────────────────────────")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    OUT = ROOT / "results" / "lp_pareto_summary.csv"
    OUT.parent.mkdir(exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")



             
