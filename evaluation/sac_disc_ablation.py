"""
sac_disc_ablation.py — SAC action-discretisation ablation (evaluation-time only)
==================================================================================
Loads trained SAC checkpoints (lambda_ci=0.1, seeds 0-4) and re-runs the
219-day held-out test rollout with each action dimension snapped to the nearest
of 11 uniformly-spaced levels in [-1, 1] — matching the granularity of the
discrete DQN/DDQN/D3QN agents — without any retraining.

Usage:
    python evaluation/sac_disc_ablation.py [--checkpoint-dir models]
                                            [--results-dir results/test_rollouts]

Outputs:
    results/test_rollouts/sac_disc_lambda0.1_seed{N}_test_steps.csv  (5 files)
    results/test_rollouts/sac_disc_ablation_summary.csv
    Diagnostic decomposition printed to stdout.
"""

import os
import sys
import argparse
from glob import glob
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch as T

from envs.battery_env import BatteryEnv
import envs.env_config as env_config
from envs.reward_scaling import get_frozen_scales
from agents.sac_agent import SACAgent
from agents.hyperparams import SAC_HYPERPARAMS

# ── Constants ────────────────────────────────────────────────────────────────

N_TEST_DAYS    : int   = 219
TEST_START_DATE: str   = "2025-05-26"
INPUT_DIMS     : int   = 103
E_MAX_MWH      : float = 99.9
LAMBDA_CI      : float = 0.1
SEEDS          : list  = [0, 1, 2, 3, 4]

# 11 uniformly-spaced levels in [-1, 1] mirroring the discrete agents' grid.
DISC_LEVELS = np.linspace(-1.0, 1.0, 11)

# ── Discretisation helper ────────────────────────────────────────────────────

def discretise_action(action: np.ndarray) -> np.ndarray:
    """Snap each element of action (shape [3,]) to the nearest of 11 levels.

    action must already be in [-1, 1] (post-tanh, pre-physical-unit mapping).
    """
    return np.array([
        DISC_LEVELS[np.argmin(np.abs(DISC_LEVELS - a))]
        for a in action
    ])


# ── Model loading ────────────────────────────────────────────────────────────

def load_sac_checkpoint(seed: int, checkpoint_dir: str) -> SACAgent:
    """Load a trained SAC checkpoint for the given seed and lambda_ci=0.1."""
    hp = SAC_HYPERPARAMS
    lci_fragment = f"*lci{LAMBDA_CI:.1f}*"

    search_best  = f"{checkpoint_dir}/*_SAC_seed{seed}_*{lci_fragment}/best_val_model.pth"
    search_final = f"{checkpoint_dir}/*_SAC_seed{seed}_*{lci_fragment}/final_model.pth"
    matching = glob(search_best) or glob(search_final)

    if not matching:
        raise FileNotFoundError(
            f"No SAC checkpoint for seed={seed}, lambda_ci={LAMBDA_CI}.\n"
            f"Tried:\n  {search_best}\n  {search_final}"
        )

    ckpt_path = matching[0]
    print(f"  Loaded: {ckpt_path}")

    agent = SACAgent(
        gamma       = hp["gamma"],
        tau         = hp["tau"],
        lr          = hp["lr"],
        alpha_lr    = hp["alpha_lr"],
        batch_size  = hp["batch_size"],
        reward_scale= hp["reward_scale"],
        input_dims  = INPUT_DIMS,
        n_actions   = 3,
    )
    ckpt = T.load(ckpt_path, map_location="cpu")
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.actor.eval()
    return agent


# ── Environment ──────────────────────────────────────────────────────────────

def build_test_env() -> BatteryEnv:
    return BatteryEnv(
        config              = env_config,
        lambda_ci           = LAMBDA_CI,
        split               = "test",
        episode_days        = N_TEST_DAYS,
        randomize_init_soc  = False,
        randomize_start     = False,
        continuous_action   = True,
        seed                = 42,
        precomputed_scales  = get_frozen_scales(),
    )


# ── Rollout ──────────────────────────────────────────────────────────────────

def run_disc_rollout(agent: SACAgent, env: BatteryEnv) -> pd.DataFrame:
    """219-day greedy rollout with action dimensions snapped to 11 levels."""
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False

    while not done:
        # Deterministic policy mean: tanh(mu), already in [-1, 1]
        state = T.tensor(np.array([obs]), dtype=T.float32).to(agent.actor.device)
        with T.no_grad():
            mu, _ = agent.actor.forward(state)
            raw_action = T.tanh(mu).cpu().numpy()[0]

        # Snap to 11-level grid before the physical-unit mapping in env.step()
        disc_action = discretise_action(raw_action)

        obs, reward, terminated, truncated, info = env.step(disc_action)
        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    return pd.DataFrame(rows)


# ── KPI computation (mirrors evaluate.py) ───────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
    df["financial_profit_gbp"] = (
        df["Planned_Profit"] + df["Intraday_Profit"] - df["degradation_cost_gbp"]
    )
    df["rev_net_gbp"] = df["actual_reward"]
    df["E_throughput_MWh"] = df["P_applied_MW"].abs() * 0.5
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    daily_profit    = df.groupby("date")["financial_profit_gbp"].sum()
    financial_profit= df["financial_profit_gbp"].sum()
    total_net_profit= df["rev_net_gbp"].sum()
    win_rate        = (daily_profit > 0).mean() * 100

    P_MAX_MW = 49.9
    n_days   = len(daily_profit)
    annualised_profit = financial_profit * (365.0 / n_days)
    rev_per_mw_year   = annualised_profit / P_MAX_MW

    efc             = df["E_throughput_MWh"].sum() / (2 * E_MAX_MWH)
    revenue_per_efc = financial_profit / efc if efc > 0 else 0

    idle_fraction   = (df["P_applied_MW"].abs() < 1e-4).mean() * 100
    net_carbon_tco2 = df["net_carbon_tCO2"].sum()

    da_plan_util    = ((df["P_applied_MW"] - df["P_planned_MW"]).abs() < 1e-4).mean() * 100

    gross_revenue   = abs(df["Planned_Profit"].sum() + df["Intraday_Profit"].sum())
    gross_revenue   = gross_revenue if gross_revenue > 0 else 1e-9
    da_rev_pct      = df["Planned_Profit"].sum() / gross_revenue * 100
    id_rev_pct      = df["Intraday_Profit"].sum() / gross_revenue * 100
    deg_cost_pct    = df["degradation_cost_gbp"].sum() / gross_revenue * 100

    return {
        "Financial Profit (£)":  financial_profit,
        "Total Profit (£)":      total_net_profit,
        "Annualised Profit (£)": annualised_profit,
        "Win Rate %":            win_rate,
        "EFC":                   efc,
        "Rev/EFC":               revenue_per_efc,
        "Idle Fraction %":       idle_fraction,
        "Net Carbon (tCO2)":     net_carbon_tco2,
        "DA Plan Utilisation %": da_plan_util,
        "DA Rev %":              da_rev_pct,
        "ID Rev %":              id_rev_pct,
        "Deg Cost %":            deg_cost_pct,
    }


# ── LP-efficiency helper ─────────────────────────────────────────────────────

def load_lp_objective(results_dir: str) -> float | None:
    """Read LP objective for lambda_ci=0.1 from the existing evaluation summary CSV."""
    summary_path = os.path.join(results_dir, "evaluation_summary.csv")
    if not os.path.exists(summary_path):
        return None
    try:
        df = pd.read_csv(summary_path)
        row = df[df["strategy"] == f"lp_lambda{LAMBDA_CI:.1f}"]
        if not row.empty:
            return float(row.iloc[0]["Total Profit (£)"])
    except Exception:
        pass
    return None


# ── Pull existing continuous SAC and D3QN profits ───────────────────────────

def load_existing_profits(results_dir: str) -> tuple[float | None, float | None]:
    """Return (sac_continuous_mean, d3qn_mean) financial profits from saved step CSVs."""
    sac_profits, d3qn_profits = [], []

    for seed in SEEDS:
        sac_path  = os.path.join(results_dir, f"sac_lambda{LAMBDA_CI:.1f}_seed{seed}_test_steps.csv")
        d3qn_path = os.path.join(results_dir, f"d3qn_lambda{LAMBDA_CI:.1f}_seed{seed}_test_steps.csv")

        for path, store in [(sac_path, sac_profits), (d3qn_path, d3qn_profits)]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "financial_profit_gbp" not in df.columns:
                    df = add_derived_columns(df)
                store.append(df["financial_profit_gbp"].sum())

    sac_mean  = float(np.mean(sac_profits))  if sac_profits  else None
    d3qn_mean = float(np.mean(d3qn_profits)) if d3qn_profits else None
    return sac_mean, d3qn_mean


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--results-dir",    default="results/test_rollouts")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(" SAC Action-Discretisation Ablation  (lambda_ci = 0.1)")
    print(" 11 levels per dimension, seeds 0-4, 219-day test split")
    print("=" * 60)

    disc_kpis = {}
    disc_dfs  = {}

    for seed in SEEDS:
        print(f"\n[Seed {seed}] Loading checkpoint ...")
        agent = load_sac_checkpoint(seed, args.checkpoint_dir)

        print(f"[Seed {seed}] Running discretised rollout ...")
        env = build_test_env()
        df  = run_disc_rollout(agent, env)
        df  = add_derived_columns(df)

        save_path = os.path.join(
            args.results_dir,
            f"sac_disc_lambda{LAMBDA_CI:.1f}_seed{seed}_test_steps.csv"
        )
        df.to_csv(save_path, index=False)
        print(f"[Seed {seed}] Saved → {save_path}")

        kpis = compute_kpis(df)
        disc_kpis[seed] = kpis
        disc_dfs[seed]  = df
        print(f"[Seed {seed}] Financial Profit: £{kpis['Financial Profit (£)']:,.0f}  "
              f"Win Rate: {kpis['Win Rate %']:.1f}%  EFC: {kpis['EFC']:.2f}")

    # ── Aggregate across seeds ───────────────────────────────────────────────
    profits   = [disc_kpis[s]["Financial Profit (£)"]  for s in SEEDS]
    efcs      = [disc_kpis[s]["EFC"]                   for s in SEEDS]
    rev_efcs  = [disc_kpis[s]["Rev/EFC"]               for s in SEEDS]
    win_rates = [disc_kpis[s]["Win Rate %"]             for s in SEEDS]
    net_carbs = [disc_kpis[s]["Net Carbon (tCO2)"]      for s in SEEDS]
    idle_frs  = [disc_kpis[s]["Idle Fraction %"]        for s in SEEDS]
    da_utils  = [disc_kpis[s]["DA Plan Utilisation %"]  for s in SEEDS]
    da_revs   = [disc_kpis[s]["DA Rev %"]               for s in SEEDS]
    id_revs   = [disc_kpis[s]["ID Rev %"]               for s in SEEDS]
    deg_cpcts = [disc_kpis[s]["Deg Cost %"]             for s in SEEDS]

    disc_mean = np.mean(profits)
    disc_std  = np.std(profits, ddof=1)

    # ── LP efficiency ────────────────────────────────────────────────────────
    lp_obj = load_lp_objective(args.results_dir)
    lp_eff = (disc_mean / lp_obj * 100) if lp_obj else None

    # ── Load reference profits ───────────────────────────────────────────────
    sac_cont_mean, d3qn_mean = load_existing_profits(args.results_dir)

    # ── Results table ────────────────────────────────────────────────────────
    summary_rows = [{
        "Strategy":             "SAC (3D, disc)",
        "Formulation":          "Continuous 3D → disc-11",
        "Profit_mean_GBP":      round(disc_mean, 0),
        "Profit_std_GBP":       round(disc_std, 1),
        "LP_Efficiency_pct":    round(lp_eff, 1) if lp_eff else None,
        "WinRate_mean_pct":     round(np.mean(win_rates), 1),
        "WinRate_std_pct":      round(np.std(win_rates, ddof=1), 1),
        "EFC_mean":             round(np.mean(efcs), 2),
        "EFC_std":              round(np.std(efcs, ddof=1), 2),
        "Rev_EFC_mean_GBP":     round(np.mean(rev_efcs), 0),
        "Rev_EFC_std_GBP":      round(np.std(rev_efcs, ddof=1), 0),
        "NetCarbon_mean_tCO2":  round(np.mean(net_carbs), 1),
        "NetCarbon_std_tCO2":   round(np.std(net_carbs, ddof=1), 1),
        "IdleFrac_mean_pct":    round(np.mean(idle_frs), 1),
        "DAPlanUtil_mean_pct":  round(np.mean(da_utils), 1),
        "DARevPct_mean":        round(np.mean(da_revs), 1),
        "IDRevPct_mean":        round(np.mean(id_revs), 1),
        "DegCostPct_mean":      round(np.mean(deg_cpcts), 1),
    }]

    summary_df = pd.DataFrame(summary_rows)
    out_csv    = os.path.join(args.results_dir, "sac_disc_ablation_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved ablation summary → {out_csv}")

    # ── Print results table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ABLATION RESULTS  (lambda_ci = 0.1, 5 seeds)")
    print("=" * 70)
    print(f"  {'Metric':<35} {'Mean':>12}  {'Std':>10}")
    print("-" * 70)
    print(f"  {'Financial Profit (£)':<35} {disc_mean:>12,.0f}  {disc_std:>10,.0f}")
    lp_str = f"{lp_eff:.1f}%" if lp_eff else "N/A"
    print(f"  {'LP Efficiency (%)':<35} {lp_str:>12}")
    print(f"  {'Win Rate (%)':<35} {np.mean(win_rates):>12.1f}  {np.std(win_rates, ddof=1):>10.1f}")
    print(f"  {'EFC (cycles)':<35} {np.mean(efcs):>12.2f}  {np.std(efcs, ddof=1):>10.2f}")
    print(f"  {'Rev/EFC (£)':<35} {np.mean(rev_efcs):>12,.0f}  {np.std(rev_efcs, ddof=1):>10,.0f}")
    print(f"  {'Net Carbon (tCO2)':<35} {np.mean(net_carbs):>12.1f}  {np.std(net_carbs, ddof=1):>10.1f}")
    print(f"  {'Idle Fraction (%)':<35} {np.mean(idle_frs):>12.1f}")
    print(f"  {'DA Plan Utilisation (%)':<35} {np.mean(da_utils):>12.1f}")

    # ── Diagnostic decomposition ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" GAP DECOMPOSITION")
    print("=" * 70)

    if sac_cont_mean is not None and d3qn_mean is not None:
        total_gap         = sac_cont_mean - d3qn_mean
        granularity_gap   = sac_cont_mean - disc_mean
        algorithm_gap     = disc_mean     - d3qn_mean

        gran_frac  = (granularity_gap / total_gap * 100) if total_gap != 0 else float("nan")
        algo_frac  = (algorithm_gap  / total_gap * 100) if total_gap != 0 else float("nan")

        print(f"  SAC continuous profit  (mean):  £{sac_cont_mean:,.0f}")
        print(f"  SAC disc-11   profit  (mean):  £{disc_mean:,.0f}")
        print(f"  D3QN          profit  (mean):  £{d3qn_mean:,.0f}")
        print()
        print(f"  total_gap         = £{total_gap:,.0f}  (SAC cont − D3QN)")
        print(f"  granularity_gap   = £{granularity_gap:,.0f}  (SAC cont − SAC disc)")
        print(f"  algorithm_gap     = £{algorithm_gap:,.0f}  (SAC disc − D3QN)")
        print()
        print(f"  Granularity fraction : {gran_frac:.1f}%")
        print(f"  Algorithm   fraction : {algo_frac:.1f}%")
        print()
        print(
            f"  Granularity accounts for {gran_frac:.1f}% of the SAC-D3QN gap; "
            f"parameterisation/policy accounts for {algo_frac:.1f}%"
        )
    else:
        missing = []
        if sac_cont_mean is None:
            missing.append("SAC continuous (no sac_lambda0.1_seedN_test_steps.csv found)")
        if d3qn_mean is None:
            missing.append("D3QN (no d3qn_lambda0.1_seedN_test_steps.csv found)")
        print(f"  Cannot compute gap decomposition — missing: {', '.join(missing)}")
        print(f"  SAC (disc-11) mean financial profit: £{disc_mean:,.0f} ± £{disc_std:,.0f}")

    # ── Thesis-ready sentence ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    lp_eff_str = f"{lp_eff:.0f}" if lp_eff else "?"
    disc_mean_k = disc_mean / 1000
    disc_std_k  = disc_std  / 1000

    if sac_cont_mean is not None and d3qn_mean is not None:
        total_gap       = sac_cont_mean - d3qn_mean
        granularity_gap = sac_cont_mean - disc_mean
        gran_frac       = (granularity_gap / total_gap * 100) if total_gap != 0 else float("nan")

        dominant = "granularity" if gran_frac >= 50 else "parameterisation/policy"
        dominant_frac = gran_frac if gran_frac >= 50 else (100 - gran_frac)

        thesis_sentence = (
            f"Discretising SAC's 3D action space to 11 levels per dimension at "
            f"evaluation yields £{disc_mean_k:.1f}k ± £{disc_std_k:.1f}k net profit "
            f"({lp_eff_str}% LP efficiency), implying that {dominant} accounts for "
            f"{dominant_frac:.1f}% of the SAC--D3QN profit gap."
        )
    else:
        thesis_sentence = (
            f"Discretising SAC's 3D action space to 11 levels per dimension at "
            f"evaluation yields £{disc_mean_k:.1f}k ± £{disc_std_k:.1f}k net profit "
            f"({lp_eff_str}% LP efficiency)."
        )

    print(f"\nTHESIS SENTENCE:\n{thesis_sentence}\n")


if __name__ == "__main__":
    main()
