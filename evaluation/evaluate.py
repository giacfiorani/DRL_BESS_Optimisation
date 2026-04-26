"""
evaluate.py — BESS RL Agent Evaluation Pipeline (IEEE Publication Grade)
=========================================================================
Runs frozen RL checkpoints through the 219-day held-out test split
(2025-05-26 → 2026-01-01) as a single continuous rollout with no
episodic SoC resets. Logs every half-hour step to a CSV for post-hoc
KPI computation and generates publication-grade visualizations.

Protocol: See markdowns/evaluation_strategy.md (Sections I–VII)

NEW: Integrated multi-seed aggregation, Pareto frontier visualization,
action distribution analysis, and chronological dispatch plots.
"""

import os
import argparse
from glob import glob
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch as T
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# --- Project imports ---
from benchmarks.lp_benchmark import solve_lp_benchmark
from envs.battery_env import BatteryEnv
import envs.env_config as env_config
from envs.reward_scaling import get_frozen_scales
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.d3qn import D3QNAgent
from agents.d3qn_per_agent import D3QNPERAgent
from agents.sac_agent import SACAgent
from agents.hyperparams import (
    DQN_HYPERPARAMS,
    DDQN_HYPERPARAMS,
    D3QN_HYPERPARAMS,
    D3QN_PER_HYPERPARAMS,
    SAC_HYPERPARAMS,
)
from utils.action_encoding import decode, N_ACTIONS

# ============================================================
# IEEE PUBLICATION FORMATTING CONSTANTS
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    'dqn': '#1f77b4',         # blue
    'ddqn': '#ff7f0e',        # orange
    'd3qn': '#2ca02c',        # green
    'd3qn_per': '#d62728',    # red
    'idle': '#9467bd',        # purple
    'random': '#8c564b',      # brown
    'p20p80': '#e377c2',      # pink
    'sac': '#17becf',          # cyan
}

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

N_TEST_DAYS: int = 219
TEST_START_DATE: str = "2025-05-26"
RESULTS_DIR: str = "results/test_rollouts"
FIGURES_DIR: str = "results/figures"
FIGURES_MAIN_DIR: str = "results/figures_main"
FIGURES_APPENDIX_DIR: str = "results/figures_appendix"
CHECKPOINT_DIR: str = "models"
INPUT_DIMS: int = 103
SEED = 42
# System capacity: 268 CATL EnerOne cabinets × 0.3727 MWh = 99.9 MWh.
# Must match env_config.E_max exactly. Used as the EFC denominator throughout.
E_MAX_MWH: float = 99.9

AGENTS: dict[str, type] = {
    "dqn":      DQNAgent,
    "ddqn":     DDQNAgent,
    "d3qn":     D3QNAgent,
    "d3qn_per": D3QNPERAgent,
    "sac":      SACAgent,
}

HYPERPARAMS_MAP: dict[str, dict] = {
    "dqn":      DQN_HYPERPARAMS,
    "ddqn":     DDQN_HYPERPARAMS,
    "d3qn":     D3QN_HYPERPARAMS,
    "d3qn_per": D3QN_PER_HYPERPARAMS,
    "sac":      SAC_HYPERPARAMS,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURES_MAIN_DIR, exist_ok=True)
os.makedirs(FIGURES_APPENDIX_DIR, exist_ok=True)


# ============================================================
# FUNCTION DEFINITIONS
# ============================================================

def load_frozen_agent(agent_name: str, seed: int,
                      checkpoint_dir: str = CHECKPOINT_DIR) -> object:
    """
    Load a trained checkpoint and return a fully greedy agent (epsilon=0).
    Prefers best_val_model.pth (validation-selected); falls back to final_model.pth.
    """
    agent_cls = AGENTS[agent_name]
    hp = HYPERPARAMS_MAP[agent_name]

    # Prefer validation-selected model; fall back to final
    search_best = f"{checkpoint_dir}/*_{agent_name.upper()}_seed{seed}_*/best_val_model.pth"
    search_final = f"{checkpoint_dir}/*_{agent_name.upper()}_seed{seed}_*/final_model.pth"
    matching_file = glob(search_best) or glob(search_final)

    if len(matching_file) == 0:
        raise FileNotFoundError(
            f"Could not find trained model for {agent_name} seed {seed}. "
            f"Patterns tried: {search_best}, {search_final}"
        )

    ckpt_path = matching_file[0]
    print(f"Found model: {ckpt_path}")

    agent = agent_cls(
        gamma      = hp["gamma"],
        epsilon    = 0.0,
        lr         = hp["lr"],
        batch_size = hp["batch_size"],
        eps_dec    = 0.0,
        eps_min    = 0.0,
        replace_target_cnt = hp["target_update_frequency"],
        input_dims = INPUT_DIMS,
        n_actions  = N_ACTIONS,
    )

    ckpt = T.load(ckpt_path, map_location="cpu")
    agent.Q_eval.load_state_dict(ckpt["model_state_dict"])
    agent.Q_eval.eval()
    agent.Q_target.load_state_dict(ckpt["model_state_dict"])
    agent.Q_target.eval()

    return agent


def load_frozen_sac_agent(seed: int,
                          lambda_ci: float | None = None,
                          checkpoint_dir: str = CHECKPOINT_DIR) -> SACAgent:
    """Load a trained SAC checkpoint (best_val_model.pth) for greedy evaluation.

    Parameters
    ----------
    seed       : training seed (0-4)
    lambda_ci  : carbon penalty used during training (e.g. 0.1, 0.3).
                 When provided, the glob is narrowed to directories whose name
                 contains ``lci{lambda_ci:.1f}`` so that Pareto-sweep runs
                 (5 lambdas × 5 seeds = 25 dirs) pick the correct checkpoint.
                 When None the original seed-only glob is used (backwards
                 compatible for single-lambda evaluations).
    """
    hp = SAC_HYPERPARAMS

    # Build lambda-aware glob suffix when lambda_ci is supplied
    lci_fragment = f"*lci{lambda_ci:.1f}*" if lambda_ci is not None else "*"

    search_best  = f"{checkpoint_dir}/*_SAC_seed{seed}_*{lci_fragment}/best_val_model.pth"
    search_final = f"{checkpoint_dir}/*_SAC_seed{seed}_*{lci_fragment}/final_model.pth"
    matching = glob(search_best) or glob(search_final)

    if not matching:
        raise FileNotFoundError(
            f"No SAC model found for seed={seed}, lambda_ci={lambda_ci}. "
            f"Patterns tried:\n  {search_best}\n  {search_final}"
        )

    ckpt_path = matching[0]
    print(f"Found SAC model: {ckpt_path}")

    agent = SACAgent(
        gamma=hp["gamma"],
        tau=hp["tau"],
        lr=hp["lr"],
        alpha_lr=hp["alpha_lr"],
        batch_size=hp["batch_size"],
        reward_scale=hp["reward_scale"],
        input_dims=INPUT_DIMS,
        n_actions=3,
    )

    ckpt = T.load(ckpt_path, map_location="cpu")
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.actor.eval()

    return agent


def run_continuous_rollout_sac(agent: SACAgent, env: BatteryEnv) -> pd.DataFrame:
    """Run a greedy 219-day SAC rollout using deterministic (mean) policy."""
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False

    while not done:
        action = agent.choose_action_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    return pd.DataFrame(rows)


def build_test_env(lambda_ci: float = 0.9, continuous_action: bool = False) -> BatteryEnv:
    """
    Create a deterministic BatteryEnv for the 219-day test split.
    Supports lambda_ci ablation. Set continuous_action=True for SAC.
    """
    env = BatteryEnv(
        config=env_config,
        lambda_ci=lambda_ci,
        split="test",
        episode_days=N_TEST_DAYS,
        randomize_init_soc=False,
        randomize_start=False,
        continuous_action=continuous_action,
        seed=SEED,
        precomputed_scales=get_frozen_scales(),
    )
    return env


def run_continuous_rollout(agent: object, env: BatteryEnv) -> pd.DataFrame:
    """
    Run a greedy 219-day rollout and return a step-level DataFrame (~10,512 rows).
    """
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False

    while not done:
        action = agent.choose_action(obs)
        dispatch_idx, plan_idx, plan_slot = decode(action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(env_action)

        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    df = pd.DataFrame(rows)
    return df


def run_idle_baseline(env: BatteryEnv) -> pd.DataFrame:
    """Baseline 0 — Idle agent: P = 0 MW every step."""
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False

    while not done:
        env_action = np.array([5, 5, 0], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(env_action)
        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    return pd.DataFrame(rows)


def run_random_baseline(env: BatteryEnv, rng_seed: int = 0) -> pd.DataFrame:
    """Baseline 1 — Uniform random policy: samples from all 5,808 actions."""
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False
    rng = np.random.default_rng(rng_seed)

    while not done:
        flat_action = rng.integers(0, N_ACTIONS)
        dispatch_idx, plan_idx, plan_slot = decode(flat_action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)

        obs, reward, terminated, truncated, info = env.step(env_action)
        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    return pd.DataFrame(rows)


def run_p20p80_heuristic(env: BatteryEnv) -> pd.DataFrame:
    """Baseline 2 — P20/P80 daily heuristic (industry rule-based dispatch)."""
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    rows = []
    step_idx = 0
    done = False
    id_prices_list = []

    env_action = np.array([5, 5, 0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(env_action)
    rows.append({**info, "step": step_idx, "reward": reward})
    step_idx += 1
    done = terminated or truncated
    id_prices_list.append(info["id_price_now"])

    while not done:
        if info["tau"] == 1:
            id_prices_list.clear()

        id_prices_list.append(info["id_price_now"])

        p20 = np.percentile(id_prices_list, 20)
        p80 = np.percentile(id_prices_list, 80)

        if info["id_price_now"] < p20:
            env_action = np.array([0, 5, 0], dtype=np.int64)
        elif info["id_price_now"] > p80:
            env_action = np.array([10, 5, 0], dtype=np.int64)
        else:
            env_action = np.array([5, 5, 0], dtype=np.int64)

        obs, reward, terminated, truncated, info = env.step(env_action)
        rows.append({**info, "step": step_idx, "reward": reward})

        step_idx += 1
        done = terminated or truncated

    return pd.DataFrame(rows)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append 5 derived columns for analysis."""
    df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
    df["month"] = pd.to_datetime(df["delivery_ts"]).dt.month_name()
    df["day_of_week"] = pd.to_datetime(df["delivery_ts"]).dt.day_name()
    # financial_profit_gbp: pure market-realised profit, no lambda_ci weighting.
    # = R_DA + R_ID - deg_cost.  Use this for paper tables comparing agents fairly
    # across different lambda_ci values (the weighted objective is not comparable).
    df["financial_profit_gbp"] = (
        df["Planned_Profit"] + df["Intraday_Profit"] - df["degradation_cost_gbp"]
    )
    # rev_net_gbp: lambda-weighted objective (= actual_reward = R_total_gbp).
    # Matches the training signal; used for cumulative-profit plots and Sharpe.
    df["rev_net_gbp"] = df["actual_reward"]
    df["E_throughput_MWh"] = df["P_applied_MW"].abs() * 0.5

    # Action classification for distribution analysis
    df["action_type"] = df["P_applied_MW"].apply(
        lambda p: "Idle" if abs(p) < 1e-4 else ("Charge" if p < -1e-4 else "Discharge")
    )

    return df


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Compute all KPIs from a step-level DataFrame.

    Sharpe ratio is intentionally omitted. There is no principled capital
    denominator for a physical BESS asset: CAPEX (£31M) produces a near-zero
    or negative result; stored-energy value (£1M) produces ~14, which is a
    dimensionless P&L Information Ratio (Lo et al., PLoS ONE 2009), not a
    portfolio Sharpe and not comparable to published benchmarks. No surveyed
    BESS RL paper (IEEE TSG, Applied Energy, Energies 2020-2025) reports a
    Sharpe ratio. Domain-standard KPIs are used instead: £/MW/year (Modo
    Energy GB BESS Index), win rate, and max drawdown.
    """
    # Use financial_profit_gbp (R_DA + R_ID - deg, no lambda_ci weighting)
    # for all economic KPIs so they are comparable across lambda_ci values.
    if "financial_profit_gbp" in df.columns:
        daily_profit = df.groupby('date')['financial_profit_gbp'].sum()
        financial_profit = df["financial_profit_gbp"].sum()
    else:
        daily_profit = df.groupby('date')['rev_net_gbp'].sum()
        financial_profit = None

    total_net_profit = df["rev_net_gbp"].sum()   # lambda-weighted RL objective

    profit_volatility = daily_profit.std()
    mean_daily_profit = daily_profit.mean()

    # Win rate: fraction of test days with positive financial P&L
    win_rate = (daily_profit > 0).mean() * 100

    # £/MW/year: GB industry standard (Modo Energy BESS Index).
    # Annualise from the actual test-period length then normalise by power capacity.
    P_MAX_MW = 49.9  # system power capacity (0.5C × 99.9 MWh)
    n_days = len(daily_profit)
    annualised_profit = financial_profit * (365.0 / n_days) if financial_profit is not None else None
    rev_per_mw_year = annualised_profit / P_MAX_MW if annualised_profit is not None else None

    cum_profit = daily_profit.cumsum()
    drawdown = cum_profit.cummax() - cum_profit
    max_drawdown = drawdown.max()

    gross_revenue = abs(df["Planned_Profit"].sum() + df["Intraday_Profit"].sum())
    if gross_revenue == 0:
        gross_revenue = 1e-9

    da_revenue_pct = (df["Planned_Profit"].sum() / gross_revenue) * 100
    id_revenue_pct = (df["Intraday_Profit"].sum() / gross_revenue) * 100
    deg_cost_pct = (df["degradation_cost_gbp"].sum() / gross_revenue) * 100
    carbon_cost_pct = (df["carbon_cashflow"].sum() / gross_revenue) * 100 if "carbon_cashflow" in df.columns else 0

    # EFC = total energy throughput (one-way MWh) / (2 × system capacity MWh).
    # Denominator = 2 × 99.9 MWh = 199.8 MWh  (charge + discharge per full cycle).
    efc = df["E_throughput_MWh"].sum() / (2 * E_MAX_MWH)
    revenue_per_efc = financial_profit / efc if (efc != 0 and financial_profit is not None) else 0
    soc_mean, soc_std = df["soc"].mean(), df["soc"].std()

    idle_fraction = (df["P_applied_MW"].abs() < 1e-4).mean() * 100
    n_charge = (df["P_applied_MW"] < -1e-4).sum()
    n_discharge = (df["P_applied_MW"] > 1e-4).sum()
    charge_discharge_ratio = n_charge / n_discharge if n_discharge != 0 else 0
    clipping_rate = ((df["P_req_MW"] - df["P_applied_MW"]).abs() > 1e-4).mean() * 100

    p75_id_price = np.percentile(df["id_price_now"], 75)
    smart_discharges = ((df["P_applied_MW"] > 1e-4) & (df["id_price_now"] >= p75_id_price)).sum()
    market_timing_score = (smart_discharges / n_discharge * 100) if n_discharge > 0 else 0

    da_plan_utilisation = ((df["P_applied_MW"] - df["P_planned_MW"]).abs() < 1e-4).mean() * 100

    total_throughput_MWh = df["E_throughput_MWh"].sum()
    net_carbon_tco2 = df["net_carbon_tCO2"].sum()
    carbon_intensity = net_carbon_tco2 / total_throughput_MWh if total_throughput_MWh > 0 else 0
    carbon_penalty_gbp = df["carbon_cashflow"].sum()

    return {
        "Financial Profit (£)":     financial_profit,    # R_DA + R_ID - deg, no λ weighting
        "Total Profit (£)":         total_net_profit,     # lambda-weighted RL objective
        "Annualised Profit (£)":    annualised_profit,
        "£/MW/year":                rev_per_mw_year,      # annualised financial profit / P_max_MW
        "Mean Daily Profit":        mean_daily_profit,
        "Profit Volatility":        profit_volatility,
        "Win Rate %":               win_rate,
        "Max Drawdown (£)":         max_drawdown,
        "DA Rev %":                 da_revenue_pct,
        "ID Rev %":                 id_revenue_pct,
        "Deg Cost %":               deg_cost_pct,
        "Carbon Cost %":            carbon_cost_pct,
        "EFC":                      efc,
        "Rev/EFC":                  revenue_per_efc,
        "SoC Mean":                 soc_mean,
        "SoC Std":                  soc_std,
        "Idle Fraction %":          idle_fraction,
        "Charge/Discharge Ratio":   charge_discharge_ratio,
        "Clipping Rate %":          clipping_rate,
        "Net Carbon (tCO2)":        net_carbon_tco2,
        "Carbon Intensity (tCO2/MWh)": carbon_intensity,
        "Carbon Penalty (£)":       carbon_penalty_gbp,
        "Market Timing Score %":    market_timing_score,
        "DA Plan Utilisation %":    da_plan_utilisation,
    }


def run_lp_benchmarks(lambda_values: list[float]) -> dict[float, dict]:
    """Solve the perfect-foresight LP once per lambda_ci on the test split.

    Uses the full 2022+ dataset with a 70/15/15 chronological split, identical
    to the training scripts. The test rows align with 2025-05-26 → 2026-01-01.

    Returns a dict keyed by lambda_ci value, each entry being the full result
    dict from solve_lp_benchmark (keys: lp_status, objective_gbp, revenue_gbp,
    deg_cost_gbp, carbon_cost_gbp, schedule).
    """
    ROOT = Path(__file__).resolve().parents[1]
    data_path = ROOT / "data" / "data.parquet"

    df_full = pd.read_parquet(data_path).copy()
    df_full["delivery_ts"] = pd.to_datetime(df_full["delivery_ts"])

    n_total = df_full["delivery_ts"].dt.date.nunique()
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)

    all_dates  = sorted(df_full["delivery_ts"].dt.date.unique())
    test_dates = set(all_dates[n_train + n_val:])
    df_test = df_full[
        df_full["delivery_ts"].dt.date.apply(lambda d: d in test_dates)
    ].copy()

    print(f"  [LP] Test split: {len(test_dates)} days, {len(df_test)} half-hours "
          f"({all_dates[n_train + n_val]} → {all_dates[-1]})")

    lp_results: dict[float, dict] = {}
    for lam in lambda_values:
        print(f"  [LP] Solving λ_ci = {lam:.1f} …", end=" ", flush=True)
        result = solve_lp_benchmark(df_test, lambda_ci=lam)
        print(result["lp_status"])
        lp_results[lam] = result

    return lp_results


def build_comparison_table(all_results: dict[str, pd.DataFrame],
                           lp_objectives: dict[float, dict] | None = None,
                           baseline_lambda: float | None = None) -> pd.DataFrame:
    """Build the master comparison table.

    If lp_objectives is provided (dict keyed by lambda_ci → solve_lp_benchmark result),
    a 'LP Efficiency Ratio (%)' column is appended to every agent/baseline row and
    one 'lp_lambda{x}' row is added per lambda_ci value at the bottom of the table.
    """
    import re

    kpi_rows = []

    for label, df in all_results.items():
        if "rev_net_gbp" not in df.columns:
            df = add_derived_columns(df)

        kpi_dict = compute_kpis(df)

        # LP Efficiency Ratio: agent_profit / lp_objective × 100
        if lp_objectives:
            m = re.search(r"lambda([\d.]+)", label)
            lam = float(m.group(1)) if m else baseline_lambda
            lp_res = lp_objectives.get(lam) if lam is not None else None
            if lp_res and lp_res.get("objective_gbp"):
                lp_obj = lp_res["objective_gbp"]
                ratio = (kpi_dict["Total Profit (£)"] / lp_obj * 100) if lp_obj != 0 else None
            else:
                ratio = None
            kpi_dict["LP Efficiency Ratio (%)"] = ratio

        kpi_rows.append({"strategy": label, **kpi_dict})

    # Append one LP oracle row per lambda_ci so it appears in the table
    if lp_objectives:
        for lam, lp_res in sorted(lp_objectives.items()):
            if lp_res.get("lp_status") != "Optimal":
                continue
            P_MAX_MW = 49.9
            lp_obj = lp_res["objective_gbp"]
            lp_annualised = lp_obj * (365.0 / 219)
            lp_row = {
                "strategy": f"lp_lambda{lam:.1f}",
                "Financial Profit (£)": lp_obj,
                "Total Profit (£)": lp_obj,
                "Annualised Profit (£)": lp_annualised,
                "£/MW/year": lp_annualised / P_MAX_MW,
                "Mean Daily Profit": lp_obj / 219,
                "Profit Volatility": None,
                "Win Rate %": None,
                "Max Drawdown (£)": None,
                "DA Rev %": None,
                "ID Rev %": None,
                "Deg Cost %": (lp_res["deg_cost_gbp"] / lp_res["revenue_gbp"] * 100)
                               if lp_res.get("revenue_gbp") else None,
                "Carbon Cost %": (lp_res["carbon_cost_gbp"] / lp_res["revenue_gbp"] * 100)
                                  if lp_res.get("revenue_gbp") else None,
                "EFC": None,
                "Rev/EFC": None,
                "SoC Mean": None,
                "SoC Std": None,
                "Idle Fraction %": None,
                "Charge/Discharge Ratio": None,
                "Clipping Rate %": None,
                "Net Carbon (tCO2)": None,
                "Carbon Intensity (tCO2/MWh)": None,
                "Carbon Penalty (£)": lp_res["carbon_cost_gbp"],
                "Market Timing Score %": None,
                "DA Plan Utilisation %": None,
                "LP Efficiency Ratio (%)": 100.0,
            }
            kpi_rows.append(lp_row)

    summary_df = pd.DataFrame(kpi_rows)
    save_path = f"{RESULTS_DIR}/evaluation_summary.csv"
    summary_df.to_csv(save_path, index=False)
    print(f"Saved master evaluation table to {save_path}")

    return summary_df


def save_step_csv(df: pd.DataFrame, agent_name: str, seed: int, lambda_ci: float = 0.9) -> str:
    """Save the step-level DataFrame to CSV."""
    save_path = f"{RESULTS_DIR}/{agent_name}_lambda{lambda_ci:.1f}_seed{seed}_test_steps.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved step CSV to {save_path}")
    return save_path


# ============================================================
# IEEE PUBLICATION VISUALIZATIONS
# ============================================================

def plot_pareto_frontier(all_results: dict[str, dict], agent_names: list[str]) -> None:
    """
    NEW-1: Pareto Frontier — Profit vs. Carbon across lambda_ci values.

    Shows mean ± std across 5 seeds for each lambda_ci value.
    Multi-objective trade-off visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    lambda_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    for agent_name in agent_names:
        profits = []
        carbons = []
        profit_stds = []
        carbon_stds = []

        for lam in lambda_values:
            agent_profits = []
            agent_carbons = []

            for seed in range(5):
                key = f"{agent_name}_lambda{lam:.1f}_seed{seed}"
                if key in all_results:
                    kpis = all_results[key]
                    agent_profits.append(kpis["Total Profit (£)"])
                    agent_carbons.append(kpis["Net Carbon (tCO2)"])

            if agent_profits:
                profits.append(np.mean(agent_profits))
                carbons.append(np.mean(agent_carbons))
                profit_stds.append(np.std(agent_profits))
                carbon_stds.append(np.std(agent_carbons))

        if profits:
            ax.errorbar(carbons, profits, xerr=carbon_stds, yerr=profit_stds,
                       label=agent_name.upper(), marker='o', linewidth=2,
                       markersize=8, capsize=5, color=COLORS.get(agent_name, 'black'))

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    ax.set_xlabel("Net Carbon Emissions (tCO₂)", fontsize=11)
    ax.set_ylabel("Annual Profit (£)", fontsize=11)
    ax.set_title("Multi-Objective Pareto Frontier (λ_CI Ablation)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_pareto_frontier.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/01_pareto_frontier.png")
    plt.close()


def plot_action_distribution(all_results: dict[str, pd.DataFrame], agent_names: list[str],
                             lambda_values: list[float]) -> None:
    """
    Action Distribution Across Lambda_CI Values.

    Stacked bar chart: % Charge, Discharge, Idle for each lambda_ci.
    Proves the linear threshold changed agent behavior.
    """
    fig, axes = plt.subplots(1, len(agent_names), figsize=(4*len(agent_names), 4))
    if len(agent_names) == 1:
        axes = [axes]

    for idx, agent_name in enumerate(agent_names):
        ax = axes[idx]

        charges = []
        discharges = []
        idles = []

        for lam in lambda_values:
            # Aggregate across all 5 seeds
            all_actions = {'Charge': [], 'Discharge': [], 'Idle': []}

            for seed in range(5):
                key = f"{agent_name}_lambda{lam:.1f}_seed{seed}"
                if key in all_results:
                    df = all_results[key]
                    if "action_type" not in df.columns:
                        df = add_derived_columns(df)

                    action_counts = df['action_type'].value_counts()
                    total = len(df)
                    all_actions['Charge'].append(action_counts.get('Charge', 0) / total * 100)
                    all_actions['Discharge'].append(action_counts.get('Discharge', 0) / total * 100)
                    all_actions['Idle'].append(action_counts.get('Idle', 0) / total * 100)

            if all_actions['Charge']:
                charges.append(np.mean(all_actions['Charge']))
                discharges.append(np.mean(all_actions['Discharge']))
                idles.append(np.mean(all_actions['Idle']))

        if charges:
            x = np.arange(len(lambda_values))
            width = 0.6

            ax.bar(x, charges, width, label='Charge', color='#1f77b4', alpha=0.8)
            ax.bar(x, discharges, width, bottom=charges, label='Discharge', color='#ff7f0e', alpha=0.8)
            ax.bar(x, idles, width, bottom=np.array(charges)+np.array(discharges),
                  label='Idle', color='#2ca02c', alpha=0.8)

            ax.set_xlabel("λ_CI (Carbon Penalty Weight)", fontsize=10)
            ax.set_ylabel("Fraction of Steps (%)", fontsize=10)
            ax.set_title(f"{agent_name.upper()}", fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.1f}" for v in lambda_values])
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)
            if idx == 0:
                ax.legend(loc='upper left', framealpha=0.9)

    plt.suptitle("Action Distribution Across Carbon Penalty Values",
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_action_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/02_action_distribution.png")
    plt.close()


def plot_chronological_dispatch(all_results: dict[str, pd.DataFrame], agent_name: str,
                                lambda_ci: float, seed: int) -> None:
    """
    7-Day Chronological Case Study (Winter + Summer).

    Top: DA Price, ID Price, Carbon Intensity
    Bottom: SoC trajectory and P_applied dispatch decisions
    """
    key = f"{agent_name}_lambda{lambda_ci:.1f}_seed{seed}"
    if key not in all_results:
        print(f"Skipping chronological plot: {key} not found")
        return

    df = all_results[key]
    df['delivery_ts'] = pd.to_datetime(df['delivery_ts'])

    # Select winter (Jan-Feb) and summer (Jul-Aug) weeks
    winter_start = df[df['delivery_ts'].dt.month == 1].iloc[0]['delivery_ts'] if any(df['delivery_ts'].dt.month == 1) else df.iloc[0]['delivery_ts']
    summer_start = df[df['delivery_ts'].dt.month == 7].iloc[0]['delivery_ts'] if any(df['delivery_ts'].dt.month == 7) else df.iloc[len(df)//2]['delivery_ts']

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    for season_idx, (season_name, start_ts) in enumerate([("Winter", winter_start), ("Summer", summer_start)]):
        # Filter 7-day window
        end_ts = start_ts + pd.Timedelta(days=7)
        subset = df[(df['delivery_ts'] >= start_ts) & (df['delivery_ts'] < end_ts)].copy()

        if len(subset) == 0:
            continue

        # Top subplot: Prices and Carbon
        ax_top = fig.add_subplot(gs[0, season_idx])
        ax_top_twin1 = ax_top.twinx()

        ax_top.plot(subset['delivery_ts'], subset['da_price_now'], label='DA Price',
                   color='#1f77b4', linewidth=1.5)
        ax_top.plot(subset['delivery_ts'], subset['id_price_now'], label='ID Price',
                   color='#ff7f0e', linewidth=1.5, linestyle='--')
        ax_top_twin1.bar(subset['delivery_ts'], subset['ci_now'], alpha=0.3,
                        label='Carbon Intensity', color='#d62728')

        ax_top.set_ylabel("Price (GBP/MWh)", fontsize=10)
        ax_top_twin1.set_ylabel("CI (gCO₂/kWh)", fontsize=10, color='#d62728')
        ax_top.set_title(f"{season_name} (7-day case study)", fontsize=11, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='upper left', fontsize=8)

        # Bottom subplot: SoC and Dispatch
        ax_bot = fig.add_subplot(gs[1, season_idx])
        ax_bot_twin = ax_bot.twinx()

        ax_bot.fill_between(subset['delivery_ts'], 0, subset['soc'], alpha=0.4,
                           label='SoC', color='#2ca02c')

        # Dispatch scatter (charge=blue, discharge=red)
        charge_mask = subset['P_applied_MW'] < -1e-4
        discharge_mask = subset['P_applied_MW'] > 1e-4

        ax_bot_twin.scatter(subset.loc[charge_mask, 'delivery_ts'],
                           subset.loc[charge_mask, 'P_applied_MW'],
                           s=20, alpha=0.6, color='#1f77b4', label='Charge')
        ax_bot_twin.scatter(subset.loc[discharge_mask, 'delivery_ts'],
                           subset.loc[discharge_mask, 'P_applied_MW'],
                           s=20, alpha=0.6, color='#d62728', label='Discharge')

        ax_bot.set_ylabel("State of Charge (SoC)", fontsize=10)
        ax_bot_twin.set_ylabel("Power Dispatch (MW)", fontsize=10)
        ax_bot.set_xlabel("Time", fontsize=10)
        ax_bot.grid(True, alpha=0.3, axis='y')
        ax_bot.legend(loc='upper left', fontsize=8)
        ax_bot_twin.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"{agent_name.upper()} — Contextual Dispatch Behavior (λ_CI={lambda_ci})",
                 fontsize=12, fontweight='bold')
    plt.savefig(f"{FIGURES_DIR}/03_chronological_dispatch_{agent_name}_lambda{lambda_ci:.1f}.png",
               dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/03_chronological_dispatch_{agent_name}_lambda{lambda_ci:.1f}.png")
    plt.close()


def plot_degradation_accumulation(all_results: dict[str, pd.DataFrame], agent_name: str,
                                  seed: int) -> None:
    """
    Degradation Cost Accumulation over 219 days.

    Compares λ=0.0 (no carbon penalty, heavy cycling expected)
    vs λ=1.0 (heavy carbon penalty, minimal cycling expected).
    """
    key_low = f"{agent_name}_lambda0.0_seed{seed}"
    key_high = f"{agent_name}_lambda1.0_seed{seed}"

    if key_low not in all_results or key_high not in all_results:
        print(f"Skipping degradation plot: missing data for {agent_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, label, color in [(key_low, "λ_CI = 0.0 (No Carbon Penalty)", '#1f77b4'),
                               (key_high, "λ_CI = 1.0 (Heavy Carbon Penalty)", '#d62728')]:
        df = all_results[key]
        df['date'] = pd.to_datetime(df['delivery_ts']).dt.date

        daily_deg = df.groupby('date')['degradation_cost_gbp'].sum()
        cumulative_deg = daily_deg.cumsum()

        ax.plot(range(len(cumulative_deg)), cumulative_deg.values, label=label,
               linewidth=2.5, color=color, alpha=0.8)

    ax.set_xlabel("Days into Test Period", fontsize=11)
    ax.set_ylabel("Cumulative Degradation Cost (£)", fontsize=11)
    ax.set_title("Degradation Cost Accumulation: λ_CI Trade-off Effect",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_degradation_accumulation_{agent_name}_seed{seed}.png",
               dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/04_degradation_accumulation_{agent_name}_seed{seed}.png")
    plt.close()

def plot_agent_vs_baselines_bar(all_results: dict[str, pd.DataFrame],
                                 agent_names: list[str],
                                 target_lambda: float = 0.1,
                                 baseline_names: list[str] = None) -> None:
    """
    Figure 2: Agent vs. Baselines Normalised Bar Chart
    
    X-axis: Strategy (Idle, Random, P20/P80, DQN, DDQN, D3QN, D3QN+PER, LP Oracle)
    Y-axis: Three grouped metrics (Net Profit, EFC, Net Carbon), each normalised 
            to LP Oracle = 100%
    
    Variables computed:
        - Net Profit = sum(Planned_Profit + Intraday_Profit - degradation_cost_gbp - carbon_cashflow)
        - EFC (Equivalent Full Cycles) = sum(abs(P_applied_MW)) * 0.5 / E_cap
        - Net Carbon = sum(net_carbon_tCO2)
    """

    if baseline_names is None:
        baseline_names = ["idle", "random", "p20p80"]

    #compute aggregate metrics for each strategy
    baseline_metrics = {}

    for baseline_name in baseline_names:
        if baseline_name not in all_results:
            print(f"  ⚠ Warning: baseline '{baseline_name}' not found")
            continue

        df = all_results[baseline_name]

        net_profit = df["actual_reward"].sum()

        efc = df["P_applied_MW"].abs().sum() * 0.5 / E_MAX_MWH
        net_carbon = df["net_carbon_tCO2"].sum()

        baseline_metrics[baseline_name] = {
            "Net Profit (£)": net_profit,
            "EFC (cycles)": efc,
            "Net Carbon (tCO2)": net_carbon,
        }
    # -- Extracting RL Agent Metrics by reconstructing Keys ---

    rl_metrics = {}

    for agent_name in agent_names:
        # Collect metrics for each seed individually (DO NOT CONCATENATE)
        seed_profits = []
        seed_efcs = []
        seed_carbons = []

        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_results:
                continue

            df_seed = all_results[key]

            # Calculate for THIS SEED ONLY — must use df_seed, not the outer `df`
            # which holds the last baseline's dataframe (was a silent data-corruption bug).
            net_profit = df_seed["actual_reward"].sum()
            efc = df_seed["P_applied_MW"].abs().sum() * 0.5 / E_MAX_MWH
            net_carbon = df_seed["net_carbon_tCO2"].sum()

            seed_profits.append(net_profit)
            seed_efcs.append(efc)
            seed_carbons.append(net_carbon)

        if not seed_profits:
            print(f"  ⚠ Skipping {agent_name} (no seeds at lambda={target_lambda})")
            continue

        # average across seeds
        rl_metrics[agent_name] = {
            "Net Profit (£)": np.mean(seed_profits),
            "EFC (cycles)": np.mean(seed_efcs),
            "Net Carbon (tCO2)": np.mean(seed_carbons),
        }
    
    # -- Normalise against p20/P80 baseline ---
    all_metrics = {**baseline_metrics, **rl_metrics}
    
    if "p20p80" not in baseline_metrics:
        print("  ⚠ ERROR: P20/P80 baseline not found. Cannot normalize.")
        return
    
    p20p80_metrics = baseline_metrics["p20p80"]
    normalized_metrics = {}
    
    for strategy, m in all_metrics.items():
        normalized_metrics[strategy] = {}
        for metric_name, value in m.items():
            baseline_val = p20p80_metrics.get(metric_name, 1.0)
            if baseline_val == 0.0:
                normalized_val = 0.0 if value == 0.0 else (100.0 if value > 0.0 else -100.0)
            else:
                normalized_val = (value / baseline_val) * 100.0
            normalized_metrics[strategy][metric_name] = normalized_val
    
    
    # -- Construct Long-fomr dataframe
    rows_for_plot = []
    for strategy, metrics_dict in normalized_metrics.items():
        for metric_name, value in metrics_dict.items():
            rows_for_plot.append({
                "Strategy": strategy,
                "Metric": metric_name,
                "Value (% of P20/P80)": value
            })
    
    df_plot = pd.DataFrame(rows_for_plot)

    # -- Plotting ---

    fig, ax = plt.subplots(figsize=(13,6))

    strategies_ordered = baseline_names + agent_names
    strategies_ordered = [s for s in strategies_ordered if s in normalized_metrics]

    df_plot["Strategy"] = pd.Categorical(
        df_plot["Strategy"],
        categories=strategies_ordered,
        ordered=True
    )

    sns.barplot(data=df_plot, x="Strategy", y="Value (% of P20/P80)",
                hue="Metric", palette="Set2", ax=ax)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Normalised Score (% of P20/P80)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=11, fontweight="bold")
    ax.set_title(f"Figure 2: Agent vs. Baselines (λ_CI = {target_lambda})", 
                fontsize=12, fontweight="bold")
    ax.legend(title="Metric", loc="upper left", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"02_agent_vs_baselines_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 2: Agent vs. Baselines Bar Chart (λ = {target_lambda})")


def plot_cumulative_profit_timeseries(all_results: dict[str, pd.DataFrame],
                                       agent_names: list[str],
                                       target_lambda: float = 0.1,
                                       baseline_names: list[str] = None) -> None:
    """
    Figure 3: Cumulative Profit Time Series (STATISTICALLY CORRECTED)

    For each seed, calculate daily_profit Series. Average across seeds FIRST.
    Then apply cumsum() on the averaged daily profit.
    """
    if baseline_names is None:
        baseline_names = ["idle", "random", "p20p80"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot baselines (no seed aggregation)
    # ─────────────────────────────────────────────────────────────────────────────
    for baseline_name in baseline_names:
        if baseline_name not in all_results:
            continue

        df = all_results[baseline_name]
        df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
        daily_profit = df.groupby("date")["actual_reward"].sum()
        cumulative = daily_profit.cumsum()

        color = COLORS.get(baseline_name, "#cccccc")
        ax.plot(cumulative.index, cumulative.values,
               label=baseline_name.upper(), color=color, linewidth=1.5, alpha=0.8)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot RL agents — AVERAGE SEEDS FIRST, THEN CUMSUM
    # ─────────────────────────────────────────────────────────────────────────────
    for agent_name in agent_names:
        # Collect daily_profit Series for each seed (DO NOT CONCATENATE YET)
        daily_profit_seeds = []

        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_results:
                continue

            df_seed = all_results[key]
            df_seed["date"] = pd.to_datetime(df_seed["delivery_ts"]).dt.date
            # Use actual_reward (= R_DA + R_ID - deg - lambda_ci * carbon_cashflow).
            # This matches compute_kpis() and the comparison table.  The previous
            # formula subtracted the full carbon_cashflow (effectively lambda_ci=1),
            # making this plot inconsistent with the reported "Total Profit" numbers.
            daily_profit = df_seed.groupby("date")["actual_reward"].sum()
            daily_profit_seeds.append(daily_profit)

        if not daily_profit_seeds:
            continue

        # Average the daily profit Series across seeds FIRST
        mean_daily_profit = pd.concat(daily_profit_seeds, axis=1).mean(axis=1)
        cumulative = mean_daily_profit.cumsum()

        color = COLORS.get(agent_name, "#cccccc")
        ax.plot(cumulative.index, cumulative.values,
               label=agent_name.upper(), color=color, linewidth=2.5, alpha=1.0)
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Net Profit (GBP)", fontsize=11)
    ax.set_title(f"Figure 3: Cumulative Profit Time Series (λ = {target_lambda})", 
                fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"03_cumulative_profit_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 3: Cumulative Profit Time Series (λ = {target_lambda})")


def plot_rolling_risk_drawdown(all_results: dict[str, pd.DataFrame],
                                agent_names: list[str],
                                target_lambda: float = 0.1) -> None:
    """
    Figure 12: Rolling Risk Profile with Drawdown (STATISTICALLY CORRECTED)

    For each seed, calculate daily_profit Series. Average across seeds FIRST.
    Then compute cumulative, drawdown, and rolling Sharpe from the averaged daily profit.
    """
    fig, ax_cum = plt.subplots(figsize=(14, 6))
    ax_sharpe = ax_cum.twinx()

    sharpe_data = {}

    for agent_name in agent_names:
        # Collect daily_profit Series for each seed
        daily_profit_seeds = []

        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_results:
                continue

            df_seed = all_results[key]
            df_seed["date"] = pd.to_datetime(df_seed["delivery_ts"]).dt.date
            daily_profit = df_seed.groupby("date")["actual_reward"].sum()  # was `df` — wrong scope
            daily_profit_seeds.append(daily_profit)

        if not daily_profit_seeds:
            continue

        # Compute cumulative profit and rolling Sharpe PER SEED, then average the
        # resulting series.  Computing Sharpe on a seed-mean time series compresses
        # variance artificially and systematically overestimates the Sharpe ratio.
        cum_profit_seeds = []
        sharpe_seeds = []
        for dp in daily_profit_seeds:
            cp = dp.cumsum()
            cum_profit_seeds.append(cp)
            rm = dp.rolling(window=30).mean()
            rs = dp.rolling(window=30).std()
            sharpe_seeds.append(np.sqrt(365) * (rm / rs.replace(0, np.nan)))

        mean_cum_profit = pd.concat(cum_profit_seeds, axis=1).mean(axis=1)
        running_max = mean_cum_profit.cummax()
        drawdown = mean_cum_profit - running_max

        color = COLORS.get(agent_name, "#cccccc")
        ax_cum.plot(mean_cum_profit.index, mean_cum_profit.values,
                   label=agent_name.upper(), color=color, linewidth=2.5)

        # Drawdown shading on the mean cumulative trajectory
        ax_cum.fill_between(mean_cum_profit.index, mean_cum_profit.values, running_max.values,
                           where=(drawdown <= 0), alpha=0.2, color="red")

        # Rolling 14-day Sharpe: average of per-seed Sharpe series
        mean_sharpe = pd.concat(sharpe_seeds, axis=1).mean(axis=1)
        sharpe_data[agent_name] = mean_sharpe
    
    ax_cum.set_xlabel("Date", fontsize=11)
    ax_cum.set_ylabel("Cumulative Net Profit (GBP)", fontsize=11, color="black")
    ax_cum.set_title(f"Figure 12: Rolling Risk Profile & Drawdown (λ = {target_lambda})", 
                    fontsize=12, fontweight="bold")
    ax_cum.tick_params(axis="y", labelcolor="black")
    ax_cum.grid(alpha=0.3)
    
    if sharpe_data:
        for agent_name, sharpe_series in sharpe_data.items():
            color = COLORS.get(agent_name, "#cccccc")
            ax_sharpe.plot(sharpe_series.index, sharpe_series.values, 
                          linestyle="--", linewidth=1.5, color=color, alpha=0.6)
        ax_sharpe.set_ylabel("Rolling 14-day Sharpe Ratio", fontsize=11, color="blue")
        ax_sharpe.tick_params(axis="y", labelcolor="blue")
    
    lines1, labels1 = ax_cum.get_legend_handles_labels()
    ax_cum.legend(lines1, labels1, loc="upper left", framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"12_rolling_risk_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 12: Rolling Risk Profile & Drawdown (λ = {target_lambda})")


def plot_revenue_decomposition_stack(all_results: dict[str, pd.DataFrame], 
                                      agent_name: str,
                                      target_lambda: float = 0.1,
                                      seed: int = 0) -> None:
    """
    Figure 14: Revenue Decomposition Stacked Area (CORRECTED)
    
    Pulls a single seed for a specific agent. Can be called for each seed separately.
    
    Args:
        agent_name: Base agent name (e.g., "dqn")
        target_lambda: Lambda value
        seed: Specific seed to visualize (0-4)
    """
    key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
    
    if key not in all_results:
        print(f"Warning: {key} not in results. Skipping Figure 14.")
        return
    
    df = all_results[key]
    
    df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
    daily = df.groupby("date").agg({
        "Planned_Profit": "sum",
        "Intraday_Profit": "sum",
        "degradation_cost_gbp": "sum",
        "carbon_cashflow": "sum",
        "actual_reward": "sum" # Pull the true net profit directly
    }).reset_index()
    
    # Scale the carbon cashflow by lambda for the visual stack
    daily["scaled_carbon_cost"] = daily["carbon_cashflow"] * target_lambda
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.stackplot(
        daily["date"],
        daily["Planned_Profit"],
        daily["Intraday_Profit"],
        -daily["degradation_cost_gbp"],
        -daily["scaled_carbon_cost"], # Use the scaled version here!
        labels=["DA Revenue", "ID Revenue", "Degradation Cost", f"Carbon Cost (λ={target_lambda})"],
        colors=["#2ca02c", "#1f77b4", "#d62728", "#7f7f7f"],
        alpha=0.7
    )
    
    # Plot the true net profit overlay directly from the env
    ax.plot(daily["date"], daily["actual_reward"], 
           color="black", linewidth=2.5, label="Net Profit", zorder=10)
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Revenue / Cost (GBP)", fontsize=11)
    ax.set_title(f"Figure 14: Revenue Decomposition — {agent_name.upper()} (λ={target_lambda}, seed={seed})", 
                fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"14_revenue_decomposition_{agent_name}_lambda{target_lambda:.1f}_seed{seed}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 14: Revenue Decomposition ({agent_name.upper()}, λ={target_lambda}, seed={seed})")


def plot_dispatch_heatmap_48x7(all_results: dict[str, pd.DataFrame], 
                                agent_name: str,
                                target_lambda: float = 0.1) -> None:
    """
    Figure 4: Dispatch Heatmap (48 x 7) (CORRECTED)
    
    Aggregates across all 5 seeds for given agent and lambda.
    """
    seed_dfs = []
    for seed in range(5):
        key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
        if key in all_results:
            seed_dfs.append(all_results[key])
    
    if not seed_dfs:
        print(f"Warning: No seeds found for {agent_name} at lambda={target_lambda}. Skipping Figure 4.")
        return
    
    df = pd.concat(seed_dfs, ignore_index=True)
    
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
    df["tau"] = df["delivery_ts"].dt.hour * 2 + df["delivery_ts"].dt.minute // 30
    df["day_of_week"] = df["delivery_ts"].dt.day_name()
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(
        index="tau", 
        columns="day_of_week", 
        values="P_applied_MW", 
        aggfunc="mean"
    )
    pivot = pivot[day_order]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(pivot, cmap="RdBu_r", center=0, 
               vmin=-0.186, vmax=0.186,
               cbar_kws={"label": "Mean Power (MW)"}, ax=ax)
    
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Settlement Period (τ)", fontsize=11)
    ax.set_title(f"Figure 4: Dispatch Heatmap (48 × 7) — {agent_name.upper()} (λ={target_lambda})", 
                fontsize=12, fontweight="bold")
    
    tau_labels = [f"{int(i*30//60):02d}:{int(i*30%60):02d}" for i in range(0, 49, 4)]
    ax.set_yticks(range(0, 49, 4))
    ax.set_yticklabels(tau_labels, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"04_dispatch_heatmap_{agent_name}_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 4: Dispatch Heatmap ({agent_name.upper()}, λ={target_lambda})")


def plot_soc_distribution(all_results: dict[str, pd.DataFrame], 
                          agent_names: list[str],
                          target_lambda: float = 0.1) -> None:
    """
    Figure 7: SoC Distribution Histogram (CORRECTED)
    
    Aggregates across all 5 seeds for each agent.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for agent_name in agent_names:
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])
        
        if not seed_dfs:
            continue
        
        df = pd.concat(seed_dfs, ignore_index=True)
        soc = df["soc"].values
        color = COLORS.get(agent_name, "#cccccc")
        
        ax.hist(soc, bins=30, alpha=0.6, label=agent_name.upper(), 
               color=color, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("State of Charge", fontsize=11)
    ax.set_ylabel("Frequency (count)", fontsize=11)
    ax.set_title(f"Figure 7: SoC Distribution (λ={target_lambda})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"07_soc_distribution_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 7: SoC Distribution (λ={target_lambda})")


def plot_price_dispatch_scatter(all_results: dict[str, pd.DataFrame], 
                                agent_name: str,
                                target_lambda: float = 0.1) -> None:
    """
    Figure 5: Price--Dispatch Scatter (CORRECTED)
    
    Aggregates across all 5 seeds.
    """
    seed_dfs = []
    for seed in range(5):
        key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
        if key in all_results:
            seed_dfs.append(all_results[key])
    
    if not seed_dfs:
        print(f"Warning: No seeds found for {agent_name}. Skipping Figure 5.")
        return
    
    df = pd.concat(seed_dfs, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.scatter(df["id_price_now"], df["P_applied_MW"], 
              alpha=0.3, s=10, color=COLORS.get(agent_name, "#1f77b4"), edgecolors="none")
    
    # LOESS-like trend (rolling mean)
    sorted_idx = np.argsort(df["id_price_now"].values)
    x_sorted = df["id_price_now"].values[sorted_idx]
    y_sorted = df["P_applied_MW"].values[sorted_idx]
    
    window = 500
    x_loess = x_sorted[::window//2]
    y_loess = []
    for i in range(0, len(x_sorted), window//2):
        y_loess.append(y_sorted[max(0, i-window//2):min(len(y_sorted), i+window//2)].mean())
    
    ax.plot(x_loess, y_loess, color="red", linewidth=2.5, label="Trend (rolling mean)")
    
    ax.set_xlabel("Intraday Price (GBP/MWh)", fontsize=11)
    ax.set_ylabel("Dispatched Power (MW)", fontsize=11)
    ax.set_title(f"Figure 5: Price--Dispatch Sensitivity — {agent_name.upper()} (λ={target_lambda})", 
                fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"05_price_dispatch_scatter_{agent_name}_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 5: Price--Dispatch Scatter ({agent_name.upper()}, λ={target_lambda})")


def plot_policy_surface_heatmap(all_results: dict[str, pd.DataFrame], 
                                 agent_names: list[str],
                                 target_lambda: float = 0.1) -> None:
    """
    Figure 8: SoC--Price Policy Surface Heatmap (CORRECTED)
    
    Aggregates across all 5 seeds for each agent.
    """
    n_agents = len(agent_names)
    n_cols = min(2, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    if n_agents == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent_name in enumerate(agent_names):
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])
        
        if not seed_dfs:
            continue
        
        df = pd.concat(seed_dfs, ignore_index=True)
        
        df_binned = df.assign(
            soc_bin=pd.cut(df["soc"], bins=np.linspace(0.1, 0.9, 11)),
            price_bin=pd.qcut(df["id_price_now"], q=20, duplicates="drop")
        )
        
        pivot = df_binned.groupby(["soc_bin", "price_bin"])["P_applied_MW"].mean().unstack()
        
        ax = axes[idx // n_cols, idx % n_cols]
        sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-0.186, vmax=0.186,
                   cbar_kws={"label": "Mean P (MW)"}, ax=ax)
        
        ax.set_xlabel("Price Quantile (Low → High)", fontsize=10)
        ax.set_ylabel("SoC", fontsize=10)
        ax.set_title(f"{agent_name.upper()}", fontsize=11, fontweight="bold")
    
    for idx in range(n_agents, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    fig.suptitle(f"Figure 8: SoC--Price Policy Surface (λ={target_lambda})", 
                fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"08_policy_surface_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 8: Policy Surface Heatmap (λ={target_lambda})")


def plot_da_plan_fidelity_hexbin(all_results: dict[str, pd.DataFrame], 
                                  agent_names: list[str],
                                  target_lambda: float = 0.1) -> None:
    """
    Figure 9: DA Plan Commitment Fidelity (CORRECTED)
    
    Aggregates across all 5 seeds for each agent.
    """
    n_agents = len(agent_names)
    n_cols = min(2, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    if n_agents == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent_name in enumerate(agent_names):
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])
        
        if not seed_dfs:
            continue
        
        df = pd.concat(seed_dfs, ignore_index=True)
        
        ax = axes[idx // n_cols, idx % n_cols]
        
        hexbin = ax.hexbin(df["P_planned_MW"], df["P_applied_MW"],
                          gridsize=20, cmap="YlOrRd", mincnt=1, edgecolors="none")
        
        ax.plot([-0.2, 0.2], [-0.2, 0.2], "r--", linewidth=1.5, alpha=0.7, label="Perfect execution")
        
        ax.set_xlabel("P_planned (MW)", fontsize=10)
        ax.set_ylabel("P_applied (MW)", fontsize=10)
        ax.set_title(f"{agent_name.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label("Count", fontsize=9)
    
    for idx in range(n_agents, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    fig.suptitle(f"Figure 9: DA Plan Fidelity (λ={target_lambda})", 
                fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"09_da_plan_fidelity_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 9: DA Plan Fidelity Hexbin (λ={target_lambda})")


def plot_da_window_shift_violin(all_results: dict[str, pd.DataFrame], 
                                 agent_names: list[str],
                                 target_lambda: float = 0.1) -> None:
    """
    Figure 10: DA Window Behavioural Shift (Pre- vs. Post-Noon) (CORRECTED)
    
    Aggregates across all 5 seeds for each agent.
    """
    n_agents = len(agent_names)
    n_cols = min(2, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    if n_agents == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent_name in enumerate(agent_names):
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])
        
        if not seed_dfs:
            continue
        
        df = pd.concat(seed_dfs, ignore_index=True)
        
        ax = axes[idx // n_cols, idx % n_cols]
        
        df["da_state"] = df["da_available"].map({
            True: "Post-noon\n(DA visible)",
            False: "Pre-noon\n(no DA)"
        })
        
        sns.violinplot(data=df, x="da_state", y="P_applied_MW",
                      palette=["#4C72B0", "#DD8452"], ax=ax, inner="quart")
        
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Dispatched Power (MW)", fontsize=10)
        ax.set_title(f"{agent_name.upper()}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(n_agents, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    fig.suptitle(f"Figure 10: DA Window Shift (λ={target_lambda})", 
                fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"10_da_window_shift_lambda{target_lambda:.1f}.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 10: DA Window Shift Violin Plot (λ={target_lambda})")



# ============================================================
# CONSOLIDATED THEME-BASED FIGURES (IEEE PUBLICATION GRADE)
# ============================================================

def plot_theme_a_financial(all_results: dict, all_kpis: dict, agent_names: list,
                           lambda_values: list, target_lambda: float = 0.1) -> None:
    """
    THEME A: Financial Performance (Consolidated into 1 large figure)

    Subplots:
    - (1,1): Pareto Frontier (λ ablation)
    - (1,2): Agent vs Baselines Bar Chart
    - (1,3): Revenue Decomposition (Stacked Area) — NEW Figure 14
    - (2,1): Cumulative Profit Time Series
    - (2,2): Rolling Risk & Drawdown
    - (2,3): [Reserved for future expansion]
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,1) Pareto Frontier
    # ─────────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for agent_name in agent_names:
        profits = []
        carbons = []
        profit_stds = []
        carbon_stds = []

        for lam in lambda_values:
            agent_profits = []
            agent_carbons = []
            for seed in range(5):
                key = f"{agent_name}_lambda{lam:.1f}_seed{seed}"
                if key in all_kpis:
                    agent_profits.append(all_kpis[key]["Total Profit (£)"])
                    agent_carbons.append(all_kpis[key]["Net Carbon (tCO2)"])

            if agent_profits:
                profits.append(np.mean(agent_profits))
                carbons.append(np.mean(agent_carbons))
                profit_stds.append(np.std(agent_profits))
                carbon_stds.append(np.std(agent_carbons))

        if profits:
            ax1.errorbar(carbons, profits, xerr=carbon_stds, yerr=profit_stds,
                       label=agent_name.upper(), marker='o', linewidth=2,
                       markersize=8, capsize=5, color=COLORS.get(agent_name, 'black'))

    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Net Carbon (tCO₂)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Annual Profit (£)", fontsize=10, fontweight='bold')
    ax1.set_title("(a) Pareto Frontier — Multi-Objective Trade-off", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,2) Agent vs Baselines Bar Chart (3 KEY METRICS)
    # ─────────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    baseline_names = ["idle", "random", "p20p80"]
    baseline_metrics = {}
    for baseline_name in baseline_names:
        if baseline_name in all_results:
            df = all_results[baseline_name]
            net_profit = df["actual_reward"].sum()
            efc = df["P_applied_MW"].abs().sum() * 0.5 / E_MAX_MWH
            net_carbon = df["net_carbon_tCO2"].sum()
            baseline_metrics[baseline_name] = {
                "Net Profit (£)": net_profit,
                "EFC": efc,
                "Net Carbon (tCO₂)": net_carbon,
            }

    rl_metrics = {}
    for agent_name in agent_names:
        seed_profits = []
        seed_efcs = []
        seed_carbons = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                df_seed = all_results[key]
                seed_profits.append(df_seed["actual_reward"].sum())
                seed_efcs.append(df_seed["P_applied_MW"].abs().sum() * 0.5 / E_MAX_MWH)
                seed_carbons.append(df_seed["net_carbon_tCO2"].sum())

        if seed_profits:
            rl_metrics[agent_name] = {
                "Net Profit (£)": np.mean(seed_profits),
                "EFC": np.mean(seed_efcs),
                "Net Carbon (tCO₂)": np.mean(seed_carbons),
            }

    all_metrics = {**baseline_metrics, **rl_metrics}
    p20p80_baseline = baseline_metrics.get("p20p80", {"Net Profit (£)": 1, "EFC": 1, "Net Carbon (tCO₂)": 1})

    strategies = list(all_metrics.keys())
    profits_norm = [(all_metrics[s]["Net Profit (£)"] / p20p80_baseline["Net Profit (£)"]) * 100
                    if p20p80_baseline["Net Profit (£)"] != 0 else 0
                    for s in strategies]
    efcs_norm = [(all_metrics[s]["EFC"] / p20p80_baseline["EFC"]) * 100
                 if p20p80_baseline["EFC"] != 0 else 0
                 for s in strategies]
    carbons_norm = [(all_metrics[s]["Net Carbon (tCO₂)"] / p20p80_baseline["Net Carbon (tCO₂)"]) * 100
                    if p20p80_baseline["Net Carbon (tCO₂)"] != 0 else 0
                    for s in strategies]

    x = np.arange(len(strategies))
    width = 0.25
    ax2.bar(x - width, profits_norm, width, label='Net Profit (£)', color='#2ca02c', alpha=0.85)
    ax2.bar(x, efcs_norm, width, label='EFC (cycles)', color='#1f77b4', alpha=0.85)
    ax2.bar(x + width, carbons_norm, width, label='Net Carbon (tCO₂)', color='#d62728', alpha=0.85)
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='P20/P80 (100%)')
    ax2.set_ylabel("Normalised Score (% of P20/P80)", fontsize=10, fontweight='bold')
    ax2.set_title("(b) Agent vs Baselines (3 Key Metrics)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in strategies], rotation=30, ha='right', fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,1) Cumulative Profit Time Series
    # ─────────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    for baseline_name in baseline_names:
        if baseline_name in all_results:
            df = all_results[baseline_name]
            df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
            daily_profit = df.groupby("date")["actual_reward"].sum()
            cumulative = daily_profit.cumsum()
            color = COLORS.get(baseline_name, "#cccccc")
            ax3.plot(cumulative.index, cumulative.values, label=baseline_name.upper(),
                    color=color, linewidth=1.5, alpha=0.8)

    for agent_name in agent_names:
        daily_profit_seeds = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                df_seed = all_results[key]
                df_seed["date"] = pd.to_datetime(df_seed["delivery_ts"]).dt.date
                daily_profit_seeds.append(df_seed.groupby("date")["actual_reward"].sum())

        if daily_profit_seeds:
            mean_daily = pd.concat(daily_profit_seeds, axis=1).mean(axis=1)
            cumulative = mean_daily.cumsum()
            color = COLORS.get(agent_name, "#cccccc")
            ax3.plot(cumulative.index, cumulative.values, label=agent_name.upper(),
                    color=color, linewidth=2.5, alpha=1.0)

    ax3.set_xlabel("Date", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Cumulative Profit (£)", fontsize=10, fontweight='bold')
    ax3.set_title("(c) Cumulative Profit Trajectory", fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    ax3.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,2) Rolling Risk & Drawdown (with Rolling 14-Day Sharpe Ratio)
    # ─────────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_sharpe = ax4.twinx()

    sharpe_plotted = []

    for agent_name in agent_names:
        daily_profit_seeds = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                df_seed = all_results[key]
                df_seed["date"] = pd.to_datetime(df_seed["delivery_ts"]).dt.date
                daily_profit_seeds.append(df_seed.groupby("date")["actual_reward"].sum())

        if daily_profit_seeds:
            # Compute cumulative profit and rolling Sharpe PER SEED, then average.
            # Computing Sharpe on the seed-mean compresses variance and inflates the ratio.
            cum_seeds = []
            sharpe_seeds = []
            for dp in daily_profit_seeds:
                cp = dp.cumsum()
                cum_seeds.append(cp)
                rm = dp.rolling(window=130).mean()
                rs = dp.rolling(window=30).std()
                sharpe_seeds.append(np.sqrt(365) * (rm / rs.replace(0, np.nan)))

            mean_cum = pd.concat(cum_seeds, axis=1).mean(axis=1)
            running_max = mean_cum.cummax()
            drawdown = mean_cum - running_max

            color = COLORS.get(agent_name, "#cccccc")

            # Plot cumulative profit and drawdown on left axis
            ax4.plot(mean_cum.index, mean_cum.values, label=agent_name.upper(),
                    color=color, linewidth=2.5, zorder=10)
            ax4.fill_between(mean_cum.index, mean_cum.values, running_max.values,
                           where=(drawdown <= 0), alpha=0.15, color="red", zorder=5)

            # Rolling 14-day Sharpe: average of per-seed Sharpe series
            mean_sharpe = pd.concat(sharpe_seeds, axis=1).mean(axis=1)

            ax4_sharpe.plot(mean_sharpe.index, mean_sharpe.values,
                           linestyle='--', linewidth=1.5, color=color, alpha=0.6,
                           label=f'{agent_name.upper()} (Sharpe)', zorder=8)
            sharpe_plotted.append(True)

    ax4.set_xlabel("Date", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Cumulative Profit (£)", fontsize=10, fontweight='bold', color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4_sharpe.set_ylabel("Rolling 30-Day Annualised Sharpe Ratio", fontsize=10, fontweight='bold', color='#1f77b4')
    ax4_sharpe.tick_params(axis='y', labelcolor='#1f77b4')
    ax4.set_title("(d) Rolling Risk & Drawdown", fontsize=11, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_sharpe.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7,
              framealpha=0.9, ncol=2)
    ax4.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,3) Revenue Decomposition (Stacked Area) — Figure 14
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            # Pool all seeds
            df = pd.concat(seed_dfs, ignore_index=True)
            df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date

            # Aggregate by date
            daily = df.groupby("date").agg({
                "Planned_Profit": "sum",     # DA revenue
                "Intraday_Profit": "sum",    # ID revenue
                "degradation_cost_gbp": "sum",
                "carbon_cashflow": "sum"
            }).reset_index()

            daily["net_profit"] = (
                daily["Planned_Profit"] + daily["Intraday_Profit"]
                - daily["degradation_cost_gbp"] - daily["carbon_cashflow"]
            )

            ax5 = fig.add_subplot(gs[0, 2])

            # Stacked area: DA (green, positive), ID (blue, positive)
            ax5.fill_between(
                range(len(daily)), 0, daily["Planned_Profit"],
                label="DA Revenue", color="#2ca02c", alpha=0.7
            )
            ax5.fill_between(
                range(len(daily)), daily["Planned_Profit"],
                daily["Planned_Profit"] + daily["Intraday_Profit"],
                label="ID Revenue", color="#1f77b4", alpha=0.7
            )

            # Stacked downward: -Degradation (red), -Carbon (grey)
            ax5.fill_between(
                range(len(daily)), daily["Planned_Profit"] + daily["Intraday_Profit"],
                daily["Planned_Profit"] + daily["Intraday_Profit"] - daily["degradation_cost_gbp"],
                label="−Degradation", color="#d62728", alpha=0.7
            )
            ax5.fill_between(
                range(len(daily)),
                daily["Planned_Profit"] + daily["Intraday_Profit"] - daily["degradation_cost_gbp"],
                daily["net_profit"],
                label="−Carbon", color="#999999", alpha=0.7
            )

            # Overlay: daily net profit (black line)
            ax5.plot(range(len(daily)), daily["net_profit"],
                    color="black", linewidth=2, label="Net Profit", zorder=10)

            ax5.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax5.set_xlabel("Day (May 2025 → Jan 2026)", fontsize=10, fontweight='bold')
            ax5.set_ylabel("Daily Revenue/Cost (£)", fontsize=10, fontweight='bold')
            ax5.set_title(f"(e) Revenue Decomposition ({agent_name.upper()})", fontsize=11, fontweight='bold')
            ax5.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax5.grid(axis='y', alpha=0.3)

    fig.suptitle("THEME A: Financial Performance & Risk", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f"{FIGURES_DIR}/THEME_A_Financial_Performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ THEME A saved: {FIGURES_DIR}/THEME_A_Financial_Performance.png")
    plt.close()


def plot_theme_b_operations(all_results: dict, agent_names: list,
                           lambda_values: list, target_lambda: float = 0.1) -> None:
    """
    THEME B: Battery Operations (Consolidated into 1 large figure)

    Subplots:
    - (1,1): Action Distribution (stacked bar)
    - (1,2): SoC Distribution (histogram)
    - (2,1): Dispatch Heatmap (first agent)
    - (2,2): Degradation Accumulation (first agent)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,1) Action Distribution
    # ─────────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    for agent_name in agent_names:
        charges = []
        discharges = []
        idles = []

        for lam in lambda_values:
            all_actions = {'Charge': [], 'Discharge': [], 'Idle': []}
            for seed in range(5):
                key = f"{agent_name}_lambda{lam:.1f}_seed{seed}"
                if key in all_results:
                    df = all_results[key]
                    if "action_type" not in df.columns:
                        df = add_derived_columns(df)
                    action_counts = df['action_type'].value_counts()
                    total = len(df)
                    all_actions['Charge'].append(action_counts.get('Charge', 0) / total * 100)
                    all_actions['Discharge'].append(action_counts.get('Discharge', 0) / total * 100)
                    all_actions['Idle'].append(action_counts.get('Idle', 0) / total * 100)

            if all_actions['Charge']:
                charges.append(np.mean(all_actions['Charge']))
                discharges.append(np.mean(all_actions['Discharge']))
                idles.append(np.mean(all_actions['Idle']))

        if charges:
            x = np.arange(len(lambda_values))
            width = 0.2
            agent_idx = agent_names.index(agent_name)
            x_pos = x + (agent_idx - len(agent_names)/2 + 0.5) * width

            ax1.bar(x_pos, charges, width, label=f'{agent_name.upper()} Charge' if agent_idx == 0 else "",
                   color=COLORS.get(agent_name, '#cccccc'), alpha=0.6)
            ax1.bar(x_pos, discharges, width, bottom=charges,
                   label=f'{agent_name.upper()} Discharge' if agent_idx == 0 else "",
                   color=COLORS.get(agent_name, '#cccccc'), alpha=0.9)

    ax1.set_xlabel("λ_CI (Carbon Penalty)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Fraction of Steps (%)", fontsize=10, fontweight='bold')
    ax1.set_title("(a) Action Distribution Across Carbon Penalty", fontsize=11, fontweight='bold')
    ax1.set_xticks(np.arange(len(lambda_values)))
    ax1.set_xticklabels([f"{v:.1f}" for v in lambda_values], fontsize=8)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    ax1.grid(axis='y', alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,2) SoC Distribution
    # ─────────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    for agent_name in agent_names:
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            color = COLORS.get(agent_name, "#cccccc")
            ax2.hist(df["soc"].values, bins=25, alpha=0.6, label=agent_name.upper(),
                    color=color, edgecolor="black", linewidth=0.5)

    ax2.set_xlabel("State of Charge (SoC)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Frequency", fontsize=10, fontweight='bold')
    ax2.set_title("(b) SoC Distribution", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,1) Dispatch Heatmap (First Agent)
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
            df["tau"] = df["delivery_ts"].dt.hour * 2 + df["delivery_ts"].dt.minute // 30
            df["day_of_week"] = df["delivery_ts"].dt.day_name()

            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = df.pivot_table(index="tau", columns="day_of_week", values="P_applied_MW", aggfunc="mean")
            pivot = pivot[day_order]

            ax3 = fig.add_subplot(gs[1, 0])
            sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-0.2, vmax=0.2,
                       cbar_kws={"label": "Mean Power (MW)"}, ax=ax3)
            ax3.set_xlabel("Day of Week", fontsize=10, fontweight='bold')
            ax3.set_ylabel("Settlement Period", fontsize=10, fontweight='bold')
            ax3.set_title(f"(c) Dispatch Heatmap — {agent_name.upper()}", fontsize=11, fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,2) Degradation Accumulation
    # ─────────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    if agent_names:
        agent_name = agent_names[0]
        for lam, color_deg in [(0.0, '#1f77b4'), (1.0, '#d62728')]:
            key = f"{agent_name}_lambda{lam:.1f}_seed0"
            if key in all_results:
                df = all_results[key]
                df['date'] = pd.to_datetime(df['delivery_ts']).dt.date
                daily_deg = df.groupby('date')['degradation_cost_gbp'].sum()
                cumulative_deg = daily_deg.cumsum()
                label = f"λ_CI = {lam:.1f}"
                ax4.plot(range(len(cumulative_deg)), cumulative_deg.values, label=label,
                       linewidth=2.5, color=color_deg, alpha=0.8)

    ax4.set_xlabel("Days into Test Period", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Cumulative Degradation Cost (£)", fontsize=10, fontweight='bold')
    ax4.set_title("(d) Degradation Cost Trade-off", fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)

    fig.suptitle("THEME B: Battery Operations & Degradation", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f"{FIGURES_DIR}/THEME_B_Battery_Operations.png", dpi=300, bbox_inches='tight')
    print(f"✓ THEME B saved: {FIGURES_DIR}/THEME_B_Battery_Operations.png")
    plt.close()


def plot_theme_c_policy(all_results: dict, agent_names: list,
                       target_lambda: float = 0.1) -> None:
    """
    THEME C: Market Timing & Policy Intelligence (Consolidated into 1 large figure)

    Subplots:
    - (1,1): Price--Dispatch Scatter
    - (1,2): Policy Surface Heatmap
    - (1,3): Action Space Utilization (Heatmap) — NEW Figure 13
    - (2,1): DA Plan Fidelity
    - (2,2): DA Window Behavioral Shift
    - (2,3): [Reserved for future expansion]
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,1) Price--Dispatch Scatter (First Agent)
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(df["id_price_now"], df["P_applied_MW"], alpha=0.2, s=5,
                       color=COLORS.get(agent_name, "#1f77b4"), edgecolors="none")

            sorted_idx = np.argsort(df["id_price_now"].values)
            x_sorted = df["id_price_now"].values[sorted_idx]
            y_sorted = df["P_applied_MW"].values[sorted_idx]
            window = 500
            x_loess = x_sorted[::window//2]
            y_loess = [y_sorted[max(0, i-window//2):min(len(y_sorted), i+window//2)].mean()
                      for i in range(0, len(x_sorted), window//2)]

            ax1.plot(x_loess, y_loess, color="red", linewidth=2.5, label="Trend", alpha=0.9)
            ax1.set_xlabel("Intraday Price (£/MWh)", fontsize=10, fontweight='bold')
            ax1.set_ylabel("Dispatched Power (MW)", fontsize=10, fontweight='bold')
            ax1.set_title("(a) Price--Dispatch Sensitivity", fontsize=11, fontweight='bold')
            ax1.legend(loc='best', fontsize=8, framealpha=0.9)
            ax1.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,2) Policy Surface Heatmap
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            df_binned = df.assign(
                soc_bin=pd.cut(df["soc"], bins=np.linspace(0.1, 0.9, 11)),
                price_bin=pd.qcut(df["id_price_now"], q=15, duplicates="drop")
            )
            pivot = df_binned.groupby(["soc_bin", "price_bin"])["P_applied_MW"].mean().unstack()

            ax2 = fig.add_subplot(gs[0, 1])
            sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-0.2, vmax=0.2,
                       cbar_kws={"label": "Mean Power (MW)"}, ax=ax2)
            ax2.set_xlabel("Price Quantile", fontsize=10, fontweight='bold')
            ax2.set_ylabel("SoC", fontsize=10, fontweight='bold')
            ax2.set_title("(b) SoC--Price Policy Surface", fontsize=11, fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,3) Action Space Utilization (Heatmap) — Figure 13
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)

            # Need to decode discrete actions from the CSV
            # For now, approximate using dispatch power levels and DA availability
            # This creates a 2D distribution of dispatch_power × da_available
            n_power_levels = 11  # -P_max to +P_max in 11 steps
            n_da_slots = 48      # 48 settlement periods per day

            # Create action frequency matrix
            action_freq = np.zeros((n_power_levels, n_da_slots))

            # Bin dispatch power to 11 levels
            df_power_binned = pd.cut(df["P_applied_MW"], bins=11, labels=range(11))

            # tau in the CSV is 1-indexed (1..48 from the env info dict).
            # Convert to 0-indexed (0..47) before binning; without the -1,
            # slot 48 maps to 0 and the last period merges into the first bucket.
            df_tau_binned = (df["tau"] - 1) % 48

            # Count action frequencies
            for power_bin, tau_slot in zip(df_power_binned, df_tau_binned):
                if pd.notna(power_bin):
                    action_freq[int(power_bin), int(tau_slot)] += 1

            ax6 = fig.add_subplot(gs[0, 2])
            im = ax6.imshow(action_freq, cmap="YlOrRd", aspect="auto", interpolation="nearest")
            ax6.set_xlabel("DA Plan Slot (0–47)", fontsize=10, fontweight='bold')
            ax6.set_ylabel("Dispatch Power Level (−P_max → +P_max)", fontsize=10, fontweight='bold')
            ax6.set_title(f"(c) Action Space Utilization ({agent_name.upper()})", fontsize=11, fontweight='bold')
            ax6.set_yticks(range(0, 11, 2))
            ax6.set_yticklabels([f"{i}" for i in range(0, 11, 2)])
            cbar = plt.colorbar(im, ax=ax6)
            cbar.set_label("Frequency (counts)", fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,1) DA Plan Fidelity
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            ax3 = fig.add_subplot(gs[1, 0])

            hexbin = ax3.hexbin(df["P_planned_MW"], df["P_applied_MW"],
                               gridsize=15, cmap="YlOrRd", mincnt=1, edgecolors="none")
            ax3.plot([-0.2, 0.2], [-0.2, 0.2], "r--", linewidth=2, alpha=0.7, label="Perfect execution")

            ax3.set_xlabel("P_planned (MW)", fontsize=10, fontweight='bold')
            ax3.set_ylabel("P_applied (MW)", fontsize=10, fontweight='bold')
            ax3.set_title("(c) DA Plan Commitment Fidelity", fontsize=11, fontweight='bold')
            ax3.set_xlim(-0.2, 0.2)
            ax3.set_ylim(-0.2, 0.2)
            ax3.legend(fontsize=8, loc='upper left', framealpha=0.9)
            ax3.grid(alpha=0.3)

            cbar = plt.colorbar(hexbin, ax=ax3)
            cbar.set_label("Count", fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────────
    # (2,2) DA Window Behavioral Shift
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            df["da_state"] = df["da_available"].map({
                True: "Post-noon\n(DA visible)",
                False: "Pre-noon\n(no DA)"
            })

            ax4 = fig.add_subplot(gs[1, 1])
            sns.violinplot(data=df, x="da_state", y="P_applied_MW",
                          palette=["#4C72B0", "#DD8452"], ax=ax4, inner="quart")

            ax4.set_xlabel("", fontsize=10, fontweight='bold')
            ax4.set_ylabel("Dispatched Power (MW)", fontsize=10, fontweight='bold')
            ax4.set_title("(d) DA Window Behavioral Shift", fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)

    fig.suptitle("THEME C: Market Timing & Policy Intelligence", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f"{FIGURES_DIR}/THEME_C_Policy_Intelligence.png", dpi=300, bbox_inches='tight')
    print(f"✓ THEME C saved: {FIGURES_DIR}/THEME_C_Policy_Intelligence.png")
    plt.close()


def plot_theme_d_regime(all_results: dict, all_kpis: dict, agent_names: list,
                        lambda_values: list, target_lambda: float = 0.1) -> None:
    """
    THEME D: Regime Adaptation & Carbon Intensity (Consolidated into 1 figure)

    Subplots:
    - (1,1): Pareto Frontier (λ_CI ablation) — Figure 6 (moved from THEME A)
    - (1,2): Seasonal Dispatch Strategy (Violin/Ridgeline) — Figure 11
    """
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, hspace=0.35, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,1) Pareto Frontier (λ_CI ablation)
    # ─────────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for agent_name in agent_names:
        profits = []
        carbons = []
        profit_stds = []
        carbon_stds = []

        for lam in lambda_values:
            agent_profits = []
            agent_carbons = []
            for seed in range(5):
                key = f"{agent_name}_lambda{lam:.1f}_seed{seed}"
                if key in all_kpis:
                    agent_profits.append(all_kpis[key]["Total Profit (£)"])
                    agent_carbons.append(all_kpis[key]["Net Carbon (tCO2)"])

            if agent_profits:
                profits.append(np.mean(agent_profits))
                carbons.append(np.mean(agent_carbons))
                profit_stds.append(np.std(agent_profits))
                carbon_stds.append(np.std(agent_carbons))

        if profits:
            ax1.errorbar(carbons, profits, xerr=carbon_stds, yerr=profit_stds,
                       label=agent_name.upper(), marker='o', linewidth=2,
                       markersize=8, capsize=5, color=COLORS.get(agent_name, 'black'))

    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Net Carbon (tCO₂)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Annual Profit (£)", fontsize=10, fontweight='bold')
    ax1.set_title("(a) Carbon vs Profit Trade-off (λ_CI Ablation)", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────────
    # (1,2) Seasonal Dispatch Strategy (Violin by Month) — Figure 11
    # ─────────────────────────────────────────────────────────────────────────────
    if agent_names:
        agent_name = agent_names[0]
        seed_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                seed_dfs.append(all_results[key])

        if seed_dfs:
            df = pd.concat(seed_dfs, ignore_index=True)
            df["date"] = pd.to_datetime(df["delivery_ts"])
            df["month"] = df["date"].dt.strftime("%B")

            # Order months from May 2025 to January 2026
            month_order = ["May", "June", "July", "August", "September",
                          "October", "November", "December", "January"]
            df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
            df = df.sort_values("month")

            ax2 = fig.add_subplot(gs[0, 1])
            sns.violinplot(data=df, x="month", y="P_applied_MW",
                          palette="Set2", ax=ax2, inner="quart")

            ax2.set_xlabel("Calendar Month", fontsize=10, fontweight='bold')
            ax2.set_ylabel("Dispatched Power (MW)", fontsize=10, fontweight='bold')
            ax2.set_title(f"(b) Seasonal Dispatch Patterns ({agent_name.upper()})", fontsize=11, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

    fig.suptitle("THEME D: Regime Adaptation & Carbon Intensity", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f"{FIGURES_DIR}/THEME_D_Regime_Adaptation.png", dpi=300, bbox_inches='tight')
    print(f"✓ THEME D saved: {FIGURES_DIR}/THEME_D_Regime_Adaptation.png")
    plt.close()


# ============================================================
# IEEE PUBLICATION: 5 MAIN FIGURES + 4 APPENDIX FIGURES
# Hypothesis: Continuous 3D SAC outperforms discrete DDQN
# ============================================================

def save_kpi_table(all_results: dict, all_kpis: dict,
                   lp_objectives: dict | None,
                   agent_names: list,
                   target_lambda: float = 0.9) -> None:
    """Save Table_1_KPI_Summary.csv — all mean ± std columns for Table 3 in the thesis.

    Columns exported:
        Strategy, Formulation,
        Profit_mean_GBP, Profit_std_GBP,
        WinRate_mean_pct, WinRate_std_pct,
        EFC_mean, EFC_std,
        Rev_EFC_mean_GBP, Rev_EFC_std_GBP,
        NetCarbon_mean_tCO2, NetCarbon_std_tCO2,
        DegCost_mean_GBP, DegCost_std_GBP,
        CarbonCost_mean_GBP, CarbonCost_std_GBP,
        LP_Oracle_GBP,          ← reference ceiling for the target lambda
        LP_Oracle_DegCost_GBP,  ← LP oracle degradation cost
        LP_Oracle_CarbonCost_GBP, ← LP oracle carbon cost
        LP_Efficiency_Ratio_pct
    """
    # Resolve LP oracle values once for the target lambda
    lp_obj_gbp: float | None = None
    lp_deg_gbp: float | None = None
    lp_carbon_gbp: float | None = None
    lp_efc: float | None = None
    lp_rev_efc: float | None = None
    lp_net_carbon_tco2: float | None = None
    if lp_objectives and target_lambda in lp_objectives:
        _lp = lp_objectives[target_lambda]
        if _lp.get("lp_status") == "Optimal":
            lp_obj_gbp    = _lp.get("objective_gbp")
            lp_deg_gbp    = _lp.get("deg_cost_gbp")
            lp_carbon_gbp = _lp.get("carbon_cost_gbp")
            # Derive EFC, Rev/EFC, Net Carbon from LP schedule
            _sched = _lp.get("schedule")
            if _sched is not None:
                _dt = 0.5  # half-hour settlement periods
                _throughput = (_sched["P_ch_MW"] + _sched["P_dis_MW"]).sum() * _dt
                lp_efc = _throughput / (2 * E_MAX_MWH)
                lp_rev_efc = (lp_obj_gbp / lp_efc) if lp_efc > 0 else None
                # Net carbon: import absorbs CI, export displaces MEF (tCO2)
                _e_imp_kwh = _sched["P_ch_MW"]  * _dt * 1000  # MWh → kWh
                _e_exp_kwh = _sched["P_dis_MW"] * _dt * 1000
                _net_co2_kg = (_e_imp_kwh * _sched["ci_gco2_kwh"]
                               - _e_exp_kwh * _sched["mef_gco2_kwh"]).sum()
                lp_net_carbon_tco2 = _net_co2_kg / 1e6  # gCO2 → tCO2
            else:
                lp_efc = lp_rev_efc = lp_net_carbon_tco2 = None

    FORMULATION = {
        "sac":  "Continuous 3D",
        "ddqn": "Discrete 5,808",
        "dqn":  "Discrete 5,808",
        "d3qn": "Discrete 5,808",
    }

    rows = []

    # ── RL agents (multi-seed) ──────────────────────────────────────────
    for agent_name in agent_names:
        profits, efcs, rev_efcs, win_rates, net_carbons = [], [], [], [], []
        deg_costs, carbon_costs = [], []
        idle_fracs, da_rev_pcts, id_rev_pcts, deg_cost_pcts, da_plan_utils = [], [], [], [], []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_kpis:
                continue
            k = all_kpis[key]
            profits.append(k["Financial Profit (£)"])
            efcs.append(k["EFC"])
            rev_efcs.append(k["Rev/EFC"])
            win_rates.append(k["Win Rate %"])
            net_carbons.append(k["Net Carbon (tCO2)"])
            idle_fracs.append(k["Idle Fraction %"])
            da_rev_pcts.append(k["DA Rev %"])
            id_rev_pcts.append(k["ID Rev %"])
            deg_cost_pcts.append(k["Deg Cost %"])
            da_plan_utils.append(k["DA Plan Utilisation %"])
            # Absolute cost values from the step-level dataframe
            df_seed = all_results.get(key)
            if df_seed is not None:
                deg_costs.append(df_seed["degradation_cost_gbp"].sum())
                carbon_costs.append(df_seed["carbon_cashflow"].sum())

        if not profits:
            continue

        lp_eff = (np.mean(profits) / lp_obj_gbp * 100) if lp_obj_gbp else None
        rows.append({
            "Strategy":               agent_name.upper(),
            "Formulation":            FORMULATION.get(agent_name.lower(), "—"),
            "Profit_mean_GBP":        round(np.mean(profits), 0),
            "Profit_std_GBP":         round(np.std(profits, ddof=1), 0),
            "WinRate_mean_pct":       round(np.mean(win_rates), 1),
            "WinRate_std_pct":        round(np.std(win_rates, ddof=1), 1),
            "EFC_mean":               round(np.mean(efcs), 2),
            "EFC_std":                round(np.std(efcs, ddof=1), 2),
            "Rev_EFC_mean_GBP":       round(np.mean(rev_efcs), 0),
            "Rev_EFC_std_GBP":        round(np.std(rev_efcs, ddof=1), 0),
            "NetCarbon_mean_tCO2":    round(np.mean(net_carbons), 1),
            "NetCarbon_std_tCO2":     round(np.std(net_carbons, ddof=1), 1),
            "DegCost_mean_GBP":       round(np.mean(deg_costs), 0) if deg_costs else None,
            "DegCost_std_GBP":        round(np.std(deg_costs, ddof=1), 0) if len(deg_costs) > 1 else 0,
            "CarbonCost_mean_GBP":    round(np.mean(carbon_costs), 0) if carbon_costs else None,
            "CarbonCost_std_GBP":     round(np.std(carbon_costs, ddof=1), 0) if len(carbon_costs) > 1 else 0,
            "LP_Oracle_GBP":          round(lp_obj_gbp, 0) if lp_obj_gbp else None,
            "LP_Oracle_DegCost_GBP":  round(lp_deg_gbp, 0) if lp_deg_gbp else None,
            "LP_Oracle_CarbonCost_GBP": round(lp_carbon_gbp, 0) if lp_carbon_gbp else None,
            "LP_Efficiency_Ratio_pct": round(lp_eff, 1) if lp_eff is not None else None,
            # ── Extended operational metrics (pivoted table) ──────────
            "IdleFrac_mean_pct":      round(np.mean(idle_fracs), 1) if idle_fracs else None,
            "IdleFrac_std_pct":       round(np.std(idle_fracs, ddof=1), 1) if len(idle_fracs) > 1 else 0,
            "DARevPct_mean":          round(np.mean(da_rev_pcts), 1) if da_rev_pcts else None,
            "DARevPct_std":           round(np.std(da_rev_pcts, ddof=1), 1) if len(da_rev_pcts) > 1 else 0,
            "IDRevPct_mean":          round(np.mean(id_rev_pcts), 1) if id_rev_pcts else None,
            "IDRevPct_std":           round(np.std(id_rev_pcts, ddof=1), 1) if len(id_rev_pcts) > 1 else 0,
            "DegCostPct_mean":        round(np.mean(deg_cost_pcts), 1) if deg_cost_pcts else None,
            "DegCostPct_std":         round(np.std(deg_cost_pcts, ddof=1), 1) if len(deg_cost_pcts) > 1 else 0,
            "DAPlanUtil_mean_pct":    round(np.mean(da_plan_utils), 1) if da_plan_utils else None,
            "DAPlanUtil_std_pct":     round(np.std(da_plan_utils, ddof=1), 1) if len(da_plan_utils) > 1 else 0,
        })

    # ── Deterministic baselines (single rollout — std = 0) ─────────────
    BASELINE_FORMULATION = {
        "p20p80": "Rule-based",
        "idle":   "Null",
        "random": "Null",
    }
    for label in ["p20p80", "idle", "random"]:
        if label not in all_results:
            continue
        df_bl = all_results[label]
        if "financial_profit_gbp" not in df_bl.columns:
            df_bl = add_derived_columns(df_bl.copy())
        k = compute_kpis(df_bl)
        lp_eff = (k["Financial Profit (£)"] / lp_obj_gbp * 100) if lp_obj_gbp else None
        bl_deg_cost    = df_bl["degradation_cost_gbp"].sum() if "degradation_cost_gbp" in df_bl.columns else None
        bl_carbon_cost = df_bl["carbon_cashflow"].sum() if "carbon_cashflow" in df_bl.columns else None
        rows.append({
            "Strategy":               label.upper(),
            "Formulation":            BASELINE_FORMULATION[label],
            "Profit_mean_GBP":        round(k["Financial Profit (£)"], 0),
            "Profit_std_GBP":         0,
            "WinRate_mean_pct":       round(k["Win Rate %"], 1),
            "WinRate_std_pct":        0,
            "EFC_mean":               round(k["EFC"], 2),
            "EFC_std":                0,
            "Rev_EFC_mean_GBP":       round(k["Rev/EFC"], 0),
            "Rev_EFC_std_GBP":        0,
            "NetCarbon_mean_tCO2":    round(k["Net Carbon (tCO2)"], 1),
            "NetCarbon_std_tCO2":     0,
            "DegCost_mean_GBP":       round(bl_deg_cost, 0) if bl_deg_cost is not None else None,
            "DegCost_std_GBP":        0,
            "CarbonCost_mean_GBP":    round(bl_carbon_cost, 0) if bl_carbon_cost is not None else None,
            "CarbonCost_std_GBP":     0,
            "LP_Oracle_GBP":          round(lp_obj_gbp, 0) if lp_obj_gbp else None,
            "LP_Oracle_DegCost_GBP":  round(lp_deg_gbp, 0) if lp_deg_gbp else None,
            "LP_Oracle_CarbonCost_GBP": round(lp_carbon_gbp, 0) if lp_carbon_gbp else None,
            "LP_Efficiency_Ratio_pct": round(lp_eff, 1) if lp_eff is not None else None,
            # ── Extended operational metrics ──────────────────────────
            "IdleFrac_mean_pct":      round(k["Idle Fraction %"], 1),
            "IdleFrac_std_pct":       0,
            "DARevPct_mean":          round(k["DA Rev %"], 1),
            "DARevPct_std":           0,
            "IDRevPct_mean":          round(k["ID Rev %"], 1),
            "IDRevPct_std":           0,
            "DegCostPct_mean":        round(k["Deg Cost %"], 1),
            "DegCostPct_std":         0,
            "DAPlanUtil_mean_pct":    round(k["DA Plan Utilisation %"], 1),
            "DAPlanUtil_std_pct":     0,
        })

    # ── LP Oracle row ───────────────────────────────────────────────────
    if lp_obj_gbp is not None:
        # Derive LP Deg Cost % from schedule if available
        _lp_data = (lp_objectives or {}).get(target_lambda, {})
        _lp_sched = _lp_data.get("schedule")
        lp_idle_frac = None
        lp_deg_cost_pct = None
        if _lp_sched is not None and lp_deg_gbp is not None:
            _total_steps = len(_lp_sched)
            _active = ((_lp_sched["P_ch_MW"] + _lp_sched["P_dis_MW"]) > 1e-4).sum()
            lp_idle_frac = round((1.0 - _active / _total_steps) * 100, 1) if _total_steps > 0 else None
            _lp_rev = _lp_data.get("revenue_gbp")
            if _lp_rev and _lp_rev > 0:
                lp_deg_cost_pct = round(lp_deg_gbp / _lp_rev * 100, 1)

        rows.append({
            "Strategy":               "LP_ORACLE",
            "Formulation":            "Perfect foresight",
            "Profit_mean_GBP":        round(lp_obj_gbp, 0),
            "Profit_std_GBP":         None,
            "WinRate_mean_pct":       None,
            "WinRate_std_pct":        None,
            "EFC_mean":               round(lp_efc, 2) if lp_efc is not None else None,
            "EFC_std":                None,
            "Rev_EFC_mean_GBP":       round(lp_rev_efc, 0) if lp_rev_efc is not None else None,
            "Rev_EFC_std_GBP":        None,
            "NetCarbon_mean_tCO2":    round(lp_net_carbon_tco2, 1) if lp_net_carbon_tco2 is not None else None,
            "NetCarbon_std_tCO2":     None,
            "DegCost_mean_GBP":       round(lp_deg_gbp, 0) if lp_deg_gbp is not None else None,
            "DegCost_std_GBP":        None,
            "CarbonCost_mean_GBP":    round(lp_carbon_gbp, 0) if lp_carbon_gbp is not None else None,
            "CarbonCost_std_GBP":     None,
            "LP_Oracle_GBP":          round(lp_obj_gbp, 0),
            "LP_Oracle_DegCost_GBP":  round(lp_deg_gbp, 0) if lp_deg_gbp is not None else None,
            "LP_Oracle_CarbonCost_GBP": round(lp_carbon_gbp, 0) if lp_carbon_gbp is not None else None,
            "LP_Efficiency_Ratio_pct": 100.0,
            # ── Extended operational metrics ──────────────────────────
            "IdleFrac_mean_pct":      lp_idle_frac,
            "IdleFrac_std_pct":       None,
            "DARevPct_mean":          None,
            "DARevPct_std":           None,
            "IDRevPct_mean":          None,
            "IDRevPct_std":           None,
            "DegCostPct_mean":        lp_deg_cost_pct,
            "DegCostPct_std":         None,
            "DAPlanUtil_mean_pct":    None,
            "DAPlanUtil_std_pct":     None,
        })

    out_path = os.path.join(FIGURES_MAIN_DIR, "Table_1_KPI_Summary.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✓ Table_1_KPI_Summary.csv saved → {out_path}")
    # ── Human-readable summary: Financial block ──────────────────────
    print("\n  ── KPI Summary: Financial Block ──")
    print(f"  {'Strategy':<14} {'Profit(£k)':<13} {'Win%':<10} {'EFC':<8} "
          f"{'Rev/EFC':<9} {'DegCost(£k)':<13} {'LP Eff%':<9}")
    print("  " + "-" * 80)
    for r in rows:
        prof  = f"{r['Profit_mean_GBP']/1000:.1f}±{(r['Profit_std_GBP'] or 0)/1000:.1f}"
        win   = f"{r['WinRate_mean_pct']}±{r['WinRate_std_pct']}" if r['WinRate_mean_pct'] is not None else "—"
        efc   = f"{r['EFC_mean']}±{r['EFC_std']}" if r['EFC_mean'] is not None else "—"
        refc  = f"{r['Rev_EFC_mean_GBP']}±{r['Rev_EFC_std_GBP']}" if r['Rev_EFC_mean_GBP'] is not None else "—"
        deg   = f"{(r['DegCost_mean_GBP'] or 0)/1000:.1f}" if r.get('DegCost_mean_GBP') is not None else "—"
        lpeff = f"{r['LP_Efficiency_Ratio_pct']}" if r['LP_Efficiency_Ratio_pct'] is not None else "—"
        print(f"  {r['Strategy']:<14} {prof:<13} {win:<10} {efc:<8} {refc:<9} {deg:<13} {lpeff:<9}")
    # ── Human-readable summary: Operational block ────────────────────
    print("\n  ── KPI Summary: Operational Block (Pivoted) ──")
    print(f"  {'Strategy':<14} {'Idle%':<10} {'DA Rev%':<10} {'ID Rev%':<10} "
          f"{'DegCost%':<11} {'DAPlanUtil%':<13}")
    print("  " + "-" * 70)
    for r in rows:
        idle  = f"{r.get('IdleFrac_mean_pct', '—')}±{r.get('IdleFrac_std_pct', 0)}" if r.get('IdleFrac_mean_pct') is not None else "—"
        da    = f"{r.get('DARevPct_mean', '—')}±{r.get('DARevPct_std', 0)}" if r.get('DARevPct_mean') is not None else "—"
        iid   = f"{r.get('IDRevPct_mean', '—')}±{r.get('IDRevPct_std', 0)}" if r.get('IDRevPct_mean') is not None else "—"
        degp  = f"{r.get('DegCostPct_mean', '—')}±{r.get('DegCostPct_std', 0)}" if r.get('DegCostPct_mean') is not None else "—"
        dapu  = f"{r.get('DAPlanUtil_mean_pct', '—')}±{r.get('DAPlanUtil_std_pct', 0)}" if r.get('DAPlanUtil_mean_pct') is not None else "—"
        print(f"  {r['Strategy']:<14} {idle:<10} {da:<10} {iid:<10} {degp:<11} {dapu:<13}")


def plot_fig1_cumulative_profit(all_results: dict,
                                agent_names: list,
                                lp_objectives: dict | None,
                                target_lambda: float = 0.9) -> None:
    """
    Fig1_Cumulative_Profit.pdf
    Cumulative net profit over 219 days.
    - ±1 std shading across 5 seeds for each RL agent
    - LP Oracle ceiling (perfect-foresight upper bound)
    - P20/P80 and Idle baselines for context

    Sized to exactly one LaTeX column (A4, 1.75 cm L/R margins, default columnsep):
        columnwidth = (210 - 2×17.5 - 3.5) mm / 2 = 85.75 mm = 3.375 in
    Using \includegraphics[width=\columnwidth]{...} in LaTeX applies no scaling,
    so all font sizes here are the true rendered sizes.
    """
    # --- Column-exact dimensions ---
    COL_WIDTH_IN = 3.375   # 85.75 mm — matches LaTeX \columnwidth exactly
    COL_HEIGHT_IN = 3.0    # tall enough for all lines + legend without crowding

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_HEIGHT_IN))

    # Font sizes tuned for COL_WIDTH_IN rendering (no LaTeX rescaling)
    LABEL_FS  = 7.5
    TICK_FS   = 6.5
    LEGEND_FS = 6.0
    LINE_W    = 1.2   # slightly thinner than default to avoid crowding at small size

    # LP Oracle ceiling
    if lp_objectives and target_lambda in lp_objectives:
        lp = lp_objectives[target_lambda]
        if lp.get("lp_status") == "Optimal":
            lp_total = lp["objective_gbp"]
            lp_daily = lp_total / N_TEST_DAYS
            lp_cum = np.arange(1, N_TEST_DAYS + 1) * lp_daily
            ax.plot(range(N_TEST_DAYS), lp_cum, color="#000000",
                    linestyle=":", linewidth=LINE_W, label="LP Oracle", zorder=3)
            ax.fill_between(range(N_TEST_DAYS), lp_cum, alpha=0.06, color="#000000")

    # Baselines — use financial_profit_gbp (R_DA + R_ID - deg, no λ weighting)
    for baseline_name in ["idle", "p20p80"]:
        if baseline_name not in all_results:
            continue
        df = all_results[baseline_name]
        df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
        profit_col = "financial_profit_gbp" if "financial_profit_gbp" in df.columns else "actual_reward"
        daily = df.groupby("date")[profit_col].sum()
        cum = daily.cumsum().values
        n = len(cum)
        ax.plot(range(n), cum, color=COLORS.get(baseline_name, "#aaaaaa"),
                linewidth=LINE_W, linestyle="--", alpha=0.8,
                label=baseline_name.upper())

    # RL agents: mean ± 1σ across 5 seeds
    for agent_name in agent_names:
        cum_seeds = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_results:
                continue
            df_s = all_results[key]
            df_s["date"] = pd.to_datetime(df_s["delivery_ts"]).dt.date
            profit_col = "financial_profit_gbp" if "financial_profit_gbp" in df_s.columns else "actual_reward"
            daily = df_s.groupby("date")[profit_col].sum()
            cum_seeds.append(daily.cumsum().values)

        if not cum_seeds:
            continue

        arr = np.array(cum_seeds)       # (n_seeds, n_days)
        mean_cum = arr.mean(axis=0)
        std_cum  = arr.std(axis=0)
        n = len(mean_cum)
        color = COLORS.get(agent_name, "#333333")

        ax.plot(range(n), mean_cum, color=color, linewidth=LINE_W + 0.4,
                label=agent_name.upper(), zorder=5)
        ax.fill_between(range(n),
                        mean_cum - std_cum,
                        mean_cum + std_cum,
                        alpha=0.18, color=color, zorder=4)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="-")

    # Compact k-formatted y-axis tick labels (£0, £100k, £200k …)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"£{val/1000:.0f}k" if val != 0 else "£0")
    )

    ax.set_xlabel("Day (May 2025 – Jan 2026)", fontsize=LABEL_FS)
    ax.set_ylabel("Cumulative profit", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(loc="upper left", fontsize=LEGEND_FS, framealpha=0.9,
              ncol=2, handlelength=1.4, columnspacing=0.8, borderpad=0.4)
    ax.grid(alpha=0.2, linewidth=0.4)

    plt.tight_layout(pad=0.4)
    out = os.path.join(FIGURES_MAIN_DIR, "Fig1_Cumulative_Profit.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Fig1_Cumulative_Profit.pdf → {FIGURES_MAIN_DIR}/")


def plot_fig2_revenue_decomposition(all_results: dict,
                                    all_kpis: dict,
                                    lp_objectives: dict | None,
                                    agent_names: list,
                                    target_lambda: float = 0.9) -> None:
    """
    Fig2_Revenue_Decomposition.pdf
    Stacked bar chart: DA Rev, ID Rev, −Degradation Cost, −Carbon Cost.
    Each bar = mean across 5 seeds over the full 219-day test period.
    LP Oracle bar included. FIX: uses ax.bar() (not stackplot) so that
    degradation cost renders correctly as a downward band.
    """
    labels = []
    da_means, id_means, deg_means, carbon_means = [], [], [], []

    for agent_name in agent_names:
        da_s, id_s, deg_s, car_s = [], [], [], []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key not in all_results:
                continue
            df = all_results[key]
            da_s.append(df["Planned_Profit"].sum())
            id_s.append(df["Intraday_Profit"].sum())
            deg_s.append(df["degradation_cost_gbp"].sum())
            car_s.append(df["carbon_cashflow"].sum())
        if da_s:
            labels.append(agent_name.upper())
            da_means.append(np.mean(da_s))
            id_means.append(np.mean(id_s))
            deg_means.append(np.mean(deg_s))
            carbon_means.append(np.mean(car_s))

    for bl in ["p20p80", "idle", "random"]:
        if bl in all_results:
            df = all_results[bl]
            if "financial_profit_gbp" not in df.columns:
                df = add_derived_columns(df.copy())
            labels.append(bl.upper())
            da_means.append(df["Planned_Profit"].sum())
            id_means.append(df["Intraday_Profit"].sum())
            deg_means.append(df["degradation_cost_gbp"].sum())
            carbon_means.append(df["carbon_cashflow"].sum())

    if lp_objectives and target_lambda in lp_objectives:
        lp = lp_objectives[target_lambda]
        if lp.get("lp_status") == "Optimal":
            labels.append("LP Oracle")
            da_means.append(lp.get("revenue_gbp", 0))
            id_means.append(0.0)
            deg_means.append(lp.get("deg_cost_gbp", 0))
            carbon_means.append(lp.get("carbon_cost_gbp", 0))

    if not labels:
        print("Warning: No data for Fig2 — skipping.")
        return

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Positive bars: DA revenue (bottom) + ID revenue (stacked on top)
    da_arr  = np.array(da_means)
    id_arr  = np.array(id_means)
    deg_arr = np.array(deg_means)   # positive cost → plot downward
    car_arr = np.array(carbon_means)

    # Only stack positive ID values upward; negative ID values go downward
    id_pos = np.maximum(id_arr, 0)
    id_neg = np.minimum(id_arr, 0)

    ax.bar(x, da_arr, width, label="DA Revenue", color="#2ca02c", alpha=0.85)
    ax.bar(x, id_pos, width, bottom=da_arr,
           label="ID Revenue (+)", color="#1f77b4", alpha=0.85)
    ax.bar(x, id_neg, width, bottom=da_arr,
           label="ID Revenue (−)", color="#aec7e8", alpha=0.85)
    # Degradation cost: downward from zero baseline
    ax.bar(x, -deg_arr, width, label="Degradation Cost", color="#d62728", alpha=0.85)
    # Carbon cost: stacked downward below degradation
    ax.bar(x, -car_arr, width, bottom=-deg_arr,
           label="Carbon Cost", color="#7f7f7f", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Total revenue / cost over 219 days (£)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIGURES_MAIN_DIR, "Fig2_Revenue_Decomposition.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Fig2_Revenue_Decomposition.pdf → {FIGURES_MAIN_DIR}/")


def plot_fig3_dispatch_case_study(all_results: dict,
                                  target_lambda: float = 0.9,
                                  window_start_day: int = 185) -> None:
    """
    Fig3_Dispatch_Case_Study.pdf
    7-day physical dispatch comparison: SAC (continuous 3D) vs DDQN (discrete 5808).
    Layout: 2 rows × 2 cols
      (top-left)  Prices + carbon intensity for SAC window
      (top-right) Prices + carbon intensity for DDQN window
      (bot-left)  SoC + dispatched power for SAC
      (bot-right) SoC + dispatched power for DDQN
    window_start_day: index into the 219-day test period (default = ~day 185 ≈ November 2025)
    """
    window_steps = 7 * 48   # 336 half-hour slots

    agents_to_plot = []
    for a in ["sac", "d3qn"]:
        key = f"{a}_lambda{target_lambda:.1f}_seed0"
        if key in all_results:
            agents_to_plot.append((a, all_results[key]))

    if len(agents_to_plot) < 2:
        print("Warning: need both SAC and DDQN results for Fig3 — skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    fig.subplots_adjust(hspace=0.12, wspace=0.28)

    for col_idx, (agent_name, df) in enumerate(agents_to_plot):
        start = window_start_day * 48
        end   = start + window_steps
        if end > len(df):
            start = max(0, len(df) - window_steps)
            end = len(df)
        window = df.iloc[start:end].reset_index(drop=True)
        t = np.arange(len(window)) / 2.0   # hours

        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        top_label = "(a)" if col_idx == 0 else "(b)"
        bot_label = "(c)" if col_idx == 0 else "(d)"
        ax_top.set_title(top_label, loc="left", fontsize=10, fontweight="bold")
        ax_bot.set_title(bot_label, loc="left", fontsize=10, fontweight="bold")

        # Top: prices + carbon intensity
        ax_top.plot(t, window["da_price_now"], color="#2ca02c", linewidth=1.0,
                    label="DA price")
        ax_top.plot(t, window["id_price_now"], color="#1f77b4", linewidth=1.0,
                    linestyle="--", label="ID price")
        ax_top2 = ax_top.twinx()
        ax_top2.fill_between(t, window["ci_now"], alpha=0.12, color="#7f7f7f")
        ax_top2.plot(t, window["ci_now"], color="#7f7f7f", linewidth=0.8,
                     alpha=0.7, label="CI (gCO₂/kWh)")
        ax_top2.set_ylabel("CI (gCO₂/kWh)", fontsize=8, color="#555555")
        ax_top2.tick_params(axis="y", labelcolor="#555555", labelsize=7)
        ax_top.set_ylabel("Price (£/MWh)", fontsize=8)
        ax_top.tick_params(labelsize=7)
        ax_top.grid(alpha=0.2)
        title_str = agent_name.upper()
        if agent_name == "sac":
            title_str += " (Continuous 3D)"
        else:
            title_str += " (Discrete 5,808)"
        ax_top.set_title(title_str, fontsize=9, fontweight="bold")
        if col_idx == 0:
            h1, l1 = ax_top.get_legend_handles_labels()
            h2, l2 = ax_top2.get_legend_handles_labels()
            ax_top.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, framealpha=0.85)

        # Bottom: SoC + dispatched power
        ax_bot.fill_between(t, window["soc"], alpha=0.35, color="#ff7f0e",
                            label="SoC")
        ax_bot.set_ylabel("SoC", fontsize=8)
        ax_bot.set_ylim(0, 1)
        ax_bot2 = ax_bot.twinx()
        ax_bot2.bar(t, window["P_applied_MW"],
                    width=0.4,
                    color=["#d62728" if p > 0 else "#1f77b4" for p in window["P_applied_MW"]],
                    alpha=0.6, label="Dispatch (MW)")
        ax_bot2.axhline(0, color="black", linewidth=0.5)
        ax_bot2.set_ylabel("Dispatch (MW)", fontsize=8)
        ax_bot2.tick_params(labelsize=7)
        ax_bot.tick_params(labelsize=7)
        ax_bot.set_xlabel("Hours into 7-day window", fontsize=8)
        ax_bot.grid(alpha=0.2)
        if col_idx == 0:
            ax_bot.legend(loc="upper left", fontsize=7, framealpha=0.85)

    out = os.path.join(FIGURES_MAIN_DIR, "Fig3_Dispatch_Case_Study.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Fig3_Dispatch_Case_Study.pdf → {FIGURES_MAIN_DIR}/")


def plot_fig4_reward_ablation(all_kpis: dict,
                              agent_names: list,
                              lambda_values: list) -> None:
    """
    Fig4_Reward_Ablation.pdf
    Pareto frontier: Financial Profit vs Net Carbon across lambda_CI values.
    Each agent = one coloured line; each point = mean across 5 seeds at one lambda.
    Error bars show ±1σ. Proves the profit-carbon trade-off is controllable
    and compares both formulations across the λ ablation.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Pareto frontier is SAC-only: the continuous 3D formulation is the paper's
    # contribution; discrete agents are evaluated at primary lambda only.
    sac_agents = [n for n in agent_names if n.lower() == "sac"]
    if not sac_agents:
        print("  [Fig4] No SAC agent in agent_names — skipping Pareto plot.")
        return

    for agent_name in sac_agents:
        profits, carbons = [], []
        prof_stds, carb_stds = [], []
        lams_plotted = []

        for lam in sorted(lambda_values):
            p_seeds = [all_kpis[f"{agent_name}_lambda{lam:.1f}_seed{s}"]["Financial Profit (£)"]
                       for s in range(5)
                       if f"{agent_name}_lambda{lam:.1f}_seed{s}" in all_kpis]
            c_seeds = [all_kpis[f"{agent_name}_lambda{lam:.1f}_seed{s}"]["Net Carbon (tCO2)"]
                       for s in range(5)
                       if f"{agent_name}_lambda{lam:.1f}_seed{s}" in all_kpis]
            if p_seeds:
                profits.append(np.mean(p_seeds))
                carbons.append(np.mean(c_seeds))
                prof_stds.append(np.std(p_seeds))
                carb_stds.append(np.std(c_seeds))
                lams_plotted.append(lam)

        if not profits:
            continue

        color = COLORS.get(agent_name, "#333333")
        label = "SAC (3D continuous)"

        ax.errorbar(carbons, profits,
                    xerr=carb_stds, yerr=prof_stds,
                    label=label, marker="o", linewidth=1.8, markersize=6,
                    capsize=4, color=color)

        # Annotate every λ point
        for lam, cx, py in zip(lams_plotted, carbons, profits):
            ax.annotate(f"λ={lam:.1f}", (cx, py),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6.5, color="#555555")

    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Net carbon displacement (tCO₂, 219 days)", fontsize=10)
    ax.set_ylabel("Financial profit (£, 219 days)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIGURES_MAIN_DIR, "Fig4_Reward_Ablation.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Fig4_Reward_Ablation.pdf → {FIGURES_MAIN_DIR}/")


# ============================================================
# APPENDIX FIGURES (results/figures_appendix/)
# ============================================================

def plot_appx_soc_duration_curve(all_results: dict,
                                  agent_names: list,
                                  target_lambda: float = 0.9) -> None:
    """Appx_SoC_Duration_Curve.pdf — sorted SoC time-series (duration curve)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for agent_name in agent_names:
        soc_all = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                soc_all.extend(all_results[key]["soc"].tolist())
        if not soc_all:
            continue
        sorted_soc = np.sort(soc_all)[::-1]
        pct = np.linspace(0, 100, len(sorted_soc))
        ax.plot(pct, sorted_soc, color=COLORS.get(agent_name, "#333333"),
                linewidth=1.5, label=agent_name.upper())

    ax.set_xlabel("% of time above SoC level", fontsize=10)
    ax.set_ylabel("State of Charge (SoC)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIGURES_APPENDIX_DIR, "Appx_SoC_Duration_Curve.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Appx_SoC_Duration_Curve.pdf → {FIGURES_APPENDIX_DIR}/")


def plot_appx_da_fidelity_hexbin(all_results: dict,
                                  agent_names: list,
                                  target_lambda: float = 0.9) -> None:
    """Appx_DA_Fidelity_Hexbin.pdf — planned vs actual power (hexbin density).

    Layout: 2×2 grid — top row: SAC, D3QN; bottom row: DDQN, DQN.
    """
    layout = [["sac", "d3qn"], ["ddqn", "dqn"]]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)

    for row, row_agents in enumerate(layout):
        for col, agent_name in enumerate(row_agents):
            ax = axes[row, col]

            label_letter = chr(ord('a')+ row * 2 + col)
            ax.set_title(f"({label_letter})", loc="left", fontsize=10, fontweight="bold")
        
            if agent_name not in agent_names:
                ax.set_visible(False)
                continue
            planned_all, actual_all = [], []
            for seed in range(5):
                key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
                if key in all_results:
                    df = all_results[key]
                    planned_all.extend(df["P_planned_MW"].tolist())
                    actual_all.extend(df["P_applied_MW"].tolist())
            if not planned_all:
                ax.set_visible(False)
                continue
            hb = ax.hexbin(planned_all, actual_all, gridsize=40, cmap="Blues",
                           mincnt=1, linewidths=0.2)
            ax.plot([-60, 60], [-60, 60], "r--", linewidth=1.0, alpha=0.7, label="Perfect fidelity")
            ax.set_xlabel("Planned power (MW)", fontsize=9)
            ax.set_ylabel("Actual power (MW)", fontsize=9)
            ax.set_title(agent_name.upper(), fontsize=9, fontweight="bold")
            ax.legend(fontsize=7)
            fig.colorbar(hb, ax=ax, label="count")

    plt.tight_layout()
    out = os.path.join(FIGURES_APPENDIX_DIR, "Appx_DA_Fidelity_Hexbin.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Appx_DA_Fidelity_Hexbin.pdf → {FIGURES_APPENDIX_DIR}/")


def plot_appx_dispatch_price_heatmap(all_results: dict,
                                      agent_names: list,
                                      target_lambda: float = 0.9) -> None:
    """Appx_Dispatch_Price_Heatmap.pdf — mean dispatch by settlement period and day-of-week.

    Layout: 2×2 grid — top row: SAC, D3QN; bottom row: DDQN, DQN.
    """
    layout = [["sac", "d3qn"], ["ddqn", "dqn"]]
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)

    for row, row_agents in enumerate(layout):
        for col, agent_name in enumerate(row_agents):
            ax = axes[row, col]
            
            label_letter = chr(ord('a') + row * 2 + col)
            ax.set_title(f"({label_letter})", loc="left", fontsize=10, fontweight="bold")

            if agent_name not in agent_names:
                ax.set_visible(False)
                continue
            dfs = []
            for seed in range(5):
                key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
                if key in all_results:
                    dfs.append(all_results[key])
            if not dfs:
                ax.set_visible(False)
                continue
            df = pd.concat(dfs, ignore_index=True)
            df["day_of_week"] = pd.to_datetime(df["delivery_ts"]).dt.day_name()
            pivot = df.pivot_table(values="P_applied_MW", index="tau",
                                   columns="day_of_week", aggfunc="mean")
            for d in day_order:
                if d not in pivot.columns:
                    pivot[d] = np.nan
            pivot = pivot[[d for d in day_order if d in pivot.columns]]
            sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax,
                        cbar_kws={"label": "Mean dispatch (MW)", "shrink": 0.7},
                        linewidths=0)
            ax.set_xlabel("Day of week", fontsize=9)
            ax.set_ylabel("Settlement period (1–48)", fontsize=9)
            ax.set_title(agent_name.upper(), fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(FIGURES_APPENDIX_DIR, "Appx_Dispatch_Price_Heatmap.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Appx_Dispatch_Price_Heatmap.pdf → {FIGURES_APPENDIX_DIR}/")


def plot_appx_action_distribution(all_results: dict,
                                   agent_names: list,
                                   target_lambda: float = 0.9) -> None:
    """Appx_Action_Distribution.pdf — Charge / Idle / Discharge fractions per agent."""
    labels, charge_pct, idle_pct, discharge_pct = [], [], [], []

    for agent_name in agent_names:
        all_dfs = []
        for seed in range(5):
            key = f"{agent_name}_lambda{target_lambda:.1f}_seed{seed}"
            if key in all_results:
                all_dfs.append(all_results[key])
        if not all_dfs:
            continue
        df = pd.concat(all_dfs, ignore_index=True)
        n = len(df)
        labels.append(agent_name.upper())
        charge_pct.append((df["P_applied_MW"] < -1e-4).sum() / n * 100)
        idle_pct.append((df["P_applied_MW"].abs() < 1e-4).sum() / n * 100)
        discharge_pct.append((df["P_applied_MW"] > 1e-4).sum() / n * 100)

    if not labels:
        print("Warning: No data for Appx_Action_Distribution — skipping.")
        return

    x = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(x, charge_pct, width, label="Charge", color="#1f77b4", alpha=0.85)
    ax.bar(x, idle_pct, width, bottom=charge_pct, label="Idle", color="#9467bd", alpha=0.85)
    bottom2 = [c + i for c, i in zip(charge_pct, idle_pct)]
    ax.bar(x, discharge_pct, width, bottom=bottom2, label="Discharge",
           color="#d62728", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fraction of steps (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIGURES_APPENDIX_DIR, "Appx_Action_Distribution.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✓ Appx_Action_Distribution.pdf → {FIGURES_APPENDIX_DIR}/")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BESS RL Evaluation — IEEE Publication Grade")
    parser.add_argument("--agents", nargs="+", type=str,
                       default=[k for k in AGENTS.keys() if k != "sac"],
                       choices=list(AGENTS.keys()),
                       help="Agent(s) to evaluate (use --agents sac for SAC)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                       help="Seeds to evaluate per agent")
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.6, 1.0],
                       help="Lambda_ci values for Pareto ablation")
    parser.add_argument("--no-baselines", action="store_true", default=False,
                       help="Skip Idle, Random, and P20/P80 baselines")
    parser.add_argument("--run-lp", action="store_true", default=False,
                       help="Solve the perfect-foresight LP benchmark for each lambda_ci "
                            "and add LP Efficiency Ratio (%) to the comparison table")
    parser.add_argument("--generate-figures", action="store_true", default=False,
                       help="Generate publication-grade visualizations")
    parser.add_argument("--primary-lambda", type=float, default=0.1,
                       help="Primary lambda_ci for main figures (should match training lambda). "
                            "Default: 0.1 (matches training configuration). "
                            "Must be present in --lambda-values if --generate-figures is used.")
    args = parser.parse_args()

    # Ensure primary_lambda is included in lambda_values if generating figures
    if args.generate_figures and args.primary_lambda not in args.lambda_values:
        args.lambda_values = sorted(set(args.lambda_values) | {args.primary_lambda})

    all_results = {}
    all_kpis    = {}
    lp_objectives: dict | None = None

    if args.run_lp:
        print("\n" + "="*50)
        print(" RUNNING LP BENCHMARK (perfect foresight)")
        print("="*50)
        lp_objectives = run_lp_benchmarks(
            lambda_values=args.lambda_values,
        )
        for lam, res in lp_objectives.items():
            if res["lp_status"] == "Optimal":
                print(f"  λ={lam:.1f}  LP Objective: £{res['objective_gbp']:,.0f}  "
                      f"Revenue: £{res['revenue_gbp']:,.0f}  "
                      f"DegCost: £{res['deg_cost_gbp']:,.0f}  "
                      f"CarbonCost: £{res['carbon_cost_gbp']:,.0f}")

    if not args.no_baselines:
        print("\n" + "="*50)
        print(" RUNNING BASELINES")
        print("="*50)

        # Use primary_lambda for baselines so they are comparable to the main RL results
        baseline_lambda = args.primary_lambda
        env = build_test_env(lambda_ci=baseline_lambda)

        print(" -> Running Idle Baseline...")
        idle_df = run_idle_baseline(env)
        idle_df = add_derived_columns(idle_df)
        save_step_csv(idle_df, agent_name="idle", seed=0, lambda_ci=baseline_lambda)
        all_results["idle"] = idle_df

        print(" -> Running Random Baseline...")
        random_df = run_random_baseline(env, rng_seed=42)
        random_df = add_derived_columns(random_df)
        save_step_csv(random_df, agent_name="random", seed=0, lambda_ci=baseline_lambda)
        all_results["random"] = random_df

        print(" -> Running P20/P80 Heuristic...")
        p20p80_df = run_p20p80_heuristic(env)
        p20p80_df = add_derived_columns(p20p80_df)
        save_step_csv(p20p80_df, agent_name="p20p80", seed=0, lambda_ci=baseline_lambda)
        all_results["p20p80"] = p20p80_df

    # Loop through agents and lambda_ci values
    for agent_name in args.agents:
        for lambda_ci in args.lambda_values:
            for seed in args.seeds:
                label = f"{agent_name}_lambda{lambda_ci:.1f}_seed{seed}"
                print(f"\n -> Evaluating {label}...")
                try:
                    if agent_name == "sac":
                        agent = load_frozen_sac_agent(seed, lambda_ci=lambda_ci)
                        env = build_test_env(lambda_ci=lambda_ci, continuous_action=True)
                        df = run_continuous_rollout_sac(agent, env)
                    else:
                        agent = load_frozen_agent(agent_name, seed)
                        env = build_test_env(lambda_ci=lambda_ci)
                        df = run_continuous_rollout(agent, env)
                    df = add_derived_columns(df)
                    save_step_csv(df, agent_name, seed, lambda_ci=lambda_ci)

                    all_results[label] = df
                    all_kpis[label] = compute_kpis(df)

                    net_profit = df["rev_net_gbp"].sum()
                    print(f"    [SUCCESS] Net Profit: £{net_profit:,.2f}")

                except Exception as e:
                    print(f"    [FAILED] {label}. Error: {e}")

    # Build comparison table
    print("\n" + "="*50)
    print(" EVALUATION COMPLETE")
    print("="*50)

    baseline_lambda = args.lambda_values[0] if args.lambda_values else 0.1
    summary_table = build_comparison_table(
        all_results,
        lp_objectives=lp_objectives,
        baseline_lambda=baseline_lambda,
    )
    print(summary_table.to_string())

    # Generate publication figures
    if args.generate_figures:
        target_lam = args.primary_lambda

        print("\n" + "="*80)
        print(" GENERATING IEEE PUBLICATION FIGURES")
        print(f" Target λ = {target_lam}  |  Main → {FIGURES_MAIN_DIR}  |  Appendix → {FIGURES_APPENDIX_DIR}")
        print("="*80)

        # ── MAIN FIGURES ──────────────────────────────────────────────────────
        print("\n[MAIN] Table 1: KPI Summary CSV ...")
        save_kpi_table(all_results, all_kpis, lp_objectives, args.agents, target_lam)

        print("\n[MAIN] Fig 1: Cumulative Profit (LP Oracle + ±1σ bands)...")
        plot_fig1_cumulative_profit(all_results, args.agents, lp_objectives, target_lam)

        print("\n[MAIN] Fig 2: Revenue Decomposition (DA/ID/Deg/Carbon + LP) ...")
        plot_fig2_revenue_decomposition(all_results, all_kpis, lp_objectives, args.agents, target_lam)

        print("\n[MAIN] Fig 3: Dispatch Case Study (7-day SAC vs DDQN) ...")
        plot_fig3_dispatch_case_study(all_results, target_lambda=target_lam)

        print("\n[MAIN] Fig 4: Reward Ablation (Pareto frontier) ...")
        plot_fig4_reward_ablation(all_kpis, args.agents, args.lambda_values)

        # ── APPENDIX FIGURES ──────────────────────────────────────────────────
        print("\n[APPENDIX] SoC Duration Curve ...")
        plot_appx_soc_duration_curve(all_results, args.agents, target_lam)

        print("\n[APPENDIX] DA Fidelity Hexbin ...")
        plot_appx_da_fidelity_hexbin(all_results, args.agents, target_lam)

        print("\n[APPENDIX] Dispatch Price Heatmap ...")
        plot_appx_dispatch_price_heatmap(all_results, args.agents, target_lam)

        print("\n[APPENDIX] Action Distribution ...")
        plot_appx_action_distribution(all_results, args.agents, target_lam)

        print(f"\n{'='*80}")
        print("[SUCCESS] Publication figures generated:")
        print(f"  MAIN  ({FIGURES_MAIN_DIR}/):")
        print("    Table_1_KPI_Summary.csv")
        print("    Fig1_Cumulative_Profit.pdf  (LP Oracle ceiling + ±1σ)")
        print("    Fig2_Revenue_Decomposition.pdf  (DA / ID / Deg / Carbon + LP)")
        print("    Fig3_Dispatch_Case_Study.pdf  (7-day SAC vs DDQN)")
        print("    Fig4_Reward_Ablation.pdf  (Pareto frontier)")
        print(f"  APPENDIX  ({FIGURES_APPENDIX_DIR}/):")
        print("    Appx_SoC_Duration_Curve.pdf")
        print("    Appx_DA_Fidelity_Hexbin.pdf")
        print("    Appx_Dispatch_Price_Heatmap.pdf")
        print("    Appx_Action_Distribution.pdf")
        print(f"{'='*80}")
