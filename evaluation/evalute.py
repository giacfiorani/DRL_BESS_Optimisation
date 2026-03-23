"""
evaluate.py — BESS RL Agent Evaluation Pipeline
================================================
Runs frozen RL checkpoints through the 219-day held-out test split
(2025-05-26 → 2026-01-01) as a single continuous rollout with no
episodic SoC resets. Logs every half-hour step to a CSV for post-hoc
KPI computation and figure generation.

Protocol:  See markdowns/evaluation_strategy.md (Sections I–VII)
"""

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch as T

# --- Project imports ---
from envs.battery_env import BatteryEnv
import envs.env_config as env_config
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.d3qn import D3QNAgent
from agents.d3qn_per_agent import D3QNPERAgent
from agents.hyperparams import (
    DQN_HYPERPARAMS,
    DDQN_HYPERPARAMS,
    D3QN_HYPERPARAMS,
    D3QN_PER_HYPERPARAMS,
)
from utils.action_encoding import decode, N_ACTIONS

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

N_TEST_DAYS: int = 219                   # 2025-05-26 → 2026-01-01
TEST_START_DATE: str = "2025-05-26"      # first delivery day of test split
RESULTS_DIR: str = "results/test_rollouts"
CHECKPOINT_DIR: str = "models"
LAMBDA_CI: float = 0.9
INPUT_DIMS: int = 103                    # observation vector length
SEED = 42

AGENTS: dict[str, type] = {
    "dqn":      DQNAgent,
    "ddqn":     DDQNAgent,
    "d3qn":     D3QNAgent,
    "d3qn_per": D3QNPERAgent,
}

HYPERPARAMS_MAP: dict[str, dict] = {
    "dqn":      DQN_HYPERPARAMS,
    "ddqn":     DDQN_HYPERPARAMS,
    "d3qn":     D3QN_HYPERPARAMS,
    "d3qn_per": D3QN_PER_HYPERPARAMS,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FUNCTION DEFINITIONS
# ============================================================

def load_frozen_agent(agent_name: str, seed: int,
                      checkpoint_dir: str = CHECKPOINT_DIR) -> object:
    """
    Load a trained checkpoint and return a fully greedy agent (epsilon=0).

    Globs the models/ directory for the matching run folder, constructs the
    agent with frozen exploration, loads Q_eval + Q_target weights, and sets
    both networks to eval() mode.

    Raises FileNotFoundError if no checkpoint matches the agent/seed pattern.
    """
    # Lookup of agent class from the two dictionaries
    agent_cls = AGENTS[agent_name]
    hp = HYPERPARAMS_MAP[agent_name]

    search_pattern = f"{checkpoint_dir}/*_{agent_name.upper()}_seed{seed}_*/final_model.pth"
    matching_file = glob(search_pattern)

    # Verify Glob found the file
    if len(matching_file) == 0:
        raise FileNotFoundError(f"Could not find a trained model for {agent_name} (Seed {seed}). Looked for pattern: {search_pattern}")
    
    #grab the first match of the file
    ckpt_path = matching_file[0]
    print(f"Found model: {ckpt_path} ")

    # Constructing the agent instance
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

    # Load weights and lock to inference mode
    ckpt = T.load(ckpt_path, map_location="cpu")
    agent.Q_eval.load_state_dict(ckpt["model_state_dict"])
    agent.Q_eval.eval()

    agent.Q_target.load_state_dict(ckpt["model_state_dict"])
    agent.Q_target.eval()

    #Return the frozen agent
    return agent

def build_test_env() -> BatteryEnv:
    """
    Create a deterministic BatteryEnv for the 219-day test split.

    Fixed SoC_0 = 0.5, no random start, no random initial SoC.
    """
    
    env = BatteryEnv(
        config  =   env_config,
        lambda_ci   =   LAMBDA_CI,
        split   =   "test",
        episode_days=N_TEST_DAYS,
        randomize_init_soc  =   False,
        randomize_start =   False,
        seed = SEED,
    )
    
    return env


def run_continuous_rollout(agent: object, env: BatteryEnv) -> pd.DataFrame:
    """
    Run a greedy 219-day rollout and return a step-level DataFrame (~10,512 rows).

    Each step: agent selects a flat action, it is decoded to (dispatch, plan, slot),
    the env is stepped, and the full info dict + reward are recorded.
    """
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})
    
    # 2. Initialise empty list and step counter (OUTSIDE the loop)
    rows = []
    step_idx = 0
    done = False
    
    # 3. Enter the while loop for the single continuous rollout
    while not done:
        # a. Get action
        action = agent.choose_action(obs)
        
        # b. Decode
        dispatch_idx, plan_idx, plan_slot = decode(action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
        
        # c. Step env
        obs, reward, terminated, truncated, info = env.step(env_action)
        
        # d. Append record
        rows.append({**info, "step": step_idx, "reward": reward})
        
        # e. Increment and check termination
        step_idx += 1
        done = terminated or truncated
    
    # 4. Convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def run_idle_baseline(env: BatteryEnv) -> pd.DataFrame:
    """
    Baseline 0 — Idle agent: dispatches P = 0 MW every step (dispatch_idx=5).

    Lower bound on performance; any learning agent should beat this.
    """
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})
    
    # 2. Initialise empty list and step counter (OUTSIDE the loop)
    rows = []
    step_idx = 0
    done = False
    
    # 3. Enter the while loop for the single continuous rollout
    while not done:
        # Setting idle action (idx 5 = IDLE = 0 Power)
        env_action = np.array([5, 5, 0], dtype=np.int64)
        
        # c. Step env
        obs, reward, terminated, truncated, info = env.step(env_action)
        
        # d. Append record
        rows.append({**info, "step": step_idx, "reward": reward})
        
        # e. Increment and check termination
        step_idx += 1
        done = terminated or truncated
    
    # 4. Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def run_random_baseline(env: BatteryEnv, rng_seed: int = 0) -> pd.DataFrame:
    """
    Baseline 1 — Uniform random policy: samples from all 5,808 actions each step.

    Deterministic via rng_seed for reproducibility.
    """
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})
    
    # 2. Initialise empty list and step counter (OUTSIDE the loop)
    rows = []
    step_idx = 0
    done = False

    rng = np.random.default_rng(rng_seed)
    
    # 3. Enter the while loop for the single continuous rollout
    while not done:
        # create random number for random action
        flat_action = rng.integers(0, N_ACTIONS)
        dispatch_idx, plan_idx, plan_slot = decode(flat_action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
        
        # c. Step env
        obs, reward, terminated, truncated, info = env.step(env_action)
        
        # d. Append record
        rows.append({**info, "step": step_idx, "reward": reward})
        
        # e. Increment and check termination
        step_idx += 1
        done = terminated or truncated
    
    # 4. Convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def run_p20p80_heuristic(env: BatteryEnv) -> pd.DataFrame:
    """
    Baseline 2 — P20/P80 daily heuristic (industry rule-based dispatch).

    Charges at full power when ID price < P20 of today's prices so far,
    discharges at full power when > P80, idles otherwise. Price list resets
    each delivery day. First step defaults to idle (no price history yet).
    """
    obs, info = env.reset(options={"delivery_day": TEST_START_DATE})

    # 2. Initialise empty list and step counter (OUTSIDE the loop)
    rows = []
    step_idx = 0
    done = False
    id_prices_list = []

    # First step: reset() info has no market keys — default to idle
    # so we take one env step to get the first real info dict
    env_action = np.array([5, 5, 0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(env_action)
    rows.append({**info, "step": step_idx, "reward": reward})
    step_idx += 1
    done = terminated or truncated
    id_prices_list.append(info["id_price_now"])

    # 3. Enter the while loop for the remaining steps
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
            
        # c. Step env
        obs, reward, terminated, truncated, info = env.step(env_action)
        
        # d. Append record
        rows.append({**info, "step": step_idx, "reward": reward})
        
        # e. Increment and check termination
        step_idx += 1
        done = terminated or truncated
    
    # 4. Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

    
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 5 derived columns: date, month, day_of_week, rev_net_gbp, E_throughput_MWh.

    rev_net_gbp = DA + ID revenue - degradation - carbon costs.
    E_throughput_MWh = |P_applied| * 0.5h (energy per settlement period).
    """
    df["date"] = pd.to_datetime(df["delivery_ts"]).dt.date
    df["month"] = pd.to_datetime(df["delivery_ts"]).dt.month_name()
    df["day_of_week"] = pd.to_datetime(df["delivery_ts"]).dt.day_name()
    df["rev_net_gbp"] = (df["Planned_Profit"] + df["Intraday_Profit"] - df["degradation_cost_gbp"] - df["carbon_cashflow"])
    df["E_throughput_MWh"] = df["P_applied_MW"].abs() * 0.5

    return df

def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute all KPIs from a step-level DataFrame. Returns a flat dict.

    Group A — Financial: profit, Sharpe, max drawdown, DA/ID revenue split.
    Group B — Battery ops: EFC, SoC stats, idle fraction, clipping rate.
    Group C — Environmental: net carbon, carbon intensity, carbon penalty.
    Group D — Policy intelligence: market timing score, DA plan utilisation.
    """
    
    # --- HELPER: Daily Aggregation ---
    # First, sum up the net profit for every half-hour to get 219 Daily Totals
    daily_profit = df.groupby('date')['rev_net_gbp'].sum()

    # --- Group A: FINANCIAL PERFORMANCE ---
    total_net_profit = df["rev_net_gbp"].sum()
    mean_daily_profit = daily_profit.mean()
    profit_volatility = daily_profit.std()
    
    # Safe division for Sharpe
    sharpe_ratio = mean_daily_profit / profit_volatility if profit_volatility != 0 else 0
    
    # Max Drawdown (Peak to Trough in £)
    cum_profit = daily_profit.cumsum()
    drawdown = cum_profit.cummax() - cum_profit
    max_drawdown = drawdown.max()

    # Revenue Breakdown (Use gross revenue as the denominator)
    gross_revenue = abs(df["Planned_Profit"].sum() + df["Intraday_Profit"].sum())
    if gross_revenue == 0: gross_revenue = 1e-9 # Prevent division by zero for Idle Agent
    
    da_revenue_pct = (df["Planned_Profit"].sum() / gross_revenue) * 100
    id_revenue_pct = (df["Intraday_Profit"].sum() / gross_revenue) * 100
    deg_cost_pct = (df["degradation_cost_gbp"].sum() / gross_revenue) * 100
    
    # Safely get carbon if it exists
    carbon_cost_pct = (df["carbon_cashflow"].sum() / gross_revenue) * 100 if "carbon_cashflow" in df.columns else 0

    # --- Group B: BATTERY OPERATIONS ---
    efc = df["E_throughput_MWh"].sum() / (2 * 0.3727)
    revenue_per_efc = total_net_profit / efc if efc != 0 else 0
    soc_mean, soc_std = df["soc"].mean(), df["soc"].std()
    
    idle_fraction = (df["P_applied_MW"].abs() < 1e-4).mean() * 100
    n_charge = (df["P_applied_MW"] < -1e-4).sum()
    n_discharge = (df["P_applied_MW"] > 1e-4).sum()
    charge_discharge_ratio = n_charge / n_discharge if n_discharge != 0 else 0
    clipping_rate = ((df["P_req_MW"] - df["P_applied_MW"]).abs() > 1e-4).mean() * 100

    # --- Group C & D: Policy Intelligence ---
    # Top 25% of prices (75th percentile)
    p75_id_price = np.percentile(df["id_price_now"], 75)
    smart_discharges = ((df["P_applied_MW"] > 1e-4) & (df["id_price_now"] >= p75_id_price)).sum()
    market_timing_score = (smart_discharges / n_discharge * 100) if n_discharge > 0 else 0
    
    da_plan_utilisation = ((df["P_applied_MW"] - df["P_planned_MW"]).abs() < 1e-4).mean() * 100

    # --- Group C: ENVIRONMENTAL IMPACT ---
    total_throughput_MWh = df["E_throughput_MWh"].sum()
    net_carbon_tco2 = df["net_carbon_tCO2"].sum()
    carbon_intensity = net_carbon_tco2 / total_throughput_MWh if total_throughput_MWh > 0 else 0
    carbon_penalty_gbp = df["carbon_cashflow"].sum()

    # Returning all KPIs in a properly populated flat dict
    return {
        "Total Profit (£)": total_net_profit,
        "Mean Daily Profit": mean_daily_profit,
        "Profit Volatility": profit_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (£)": max_drawdown,
        "DA Rev %": da_revenue_pct,
        "ID Rev %": id_revenue_pct,
        "Deg Cost %": deg_cost_pct,
        "Carbon Cost %": carbon_cost_pct,
        "EFC": efc,
        "Rev/EFC": revenue_per_efc,
        "SoC Mean": soc_mean,
        "SoC Std": soc_std,
        "Idle Fraction %": idle_fraction,
        "Charge/Discharge Ratio": charge_discharge_ratio,
        "Clipping Rate %": clipping_rate,
        "Net Carbon (tCO2)": net_carbon_tco2,
        "Carbon Intensity (tCO2/MWh)": carbon_intensity,
        "Carbon Penalty (£)": carbon_penalty_gbp,
        "Market Timing Score %": market_timing_score,
        "DA Plan Utilisation %": da_plan_utilisation,
    }

    

def build_comparison_table(
    all_results: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build the master comparison table (one row per strategy/seed).

    Iterates over all_results, ensures derived columns exist, computes KPIs,
    and saves the summary to evaluation_summary.csv.
    """
    kpi_rows = []

    for label, df  in all_results.items():
        
        if "rev_net_gbp" not in df.columns:
            df = add_derived_columns(df)
        
        kpi_dict = compute_kpis(df)

        final_row = {"strategy":label, **kpi_dict}

        kpi_rows.append(final_row)
    
    summary_df = pd.DataFrame(kpi_rows)

    save_path = f"{RESULTS_DIR}/evaluation_summary.csv"
    summary_df.to_csv(save_path, index=False)
    print(f"Saved master evaluation table to {save_path}")
    
    return summary_df

    


def save_step_csv(df: pd.DataFrame, agent_name: str, seed: int) -> str:
    """Save the step-level DataFrame to CSV and return the file path."""
    
    save_path = f"{RESULTS_DIR}/{agent_name}_seed{seed}_test_steps.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved CSV Step to {save_path}")
    return save_path

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BESS RL Evaluation — 219-day test rollout"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        default=list(AGENTS.keys()),
        choices=list(AGENTS.keys()),
        help="Agent(s) to evaluate (e.g. --agents dqn ddqn d3qn d3qn_per)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to evaluate per agent (e.g. --seeds 0 1 2 3 4)",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        default=False,
        help="Also run Idle, Random, and P20/P80 baselines",
    )
    args = parser.parse_args()

    all_results = {}

    if args.baselines:
        print("\n" + "="*50)
        print(" RUNNING BASELINES")
        print("="*50)
        
        # Build a fresh test env
        env = build_test_env()
        
        # --- IDLE BASELINE ---
        print(" -> Running Idle Baseline...")
        idle_df = run_idle_baseline(env)
        save_step_csv(idle_df, agent_name="idle", seed=0)
        all_results["idle"] = idle_df

        # --- RANDOM BASELINE ---
        print(" -> Running Random Baseline...")
        random_df = run_random_baseline(env, rng_seed=42)
        save_step_csv(random_df, agent_name="random", seed=0)
        all_results["random"] = random_df
        
        # --- P20/P80 HEURISTIC ---
        print(" -> Running P20/P80 Heuristic...")
        p20p80_df = run_p20p80_heuristic(env)
        save_step_csv(p20p80_df, agent_name="p20p80", seed=0)
        all_results["p20p80"] = p20p80_df
    
    # 3. Loop through RL Agents
    for agent_name in args.agents:
        for seed in args.seeds:
            print(f"\n -> Evaluating {agent_name.upper()} (Seed {seed})...")
            try:
                # a. Load the frozen agent
                agent = load_frozen_agent(agent_name, seed)
                
                # b. Build a fresh test env
                env = build_test_env()
                
                # c. Run the rollout using the agent object
                df = run_continuous_rollout(agent, env)
                
                # d. Add derived columns
                df = add_derived_columns(df)
                
                # e. Save step CSV
                save_step_csv(df, agent_name, seed)
                
                # f. Store it IN the dictionary
                all_results[f"{agent_name}_seed{seed}"] = df
                
                # g. Print progress
                net_profit = df["rev_net_gbp"].sum()
                print(f"    [SUCCESS] Net Profit: £{net_profit:,.2f}")
                
            except Exception as e:
                # If one seed fails (e.g., file not found), don't crash the whole script!
                print(f"    [FAILED] Could not evaluate {agent_name} seed {seed}. Error: {e}")

    # 4 & 5. Build and print the comparison table
    print("\n" + "="*50)
    print(" EVALUATION COMPLETE. GENERATING MASTER TABLE...")
    print("="*50)
    
    summary_table = build_comparison_table(all_results)
    
    # Print the table neatly to the terminal
    print(summary_table.to_string())






