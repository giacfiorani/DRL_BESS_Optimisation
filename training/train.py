import sys
import os
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
import envs.env_config

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.d3qn import D3QNAgent
from agents.d3qn_per_agent import D3QNPERAgent
from envs.battery_env import BatteryEnv
from utils.action_encoding import decode, N_ACTIONS
from agents.hyperparams import DQN_HYPERPARAMS, DDQN_HYPERPARAMS, D3QN_HYPERPARAMS, D3QN_PER_HYPERPARAMS

# ============================================================
# REPRODUCIBILITY SEEDS — 5-seed averaging for academic robustness
# ============================================================
SEEDS = [0, 1, 2, 3, 4, 42]

# ============================================================
# STEP-LEVEL MONITORING
# Every STEP_LOG_INTERVAL episodes, every step of that episode
# is logged under Step/ tags in TensorBoard.
# ============================================================
STEP_LOG_INTERVAL = 250

# ============================================================
# HYPERPARAMETERS — per-agent Optuna-tuned configs
# ============================================================
HYPERPARAMS_MAP = {
    "dqn":      DQN_HYPERPARAMS,
    "ddqn":     DDQN_HYPERPARAMS,
    "d3qn":     D3QN_HYPERPARAMS,
    "d3qn_per": D3QN_PER_HYPERPARAMS,
}

# ============================================================
# AGENT REGISTRY  —  add new agents here as they are built
# ============================================================
AGENTS = {
    "dqn":  DQNAgent,
    "ddqn": DDQNAgent,
    "d3qn": D3QNAgent,
    "d3qn_per": D3QNPERAgent,
}

env_config = envs.env_config


def build_agent(agent_name: str, hp: dict):
    cls = AGENTS[agent_name]
    return cls(
        gamma      = hp["gamma"],
        epsilon    = hp["epsilon"],
        lr         = hp["lr"],
        batch_size = hp["batch_size"],
        eps_dec    = hp["eps_dec"],
        eps_min    = hp["eps_min"],
        replace_target_cnt = hp["target_update_frequency"],
        input_dims = 103,
        n_actions  = N_ACTIONS,
    )


def train(agent_name: str, run_id: int, seed: int = 42, resume_path: str = None):
    hp = HYPERPARAMS_MAP[agent_name]

    # Freeze all randomness for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.backends.mps.is_available():
        T.mps.manual_seed(seed)

    run_name = (
        f"{run_id:02d}_{agent_name.upper()}"
        f"_seed{seed}"
        f"_lci{hp['lambda_ci']}"
        f"_lr{hp['lr']}"
        f"_g{hp['gamma']}"
        f"_eps{hp['n_episodes']}_train70"
    )
    os.makedirs(f"models/{run_name}", exist_ok=True)

    writer = SummaryWriter(f"runs/{run_name}")
    env = BatteryEnv(
        config=env_config,
        lambda_ci=hp["lambda_ci"],
        split="train",
        randomize_init_soc=True,
        randomize_start=True,
        seed=seed,
    )
    agent = build_agent(agent_name, hp)

    start_episode = 0
    if resume_path and os.path.exists(resume_path):
        print(f"LOADING CRASH SAVE: Resuming from {resume_path}...")
        checkpoint = T.load(resume_path)

        # Restore the brain (weights and optimiser)
        agent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        agent.Q_eval.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        # Both DQN and DDQN have Q_target — sync it from the restored eval weights
        agent.Q_target.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore the memory of where we were in time
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1  # Start at the next episode
        print(f"Successfully restored! Starting at Episode {start_episode} with Epsilon {agent.epsilon:.4f}")

    global_step = 0
    scores = []
    log_this_episode = False   # set per episode below

    for i in range(start_episode, hp["n_episodes"]):
        obs, info = env.reset()
        done = False
        step_in_ep = 0
        log_this_episode = (STEP_LOG_INTERVAL > 0 and i % STEP_LOG_INTERVAL == 0)

        # --- Episode Accumulators ---
        ep_reward         = 0.0
        ep_da_revenue     = 0.0
        ep_id_revenue     = 0.0
        ep_carbon_gbp     = 0.0
        ep_deg_cost_gbp   = 0.0
        ep_throughput_mwh = 0.0
        ep_idle_steps     = 0
        ep_carbon_kg      = 0.0
        ep_p_req_list     = []
        ep_p_applied_list = []
        ep_p_dev_list     = []
        ep_soc_list       = []
        ep_clip_events    = 0
        ep_da_plan_edits  = 0
        ep_losses         = []
        ep_grad_norms     = []
        ep_q_means        = []

        while not done:
            action = agent.choose_action(obs)
            dispatch_idx, plan_idx, plan_slot = decode(action)
            env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
            obs_, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, obs_, done)
            loss, grad_norm, q_mean = agent.learn()

            # -- Accumulate --
            ep_reward         += reward
            ep_da_revenue     += info["Planned_Profit"]
            ep_id_revenue     += info["Intraday_Profit"]
            ep_carbon_gbp     += info["carbon_cashflow"]
            ep_deg_cost_gbp   += info["degradation_cost_gbp"]
            ep_throughput_mwh += abs(info["P_applied_MW"]) * 0.5
            ep_carbon_kg      += info["net_carbon_tCO2"]
            ep_p_req_list.append(info["P_req_MW"])
            ep_p_applied_list.append(info["P_applied_MW"])
            ep_soc_list.append(info["soc"])
            ep_p_dev_list.append(info["P_dev_MW"])

            if abs(info["P_req_MW"] - info["P_applied_MW"]) > 1e-4:
                ep_clip_events += 1
            if info["da_available"] and info["tomorrow_plan_value_written"] >= 0:
                ep_da_plan_edits += 1
            if loss is not None:
                ep_losses.append(loss)
                ep_grad_norms.append(grad_norm)
                ep_q_means.append(q_mean)
            if abs(info["P_applied_MW"]) < 1e-4:
                ep_idle_steps += 1

            # ----------------------------------------------------------------
            # STEP-LEVEL SNAPSHOT — logged every STEP_LOG_INTERVAL episodes.
            # ----------------------------------------------------------------
            if log_this_episode:
                s = step_in_ep

                writer.add_scalar(f"Step_ep{i:04d}/Bat_SoC",          info["soc"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_Delta_SoC",    info["delta_soc"],    s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_applied_MW", info["P_applied_MW"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_req_MW",     info["P_req_MW"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_planned_MW", info["P_planned_MW"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_dev_MW",     info["P_dev_MW"],     s)

                writer.add_scalar(f"Step_ep{i:04d}/Act_Dispatch_idx",    info["dispatch_idx_agent"],             s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_Planned_Today",   info["planned_idx_today"],              s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_DA_Available",    float(info["da_available"]),            s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_Plan_Written",    info["tomorrow_plan_value_written"],    s)

                writer.add_scalar(f"Step_ep{i:04d}/Mkt_DA_Price",      info["da_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_ID_Price",      info["id_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_CI",            info["ci_now"],           s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_MEF",           info["mef_now"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_Carbon_Price",  info["carbon_price_now"], s)

                R_gross = info["Planned_Profit"] + info["Intraday_Profit"]
                R_net   = R_gross - info["degradation_cost_gbp"] - info["carbon_cashflow"]
                writer.add_scalar(f"Step_ep{i:04d}/Rev_DA_GBP",       info["Planned_Profit"],       s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_ID_GBP",       info["Intraday_Profit"],      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Gross_GBP",    R_gross,                      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Deg_GBP",      info["degradation_cost_gbp"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Carbon_GBP",   info["carbon_cashflow"],      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Net_GBP",      R_net,                        s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Carbon_tCO2",  info["net_carbon_tCO2"],      s)

                # --- NEW LINEAR CLIPPED REWARD LOGGING ---
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Total_GBP",    info["R_total_gbp"],    s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Thresh_Pen",   info["P_thresh_gbp"],   s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_r_MWh",        info["r_mwh"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Final_Clipped",reward,                 s)

                writer.add_scalar(f"Step_ep{i:04d}/Pos_Tau",       info["tau"],       s)
                writer.add_scalar(f"Step_ep{i:04d}/Pos_Days_Done", info["days_done"], s)

            step_in_ep  += 1
            global_step += 1
            obs = obs_

        # ============================================================
        # EPISODE-LEVEL LOGGING
        # ============================================================
        scores.append(ep_reward)
        avg_score = np.mean(scores[-100:])

        # 1. Reward Decomposition
        writer.add_scalar("Reward/Total_Episode_Clipped", ep_reward,         i)
        # writer.add_scalar("Reward/DA_Revenue_GBP",      ep_da_revenue,     i)
        # writer.add_scalar("Reward/ID_Revenue_GBP",      ep_id_revenue,     i)
        # writer.add_scalar("Reward/Degradation_Cost",    ep_deg_cost_gbp,   i)
        # writer.add_scalar("Reward/Carbon_Cashflow_GBP", ep_carbon_gbp,     i)

        # profit_without_deg = ep_id_revenue + ep_da_revenue
        # profit_with_deg    = profit_without_deg - ep_deg_cost_gbp
        # writer.add_scalar("Reward/Profit_Only_Without_Deg_Costs_GBP", profit_without_deg, i)
        # writer.add_scalar("Reward/Profit_Only_With_Deg_Costs_GBP",    profit_with_deg,    i)
        # writer.add_scalar("Reward/Carbon_Only_GBP",                   ep_carbon_gbp,      i)

        # 2. Dispatch Behaviour
        # p_req = np.array(ep_p_req_list)
        # p_app = np.array(ep_p_applied_list)
        # p_dev = np.array(ep_p_dev_list)
        # writer.add_scalar("Dispatch/Mean_Requested_MW",    np.mean(p_req),  i)
        # writer.add_scalar("Dispatch/Mean_Applied_MW",      np.mean(p_app),  i)
        # writer.add_scalar("Dispatch/Mean_Deviation_MW",    np.mean(p_dev),  i)
        # writer.add_scalar("Dispatch/SoC_Shield_Activation", ep_clip_events, i)
        writer.add_scalar("Dispatch/Idle_Steps_per_Episode", ep_idle_steps, i)
        # writer.add_histogram("Dispatch/P_Applied_Dist",    p_app,           i)

        # 3. SoC Health
        # soc_arr = np.array(ep_soc_list)
        # writer.add_scalar("Battery/SoC_Mean", np.mean(soc_arr), i)
        # writer.add_scalar("Battery/SoC_Std",  np.std(soc_arr),  i)
        # writer.add_scalar("Battery/SoC_Min",  np.min(soc_arr),  i)
        # writer.add_scalar("Battery/SoC_Max",  np.max(soc_arr),  i)
        # writer.add_histogram("Battery/SoC_Dist", soc_arr,        i)

        # 4. DA Planning Behaviour
        writer.add_scalar("Planning/DA_Plan_Edits_per_Episode", ep_da_plan_edits, i)

        # 5. Training Health
        if ep_losses:
            writer.add_scalar("Training/Loss_Mean",      np.mean(ep_losses),     i)
            writer.add_scalar("Training/Grad_Norm_Mean", np.mean(ep_grad_norms), i)
            writer.add_scalar("Training/Q_Mean",         np.mean(ep_q_means),    i)

        # 6. Exploration
        writer.add_scalar("Training/Epsilon", agent.epsilon, i)

        print(f"[{agent_name.upper()}] Episode {i} | Score: {ep_reward:.2f} | "
              f"Avg: {avg_score:.2f} | ε: {agent.epsilon:.4f}")

        if (i + 1) % 50 == 0:
            ckpt_path = f"models/{run_name}/checkpoint_ep{i+1}.pth"
            T.save({
                'episode':            i,
                'epsilon':            agent.epsilon,
                'model_state_dict':   agent.Q_eval.state_dict(),
                'optimizer_state_dict': agent.Q_eval.optimiser.state_dict(),
            }, ckpt_path)
            print(f"  ✔ Checkpoint saved → {ckpt_path}")

    final_path = f"models/{run_name}/final_model.pth"
    T.save({
        'episode':              hp["n_episodes"] - 1,
        'epsilon':              agent.epsilon,
        'seed':                 seed,
        'agent':                agent_name,
        'model_state_dict':     agent.Q_eval.state_dict(),
        'optimizer_state_dict': agent.Q_eval.optimiser.state_dict(),
    }, final_path)
    print(f"  ✔ Final model saved → {final_path}")
    writer.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=list(AGENTS.keys()),
        help="Agent to train: " + ", ".join(AGENTS.keys()),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Run number prefix in the TensorBoard run name (e.g. 1 → 01_DDQN_...)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Space-separated list of seeds for 5-seed averaging (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Override the number of training episodes",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path to a .pth checkpoint to resume from (single-seed runs only)",
    )

    args = parser.parse_args()

    # Override n_episodes for all agents if passed
    if args.n_episodes is not None:
        for hp in HYPERPARAMS_MAP.values():
            hp["n_episodes"] = args.n_episodes

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  {args.agent.upper()} | seed={seed}")
        print(f"{'='*60}")
        train(args.agent, args.run_id, seed=seed, resume_path=args.resume_path)