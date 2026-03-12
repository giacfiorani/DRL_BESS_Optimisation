import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import envs.env_config

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from envs.battery_env import BatteryEnv
from utils.action_encoding import decode, N_ACTIONS

# ============================================================
# HYPERPARAMETERS  —  edit here before each run
# ============================================================
HYPERPARAMS = {
    "gamma":        0.99,
    "epsilon":      1.0,
    "lr":           1e-5,
    "batch_size":   128,
    "eps_dec":      5e-6,
    "eps_min":      0.01,
    "lambda_ci":    0.9,   # 0 = profit only, 1 = equal weight
    "n_episodes":   500,
}

# ============================================================
# AGENT REGISTRY  —  add new agents here as they are built
# ============================================================
AGENTS = {
    "dqn":  DQNAgent,
    "ddqn": DDQNAgent,
}

env_config = envs.env_config


def build_agent(agent_name: str):
    cls = AGENTS[agent_name]
    return cls(
        gamma      = HYPERPARAMS["gamma"],
        epsilon    = HYPERPARAMS["epsilon"],
        lr         = HYPERPARAMS["lr"],
        batch_size = HYPERPARAMS["batch_size"],
        eps_dec    = HYPERPARAMS["eps_dec"],
        eps_min    = HYPERPARAMS["eps_min"],
        input_dims = 103,
        n_actions  = N_ACTIONS,
    )


def train(agent_name: str, run_id: int):
    run_name = (
        f"{run_id:02d}_{agent_name.upper()}"
        f"_lci{HYPERPARAMS['lambda_ci']}"
        f"_lr{HYPERPARAMS['lr']}"
        f"_g{HYPERPARAMS['gamma']}"
        f"_eps{HYPERPARAMS['n_episodes']}_train70"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    env = BatteryEnv(config=env_config, lambda_ci=HYPERPARAMS["lambda_ci"], split="train")
    agent = build_agent(agent_name)

    global_step = 0
    scores = []

    for i in range(HYPERPARAMS["n_episodes"]):
        obs, info = env.reset()
        done = False

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
            ep_carbon_gbp     += info["carbon_penalty_norm"]
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

            # Step-level logging (uncomment when debugging early episodes)
            # writer.add_scalar("Step/SoC",          info["soc"],          global_step)
            # writer.add_scalar("Step/P_requested",  info["P_req_MW"],     global_step)
            # writer.add_scalar("Step/P_applied",    info["P_applied_MW"], global_step)
            # writer.add_scalar("Step/DA_Price",     info["da_price_now"], global_step)
            # writer.add_scalar("Step/ID_Price",     info["id_price_now"], global_step)
            # writer.add_scalar("Step/CI",           info["ci_now"],       global_step)

            global_step += 1
            obs = obs_

        # ============================================================
        # EPISODE-LEVEL LOGGING
        # ============================================================
        scores.append(ep_reward)
        avg_score = np.mean(scores[-100:])

        # 1. Reward Decomposition
        writer.add_scalar("Reward/Total_Episode",       ep_reward,         i)
        writer.add_scalar("Reward/DA_Revenue_GBP",      ep_da_revenue,     i)
        writer.add_scalar("Reward/ID_Revenue_GBP",      ep_id_revenue,     i)
        writer.add_scalar("Reward/Degradation_Cost",    ep_deg_cost_gbp,   i)
        writer.add_scalar("Reward/Carbon_Penalty_Norm", ep_carbon_gbp,     i)

        profit_without_deg = ep_id_revenue + ep_da_revenue
        profit_with_deg    = profit_without_deg - ep_deg_cost_gbp
        writer.add_scalar("Reward/Profit_Only_Without_Deg_Costs_GBP", profit_without_deg, i)
        writer.add_scalar("Reward/Profit_Only_With_Deg_Costs_GBP",    profit_with_deg,    i)
        writer.add_scalar("Reward/Carbon_Only_GBP",                   ep_carbon_gbp,      i)

        # 2. Dispatch Behaviour
        p_req = np.array(ep_p_req_list)
        p_app = np.array(ep_p_applied_list)
        p_dev = np.array(ep_p_dev_list)
        writer.add_scalar("Dispatch/Mean_Requested_MW",    np.mean(p_req),  i)
        writer.add_scalar("Dispatch/Mean_Applied_MW",      np.mean(p_app),  i)
        writer.add_scalar("Dispatch/Mean_Deviation_MW",    np.mean(p_dev),  i)
        writer.add_scalar("Dispatch/SoC_Shield_Activation", ep_clip_events, i)
        writer.add_scalar("Dispatch/Idle_Steps_per_Episode", ep_idle_steps, i)
        writer.add_histogram("Dispatch/P_Applied_Dist",    p_app,           i)

        # 3. SoC Health
        soc_arr = np.array(ep_soc_list)
        writer.add_scalar("Battery/SoC_Mean", np.mean(soc_arr), i)
        writer.add_scalar("Battery/SoC_Std",  np.std(soc_arr),  i)
        writer.add_scalar("Battery/SoC_Min",  np.min(soc_arr),  i)
        writer.add_scalar("Battery/SoC_Max",  np.max(soc_arr),  i)
        writer.add_histogram("Battery/SoC_Dist", soc_arr,        i)

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

        if (i + 1) % 10 == 0:
            agent.Q_eval.save(f'{agent_name}_checkpoint_ep{i+1}.pth')

    agent.Q_eval.save(f'{agent_name}_final.pth')
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="ddqn",
        choices=list(AGENTS.keys()),
        help="Agent to train: " + ", ".join(AGENTS.keys()),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Run number prefix in the TensorBoard run name (e.g. 1 → 01_DDQN_...)",
    )
    args = parser.parse_args()
    train(args.agent, args.run_id)
