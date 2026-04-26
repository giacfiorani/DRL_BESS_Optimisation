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
from envs.reward_scaling import get_frozen_scales

# Five fixed seeds for multi-seed averaging across evaluation runs.
SEEDS = [0, 1, 2, 3, 4]

# Every STEP_LOG_INTERVAL episodes, log all per-step metrics to TensorBoard.
STEP_LOG_INTERVAL = 250

HYPERPARAMS_MAP = {
    "dqn":      DQN_HYPERPARAMS,
    "ddqn":     DDQN_HYPERPARAMS,
    "d3qn":     D3QN_HYPERPARAMS,
    "d3qn_per": D3QN_PER_HYPERPARAMS,
}

AGENTS = {
    "dqn":      DQNAgent,
    "ddqn":     DDQNAgent,
    "d3qn":     D3QNAgent,
    "d3qn_per": D3QNPERAgent,
}

env_config = envs.env_config

DEFAULT_VAL_INTERVAL = 50   # evaluate on val split every N episodes
DEFAULT_PATIENCE     = 200  # episodes without val improvement before early stop


def evaluate_on_val(agent, val_env):
    """Run a deterministic greedy rollout over the entire validation split.

    No transitions are stored, preventing replay buffer contamination.
    ``val_env`` must be created before the training loop to avoid repeated
    parquet reads.

    Args:
        agent: Trained discrete agent with an ``epsilon`` attribute.
        val_env: Pre-constructed ``BatteryEnv`` instance using ``split="val"``.

    Returns:
        Tuple of (cumulative_reward, metrics_dict).
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    obs, _ = val_env.reset()
    done = False
    cumulative_reward    = 0.0
    cumulative_profit_gbp  = 0.0
    cumulative_carbon_tco2 = 0.0
    n_steps = 0

    while not done:
        action = agent.choose_action(obs)
        dispatch_idx, plan_idx, plan_slot = decode(action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
        obs, reward, terminated, truncated, info = val_env.step(env_action)
        done = terminated or truncated
        cumulative_reward      += reward
        cumulative_profit_gbp  += info.get("R_total_gbp", 0.0)
        cumulative_carbon_tco2 += info.get("net_carbon_tCO2", 0.0)
        n_steps += 1

    agent.epsilon = original_epsilon

    metrics = {
        "val_reward":      cumulative_reward,
        "val_profit_gbp":  cumulative_profit_gbp,
        "val_carbon_tco2": cumulative_carbon_tco2,
        "val_steps":       n_steps,
    }
    return cumulative_reward, metrics


def build_agent(agent_name: str, hp: dict):
    """Instantiate a discrete RL agent from its hyperparameter dict.

    Args:
        agent_name: Key into the ``AGENTS`` registry.
        hp: Hyperparameter dict with keys matching the agent constructor.

    Returns:
        Initialised agent instance.
    """
    cls = AGENTS[agent_name]
    return cls(
        gamma              = hp["gamma"],
        epsilon            = hp["epsilon"],
        lr                 = hp["lr"],
        batch_size         = hp["batch_size"],
        eps_dec            = hp["eps_dec"],
        eps_min            = hp["eps_min"],
        replace_target_cnt = hp["target_update_frequency"],
        input_dims         = 103,
        n_actions          = N_ACTIONS,
    )


def train(
    agent_name: str,
    run_id: int,
    seed: int = 42,
    resume_path: str = None,
    train_start: str = None,
    val_interval: int = DEFAULT_VAL_INTERVAL,
    patience: int = DEFAULT_PATIENCE,
):
    """Train a discrete RL agent on the BESS environment.

    Args:
        agent_name: One of ``"dqn"``, ``"ddqn"``, ``"d3qn"``, ``"d3qn_per"``.
        run_id: Integer prefix for the TensorBoard run name.
        seed: Random seed for reproducibility.
        resume_path: Optional path to a ``.pth`` checkpoint to resume from.
        train_start: Optional ISO date string. Data before this date is excluded.
        val_interval: Evaluate on the validation split every this many episodes.
        patience: Stop training after this many episodes without val improvement
            (counting only begins after ε reaches eps_min).

    Returns:
        Path to the saved final model checkpoint.
    """
    hp = HYPERPARAMS_MAP[agent_name]

    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.backends.mps.is_available():
        T.mps.manual_seed(seed)

    start_label = train_start if train_start else "2022"
    run_name = (
        f"{run_id:02d}_{agent_name.upper()}"
        f"_seed{seed}"
        f"_start{start_label}"
        f"_lci{hp['lambda_ci']}"
        f"_lr{hp['lr']}"
        f"_g{hp['gamma']}"
        f"_eps{hp['n_episodes']}_train70"
    )
    os.makedirs(f"models/{run_name}", exist_ok=True)

    writer        = SummaryWriter(f"runs/{run_name}")
    frozen_scales = get_frozen_scales()

    env = BatteryEnv(
        config             = env_config,
        lambda_ci          = hp["lambda_ci"],
        split              = "train",
        randomize_init_soc = True,
        randomize_start    = True,
        seed               = seed,
        precomputed_scales = frozen_scales,
    )
    agent = build_agent(agent_name, hp)

    # Construct the validation environment once to avoid repeated parquet reads.
    val_env = BatteryEnv(
        config             = env_config,
        lambda_ci          = hp["lambda_ci"],
        split              = "val",
        episode_days       = 999,
        randomize_init_soc = False,
        randomize_start    = False,
        seed               = 42,
        precomputed_scales = frozen_scales,
    )
    val_env.episode_days = len(val_env.active_valid_days)

    start_episode = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = T.load(resume_path)

        agent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        agent.Q_eval.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        # Both DQN and DDQN have Q_target — sync it from the restored eval weights.
        agent.Q_target.load_state_dict(checkpoint['model_state_dict'])

        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1
        print(f"Restored. Starting at episode {start_episode}, ε={agent.epsilon:.4f}")

    global_step       = 0
    scores            = []
    log_this_episode  = False

    best_val_reward   = -float('inf')
    best_val_episode  = -1
    patience_counter  = 0

    for i in range(start_episode, hp["n_episodes"]):
        obs, info = env.reset()
        done = False
        step_in_ep       = 0
        log_this_episode = (STEP_LOG_INTERVAL > 0 and i % STEP_LOG_INTERVAL == 0)

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

            if log_this_episode:
                s = step_in_ep

                writer.add_scalar(f"Step_ep{i:04d}/Bat_SoC",          info["soc"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_Delta_SoC",    info["delta_soc"],    s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_applied_MW", info["P_applied_MW"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_req_MW",     info["P_req_MW"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_planned_MW", info["P_planned_MW"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_dev_MW",     info["P_dev_MW"],     s)

                writer.add_scalar(f"Step_ep{i:04d}/Act_Dispatch_idx",    info["dispatch_idx_agent"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_Planned_Today",   info["planned_idx_today"],           s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_DA_Available",    float(info["da_available"]),         s)
                writer.add_scalar(f"Step_ep{i:04d}/Act_Plan_Written",    info["tomorrow_plan_value_written"], s)

                writer.add_scalar(f"Step_ep{i:04d}/Mkt_DA_Price",     info["da_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_ID_Price",     info["id_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_CI",           info["ci_now"],           s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_MEF",          info["mef_now"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_Carbon_Price", info["carbon_price_now"], s)

                R_gross = info["Planned_Profit"] + info["Intraday_Profit"]
                R_net   = R_gross - info["degradation_cost_gbp"] - info["carbon_cashflow"]
                writer.add_scalar(f"Step_ep{i:04d}/Rev_DA_GBP",      info["Planned_Profit"],       s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_ID_GBP",      info["Intraday_Profit"],      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Gross_GBP",   R_gross,                      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Deg_GBP",     info["degradation_cost_gbp"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Carbon_GBP",  info["carbon_cashflow"],      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Net_GBP",     R_net,                        s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Carbon_tCO2", info["net_carbon_tCO2"],      s)

                writer.add_scalar(f"Step_ep{i:04d}/Rew_Total_GBP",     info["R_total_gbp"],  s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Thresh_Pen",    info["P_thresh_gbp"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_r_MWh",         info["r_mwh"],        s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Final_Clipped", reward,               s)

                writer.add_scalar(f"Step_ep{i:04d}/Pos_Tau",       info["tau"],       s)
                writer.add_scalar(f"Step_ep{i:04d}/Pos_Days_Done", info["days_done"], s)

            step_in_ep  += 1
            global_step += 1
            obs = obs_

        scores.append(ep_reward)
        avg_score = np.mean(scores[-100:])

        writer.add_scalar("Reward/Total_Episode_Clipped",        ep_reward,       i)
        writer.add_scalar("Dispatch/Idle_Steps_per_Episode",     ep_idle_steps,   i)
        writer.add_scalar("Planning/DA_Plan_Edits_per_Episode",  ep_da_plan_edits, i)

        if ep_losses:
            writer.add_scalar("Training/Loss_Mean",      np.mean(ep_losses),     i)
            writer.add_scalar("Training/Grad_Norm_Mean", np.mean(ep_grad_norms), i)
            writer.add_scalar("Training/Q_Mean",         np.mean(ep_q_means),    i)

        writer.add_scalar("Training/Epsilon", agent.epsilon, i)

        print(f"[{agent_name.upper()}] Episode {i} | Score: {ep_reward:.2f} | "
              f"Avg: {avg_score:.2f} | e: {agent.epsilon:.4f}")

        if (i + 1) % val_interval == 0:
            ckpt_path = f"models/{run_name}/checkpoint_ep{i+1}.pth"
            T.save({
                'episode':              i,
                'epsilon':              agent.epsilon,
                'model_state_dict':     agent.Q_eval.state_dict(),
                'optimizer_state_dict': agent.Q_eval.optimiser.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            val_reward, val_metrics = evaluate_on_val(agent, val_env)

            writer.add_scalar("Val/Cumulative_Reward", val_reward,                      i)
            writer.add_scalar("Val/Profit_GBP",        val_metrics["val_profit_gbp"],   i)
            writer.add_scalar("Val/Carbon_tCO2",       val_metrics["val_carbon_tco2"],  i)

            print(f"  [VAL] Episode {i+1} | Val Reward: {val_reward:.2f} | "
                  f"Val Profit: £{val_metrics['val_profit_gbp']:.0f}")

            if val_reward > best_val_reward:
                best_val_reward  = val_reward
                best_val_episode = i + 1
                patience_counter = 0

                best_path = f"models/{run_name}/best_val_model.pth"
                T.save({
                    'episode':              i,
                    'epsilon':              agent.epsilon,
                    'seed':                 seed,
                    'agent':                agent_name,
                    'model_state_dict':     agent.Q_eval.state_dict(),
                    'optimizer_state_dict': agent.Q_eval.optimiser.state_dict(),
                    'val_reward':           val_reward,
                    'val_metrics':          val_metrics,
                }, best_path)
                print(f"  New best val model saved (reward={val_reward:.2f})")
            else:
                # Count patience only after exploration is complete; val rollouts
                # during high-epsilon phases reflect random behaviour, not policy quality.
                if agent.epsilon <= agent.eps_min * 1.5:
                    patience_counter += val_interval

            if patience_counter >= patience and agent.epsilon <= agent.eps_min * 1.5:
                print(f"  [EARLY STOP] No val improvement for {patience} episodes. "
                      f"Best at episode {best_val_episode}.")
                break

    print(f"\n  Training complete. Best validation model at episode {best_val_episode} "
          f"with reward {best_val_reward:.2f}")

    final_path = f"models/{run_name}/final_model.pth"
    T.save({
        'episode':              hp["n_episodes"] - 1,
        'epsilon':              agent.epsilon,
        'seed':                 seed,
        'agent':                agent_name,
        'model_state_dict':     agent.Q_eval.state_dict(),
        'optimizer_state_dict': agent.Q_eval.optimiser.state_dict(),
    }, final_path)
    print(f"  Final model saved: {final_path}")
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
        help="Run number prefix in the TensorBoard run name (e.g. 1 -> 01_DDQN_...)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Space-separated list of seeds for multi-seed averaging (default: 0 1 2 3 4)",
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
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Override degradation cost kappa (GBP/MWh) for ablation study",
    )
    parser.add_argument(
        "--lambda-ci",
        type=float,
        default=None,
        help="Override carbon penalty weight lambda_ci for ablation study",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Exclude data before this ISO date (e.g. 2023-01-01)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=DEFAULT_VAL_INTERVAL,
        help="Evaluate on val split every N episodes (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stop after N episodes without val improvement (default: 200)",
    )

    args = parser.parse_args()

    if args.n_episodes is not None:
        for hp in HYPERPARAMS_MAP.values():
            hp["n_episodes"] = args.n_episodes

    if args.kappa is not None:
        env_config.deg_kappa = args.kappa
        print(f"  [ABLATION] deg_kappa overridden to {args.kappa} GBP/MWh")

    if args.lambda_ci is not None:
        for hp in HYPERPARAMS_MAP.values():
            hp["lambda_ci"] = args.lambda_ci
        print(f"  [ABLATION] lambda_ci overridden to {args.lambda_ci}")

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  {args.agent.upper()} | seed={seed} | start={args.train_start or 'Default'}")
        print(f"{'='*60}")

        train(
            args.agent,
            args.run_id,
            seed=seed,
            resume_path=args.resume_path,
            train_start=args.train_start,
            val_interval=args.val_interval,
            patience=args.patience,
        )
