import sys
import os
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
import envs.env_config

from agents.sac_agent import SACAgent
from envs.battery_env import BatteryEnv
from agents.hyperparams import SAC_HYPERPARAMS
from envs.reward_scaling import get_frozen_scales

# ============================================================
# REPRODUCIBILITY SEEDS
# ============================================================
SEEDS = [0, 1, 2, 3, 4]

# ============================================================
# STEP-LEVEL MONITORING
# Every STEP_LOG_INTERVAL episodes, log every step.
# ============================================================
STEP_LOG_INTERVAL = 250

# ============================================================
# SAC WARMUP — random actions before learning starts
# ============================================================
DEFAULT_WARMUP_STEPS = 5000

env_config = envs.env_config

# ============================================================
# VALIDATION DEFAULTS
# ============================================================
DEFAULT_VAL_INTERVAL = 50      # evaluate on val every N episodes
DEFAULT_PATIENCE     = 200     # episodes without val improvement → early stop


def evaluate_on_val(agent, val_env):
    """Run a single deterministic rollout over the entire validation split.
    Returns (cumulative_reward, metrics_dict).
    No transitions are stored — zero buffer contamination."""
    
    agent.actor.eval()

    obs, _ = val_env.reset()
    done = False
    cumulative_reward = 0.0
    cumulative_profit_gbp = 0.0
    cumulative_carbon_tco2 = 0.0
    n_steps = 0

    while not done:
        action = agent.choose_action_deterministic(obs)
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        cumulative_profit_gbp += info.get("R_total_gbp", 0.0)
        cumulative_carbon_tco2 += info.get("net_carbon_tCO2", 0.0)
        n_steps += 1

    agent.actor.train()

    metrics = {
        "val_reward": cumulative_reward,
        "val_profit_gbp": cumulative_profit_gbp,
        "val_carbon_tco2": cumulative_carbon_tco2,
        "val_steps": n_steps,
    }
    return cumulative_reward, metrics

def train(agent_name: str, run_id: int, seed: int = 42,
          train_start: str = None, warmup_steps: int = DEFAULT_WARMUP_STEPS,
          val_interval: int = DEFAULT_VAL_INTERVAL,
          patience: int = DEFAULT_PATIENCE):
    hp = SAC_HYPERPARAMS

    # Freeze all randomness for reproducibility
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

    writer = SummaryWriter(f"runs/{run_name}")
    frozen_scales = get_frozen_scales()

    env = BatteryEnv(
        config=env_config,
        lambda_ci=hp["lambda_ci"],
        split="train",
        randomize_init_soc=True,
        randomize_start=True,
        continuous_action=True,
        seed=seed,
        precomputed_scales=frozen_scales,
    )

    # Build continuous environment (Validation) ONCE
    val_env = BatteryEnv(
        config=env_config,
        lambda_ci=hp["lambda_ci"],
        split="val",
        episode_days=999,
        randomize_init_soc=False,
        randomize_start=False,
        continuous_action=True,
        seed=42,
        precomputed_scales=frozen_scales,
    )
    val_env.episode_days = len(val_env.active_valid_days)

    # Build SAC agent
    agent = SACAgent(
        gamma=hp["gamma"],
        tau=hp["tau"],
        lr=hp["lr"],
        alpha_lr=hp["alpha_lr"],
        batch_size=hp["batch_size"],
        reward_scale=hp["reward_scale"],
        input_dims=103,
        n_actions=3,
    )

    global_step = 0
    scores = []

    # --- Validation-based model selection ---
    best_val_reward = -float('inf')
    best_val_episode = -1
    patience_counter = 0

    for i in range(hp["n_episodes"]):
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
        ep_critic_losses  = []
        ep_actor_losses   = []
        ep_alphas         = []

        while not done:
            # 1. Check if we are in the warmup phase
            is_warmup = (global_step < warmup_steps)
            
            # 2. Actor generates a 3-dim float array in [-1, 1]
            #    action[0] = real-time dispatch, action[1] = DA power fraction, action[2] = DA aggressiveness
            action = agent.choose_action(obs, warmup=is_warmup)

            # 3. Take the action in the continuous environment
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. Store the transition in the replay buffer.
            # Use `terminated` (not `done`) for the bootstrap mask:
            # - truncated (dataset exhaustion) should still bootstrap
            # - terminated (episode time limit) zeroes bootstrap — this
            #   introduces a small 3.4% downward bias but PREVENTS unbounded
            #   Q-drift from function approximation error compounding.
            agent.store_transition(obs, action, reward, obs_, terminated)

            # 5. SAC learns every single step (unlike DDQN which learns at the end)
            critic_loss, actor_loss, alpha = agent.learn()

            # -- Accumulate Metrics --
            ep_reward         += reward
            ep_da_revenue     += info["Planned_Profit"]
            ep_id_revenue     += info["Intraday_Profit"]
            ep_carbon_gbp     += info["carbon_cashflow"]
            ep_deg_cost_gbp   += info["degradation_cost_gbp"]
            ep_throughput_mwh += abs(info["P_applied_MW"]) * 0.5
            ep_carbon_kg      += info["net_carbon_tCO2"]

            if abs(info["P_applied_MW"]) < 1e-4:
                ep_idle_steps += 1

            # Only append losses if the agent actually learned (skips during warmup)
            if critic_loss is not None:
                ep_critic_losses.append(critic_loss)
                ep_actor_losses.append(actor_loss)
                ep_alphas.append(alpha)

            # ----------------------------------------------------------------
            # STEP-LEVEL SNAPSHOT
            # ----------------------------------------------------------------
            if log_this_episode:
                s = step_in_ep

                writer.add_scalar(f"Step_ep{i:04d}/Bat_SoC",          info["soc"],          s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_Delta_SoC",    info["delta_soc"],    s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_applied_MW", info["P_applied_MW"], s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_req_MW",     info["P_req_MW"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Bat_P_planned_MW", info["P_planned_MW"], s)

                writer.add_scalar(f"Step_ep{i:04d}/Mkt_DA_Price",      info["da_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_ID_Price",      info["id_price_now"],     s)
                writer.add_scalar(f"Step_ep{i:04d}/Mkt_CI",            info["ci_now"],           s)

                R_gross = info["Planned_Profit"] + info["Intraday_Profit"]
                R_net   = R_gross - info["degradation_cost_gbp"] - info["carbon_cashflow"]
                writer.add_scalar(f"Step_ep{i:04d}/Rev_DA_GBP",       info["Planned_Profit"],       s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_ID_GBP",       info["Intraday_Profit"],      s)
                writer.add_scalar(f"Step_ep{i:04d}/Rev_Net_GBP",      R_net,                        s)

                writer.add_scalar(f"Step_ep{i:04d}/Rew_Total_GBP",    info["R_total_gbp"],    s)
                writer.add_scalar(f"Step_ep{i:04d}/Rew_Final_Clipped",reward,                 s)

            # 6. Update the observation for the next step
            obs = obs_
            
            step_in_ep  += 1
            global_step += 1

        # ============================================================
        # EPISODE-LEVEL LOGGING
        # ============================================================
        scores.append(ep_reward)
        avg_score = np.mean(scores[-100:])

        writer.add_scalar("Reward/Total_Episode_Clipped", ep_reward, i)
        writer.add_scalar("Dispatch/Idle_Steps_per_Episode", ep_idle_steps, i)

        # SAC-specific training metrics
        if ep_critic_losses:
            writer.add_scalar("Training/Critic_Loss_Mean", np.mean(ep_critic_losses), i)
            writer.add_scalar("Training/Actor_Loss_Mean",  np.mean(ep_actor_losses),  i)
            writer.add_scalar("Training/Alpha",            np.mean(ep_alphas),         i)

        print(f"[{agent_name.upper()}] Episode {i} | Score: {ep_reward:.2f} | "
              f"Avg: {avg_score:.2f} | Alpha: {np.mean(ep_alphas) if ep_alphas else 0:.4f}")

        if (i + 1) % val_interval == 0:
            # --- Periodic checkpoint (unchanged) ---
            ckpt_path = f"models/{run_name}/checkpoint_ep{i+1}.pth"
            T.save({
                'episode':              i,
                'seed':                 seed,
                'agent':                agent_name,
                'actor_state_dict':     agent.actor.state_dict(),
                'critic_1_state_dict':  agent.critic_1.state_dict(),
                'critic_2_state_dict':  agent.critic_2.state_dict(),
                'actor_optim':          agent.actor.optimiser.state_dict(),
                'log_alpha':            agent.log_alpha.detach().cpu(),
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

            # --- Validation evaluation ---
            val_reward, val_metrics = evaluate_on_val(agent, val_env)

            writer.add_scalar("Val/Cumulative_Reward", val_reward, i)
            writer.add_scalar("Val/Profit_GBP", val_metrics["val_profit_gbp"], i)
            writer.add_scalar("Val/Carbon_tCO2", val_metrics["val_carbon_tco2"], i)

            print(f"  [VAL] Episode {i+1} | Val Reward: {val_reward:.2f} | "
                  f"Val Profit: £{val_metrics['val_profit_gbp']:.0f}")

            # --- Best model selection ---
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_val_episode = i + 1
                patience_counter = 0

                best_path = f"models/{run_name}/best_val_model.pth"
                T.save({
                    'episode':              i,
                    'seed':                 seed,
                    'agent':                agent_name,
                    'actor_state_dict':     agent.actor.state_dict(),
                    'critic_1_state_dict':  agent.critic_1.state_dict(),
                    'critic_2_state_dict':  agent.critic_2.state_dict(),
                    'actor_optim':          agent.actor.optimiser.state_dict(),
                    'log_alpha':            agent.log_alpha.detach().cpu(),
                    'val_reward':           val_reward,
                    'val_metrics':          val_metrics,
                }, best_path)
                print(f"  -> New best val model saved (reward={val_reward:.2f})")
            else:
                patience_counter += val_interval

            # --- Optional early stopping ---
            if patience_counter >= patience:
                print(f"  [EARLY STOP] No val improvement for {patience} episodes. "
                      f"Best at episode {best_val_episode}.")
                break

    print(f"\n  Training complete. Best validation model at episode {best_val_episode} "
          f"with reward {best_val_reward:.2f}")

    final_path = f"models/{run_name}/final_model.pth"
    T.save({
        'episode':              hp["n_episodes"] - 1,
        'seed':                 seed,
        'agent':                agent_name,
        'actor_state_dict':     agent.actor.state_dict(),
        'critic_1_state_dict':  agent.critic_1.state_dict(),
        'critic_2_state_dict':  agent.critic_2.state_dict(),
        'actor_optim':          agent.actor.optimiser.state_dict(),
        'log_alpha':            agent.log_alpha.detach().cpu(),
    }, final_path)
    print(f"  -> Final model saved: {final_path}")
    writer.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="sac", choices=["sac"])
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--lambda-ci", type=float, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--val-interval", type=int, default=DEFAULT_VAL_INTERVAL,
                        help="Evaluate on val split every N episodes (default: 50)")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="Early stop after N episodes without val improvement (default: 200)")

    args = parser.parse_args()

    if args.n_episodes is not None:
        SAC_HYPERPARAMS["n_episodes"] = args.n_episodes

    if args.kappa is not None:
        env_config.deg_kappa = args.kappa
        print(f"  [ABLATION] deg_kappa overridden to {args.kappa} GBP/MWh")

    if args.lambda_ci is not None:
        SAC_HYPERPARAMS["lambda_ci"] = args.lambda_ci
        print(f"  [ABLATION] lambda_ci overridden to {args.lambda_ci}")

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  {args.agent.upper()} | seed={seed} | start={args.train_start or 'Default'}")
        print(f"{'='*60}")

        train(
            args.agent,
            args.run_id,
            seed=seed,
            train_start=args.train_start,
            warmup_steps=args.warmup_steps,
            val_interval=args.val_interval,
            patience=args.patience,
        )
