"""
Optuna hyperparameter search for BESS RL agents.
"""

import sys
import os
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch as T
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import envs.env_config
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.d3qn import D3QNAgent
from agents.d3qn_per_agent import D3QNPERAgent
from agents.sac_agent import SACAgent
from envs.battery_env import BatteryEnv
from envs.reward_scaling import get_frozen_scales
from utils.action_encoding import N_ACTIONS
from utils.action_encoding import decode

# ── Search Config ──────────────────────────────────────────────────────────────

N_TRIAL_EPISODES     = 1000
SAC_TRIAL_EPISODES   = 1000   # SAC is slower per step — same eps, longer wall time
EVAL_WINDOW          = 100
PRUNE_INTERVAL       = 200
SAC_WARMUP_STEPS     = 5000   # Must fill buffer before SAC can learn

LAMBDA_CI    = 0.1
INPUT_DIMS   = 103
EPS_MIN      = 0.01
SEED         = 42

AGENTS = {
    "dqn":      DQNAgent,
    "ddqn":     DDQNAgent,
    "d3qn":     D3QNAgent,
    "d3qn_per": D3QNPERAgent,
    "sac":      SACAgent,
}

env_config = envs.env_config

# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, agent_name: str) -> float:
    # ── 1. Sample hyperparameters ──
    # lr range: extended down to 1e-5 — best known DDQN run uses lr=1.2e-5
    lr          = trial.suggest_float("lr",       1e-5, 5e-4, log=True)
    eps_dec     = trial.suggest_float("eps_dec", 2e-6, 1e-5)
    target_upd  = trial.suggest_int("target_update", 2000, 15000, step=1000)
    batch_size  = trial.suggest_categorical("batch_size", [128, 256, 512])
    gamma       = trial.suggest_float("gamma",   0.95, 0.999)

    # ── 2. Freeze randomness ──
    random.seed(SEED)
    np.random.seed(SEED)
    T.manual_seed(SEED)
    if T.backends.mps.is_available():
        T.mps.manual_seed(SEED)

    # ── 3. Build environment ──
    frozen_scales = get_frozen_scales()
    env = BatteryEnv(
        config             = env_config,
        lambda_ci          = LAMBDA_CI,
        split              = "train",
        randomize_init_soc = True,
        randomize_start    = True,
        seed               = SEED,
        precomputed_scales = frozen_scales,
    )

    # ── 4. Build agent ──
    cls   = AGENTS[agent_name]
    agent = cls(
        gamma      = gamma,
        epsilon    = 1.0,
        lr         = lr,
        batch_size = batch_size,
        eps_dec    = eps_dec,
        eps_min    = EPS_MIN,
        input_dims = INPUT_DIMS,
        n_actions  = N_ACTIONS,
    )
    # DDQNAgent.__init__ stores replace_target_cnt as self.target_update_frequency
    agent.target_update_frequency = target_upd

    # ── 5. Training loop ──
    scores = []

    for ep in range(N_TRIAL_EPISODES):
        obs, _ = env.reset()
        done    = False
        ep_reward = 0.0

        while not done:
            action = agent.choose_action(obs)
            dispatch_idx, plan_idx, plan_slot = decode(action)
            env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
            obs_, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            ep_reward += reward
            obs = obs_

        scores.append(ep_reward)

        # ── Pruning ──
        if PRUNE_INTERVAL > 0 and (ep + 1) % PRUNE_INTERVAL == 0:
            window = min(PRUNE_INTERVAL, len(scores))
            intermediate_value = float(np.mean(scores[-window:]))
            trial.report(intermediate_value, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned() # Fixed syntax

    # ── 6. Evaluate on VALIDATION split (not training reward) ──
    # This prevents HPO from selecting hyperparams that memorise the training calendar.
    agent.epsilon = 0.0  # greedy policy for evaluation
    val_env = BatteryEnv(
        config             = env_config,
        lambda_ci          = LAMBDA_CI,
        split              = "val",
        episode_days       = 999,  # overridden below
        randomize_init_soc = False,
        randomize_start    = False,
        seed               = SEED,
        precomputed_scales = frozen_scales,
    )
    val_env.episode_days = len(val_env.active_valid_days)

    obs, _ = val_env.reset()
    done = False
    val_reward = 0.0
    while not done:
        action = agent.choose_action(obs)
        dispatch_idx, plan_idx, plan_slot = decode(action)
        env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
        obs, reward, terminated, truncated, _ = val_env.step(env_action)
        done = terminated or truncated
        val_reward += reward

    return val_reward


# ── SAC Objective ─────────────────────────────────────────────────────────────

def sac_objective(trial: optuna.Trial) -> float:
    # ── 1. Sample hyperparameters ──
    # lr: centred on Haarnoja et al. (2018) default of 3e-4; allow wider range
    lr          = trial.suggest_float("lr",           1e-4, 1e-3,  log=True)
    alpha_lr    = trial.suggest_float("alpha_lr",     1e-4, 3e-4,  log=True)
    reward_scale = trial.suggest_int("reward_scale",  2,    15)
    batch_size  = trial.suggest_categorical("batch_size", [128, 256])
    gamma       = trial.suggest_float("gamma",        0.97, 0.999)

    # ── 2. Freeze randomness ──
    random.seed(SEED)
    np.random.seed(SEED)
    T.manual_seed(SEED)
    if T.backends.mps.is_available():
        T.mps.manual_seed(SEED)

    # ── 3. Build continuous environment ──
    frozen_scales = get_frozen_scales()
    env = BatteryEnv(
        config             = env_config,
        lambda_ci          = LAMBDA_CI,
        split              = "train",
        randomize_init_soc = True,
        randomize_start    = True,
        continuous_action  = True,
        seed               = SEED,
        precomputed_scales = frozen_scales,
    )

    # ── 4. Build SAC agent ──
    agent = SACAgent(
        input_dims   = INPUT_DIMS,
        n_actions    = 3,
        lr           = lr,
        alpha_lr     = alpha_lr,
        gamma        = gamma,
        batch_size   = batch_size,
        reward_scale = reward_scale,
    )

    # ── 5. Training loop (learn every step, with warmup) ──
    scores = []
    global_step = 0

    for ep in range(SAC_TRIAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            is_warmup = (global_step < SAC_WARMUP_STEPS)
            action = agent.choose_action(obs, warmup=is_warmup)
            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, obs_, terminated)
            agent.learn()
            ep_reward += reward
            obs = obs_
            global_step += 1

        scores.append(ep_reward)

        # ── Pruning (skip during warmup episodes to avoid misleading signals) ──
        if PRUNE_INTERVAL > 0 and (ep + 1) % PRUNE_INTERVAL == 0:
            # Only prune after warmup is well past
            if global_step > SAC_WARMUP_STEPS * 2:
                window = min(PRUNE_INTERVAL, len(scores))
                intermediate_value = float(np.mean(scores[-window:]))
                trial.report(intermediate_value, step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    # ── 6. Evaluate on validation split (greedy / deterministic policy) ──
    val_env = BatteryEnv(
        config             = env_config,
        lambda_ci          = LAMBDA_CI,
        split              = "val",
        episode_days       = 999,
        randomize_init_soc = False,
        randomize_start    = False,
        continuous_action  = True,
        seed               = SEED,
        precomputed_scales = frozen_scales,
    )
    val_env.episode_days = len(val_env.active_valid_days)

    agent.actor.eval()
    obs, _ = val_env.reset()
    done = False
    val_reward = 0.0
    while not done:
        action = agent.choose_action_deterministic(obs)
        obs, reward, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated
        val_reward += reward

    return val_reward


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",     type=str, default="ddqn",
                        choices=list(AGENTS.keys()))
    parser.add_argument("--n-trials",  type=int, default=30)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--show-best", action="store_true")
    args = parser.parse_args()

    os.makedirs("optuna_results", exist_ok=True)
    db_path    = f"optuna_results/{args.agent}_rand_study.db"
    study_name = f"bess_{args.agent}_hpo_rand"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"

    if args.show_best:
        study = optuna.load_study(study_name=study_name, storage=storage)
        _print_results(study)
        return

    if args.resume and os.path.exists(db_path):
        study = optuna.load_study(
            study_name = study_name,
            storage    = storage,
            sampler    = TPESampler(seed=SEED),
            pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=400),
        )
        print(f"Resuming study '{study_name}' — {len(study.trials)} trials already complete.")
    else:
        study = optuna.create_study(
            study_name = study_name,
            direction  = "maximize",
            storage    = storage,
            sampler    = TPESampler(seed=SEED),
            pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=400),
            load_if_exists = True,
        )

    eps_per_trial = SAC_TRIAL_EPISODES if args.agent == "sac" else N_TRIAL_EPISODES
    print(f"Study: {study_name}")
    print(f"Agent: {args.agent.upper()} | Trials: {args.n_trials} | "
          f"Episodes/trial: {eps_per_trial} | Eval window: last {EVAL_WINDOW} eps")
    print(f"Results saved to: {db_path}\n")
    print("-" * 75)

    is_sac = (args.agent == "sac")

    if is_sac:
        print(f"{'Trial':>6}  {'Score':>8}  {'lr':>10}  {'alpha_lr':>10}  "
              f"{'r_scale':>7}  {'batch':>6}  {'gamma':>7}  {'Status'}")
    else:
        print(f"{'Trial':>6}  {'Score':>8}  {'lr':>10}  {'eps_dec':>10}  "
              f"{'batch':>6}  {'tgt_upd':>8}  {'gamma':>7}  {'Status'}")

    def print_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            p = trial.params
            if is_sac:
                print(f"{trial.number:>6}  {trial.value:>8.3f}  "
                      f"{p['lr']:>10.2e}  {p['alpha_lr']:>10.2e}  "
                      f"{p['reward_scale']:>7}  {p['batch_size']:>6}  "
                      f"{p['gamma']:>7.4f}  OK")
            else:
                print(f"{trial.number:>6}  {trial.value:>8.3f}  "
                      f"{p['lr']:>10.2e}  {p['eps_dec']:>10.2e}  "
                      f"{p['batch_size']:>6}  {p['target_update']:>8}  "
                      f"{p['gamma']:>7.4f}  OK")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"{trial.number:>6}  {'—':>8}  {'—':>10}  {'—':>10}  "
                  f"{'—':>6}  {'—':>8}  {'—':>7}  PRUNED")

    # Route SAC to its own objective (different action space, warmup, params)
    if args.agent == "sac":
        study.optimize(
            sac_objective,
            n_trials          = args.n_trials,
            callbacks         = [print_trial_callback],
            show_progress_bar = False,
            catch             = (Exception,),
        )
    else:
        study.optimize(
            lambda trial: objective(trial, args.agent),
            n_trials          = args.n_trials,
            callbacks         = [print_trial_callback],
            show_progress_bar = False,
            catch             = (Exception,),
        )

    _print_results(study)

def _print_results(study: optuna.Study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{'='*75}")
    print(f"STUDY COMPLETE — {len(completed)} completed, {len(pruned)} pruned")
    print(f"{'='*75}")

    if not completed:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print(f"\nBest trial #{best.number}  →  val reward = {best.value:.4f}")
    print("\n── Best Hyperparameters ──────────────────────────────────────────────")

    is_sac = "alpha_lr" in best.params

    if is_sac:
        print(f"  lr                      = {best.params['lr']:.2e}")
        print(f"  alpha_lr                = {best.params['alpha_lr']:.2e}")
        print(f"  reward_scale            = {best.params['reward_scale']}")
        print(f"  batch_size              = {best.params['batch_size']}")
        print(f"  gamma                   = {best.params['gamma']:.4f}")
        print("\n── Copy-paste into hyperparams.py ─────────────────────────────────────")
        print("SAC_HYPERPARAMS = {")
        print(f'    "lr":           {best.params["lr"]:.2e},')
        print(f'    "alpha_lr":     {best.params["alpha_lr"]:.2e},')
        print(f'    "gamma":        {best.params["gamma"]:.4f},')
        print(f'    "tau":          0.005,')
        print(f'    "reward_scale": {best.params["reward_scale"]},')
        print(f'    "batch_size":   {best.params["batch_size"]},')
        print(f'    "warmup_steps": {SAC_WARMUP_STEPS},')
        print(f'    "n_episodes":   {SAC_TRIAL_EPISODES},')
        print(f'    "lambda_ci":    {LAMBDA_CI},')
        print("}")
    else:
        print(f"  lr                      = {best.params['lr']:.2e}")
        print(f"  eps_dec                 = {best.params['eps_dec']:.2e}")
        print(f"  batch_size              = {best.params['batch_size']}")
        print(f"  target_update_frequency = {best.params['target_update']}")
        print(f"  gamma                   = {best.params['gamma']:.4f}")

        steps_to_min = 0.99 / best.params['eps_dec']
        eps_min_ep   = int(steps_to_min / 336)
        exploit_eps  = max(0, N_TRIAL_EPISODES - eps_min_ep)
        print(f"\n── Epsilon Schedule (best trial) ──────────────────────────────────────")
        print(f"  eps_min reached at episode ~{eps_min_ep}")
        print(f"  exploitation episodes: {exploit_eps} / {N_TRIAL_EPISODES}")

        print("\n── Copy-paste into hyperparams.py ─────────────────────────────────────")
        print("HYPERPARAMS = {")
        print(f'    "gamma":        {best.params["gamma"]:.4f},')
        print(f'    "epsilon":      1.0,')
        print(f'    "lr":           {best.params["lr"]:.2e},')
        print(f'    "batch_size":   {best.params["batch_size"]},')
        print(f'    "eps_dec":      {best.params["eps_dec"]:.2e},')
        print(f'    "eps_min":      0.01,')
        print(f'    "lambda_ci":    {LAMBDA_CI},')
        print(f'    "n_episodes":   {N_TRIAL_EPISODES},')
        print(f'    "target_update_frequency": {best.params["target_update"]}')
        print("}")
    print()

if __name__ == "__main__":
    main()