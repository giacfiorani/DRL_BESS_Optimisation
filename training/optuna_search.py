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
from envs.battery_env import BatteryEnv
from utils.action_encoding import N_ACTIONS
from utils.action_encoding import decode

# ── Search Config ──────────────────────────────────────────────────────────────

N_TRIAL_EPISODES = 1000
EVAL_WINDOW = 100
PRUNE_INTERVAL = 100

LAMBDA_CI    = 0.1    
INPUT_DIMS   = 103    
EPS_MIN      = 0.01   
SEED         = 42     

AGENTS = {
    "dqn":  DQNAgent,
    "ddqn": DDQNAgent,
    "d3qn":  D3QNAgent,
    "d3qn_per": D3QNPERAgent,
}

env_config = envs.env_config

# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, agent_name: str) -> float:
    # ── 1. Sample hyperparameters ──
    lr          = trial.suggest_float("lr",       5e-5, 1e-3, log=True)
    eps_dec     = trial.suggest_float("eps_dec", 2e-6, 1e-5) # Removed log=True for linear decay ranges
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
    env = BatteryEnv(
        config             = env_config,
        lambda_ci          = LAMBDA_CI,
        split              = "train",
        randomize_init_soc = True,   
        randomize_start    = True,   
        seed               = SEED,   
        train_start        = "2023-01-01"
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
    # Note: Using the internal class variable. Check if your DDQN class uses target_update_frequency or replace_target_cnt
    agent.replace_target_cnt = target_upd 

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

    # ── 6. Return Objective ──
    return float(np.mean(scores[-EVAL_WINDOW:]))


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",     type=str, default="ddqn", choices=list(AGENTS.keys()))
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
            pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        )
        print(f"Resuming study '{study_name}' — {len(study.trials)} trials already complete.")
    else:
        study = optuna.create_study(
            study_name = study_name,
            direction  = "maximize",
            storage    = storage,
            sampler    = TPESampler(seed=SEED),
            pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=50),
            load_if_exists = True,
        )

    print(f"Study: {study_name}")
    print(f"Agent: {args.agent.upper()} | Trials: {args.n_trials} | "
          f"Episodes/trial: {N_TRIAL_EPISODES} | Eval window: last {EVAL_WINDOW} eps")
    print(f"Results saved to: {db_path}\n")
    print(f"{'Trial':>6}  {'Score':>8}  {'lr':>10}  {'eps_dec':>10}  "
          f"{'batch':>6}  {'tgt_upd':>8}  {'gamma':>7}  {'Status'}")
    print("-" * 75)

    def print_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            p = trial.params
            print(f"{trial.number:>6}  {trial.value:>8.3f}  "
                  f"{p['lr']:>10.2e}  {p['eps_dec']:>10.2e}  "
                  f"{p['batch_size']:>6}  {p['target_update']:>8}  "
                  f"{p['gamma']:>7.4f}  OK")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"{trial.number:>6}  {'—':>8}  {'—':>10}  {'—':>10}  "
                  f"{'—':>6}  {'—':>8}  {'—':>7}  PRUNED")

    # Use a lambda to pass the agent name safely
    study.optimize(
        lambda trial: objective(trial, args.agent),
        n_trials   = args.n_trials,
        callbacks  = [print_trial_callback],
        show_progress_bar = False,
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
    print(f"\nBest trial #{best.number}  →  mean reward = {best.value:.4f}")
    print("\n── Best Hyperparameters ──────────────────────────────────────────────")
    print(f"  lr                      = {best.params['lr']:.2e}")
    print(f"  eps_dec                 = {best.params['eps_dec']:.2e}")
    print(f"  batch_size              = {best.params['batch_size']}")
    print(f"  target_update_frequency = {best.params['target_update']}")
    print(f"  gamma                   = {best.params['gamma']:.4f}")

    steps_to_min  = 0.99 / best.params['eps_dec']
    eps_min_ep    = int(steps_to_min / 336)   
    exploit_eps   = max(0, N_TRIAL_EPISODES - eps_min_ep)
    
    print(f"\n── Epsilon Schedule (best trial) ──────────────────────────────────────")
    print(f"  eps_min reached at episode ~{eps_min_ep}")
    print(f"  exploitation episodes: {exploit_eps} / {N_TRIAL_EPISODES}")

    print("\n── Copy-paste into train.py ───────────────────────────────────────────")
    print("HYPERPARAMS = {")
    print(f'    "gamma":        {best.params["gamma"]:.4f},')
    print(f'    "epsilon":      1.0,')
    print(f'    "lr":           {best.params["lr"]:.2e},')
    print(f'    "batch_size":   {best.params["batch_size"]},')
    print(f'    "eps_dec":      {best.params["eps_dec"]:.2e},')
    print(f'    "eps_min":      0.01,')
    print(f'    "lambda_ci":    {LAMBDA_CI},') # Fixed
    print(f'    "n_episodes":   {N_TRIAL_EPISODES},') # Fixed
    print(f'    "target_update_frequency": {best.params["target_update"]}')
    print("}")
    print()

if __name__ == "__main__":
    main()