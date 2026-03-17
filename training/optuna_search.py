"""
Optuna hyperparameter search for BESS RL agents.

Each trial trains the agent for N_TRIAL_EPISODES on the fixed 7-day training
window, then returns the mean reward of the last EVAL_WINDOW episodes
(pure exploitation phase) as the objective to maximise.

Usage:
    python training/optuna_search.py --agent ddqn --n-trials 50
    python training/optuna_search.py --agent ddqn --n-trials 50 --resume   # continue a saved study
    python training/optuna_search.py --agent ddqn --show-best               # print best params only

Results are saved to optuna_results/<agent>_study.db (SQLite).
Re-run with --resume to add more trials without losing previous results.
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

# Episodes per trial.  Must be long enough for epsilon to decay AND for
# the agent to exploit. 300 is safe even for slow eps_dec values.
N_TRIAL_EPISODES = 300

# Window of final episodes used to score a trial.
# Must be ≤ N_TRIAL_EPISODES.  We want this to be exploitation-only.
EVAL_WINDOW = 100

# Pruning: report intermediate score every N episodes so Optuna can kill
# clearly bad trials early. Set to 0 to disable pruning.
PRUNE_INTERVAL = 25

# Fixed params (not searched — set by physics/architecture/ablation plan)
LAMBDA_CI    = 0.9    # carbon penalty weight — ablated separately later
INPUT_DIMS   = 103    # observation space size — fixed by env
EPS_MIN      = 0.01   # minimum epsilon — standard value
SEED         = 42     # same seed for all trials → fair cross-trial comparison

AGENTS = {
    "dqn":  DQNAgent,
    "ddqn": DDQNAgent,
    "d3qn":  D3QNAgent,
    "d3qn_per": D3QNPERAgent,
}

# ── Search Ranges (rationale) ───────────────────────────────────────────────────
#
#  lr        [1e-6, 5e-4]  log-uniform
#               Run 04/05 best was 1e-5; explore ±2 orders of magnitude.
#               Upper 5e-4 is high but DDQN with Huber loss tolerates it.
#
#  eps_dec   [1.5e-5, 2e-4]  log-uniform
#               Lower bound set so eps_min is reached by ep 200 at the latest:
#                 0.99 / (200 × 336 steps) = 1.47e-5  →  floor = 1.5e-5
#               This guarantees ≥ 100 exploitation episodes in EVAL_WINDOW.
#               Upper bound 2e-4 → eps_min at ep 14; 286 exploitation eps.
#
#  target_update  [500, 6000]  int step 500
#               Run 04 used 2000 (15 updates/episode with 336 steps).
#               Explore tighter (500 → almost every episode) and looser (6000).
#
#  batch_size  {64, 128, 256}  categorical
#               128 used so far.  Larger = smoother gradients; smaller = noisier.
#
#  gamma       [0.970, 0.999]  uniform
#               0.99 used so far.  Tight range — large changes break Bellman.

env_config = envs.env_config


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    # ── 1. Sample hyperparameters ──
    lr          = trial.suggest_float("lr",       1e-6, 5e-4, log=True)
    eps_dec     = trial.suggest_float("eps_dec", 1.5e-5, 2e-4, log=True)
    target_upd  = trial.suggest_int  ("target_update", 500, 6000, step=500)
    batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma       = trial.suggest_float("gamma",   0.970, 0.999)

    # ── 2. Freeze randomness (same seed → fair cross-trial comparison) ──
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
        randomize_init_soc = True,   # randomise starting SoC each episode
        randomize_start    = True,   # randomise 7-day window within train split
        seed               = SEED,   # same seed for all trials → fair comparison
    )

    # ── 4. Build agent ──
    cls   = AGENTS[trial.study.user_attrs["agent"]]
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

        # ── Pruning: report intermediate value every PRUNE_INTERVAL episodes ──
        if PRUNE_INTERVAL > 0 and (ep + 1) % PRUNE_INTERVAL == 0:
            window = min(PRUNE_INTERVAL, len(scores))
            intermediate_value = float(np.mean(scores[-window:]))
            trial.report(intermediate_value, step=ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # ── 6. Objective: mean reward over the last EVAL_WINDOW episodes ──
    # This is the exploitation-phase performance, which is what matters.
    objective_value = float(np.mean(scores[-EVAL_WINDOW:]))
    return objective_value


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for BESS RL agents")
    parser.add_argument("--agent",     type=str, default="ddqn", choices=list(AGENTS.keys()))
    parser.add_argument("--n-trials",  type=int, default=30,
                        help="Number of trials to run (default: 30)")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume an existing study from the .db file")
    parser.add_argument("--show-best", action="store_true",
                        help="Print best params from a saved study without running new trials")
    args = parser.parse_args()

    os.makedirs("optuna_results", exist_ok=True)
    db_path    = f"optuna_results/{args.agent}_rand_study.db"
    study_name = f"bess_{args.agent}_hpo_rand"

    # Suppress verbose Optuna logs — still shows trial results
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{db_path}"

    if args.show_best:
        study = optuna.load_study(study_name=study_name, storage=storage)
        _print_results(study)
        return

    # Create or load study
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

    # Store agent name so objective() can access it via trial.study.user_attrs
    study.set_user_attr("agent", args.agent)

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

    study.optimize(
        objective,
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

    # Compute epsilon schedule for the best eps_dec
    steps_to_min  = 0.99 / best.params['eps_dec']
    eps_min_ep    = int(steps_to_min / 336)   # 336 steps per 7-day episode
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
    print(f'    "lambda_ci":    0.9,')
    print(f'    "n_episodes":   500,')
    print("}")
    print(f'# agent.target_update_frequency = {best.params["target_update"]}')
    print()

    # Top 5 trials
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    print("── Top 5 Trials ───────────────────────────────────────────────────────")
    print(f"{'#':>4}  {'Score':>8}  {'lr':>10}  {'eps_dec':>10}  "
          f"{'batch':>6}  {'tgt_upd':>8}  {'gamma':>7}")
    for t in sorted_trials[:5]:
        p = t.params
        print(f"{t.number:>4}  {t.value:>8.3f}  "
              f"{p['lr']:>10.2e}  {p['eps_dec']:>10.2e}  "
              f"{p['batch_size']:>6}  {p['target_update']:>8}  "
              f"{p['gamma']:>7.4f}")


if __name__ == "__main__":
    main()
