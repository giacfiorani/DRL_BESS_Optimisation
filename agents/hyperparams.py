# =====
# Hyperparameters — INTERIM values for corrected environment (kappa=10)
# Generated: 2026-03-25
#
# NOTE: Previous Optuna HPO results (kappa=35) are INVALID — they optimised
# for an environment where idling was the rational policy. eps_dec values
# have been replaced with 1.375e-6 (eps_min at episode 500 of 1000).
# All other params retained from HPO until re-run with corrected kappa.
#
# Re-run HPO with: python training/optuna_search.py --agent <name> --n-trials 50
# Database: training/optuna_results/{dqn,ddqn,d3qn,d3qn_per}_rand_study.db
# =====


DQN_HYPERPARAMS = {
    "gamma":        0.9781,
    "epsilon":      1.0,
    "lr":           1.028e-05,
    "batch_size":   512,
    "eps_dec":      1.375e-06,  # INTERIM: eps_min at ep 500 (was 2.487e-05 from kappa=35 HPO)
    "eps_min":      0.01,
    "lambda_ci":    0.1,
    "n_episodes":   1000,
    "target_update_frequency": 7500,
}


DDQN_HYPERPARAMS = {
    "gamma":        0.9701809308246343,
    "epsilon":      1.0,
    "lr":           1.1952270129143866e-05,
    "batch_size":   128,
    "eps_dec":      5e-06,  # INTERIM: eps_min at ep 500 (was 2.91e-05 from kappa=35 HPO)
    "eps_min":      0.01,
    "lambda_ci":    0.1,
    "n_episodes":   1000,
    "target_update_frequency": 6500,
}

D3QN_HYPERPARAMS = {
    "gamma":        0.9741490704098825,
    "epsilon":      1.0,
    "lr":           1.944371379845093e-05,
    "batch_size":   128,
    "eps_dec":      1.375e-06,  # INTERIM: eps_min at ep 500 (was 2.94e-05 from kappa=35 HPO)
    "eps_min":      0.01,
    "lambda_ci":    0.1,
    "n_episodes":   1000,
    "target_update_frequency": 7500,
}

D3QN_PER_HYPERPARAMS = {
    "gamma":        0.975,      # PLACEHOLDER — will update from trial results
    "epsilon":      1.0,
    "lr":           1.45e-05,   # PLACEHOLDER
    "batch_size":   256,        # PLACEHOLDER
    "eps_dec":      1.375e-06,  # INTERIM: eps_min at ep 500 (was 1.32e-04 placeholder)
    "eps_min":      0.01,
    "lambda_ci":    0.1,
    "n_episodes":   1000,
    "target_update_frequency": 5500,  # PLACEHOLDER
}

