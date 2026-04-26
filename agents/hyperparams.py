DQN_HYPERPARAMS = {
    "gamma":                   0.9696,
    "epsilon":                 1.0,
    "lr":                      1.62e-05,
    "batch_size":              256,
    "eps_dec":                 5.88e-06,
    "eps_min":                 0.01,
    "lambda_ci":               0.1,
    "n_episodes":              1000,
    "target_update_frequency": 7000,
}

DDQN_HYPERPARAMS = {
    "gamma":                   0.9707,
    "epsilon":                 1.0,
    "lr":                      1.48e-05,
    "batch_size":              128,
    "eps_dec":                 8.63e-06,
    "eps_min":                 0.01,
    "lambda_ci":               0.1,
    "n_episodes":              1000,
    "target_update_frequency": 15000,
}

D3QN_HYPERPARAMS = {
    "gamma":                   0.9867,
    "epsilon":                 1.0,
    "lr":                      1.03e-05,
    "batch_size":              128,
    "eps_dec":                 9.95e-06,
    "eps_min":                 0.01,
    "lambda_ci":               0.1,
    "n_episodes":              1000,
    "target_update_frequency": 13000,
}

SAC_HYPERPARAMS = {
    # Key HPO findings (30-trial TPE search):
    #   - High gamma (>0.993) is essential: BESS rewards accumulate over multi-day horizons.
    #   - reward_scale in [6, 9] keeps entropy alive and prevents α collapse.
    #   - alpha_lr < 2e-4 avoids premature entropy convergence.
    #   - lr ≈ 1.5e-4 (below the default 3e-4) yields more stable critic updates.
    #   - batch_size=128 consistently outperforms 256 across the top-5 trials.
    "lr":           1.457e-04,
    "alpha_lr":     1.542e-04,
    "gamma":        0.9988,
    "tau":          0.005,
    "reward_scale": 7,
    "batch_size":   128,
    "warmup_steps": 5000,
    "n_episodes":   1000,
    "lambda_ci":    0.1,
}

# ====
# Other Hyperparameters for two other agents not used
# ====

# SAC_UVFA_HYPERPARAMS = {
#     "lr":           1.457e-04,
#     "alpha_lr":     7e-5,
#     "gamma":        0.9988,
#     "tau":          0.005,
#     "reward_scale": 4,
#     "batch_size":   128,
#     "warmup_steps": 5000,
#     "n_episodes":   1000,
#     "lambda_ci":    0.1,
# }

# D3QN_PER_HYPERPARAMS = {
#     "gamma":        0.975,      # PLACEHOLDER — will update from trial results
#     "epsilon":      1.0,
#     "lr":           1.45e-05,   # PLACEHOLDER
#     "batch_size":   256,        # PLACEHOLDER
#     "eps_dec":      1.375e-06,  # INTERIM: eps_min at ep 500 (was 1.32e-04 placeholder)
#     "eps_min":      0.01,
#     "lambda_ci":    0.1,
#     "n_episodes":   1000,
#     "target_update_frequency": 5500,  # PLACEHOLDER
# }