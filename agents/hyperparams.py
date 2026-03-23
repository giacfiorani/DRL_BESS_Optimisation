# =====
# Hyperparameters found from Hyperparameter Optimisation through Optuna Search



DQN_HYPERPARAMS = {
    "gamma":        0.9739,
    "epsilon":      1.0,
    "lr":           6.39e-06,
    "batch_size":   256,
    "eps_dec":      1.39e-04,  # Optuna Trial 24 (best score 8.835). Early exploitation is empirically
    "eps_min":      0.01,      # optimal for dense-reward BESS arbitrage — eps_min reached ~ep 21.
    "lambda_ci":    0.9,
    "n_episodes":   500,
    "target_update_frequency": 5500,
}


DDQN_HYPERPARAMS = {
    "gamma":        0.9720,
    "epsilon":      1.0,
    "lr":           1.36e-05,
    "batch_size":   256,
    "eps_dec":      1.92e-04,  # Optuna Trial 31 (best score 9.568). Early exploitation is empirically
    "eps_min":      0.01,      # optimal for dense-reward BESS arbitrage — eps_min reached ~ep 15.
    "lambda_ci":    0.9,
    "n_episodes":   500,
    "target_update_frequency": 5500,
}

D3QN_HYPERPARAMS = {
    "gamma":        0.9960,
    "epsilon":      1.0,
    "lr":           1.16e-05,
    "batch_size":   64,
    "eps_dec":      1.65e-04,  # Optuna Trial 27 (best score 10.729). Early exploitation is empirically
    "eps_min":      0.01,      # optimal for dense-reward BESS arbitrage — eps_min reached ~ep 18.
    "lambda_ci":    0.9,
    "n_episodes":   500,
    "target_update_frequency": 6000,
}

D3QN_PER_HYPERPARAMS = {
    "gamma":        0.975,
    "epsilon":      1.0,
    "lr":           1.45e-05,
    "batch_size":   256,
    "eps_dec":      1.32e-04,  # Optuna Trial 26 (best score 10.611). Early exploitation is empirically
    "eps_min":      0.01,      # optimal for dense-reward BESS arbitrage — eps_min reached ~ep 22.
    "lambda_ci":    0.9,
    "n_episodes":   500,
    "target_update_frequency": 5500,
}

