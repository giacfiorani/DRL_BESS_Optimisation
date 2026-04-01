# =====
# Hyperparameters — corrected environment (kappa=10, train_start=2023-01-01)
# Updated: 2026-03-26
#
# REWARD SCALE NOTE: With S_profit ~= 7,487 (post-crisis window), individual
# step rewards are in [-0.2, 0.2]. Episodic reward (336 steps) for a good
# agent is ~0.5-0.7 per episode. The seed-20 DDQN run achieved GBP 160k
# net profit on the 219-day test set with these DDQN params.
#
# DDQN params below are PROVEN (seed-20 achieved GBP 137k+ on test).
# Re-run HPO with FIXED optuna_search.py (lr range, target_update bug fixed).
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
    "gamma":        0.9707,
    "epsilon":      1.0,
    "lr":           1.48e-05,
    "batch_size":   128,
    "eps_dec":      8.63e-06,
    "eps_min":      0.01,
    "lambda_ci":    0.1,
    "n_episodes":   1000,
    "target_update_frequency": 15000
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

SAC_HYPERPARAMS = {
    # Best trial: #14  val_reward=50.70  (30-trial TPE search, 2026-03-30)
    # Key HPO findings:
    #   - High gamma (>0.993) essential: BESS rewards accumulate over days
    #   - reward_scale=6-9: keeps entropy alive, prevents alpha collapse
    #   - alpha_lr < 2e-4: slower entropy tuning avoids premature convergence
    #   - lr ~1.5e-4: slower than default 3e-4, more stable critic updates
    #   - batch_size=128: outperforms 256 consistently across top-5 trials
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