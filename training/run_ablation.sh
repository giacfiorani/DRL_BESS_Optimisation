#!/usr/bin/env bash
# ============================================================
# "Why Does the Agent Idle?" — Full Ablation Study
# ============================================================
#
# PART 1: κ × λ_ci grid (3×3 = 9 runs)
#   kappa ∈ {5, 10, 35} × lambda_ci ∈ {0.0, 0.1, 0.3}
#   Tests whether degradation cost and carbon penalty cause idling.
#
# PART 2: Covariate-shift comparison (2 runs)
#   Same config (κ=10, λ_ci=0.1), but:
#     - Run A: train on full 2022-2024 window (default)
#     - Run B: train on 2023-07-01 onward (post-crisis only)
#   Tests whether 2022 energy-crisis data hurts generalisation.
#
# Each run: 1000-episode DDQN, seed=42
#
# Usage:
#   cd BESS_Agent
#   source .venv/bin/activate
#   bash training/run_ablation.sh
#
# Estimated time: ~11 × 1.5h ≈ 16.5 hours (sequential on M4 Pro)
# ============================================================

set -euo pipefail

AGENT="ddqn"
SEED=42
N_EPISODES=1000

# ============================================================
# PART 1: κ × λ_ci GRID
# ============================================================
KAPPAS=(5 10 35)
LAMBDAS=(0.0 0.1 0.3)

RUN_ID=50  # start at 50 to avoid colliding with regular runs

for kappa in "${KAPPAS[@]}"; do
    for lci in "${LAMBDAS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  ABLATION P1: κ=${kappa} £/MWh | λ_ci=${lci} | seed=${SEED}"
        echo "  Run ID: ${RUN_ID}"
        echo "============================================================"

        python training/train.py \
            --agent "$AGENT" \
            --run-id "$RUN_ID" \
            --seeds "$SEED" \
            --n-episodes "$N_EPISODES" \
            --kappa "$kappa" \
            --lambda-ci "$lci"

        RUN_ID=$((RUN_ID + 1))
    done
done

# ============================================================
# PART 2: COVARIATE SHIFT — 2022-inclusive vs post-crisis
# ============================================================
echo ""
echo "============================================================"
echo "  ABLATION P2: Covariate shift — full window (2022-inclusive)"
echo "  Run ID: ${RUN_ID}"
echo "============================================================"

python training/train.py \
    --agent "$AGENT" \
    --run-id "$RUN_ID" \
    --seeds "$SEED" \
    --n-episodes "$N_EPISODES" \
    --kappa 10 \
    --lambda-ci 0.1

RUN_ID=$((RUN_ID + 1))

echo ""
echo "============================================================"
echo "  ABLATION P2: Covariate shift — post-crisis (2023-07-01+)"
echo "  Run ID: ${RUN_ID}"
echo "============================================================"

python training/train.py \
    --agent "$AGENT" \
    --run-id "$RUN_ID" \
    --seeds "$SEED" \
    --n-episodes "$N_EPISODES" \
    --kappa 10 \
    --lambda-ci 0.1 \
    --train-start "2023-07-01"

echo ""
echo "============================================================"
echo "  ABLATION COMPLETE — view results with: tensorboard --logdir runs/"
echo "============================================================"
