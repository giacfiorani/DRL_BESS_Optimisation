#!/bin/bash
#
# SAC Pareto Sweep — λ_ci ablation for profit-carbon frontier
# Trains SAC across 5 λ values × 5 seeds = 25 runs.
# λ = {0.0, 0.1, 0.3, 0.6, 1.0}: carbon-blind → strong penalty.
# λ = 0.1 is the primary training λ (already done); re-run here for
# completeness so all 25 checkpoints are in one clean sweep.
#
# Usage (background tmux session):
#   tmux new-session -d -s pareto
#   tmux send-keys -t pareto "cd ~/BESS_Agent && bash training/run_pareto_sweep.sh" Enter
#
# Estimated time: 25 runs × ~2h = ~50h sequential on M4 Pro
# Monitor: tensorboard --logdir runs/
#

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/training_logs/pareto_sweep"
TRAIN_SCRIPT="$SCRIPT_DIR/train_sac.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Pareto dimensions (5 λ × 5 seeds = 25 runs) ─────────────────────────────
# 0.0: carbon-blind extreme (upper bound on profit)
# 0.1: primary training λ — consistent with all main-result checkpoints
# 0.3: mild carbon awareness
# 0.6: moderate penalty — balanced operating point
# 1.0: strong penalty (lower bound on carbon)
LAMBDA_VALUES=(0.0 0.3 0.6 1.0)  # λ=0.1 already trained (5 seeds); skip to save ~10h
SEEDS=(0 1 2 3 4)
AGENT="sac"

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/pareto_${AGENT}_${TIMESTAMP}.log"
touch "$MAIN_LOG"

echo "==============================================================================="
echo "SAC Pareto Sweep — $(date)"
echo "==============================================================================="
echo "Agent:      $AGENT"
echo "λ values:   ${LAMBDA_VALUES[*]}"
echo "Seeds:      ${SEEDS[*]}"
echo "Total runs: ${#LAMBDA_VALUES[@]} × ${#SEEDS[@]} = $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "Main log:   $MAIN_LOG"
echo "==============================================================================="
{ echo "SAC Pareto Sweep — $(date)"; echo "Total runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"; echo ""; } >> "$MAIN_LOG"

RUN_NUMBER=0
TOTAL_RUNS=$((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))
FAILURES=()

for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NUMBER=$((RUN_NUMBER + 1))
        RUN_TAG="${AGENT}_lambda${LAMBDA}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_TAG}_${TIMESTAMP}.log"

        echo "├─ [$RUN_NUMBER/$TOTAL_RUNS] Training: $RUN_TAG"
        echo "   └─ Log: $RUN_LOG"

        if python "$TRAIN_SCRIPT" \
            --agent "$AGENT" \
            --run-id "$RUN_NUMBER" \
            --seeds "$SEED" \
            --lambda-ci "$LAMBDA" \
            > "$RUN_LOG" 2>&1; then

            echo "   ✓ COMPLETE"
            echo "[$RUN_NUMBER/$TOTAL_RUNS] ✓ $RUN_TAG — SUCCESS" >> "$MAIN_LOG"
        else
            EXIT_CODE=$?
            echo "   ✗ FAILED (exit code: $EXIT_CODE)"
            FAILURES+=("$RUN_TAG (exit code: $EXIT_CODE)")
            echo "[$RUN_NUMBER/$TOTAL_RUNS] ✗ $RUN_TAG — FAILED (exit code: $EXIT_CODE)" >> "$MAIN_LOG"
        fi

        echo "" >> "$MAIN_LOG"

        # Brief pause between runs to avoid resource thrashing
        if [ $RUN_NUMBER -lt $TOTAL_RUNS ]; then
            sleep 2
        fi
    done
done

echo ""
echo "==============================================================================="
echo "Pareto Sweep Complete — $(date)"
echo "==============================================================================="

if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "✓ All $TOTAL_RUNS runs completed successfully"
    echo "✓ All $TOTAL_RUNS runs completed successfully" >> "$MAIN_LOG"
    exit 0
else
    echo "✗ $((TOTAL_RUNS - ${#FAILURES[@]}))/$TOTAL_RUNS succeeded, ${#FAILURES[@]} failed:"
    echo "✗ Failed runs:" >> "$MAIN_LOG"
    for failure in "${FAILURES[@]}"; do
        echo "  - $failure"
        echo "  - $failure" >> "$MAIN_LOG"
    done
    exit 1
fi
