#!/bin/bash
#
# Phase 2: Pareto Sweep — Multi-Objective Training
# Trains the best agent across 25 configurations (5 lambda_ci × 5 seeds)
# Sequential execution to protect M4 Pro unified memory
# Usage: tmux new-session -d -s pareto && tmux send-keys -t pareto "cd ~/Desktop/BESS_Agent && bash training/run_pareto_sweep.sh" Enter
#

set -e  # Exit on first error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/training_logs/pareto_sweep"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Pareto dimensions (cartesian product: 5 × 5 = 25 runs)
LAMBDA_VALUES=(0.0 0.1 0.3 0.5 1.0)
SEEDS=(0 1 2 3 4)

# Agent to train (will be set by Phase 1 HPO — update this after HPO completes)
AGENT="ddqn"  # Change to: dqn | ddqn | d3qn | d3qn_per (after HPO identifies best)

# Initialize logging
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/pareto_${AGENT}_${TIMESTAMP}.log"
touch "$MAIN_LOG"

echo "==============================================================================="
echo "Phase 2: Pareto Sweep — $(date)"
echo "==============================================================================="
echo "Agent: $AGENT"
echo "Total runs: ${#LAMBDA_VALUES[@]} × ${#SEEDS[@]} = $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "Sequential execution (M4 Pro unified memory protection)"
echo "Main log: $MAIN_LOG"
echo "==============================================================================="
echo "" >> "$MAIN_LOG"
echo "Phase 2: Pareto Sweep — $(date)" >> "$MAIN_LOG"
echo "Agent: $AGENT | Total runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))" >> "$MAIN_LOG"
echo "" >> "$MAIN_LOG"

# Counters
RUN_NUMBER=0
TOTAL_RUNS=$((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))

# Track failures
FAILURES=()

# ── Execution Loop ──────────────────────────────────────────────────────────────
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NUMBER=$((RUN_NUMBER + 1))
        RUN_TAG="${AGENT}_lambda${LAMBDA}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_TAG}_${TIMESTAMP}.log"

        echo "├─ [$RUN_NUMBER/$TOTAL_RUNS] Training: $RUN_TAG"
        echo "   └─ Logging to: $RUN_LOG"

        # Run training with stderr + stdout captured
        if python "$TRAIN_SCRIPT" \
            --agent "$AGENT" \
            --lambda-ci "$LAMBDA" \
            --seed "$SEED" \
            --n-episodes 500 \
            > "$RUN_LOG" 2>&1; then

            echo "   ✓ COMPLETE"
            echo "[$RUN_NUMBER/$TOTAL_RUNS] ✓ $RUN_TAG — SUCCESS" >> "$MAIN_LOG"
        else
            EXIT_CODE=$?
            echo "   ✗ FAILED (exit code: $EXIT_CODE)"
            FAILURES+=("$RUN_TAG (exit code: $EXIT_CODE)")
            echo "[$RUN_NUMBER/$TOTAL_RUNS] ✗ $RUN_TAG — FAILED (exit code: $EXIT_CODE)" >> "$MAIN_LOG"
        fi

        # Log separator
        echo "" >> "$MAIN_LOG"

        # Brief pause between runs (avoid resource thrashing)
        if [ $RUN_NUMBER -lt $TOTAL_RUNS ]; then
            sleep 2
        fi
    done
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "==============================================================================="
echo "Pareto Sweep Complete — $(date)"
echo "==============================================================================="

if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "✓ All $TOTAL_RUNS runs completed successfully"
    echo "" >> "$MAIN_LOG"
    echo "✓ All $TOTAL_RUNS runs completed successfully" >> "$MAIN_LOG"
    EXIT_CODE=0
else
    echo "✗ $((TOTAL_RUNS - ${#FAILURES[@]}))/$TOTAL_RUNS succeeded, ${#FAILURES[@]} failed:"
    echo "" >> "$MAIN_LOG"
    echo "✗ Failed runs:" >> "$MAIN_LOG"
    for failure in "${FAILURES[@]}"; do
        echo "  - $failure"
        echo "  - $failure" >> "$MAIN_LOG"
    done
    EXIT_CODE=1
fi

echo ""
echo "Main log: $MAIN_LOG"
echo "Run logs: $LOG_DIR/${AGENT}_lambda*_seed*_${TIMESTAMP}.log"
echo ""

exit $EXIT_CODE
