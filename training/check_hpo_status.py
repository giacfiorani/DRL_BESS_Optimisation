#!/usr/bin/env python3
"""
Quick HPO status checker — no heavy dependencies, just checks trial counts.
Run: python check_hpo_status.py
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

def check_study(db_path, agent_name):
    """Query Optuna SQLite database for trial counts and best score."""
    if not os.path.exists(db_path):
        print(f"  {agent_name:10s} — ❌ No study found")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Get trial counts by state
        cur.execute("""
            SELECT state, COUNT(*) FROM trials GROUP BY state
        """)
        states = dict(cur.fetchall())
        complete = states.get('COMPLETE', 0)
        running = states.get('RUNNING', 0)
        pruned = states.get('PRUNED', 0)
        total = sum(states.values())

        # Get best completed trial (join with trial_values table)
        cur.execute("""
            SELECT t.number, tv.value FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY tv.value DESC LIMIT 1
        """)
        result = cur.fetchone()
        best_trial = result[0] if result else None
        best_score = result[1] if result else None

        conn.close()

        status_line = f"  {agent_name:10s} — "
        if running > 0:
            status_line += f"🟡 {complete} complete, {running} RUNNING, {pruned} pruned ({total} total)"
        else:
            status_line += f"✅ {complete} complete, {pruned} pruned ({total} total)"

        if best_score is not None:
            status_line += f" | Best: Trial #{best_trial} (score: {best_score:.4f})"

        print(status_line)

    except Exception as e:
        print(f"  {agent_name:10s} — ⚠️  Error: {e}")

def main():
    results_dir = Path("training/optuna_results")

    print("\n" + "=" * 85)
    print(" BESS HPO Status Monitor")
    print("=" * 85)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database dir: {results_dir}\n")

    agents = ["dqn", "ddqn", "d3qn", "d3qn_per"]

    for agent in agents:
        db_path = results_dir / f"{agent}_rand_study.db"
        check_study(db_path, agent)

    print("\n" + "=" * 85)
    print("Notes:")
    print("  • DQN & DDQN: Extended from 500 to 1000 episodes per trial")
    print("  • D3QN & D3QN+PER: Queued to start after DDQN completes")
    print("  • Next step: Once all 4 complete, run:")
    print("    python evaluation/evaluate.py --agents ddqn d3qn d3qn_per --seeds 0 1 2 3 4 \\")
    print("      --lambda-values 0.0 0.1 0.3 0.5 1.0 --baselines --generate-figures")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
