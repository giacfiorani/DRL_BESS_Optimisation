import optuna
import pandas as pd

def export_study(agent_name: str):
    db_path    = f"training/optuna_results/{agent_name}_rand_study.db"
    study_name = f"bess_{agent_name}_hpo_rand"

    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

    # Trial history
    df = study.trials_dataframe()
    df.to_csv(f"training/optuna_results/{agent_name}_trials.csv", index=False)

    # Parameter importance (requires scikit-learn)
    importance = optuna.importance.get_param_importances(study)
    imp_df = pd.DataFrame(importance.items(), columns=["param", "importance"])
    imp_df.to_csv(f"training/optuna_results/{agent_name}_importance.csv", index=False)

    print(f"{agent_name}: {len(study.trials)} trials, best={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

# export_study("d3qn_per")
export_study("ddqn")
