DATA_LOADER_CONFIG = {
    "data_name": "norman",
    "save_dir": "data",
    "shuffle": True,
    "test_split": 0.2,
    "random_seed": 42,
    "stratify": True,
}

LOGGING_DIR = "logs"

# MODEL_CONFIG = {
#     "n_estimators": 100,
#     "learning_rate": 0.1,
#     "max_depth": 5,
#     "random_state": 42,
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",
# }

HYPERPARAM_CONFIG = {
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate": [0.01, 0.1, 0.3],
}

PATH = {
    "dataloader": "data/dataloader.pkl",
    "best_hyperparameters": "logs/best_hyperparameters.json",
    "optuna_trials": "logs/optuna_trials.csv",
    "confusion_matrix": "logs/confusion_matrix.svg",
    "metrics": "logs/metrics.txt",
    "auc_roc": "logs/auc_roc.svg",
    "clf": "data/clf.json",
}
