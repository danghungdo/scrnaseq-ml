DATA_LOADER_CONFIG = {
    "data_name": "norman",
    "save_dir": "data",
    "shuffle": True,
    "test_split": 0.2,
    "random_seed": 42,
    "stratify": True
}

LOGGING_DIR = "logs"

MODEL_CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 10,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}



