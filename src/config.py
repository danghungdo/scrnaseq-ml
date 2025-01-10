DATA_LOADER_CONFIG = {
    "data_name": "norman",
    "save_dir": "data",
    "shuffle": True,
    "test_split": 0.2,
    "stratify": True,
}

RANDOM_SEED = 42

LOGGING_DIR = "logs"

HYPERPARAM_CONFIG = {
    "max_depth": [5, 7, 9],
    "min_child_weight": [3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate": [0.01, 0.1, 0.3],
}