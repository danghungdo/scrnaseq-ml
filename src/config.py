DATA_LOADER_CONFIG = {
    "data_name": "norman",
    "save_dir": "data",
    "shuffle": True,
    "test_split": 0.2,
    "random_seed": 42,
    "stratify": True
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
    'n_estimators': (50, 500),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 10),
    'lambda': (1e-8, 10.0),
    'alpha': (1e-8, 10.0),
    'min_child_weight': (1, 10)
}




