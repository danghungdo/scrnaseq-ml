import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from .data_loader import DataLoader
from .config import HYPERPARAM_CONFIG, DATA_LOADER_CONFIG, LOGGING_DIR, PATH
import json
import os
import math


class Optimizer:
    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.seed = DATA_LOADER_CONFIG["random_seed"]
        self.logging_dir = LOGGING_DIR
        self.path_best_hyperparameters = PATH["best_hyperparameters"]
        self.path_optuna_trials = PATH["optuna_trials"]
        self.search_space = HYPERPARAM_CONFIG

    def objective(self, trial):
        params = {
            "max_depth": trial.suggest_categorical(
                "max_depth", self.search_space["max_depth"]
            ),
            "min_child_weight": trial.suggest_categorical(
                "min_child_weight", self.search_space["min_child_weight"]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", self.search_space["subsample"]
            ),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", self.search_space["colsample_bytree"]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", self.search_space["learning_rate"]
            ),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier(
            **params, objective="multi:softmax", eval_metric="mlogloss"
        )
        X_train, y_train = self.data_loader.get_train_data()
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        return np.mean(scores)

    def tune_hyperparams(self, n_trials=None):
        print("Tuning hyperparameters...")

        if not os.path.exists(self.path_best_hyperparameters) and not os.path.exists(
            self.path_optuna_trials
        ):

            sampler = optuna.samplers.GridSampler(self.search_space)
            n_trials = math.prod([len(i) for i in self.search_space.values()])
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)

            best_params = study.best_params
            best_values = study.best_value
            print("Best Hyperparams: ", best_params)
            print("Best Accuracy: ", best_values)

            print("saving hyperparameters each trials and best hyperparameters")
            df = study.trials_dataframe()

            # Save to CSV
            df.to_csv(self.path_optuna_trials.format(self.logging_dir), index=False)

            # Save best hyperparameters as txt file
            with open(self.path_best_hyperparameters, "w") as file:
                json.dump(best_params, file, indent=4)

        else:
            # Open the JSON file for reading
            with open(self.path_best_hyperparameters, "r") as file:
                # Load JSON data from file
                best_params = json.load(file)
        print(
            "Best hyperparameters and trials are saved under {} and {}.".format(
                self.path_best_hyperparameters, self.path_optuna_trials
            )
        )
        return best_params
