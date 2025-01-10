import optuna
import json
import os
import math
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from .data_loader import DataLoader
from .config import HYPERPARAM_CONFIG, LOGGING_DIR, RANDOM_SEED


class Optimizer:
    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.search_space = HYPERPARAM_CONFIG

    def objective(self, trial) -> float:
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

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        model = XGBClassifier(
            **params, objective="multi:softmax", eval_metric="mlogloss"
        )
        X_train, y_train = self.data_loader.get_train_data()
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="balanced_accuracy")
        return np.mean(scores)

    def tune_hyperparams(self, n_trials: int = None) -> dict:
        print("Tuning hyperparameters...")

        if not os.path.exists(LOGGING_DIR + "/best_hyperparameters.json") and not os.path.exists(
            LOGGING_DIR + "/optuna_trials.csv"
        ):

            sampler = optuna.samplers.GridSampler(self.search_space)
            n_trials = math.prod([len(i) for i in self.search_space.values()])
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self.objective, n_trials=n_trials)

            best_params = study.best_params
            best_values = study.best_value
            print("Best Hyperparams: ", best_params)
            print("Best Accuracy: ", best_values)

            print("Saving hyperparameters each trials and best hyperparameters...")
            df = study.trials_dataframe()

            # Save to CSV
            df.to_csv(LOGGING_DIR + "/optuna_trials.csv", index=False)

            # Save best hyperparameters as txt file
            with open(LOGGING_DIR + "/best_hyperparameters.json", "w") as file:
                json.dump(best_params, file, indent=4)
            
            print(
                "Best hyperparameters and trials are saved under {} and {}.".format(
                    LOGGING_DIR + "/best_hyperparameters.json",
                    LOGGING_DIR + "/optuna_trials.csv",
                )
            )

        else:
            # Open the JSON file for reading
            print("Loading best hyperparameters from file...")
            with open(LOGGING_DIR + "/best_hyperparameters.json", "r") as file:
                # Load JSON data from file
                best_params = json.load(file)

        return best_params
