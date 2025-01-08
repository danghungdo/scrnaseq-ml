import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from .data_loader import DataLoader
from .config import HYPERPARAM_CONFIG

class Optimizer:
    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader

        
    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', HYPERPARAM_CONFIG['n_estimators'][0], HYPERPARAM_CONFIG['n_estimators'][1]),
            'max_depth': trial.suggest_int('max_depth', HYPERPARAM_CONFIG['max_depth'][0], HYPERPARAM_CONFIG['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate', HYPERPARAM_CONFIG['learning_rate'][0], HYPERPARAM_CONFIG['learning_rate'][1], log=True),
            'subsample': trial.suggest_float('subsample', HYPERPARAM_CONFIG['subsample'][0], HYPERPARAM_CONFIG['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree', HYPERPARAM_CONFIG['colsample_bytree'][0], HYPERPARAM_CONFIG['colsample_bytree'][1]),
            'gamma': trial.suggest_float('gamma', HYPERPARAM_CONFIG['gamma'][0], HYPERPARAM_CONFIG['gamma'][1]),
            'lambda': trial.suggest_float('lambda', HYPERPARAM_CONFIG['lambda'][0], HYPERPARAM_CONFIG['lambda'][1], log=True),
            'alpha': trial.suggest_float('alpha', HYPERPARAM_CONFIG['alpha'][0], HYPERPARAM_CONFIG['alpha'][1], log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', HYPERPARAM_CONFIG['min_child_weight'][0], HYPERPARAM_CONFIG['min_child_weight'][1])
        }
        
        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier(**params, objective='multi:softmax', eval_metric='mlogloss')
        X_train, y_train = self.data_loader.get_train_data()
        sample_weights = self.data_loader.sample_weights
        val_indexes = np.random.choice(len(X_train), int(0.2*len(X_train)), replace=False)
        X_val, y_val = X_train[val_indexes], y_train[val_indexes]
        X_train, y_train = np.delete(X_train, val_indexes, axis=0), np.delete(y_train, val_indexes)
        new_sample_weights = np.delete(sample_weights, val_indexes)
        model.fit(X_train, y_train, sample_weight=new_sample_weights)
        preds = model.predict(X_train)
        # scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        return accuracy_score(y_train, preds)
    
    def tune_hyperparams(self, n_trials=5):
        print("Tuning hyperparameters...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("Best Hyperparams: ", study.best_params)
        print("Best Accuracy: ", study.best_value)
        
        return study.best_params
        