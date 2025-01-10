import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple
from .config import LOGGING_DIR


class Classifier:
    def __init__(self, **kwargs):
        print("Initializing model...")
        self.clf = XGBClassifier(**kwargs)

    def train(
        self, train_data: Tuple[np.ndarray, np.ndarray], sample_weights: np.ndarray
    ) -> None:
        print("Training model...")
        X_train, y_train = train_data
        self.clf.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"Accuracy on training data: {self.clf.score(X_train, y_train)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)
