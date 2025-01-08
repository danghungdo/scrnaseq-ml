import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple
from .config import LOGGING_DIR


class Classifier:
    def __init__(self, **kwargs):
        print("Initializing model...")
        self.clf = XGBClassifier(**kwargs)

    def train(self, train_data: Tuple[np.ndarray, np.ndarray], sample_weights: np.ndarray) -> None:
        print("Training model...")
        X_train, y_train = train_data
        self.clf.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"Accuracy on training data: {self.clf.score(X_train, y_train)}")

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> None:
        print("Evaluating model...")
        X_test, y_test = test_data
        y_pred = self.clf.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")
        print(
            f"Classification report: {classification_report(y_test, y_pred)}")

        # logging to a file
        with open(f"{LOGGING_DIR}/evaluation.txt", "w") as f:
            f.write("=== Model Evaluation Results ===\n\n")
            
            # Accuracy
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
            
            # Confusion Matrix
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(y_test, y_pred)
            f.write(np.array2string(cm, separator=', ', prefix='  '))
            f.write("\n\n")
            # save detailed confusion matrix to a file
            np.savetxt(
                f"{LOGGING_DIR}/confusion_matrix.csv", 
                cm, 
                delimiter=',', 
                fmt='%d'
            )
            # Classification Report
            f.write("Classification Report:\n")
            report = classification_report(y_test, y_pred)
            f.write(report)
            

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
    
