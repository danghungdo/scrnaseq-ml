from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from .model import Classifier
from .data_loader import DataLoader
import matplotlib.pyplot as plt
import os
from .config import LOGGING_DIR
import seaborn as sns
from prettytable import PrettyTable
from sklearn.preprocessing import LabelBinarizer
import json
from .config import PATH


class EvaluationMetrics:
    def __init__(self, model: Classifier, data_loader: DataLoader):

        # get y_pred, y_prob, y_true
        self.X, self.y_true = data_loader.get_test_data()
        self.y_pred = model.predict(self.X)
        self.y_prob = model.predict_proba(self.X)
        self.y_true = self.y_true
        self.num_classes = len(set(self.y_true))
        self.label_name = data_loader.get_label_name()

        # metrics for all labels
        self.accuracy = accuracy_score(self.y_pred, self.y_true)
        self.confusion_matrix = confusion_matrix(self.y_pred, self.y_true)

        # path
        self.path_confusion_matrix = PATH["confusion_matrix"]
        self.path_auc_roc = PATH["auc_roc"]
        self.path_metrics = PATH["metrics"]

    def accuracy_each_class(self):
        # Calculate accuracy score for each class
        return self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)

    def precision_each_class(self):
        # Calculate precision score for each class
        precision_scores_each_class = {}
        for i in range(len(self.confusion_matrix)):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_scores_each_class[i] = precision

        return precision_scores_each_class

    def recall_each_class(self):
        # Calculate recall for each class
        recall_scores_each_class = {}
        for i in range(len(self.confusion_matrix)):
            tp = self.confusion_matrix[i, i]
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_scores_each_class[i] = recall

        return recall_scores_each_class

    def plot_confustion_matrix(self):
        # plot confusion matrix
        fig = plt.figure(figsize=(16, 14))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_name,
            yticklabels=self.label_name,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        fig.savefig(self.path_confusion_matrix, format="svg", bbox_inches="tight")

    def auc_roc(self):
        # create the figure
        fig = plt.figure(figsize=(8, 6))

        # Binarize the output
        lb = LabelBinarizer()
        lb.fit(self.y_true)
        y_true_binarized = lb.transform(self.y_true)

        # calculate auc_roc for each class
        auc_roc_score = {}
        for i in range(self.num_classes):
            # calculate auc roc
            y_true_label = y_true_binarized[:, i]
            y_prob_label = self.y_prob[:, i]  # assuming y_prob is given for each class
            auc_roc = roc_auc_score(y_true_label, y_prob_label, multi_class="ovr")
            fpr, tpr, _ = roc_curve(y_true_label, y_prob_label)
            auc_roc_score[i] = [auc_roc, fpr, tpr]

            # plot the auc roc
            plt.plot(fpr, tpr, label=f"{self.label_name[i]} AUC = {auc_roc:.2f}", lw=2)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUC-ROC Curve")
        plt.legend(loc="lower right")
        fig.savefig(self.path_auc_roc, format="svg", bbox_inches="tight")

        return auc_roc_score

    def save_metrics_as_text(self):
        # Save all metrics as a .txt file
        acc_total = self.accuracy
        cm = self.confusion_matrix
        acc_labels = self.accuracy_each_class()
        precision_labels = self.precision_each_class()
        recall_labels = self.recall_each_class()
        auc_roc_labels = self.auc_roc()
        self.plot_confustion_matrix()

        # Initialize the text list
        text = []
        text.append(f"Total Accuracy: {acc_total}")
        text.append(f"Confusion Matrix:\n{cm}")

        # Create a table for each label using PrettyTable
        table = PrettyTable()
        table.field_names = [
            "Num",
            "Label",
            "Accuracy",
            "Precision",
            "Recall",
            "AUC ROC",
        ]
        for i in range(self.num_classes):
            row_to_add = [
                i,
                self.label_name[i],
                f"{acc_labels[i]:.2f}",
                f"{precision_labels[i]:.2f}",
                f"{recall_labels[i]:.2f}",
                f"{auc_roc_labels[i][0]:.2f}",
            ]
            table.add_row(row_to_add)

        # Convert table to string and append to text
        text.append(str(table))

        # Join all pieces of text into a single string
        full_text = "\n".join(text)

        # Save the metrics as a .txt file
        with open(self.path_metrics, "w") as file:
            # Write the text to the file
            file.write(full_text)
