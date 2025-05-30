"""Check the metrics of a CNN."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from attrs import define
from keras.layers import Rescaling
from keras.models import Model, load_model
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@define(frozen=True)
class Evaluation:
    """An evaluation of model performance."""

    name: str
    classification: str | dict
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    recall: int
    f1: int
    precision: int
    accuracy: int
    roc_auc: int

    def __str__(self):
        """Return a string represnetation of the Evaluation."""
        return (
            f"{self.name} Evaluation:"
            f"Classification Report:\n{self.classification}"
            "Confusion Matrix:\n"
            f"{self.confusion_matrix}"
            f"Recall: {self.recall}"
            f"F1 Score: {self.f1}"
            f"Precision: {self.precision}"
            f"Accuracy: {self.accuracy}"
            f"ROC AUC Score: {self.roc_auc}"
        )

    def to_latex_table(self) -> str:
        """Return the smaller items in latex table format.

        These items are:
        - Recall
        - Precision
        - Accuracy
        - F1 Score
        - ROC AUC
        """
        items = [self.recall, self.precision, self.accuracy, self.f1, self.roc_auc]
        return " & ".join(map(str, items))


class ModelTester:
    """Tests the models."""

    def __init__(
        self, model: Model, test_folder: str, batch_size=32, target_size=(512, 512)
    ):
        """Initialise a ModelTester.

        Args:
            model (Model): The model to test.
            test_folder (str): The location of the test folder
            batch_size (int, optional): How many images to test at once. Defaults to 32.
            target_size (tuple, optional): How big the processed images should be.
            Defaults to (512, 512).
        """
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.model = model
        self.test_generator = self.create_test_generator()

    @classmethod
    def from_path(
        cls, model_path: str, test_folder: str, batch_size=32, target_size=(512, 512)
    ):
        """Create a ModelTester from a given model path.

        Args:
            model_path (str): The path to the saved model
            test_folder (str): The folder of test files
            batch_size (int, optional): How many images to test at once. Defaults to 32.
            target_size (tuple, optional): The image size. Defaults to (512, 512).
        """
        return ModelTester(
            load_model(
                model_path,
            ),
            test_folder,
            batch_size,
            target_size,
        )

    def create_test_generator(self) -> tf.data.Dataset:
        """Create a Dataset from the test dataset.

        Values are rescaled to [0,1]

        Returns:
            Dataset: the resulting dataset
        """
        unscaled: tf.data.Dataset = image_dataset_from_directory(
            self.test_folder,
            label_mode="categorical",
            image_size=self.target_size,
            batch_size=self.batch_size,
            shuffle=False,
        )
        normalization_layer = Rescaling(1.0 / 255)
        return unscaled.map(lambda x, y: (normalization_layer(x), y))

    def evaluate_model(self):
        """Evaluate the resulting model."""
        y_true = self.test_generator.classes
        y_pred = self.model.predict(self.test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)

        report = classification_report(
            y_true,
            y_pred_classes,
            target_names=self.test_generator.class_indices.keys(),
        )
        cm = confusion_matrix(y_true, y_pred_classes)
        recall = recall_score(y_true, y_pred_classes, average="weighted")
        f1 = f1_score(y_true, y_pred_classes, average="weighted")
        precision = precision_score(y_true, y_pred_classes, average="weighted")
        accuracy = np.mean(y_true == y_pred_classes)

        # Convert y_true to one-hot encoding for roc_auc_score
        y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
        y_true_one_hot[np.arange(y_true.size), y_true] = 1
        roc_auc = roc_auc_score(y_true_one_hot, y_pred, multi_class="ovr")

        return Evaluation(
            self.model_path.split(os.path.seperator)[-1],
            report,
            cm,
            recall,
            f1,
            precision,
            accuracy,
            roc_auc,
        )


# Usage
if __name__ == "__main__":
    model_path = "model/best_modelCNN.keras"
    test_folder = "archive/test/test"
    tester = ModelTester(model_path, test_folder)
    evaluation = tester.evaluate_model()
    print(evaluation)
