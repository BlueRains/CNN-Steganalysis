"""Check the metrics of a CNN."""

import logging

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelTester:
    """Tests the models."""

    def __init__(
        self, model_path: str, test_folder: str, batch_size=32, target_size=(512, 512)
    ):
        """Initialise a ModelTester.

        Args:
            model_path (str): The model to test.
            test_folder (str): The location of the test folder
            batch_size (int, optional): How many images to test at once. Defaults to 32.
            target_size (tuple, optional): How big the processed images should be.
            Defaults to (512, 512).
        """
        self.model_path = model_path
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.model = self.load_model()
        self.test_generator = self.create_test_generator()

    def load_model(self) -> Model:
        """Load a model.

        Returns:
            Model: The loaded model.
        """
        return load_model(self.model_path)  # type: ignore

    def create_test_generator(self) -> tf.data.Dataset:
        """Create a Dataset from the test dataset.

        Returns:
            _type_: _description_
        """
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        return test_datagen.flow_from_directory(
            self.test_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

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

        logging.info("Classification Report:\n%s", report)
        logging.info("Confusion Matrix:\n%s", cm)
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"ROC AUC Score: {roc_auc}")


# Usage
if __name__ == "__main__":
    model_path = "model/best_modelCNN.keras"
    test_folder = "archive/test/test"
    tester = ModelTester(model_path, test_folder)
    tester.evaluate_model()
