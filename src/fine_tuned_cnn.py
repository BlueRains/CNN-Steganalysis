"""A fine-tuned CNN that scans pngs for steganography.

Trains and tests the cnn.
"""

from __future__ import annotations

import logging
import os
from typing import IO

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from keras.optimizers import Adam
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from Dataset import DatasetFiles, load_dataset

logger = logging.getLogger(__name__)
THRESHOLD = 0.5


def preprocess_image(image_path: str | os.PathLike | IO[bytes]) -> np.ndarray:
    """Preprocess an image to the right size and colour.

    Args:
        image_path (StrOrBytesPath): The path to the image

    Returns:
        np.ndarray: _description_
    """
    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    # Resize image to 512x512 and convert to numpy array
    image = np.array(image).reshape(512, 512, 3)
    # Scale to [0,1]
    image = image / 255.0
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image - mean) / std


def preprocess_data(
    image_paths: list[str], labels: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Preprocess the given image and labels.

    Args:
        image_paths (list[str]): The paths to the image
        labels (list[int]): The category an image belongs in

    Returns:
        A list of processed images, and the labels
    """
    # Load and preprocess images
    images = [preprocess_image(path) for path in image_paths]
    images = np.array(images).reshape(
        -1, 512, 512, 3
    )  # Reshape to (num_images, 512, 512, 3)
    return images, labels


# Custom High-Pass Filter Layer
def high_pass_layer() -> Conv2D:
    """A function that creates a high-pass layer  for CNN.

    Returns:
        [type]: [description]
    """
    kernel_init = tf.constant_initializer(
        [[[[-1]], [[2]], [[-1]]], [[[2]], [[-4]], [[2]]], [[[-1]], [[2]], [[-1]]]] * 3
    )
    return Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init,
        trainable=True,
    )


def build_model(hp) -> Model:
    """Build a CNN model.

    Args:
        hp ([type]): [description]

    Returns:
        Model: [description]
    """
    inputs = Input(shape=(512, 512, 3))
    x = high_pass_layer()(inputs)
    logger.info("Added high-pass layer")
    for i in range(hp.Int("conv_blocks", 1, 3, default=2)):
        filters = hp.Choice(f"filters_{i}", [32, 64, 128])
        x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1))(x)
    logger.info("Added convolutional blocks")
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int("dense_units", 64, 256, step=64), activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    logger.info("Compiling model")
    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", [1e-3, 1e-4, 5e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_dataset(
    image_paths: list[str], labels: list[int], batch_size=32, training=False
) -> tf.data.Dataset:
    """Prepare a datset for training.

    Args:
        image_paths (list[str]): _description_
        labels (list[int]): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        training (bool, optional): _description_. Defaults to False.
    """
    logger.info("Preparing dataset with %d images", len(image_paths))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [512, 512])
        img = preprocess_input(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
    logger.info("Dataset prepared with batch size %d", batch_size)
    return dataset


def train(data: DatasetFiles):
    """Train and test CNN."""
    train_image_paths, train_labels = load_dataset(data.train)
    validation_image_paths, validation_labels = load_dataset(data.val)
    test_image_paths, test_labels = load_dataset(data.test)

    train_dataset = prepare_dataset(train_image_paths, train_labels, training=True)
    validation_dataset = prepare_dataset(validation_image_paths, validation_labels)
    test_dataset = prepare_dataset(test_image_paths, test_labels)
    logger.info("Prepared datasets")
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="my_dir",
        project_name="cnn_steganalysis",
    )
    logger.info("Created Hyperband tuner")
    logger.info("Starting hyperparameter search")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    best_model_path = "results/best_modelCNN.keras"
    model_checkpoint = ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    tuner.search(
        train_dataset,
        epochs=20,
        validation_data=validation_dataset,
        callbacks=[reduce_lr, early_stopping],
    )
    logger.info("Hyperparameter search completed")
    logger.info(
        "Best hyperparameters found: %s", tuner.get_best_hyperparameters()[0].values
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Building model with best hyperparameters")
    model = tuner.hypermodel.build(best_hps)
    logger.info("Fitting model")
    model.fit(
        train_dataset,
        epochs=25,
        validation_data=validation_dataset,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, model_checkpoint],
    )

    logger.info("Saving model")
    model.save("results/modelCNN.keras")

    logger.info("Evaluating model")
    y_true = np.array(
        [label for _, label in test_dataset.unbatch().as_numpy_iterator()]
    ).astype(float)
    y_pred = model.predict(test_dataset)
    y_pred_classes = (y_pred > THRESHOLD).astype(int).flatten()

    logger.info("CNN Classification Report:")
    logger.info(
        classification_report(y_true, y_pred_classes, target_names=["Clean", "Stego"])
    )

    logger.info("CNN Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    logger.info(conf_matrix)

    accuracy = np.mean(y_pred_classes == y_true)
    logger.info("CNN Accuracy: %d", accuracy)

    precision = precision_score(y_true, y_pred_classes)
    logger.info("CNN Precision: %d", precision)

    recall = recall_score(y_true, y_pred_classes)
    logger.info("CNN Recall: %d", recall)

    f1 = f1_score(y_true, y_pred_classes)
    logger.info("CNN F1 Score: %d", f1)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    logger.info("CNN Specificity: %d", specificity)

    fpr = fp / (fp + tn)
    logger.info("CNN False Positive Rate (FPR): %d", fpr)

    roc_auc = roc_auc_score(y_true, y_pred)
    logger.info("CNN AUC-ROC: %d", roc_auc)


if __name__ == "__main__":
    train()
