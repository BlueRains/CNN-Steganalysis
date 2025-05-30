"""Create a VGG166 model."""

import logging

import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ImagePreprocessor import preprocess_image


def preprocess_data(
    image_paths: list[str], labels: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Preprocess images.

    Args:
        image_paths (list[str]): The paths to the image
        labels (list[int]): the labels beloning to the images.

    Returns:
        tuple[np.ndarray, list[int]]: A list of processed images and a list of labels.
    """
    # Load and preprocess images
    images = [preprocess_image(path) for path in image_paths]
    images = np.array(images).reshape(
        -1, 512, 512, 3
    )  # Reshape to (num_images, 512, 512, 3)

    # Apply StandardScaler to each image individually
    scaler = StandardScaler()
    images_scaled = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        for j in range(3):  # Apply scaler to each channel separately
            images_scaled[i, :, :, j] = scaler.fit_transform(images[i, :, :, j])

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=2)

    return images_scaled, labels, scaler


def create_vgg16_model(input_shape) -> Model:
    """Create a VGG16 model.

    Args:
        input_shape (_type_): _description_

    Returns:
        Model: The resulting model
    """
    # Define a VGG16 model with pre-trained weights
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(
        256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax")(x)  # Adjusted for 2 classes
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    """Create, train and test a VG166 model."""
    # Paths to datasets
    train_folder = "archive/train/train"
    test_folder = "archive/test/test"
    validation_folder = "archive/val/val"

    # Model initialization
    model = create_vgg16_model((512, 512, 3))

    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Data generators for validation and test data (no augmentation, only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Create generators with smaller batch size
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(512, 512),
        batch_size=128,  # Decreased batch size
        class_mode="categorical",
    )

    validation_generator = val_test_datagen.flow_from_directory(
        validation_folder,
        target_size=(512, 512),
        batch_size=128,  # Decreased batch size
        class_mode="categorical",
    )

    # Create the VGG16 model
    input_shape = (512, 512, 3)
    model = create_vgg16_model(input_shape)

    # Learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Define the path to save the best model
    best_model_path = "best_model.keras"

    # Create the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model with additional options
    model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, model_checkpoint],
    )

    # Save the final model
    model.save("model/modelVGG16.keras")

    # Define the test generator
    test_generator = val_test_datagen.flow_from_directory(
        test_folder,
        target_size=(512, 512),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    # Evaluate the model
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes[: len(y_pred_classes)]

    # Print the metrics
    logging.info("VGG16 Classification Report:")
    logging.info(classification_report(y_true, y_pred_classes))

    logging.info("VGG16 Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    logging.info(conf_matrix)

    # Accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    logging.info("VGG16 Accuracy: %d", accuracy)

    # Precision
    precision = precision_score(y_true, y_pred_classes, average="binary")
    logging.info("VGG16 Precision: %d", precision)

    # Recall
    recall = recall_score(y_true, y_pred_classes, average="binary")
    logging.info("VGG16 Recall: %d", recall)

    # F1 Score
    f1 = f1_score(y_true, y_pred_classes, average="binary")
    logging.info("VGG16 F1 Score: %d", f1)

    # Area Under the Receiver Operating Characteristic curve (AUC-ROC)
    roc_auc = roc_auc_score(y_true, y_pred_classes)
    logging.info("VGG16 AUC-ROC: %d", roc_auc)


if __name__ == "__main__":
    main()
