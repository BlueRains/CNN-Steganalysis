"""Use statistical methods to predict."""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from joblib import dump
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage import feature
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize

# Record the start time for execution time measurement
start_time = time.time()


def extract_images_from_folder(folder_path: str) -> list[str]:
    """Extract all supported images from a folder.

    Args:
        folder_path (str): folder to look in

    Raises:
        FileNotFoundError: If the folder doesn't exist

    Returns:
        list[str]: Paths to all supported images.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # Define the supported image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    # Iterate over files in the folder and collect image file paths
    return [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if any(file_name.lower().endswith(ext) for ext in image_extensions)
    ]


def median_edge_detector(image_matrix):
    """Convert image matrix to int64 for calculations."""
    image_matrix = image_matrix.astype(np.int64)
    predicted_matrix = image_matrix.copy()

    # Apply the median edge detector to predict pixel values
    for i in range(1, image_matrix.shape[0]):
        for j in range(1, image_matrix.shape[1]):
            predicted_matrix[i, j] = np.median(
                [
                    image_matrix[i - 1, j],
                    image_matrix[i, j - 1],
                    image_matrix[i - 1, j]
                    + image_matrix[i, j - 1]
                    - image_matrix[i - 1, j - 1],
                ]
            )

    return predicted_matrix


def calculate_residuals(image_matrix, predicted_matrix):
    """Calculate the residuals between the original and predicted matrices."""
    return image_matrix - predicted_matrix


def calculate_rs_features(residuals):
    """Calculate RS analysis features.

    These features are mean, standard deviation, skewness, and kurtosis.
    """
    features_rs = [
        np.mean(residuals),
        np.std(residuals),
        skew(residuals.flatten()),
        kurtosis(residuals.flatten()),
    ]
    return np.array(features_rs)


def calculate_lbp_features(lbp, num_bins=256):
    """Calculate the histogram of LBP values and normalize it."""
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    lbp_hist = normalize(lbp_hist.reshape(1, -1), norm="l1")
    return lbp_hist.flatten()


def calculate_lbp(image_matrix, points=8, radius=1):
    """Calculate the Local Binary Pattern (LBP) representation of the image."""
    return feature.local_binary_pattern(image_matrix, points, radius, method="uniform")


def chi_square_attack(image):
    """Calculate the histogram of the image."""
    histogram = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
    pairs = np.zeros(128)
    for i in range(0, 256, 2):
        pairs[i // 2] = histogram[i] + histogram[i + 1]
    expected = np.sum(pairs) / 128
    # Calculate the chi-square statistic
    return np.sum((pairs - expected) ** 2 / expected)


def sample_pair_analysis(image):
    """Flatten the image and calculate differences between adjacent pixels."""
    image = image.flatten()
    differences = np.diff(image)
    same_value = np.sum(differences == 0)
    different_signs = np.sum(np.diff(np.sign(differences)) != 0)
    # Calculate the SPA statistic
    return same_value - different_signs


def process_image(image_path, label):
    """Open the image and convert to grayscale."""
    image = Image.open(image_path)
    image = np.array(image.convert("L"))

    # Perform RS analysis
    predicted_matrix = median_edge_detector(image)
    residuals = calculate_residuals(image, predicted_matrix)
    features_rs = calculate_rs_features(residuals)

    # Perform LBP analysis
    lbp = calculate_lbp(image)
    features_lbp = calculate_lbp_features(lbp)

    # Perform Chi-square attack
    features_chi = np.array([chi_square_attack(image)])  # Wrap the scalar in a 1D array

    # Perform Sample Pair Analysis
    features_spa = np.array(
        [sample_pair_analysis(image)]
    )  # Wrap the scalar in a 1D array

    # Concatenate all features into a single feature vector
    features = np.concatenate((features_rs, features_lbp, features_chi, features_spa))

    return features, label


def fuse_image_features(images, labels):
    """Process images in parallel using ProcessPoolExecutor."""
    features_list = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, images, labels)
    for features, label in results:
        features_list.append((features, label))
    features_array, labels_array = zip(*features_list)
    return np.array(features_array), np.array(labels_array)


def load_images_and_labels(base_folder):
    """Load images and labels from the specified folders."""
    clean_train = extract_images_from_folder(
        os.path.join(base_folder, "train/train/clean")
    )
    stego_train = extract_images_from_folder(
        os.path.join(base_folder, "train/train/stego")
    )
    clean_val = extract_images_from_folder(os.path.join(base_folder, "val/val/clean"))
    stego_val = extract_images_from_folder(os.path.join(base_folder, "val/val/stego"))
    clean_test = extract_images_from_folder(
        os.path.join(base_folder, "test/test/clean")
    )
    stego_test = extract_images_from_folder(
        os.path.join(base_folder, "test/test/stego")
    )

    train_images = clean_train + stego_train
    train_labels = [0] * len(clean_train) + [1] * len(stego_train)
    val_images = clean_val + stego_val
    val_labels = [0] * len(clean_val) + [1] * len(stego_val)
    test_images = clean_test + stego_test
    test_labels = [0] * len(clean_test) + [1] * len(stego_test)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def main():
    """Try and detect steganography images based on statistical features."""
    # Specify the absolute path to the archive folder
    base_folder = "E:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive"
    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        load_images_and_labels(base_folder)
    )

    # Extract and fuse features from images
    train_features, train_labels = fuse_image_features(train_images, train_labels)
    val_features, val_labels = fuse_image_features(val_images, val_labels)
    test_features, test_labels = fuse_image_features(test_images, test_labels)

    # Train an SVM classifier
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(train_features, train_labels)

    # Predict and evaluate on validation set
    val_predictions = clf.predict(val_features)
    logging.info("Validation Accuracy: %d", accuracy_score(val_labels, val_predictions))
    logging.info(
        "Validation Precision: %d", precision_score(val_labels, val_predictions)
    )
    logging.info("Validation Recall: %d", recall_score(val_labels, val_predictions))
    logging.info("Validation F1 Score: %d", f1_score(val_labels, val_predictions))

    # Predict and evaluate on test set
    test_predictions = clf.predict(test_features)
    logging.info("Test Accuracy: %d", accuracy_score(test_labels, test_predictions))
    logging.info("Test Precision: %d", precision_score(test_labels, test_predictions))
    logging.info("Test Recall: %d", recall_score(test_labels, test_predictions))
    logging.info("Test F1 Score: %d", f1_score(test_labels, test_predictions))

    # Save the trained model
    dump(clf, "svm_model.joblib")

    # Print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Execution Time: %d seconds", execution_time)


if __name__ == "__main__":
    main()
