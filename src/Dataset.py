"""Load the dataset for training a CNN."""

import os

import kagglehub
import tensorflow as tf
from attrs import define, field
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from ImagePreprocessor import preprocess_image


@define
class DatasetFiles:
    """Where a dataset is stored."""

    url: str
    _train: str = field()
    _test: str = field()
    _val: str = field()
    path: str = field(init=False, default=None)

    def download_dataset(self):
        """Download the dataset from `url`. Download location is stored in `path`."""
        self.path = kagglehub.dataset_download(self.url)

    @property
    def train(self):
        """Get or set the training folder.

        If get, combined with `path`.
        If path is not yet set, download the dataset.
        """
        if self.path is None:
            self.download_dataset()
        return os.path.join(self.path, self._train)

    @property
    def test(self):
        """Get or set the test folder.

        If get, combined with `path`.
        If path is not yet set, download the dataset.
        """
        if self.path is None:
            self.download_dataset()
        return os.path.join(self.path, self._test)

    @property
    def val(self):
        """Get or set the validation folder.

        If get, combined with `path`.
        If path is not yet set, download the dataset.
        """
        if self.path is None:
            self.download_dataset()
        return os.path.join(self.path, self._val)


stego_images_dataset = DatasetFiles(
    "marcozuppelli/stegoimagesdataset",
    os.path.join(*["train"] * 2),
    os.path.join(*["test"] * 2),
    os.path.join(*["val"] * 2),
)


def load_dataset(folder: str) -> tuple[list[str], list[int]]:
    """Load a dataset from `folder`.

    Args:
        folder (str): The folder to get things from

    Returns:
        tuple[list[str], list[int]]: A list of image paths and labels
    """
    clean_folder = os.path.join(folder, "clean")
    stego_folder = os.path.join(folder, "stego")
    clean_images = [
        os.path.join(clean_folder, f)
        for f in os.listdir(clean_folder)
        if f.lower().endswith(".png")
    ]
    stego_images = [
        os.path.join(stego_folder, f)
        for f in os.listdir(stego_folder)
        if f.lower().endswith(".png")
    ]

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    return image_paths, labels


def create_tf_dataset(
    image_paths: list[str], labels: list[int], batch_size=32, buffer_size=1000
) -> DatasetV2:
    """Create a dataset.

    Args:
        image_paths (list[str]): _description_
        labels (list[int]): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        buffer_size (int, optional): _description_. Defaults to 1000.

    Returns:
        DatasetV2: The tensorflow dataset.
    """

    def load_and_preprocess_image(path: str, label: int):
        image = tf.numpy_function(preprocess_image, [path], tf.float32)
        image.set_shape((512, 512, 3))
        return image, label

    # Create a TensorFlow dataset from image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return (
        dataset.shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
