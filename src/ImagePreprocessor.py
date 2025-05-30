"""Image preprocessor.

Changes images to be the right siz, colour etc.
"""

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Image preprocessor.

    Set to a certain size.
    """

    def __init__(self, target_size=(512, 512)):
        """Initialise a ImagePreprocessor to `target_size` x `target_size`.

        Args:
            target_size (tuple, optional): The target size for images.
            Defaults to (512, 512).
        """
        self.target_size = target_size

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a single image. Returns an numpy array.

        Sets it to RGB channels and to be `target_size`.
        Also normalises the values to [0, 1]

        Args:
            image (Image): The image to modify

        Returns:
            np.ndarray: Resulting array.
        """
        image = image.convert("RGB")
        image = image.resize(self.target_size)
        res = np.array(image).astype(np.float32)
        res /= 255.0  # Normalize to [0, 1]
        return res

    def preprocess_images(self, images: list[Image.Image]) -> np.ndarray:
        """Preprocess multiple images.

        Args:
            images (list[Image]): The images to process

        Returns:
            np.ndarray: The list of processed images. A list of arrays.
        """
        return np.array([self.preprocess_image(image) for image in images])


def preprocess_image(image_path: str) -> np.ndarray[np.float32]:
    """Preprocess an image.

    Set size to 512x512 and convert to array.

    Args:
        image_path (str): The path to the image

    Returns:
        _type_: _description_
    """
    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    # Resize image to 512x512 and convert to numpy array
    image = np.array(image).reshape(512, 512, 3)
    return image.astype(np.float32)
