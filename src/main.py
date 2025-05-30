"""The main file. Intended to automatically run StegaScanMail."""

import logging

import tensorflow as tf

from Dataset import stego_images_dataset
from fine_tuned_cnn import train

logger = logging.getLogger(__name__)


def __main():
    logging.basicConfig(filename="main.log", level="INFO")
    logger.debug("Training devices available: %d", tf.config.list_physical_devices())
    stego_images_dataset.download_dataset()
    logger.info("Downloaded dataset")
    train(stego_images_dataset)


if __name__ == "__main__":
    __main()
