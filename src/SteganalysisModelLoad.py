"""Load a model, then predict."""

import numpy as np
from keras.models import Model, load_model


class SteganalysisModel:
    """A model that can predict images."""

    def __init__(self, model_path: str):
        """Open a steganalysis model.

        Args:
            model_path (str): The path to the model
        """
        self.model: Model = load_model(model_path)

    def predict(self, images):
        """Predict whther the images contain steganalysis.

        Args:
            images (_type_): _description_

        Returns:
            _type_: _description_
        """
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)
