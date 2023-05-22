import os

import xgboost as xgb

from utils import timeit


class XGBClassificationModel:
    def __init__(self, nvidia_gpu_available: bool, hyperparams: dict) -> None:
        """
        Args:
            nvidia_gpu_available (bool): Boolean if nvidia gpu available.
            hyperparams (dict): Hyperparameters for the model.

        Returns:
            None
        """
        self.nvidia_gpu_available = nvidia_gpu_available
        self.hyperparams = hyperparams
        if self.nvidia_gpu_available:
            self.hyperparams["tree_method"] = "gpu_hist"
        self.model = None

    @timeit
    def fit(self, X_train: DaaFrame, y_train: Series) -> object:
        """Fit the model to the training data.

        Args:
            X_train (DataFrame): Training data.
            y_train (Series): Training labels.

        Returns:
            self
        """
        self.model = xgb.train(self.hyperparams, xgb.DMatrix(X_train, label=y_train))
        return self

    def predict(self, X: DataFrame) -> Series:
        """Make predictions on the test data.

        Args:
            X (DataFrame): Test data.

        Returns:
            Series: Predictions on the test data.
        """
        assert self.model is not None, "Fit the model first to make the predictions."
        probas = self.model.predict(xgb.DMatrix(X), pred_contribs=False).round(2)
        return probas
