import os

import xgbost as xgb

from src.utils import timeit


class XGBClassificationModel:
    def __init__(self, nvidia_gpu_available:bool, hyperparams:dict) -> None:
        self.nvidia_gpu_available = nvidia_gpu_available
        self.hyperparams = hyperparams
        if self.nvidia_gpu_available:
            self.hyperparams["tree_method"] = "gpu_hist"
        self.model = None

    @timeit
    def fit(self, X_train, y_train):
        self.model = xgb.train(self.hyperparams, xgb.DMatrix(X_train,label=y_train))
        return self

    def predict(self, X):
        assert self.model is None, "Fit the model first to make the predictions."
        probas = self.model.predict(xgb.DMatrix(X), pred_contribs=False).round(2)
        return probas
