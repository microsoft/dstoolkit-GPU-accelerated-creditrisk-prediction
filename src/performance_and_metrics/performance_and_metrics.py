import os
import time
import traceback
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from azureml.core import Run
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance

from utils import timeit


class ClassificationReport:
    def __init__(self, run: Run, nvidia_gpu_available: bool, threshold: float) -> None:
        """
        This class is used to generate metrics and plots for classification models.

        Args:
            run (Run): Azure ML run object.
            nvidia_gpu_available (bool): Boolean if nvidia gpu available.
            threshold (float): Threshold for classification.

        Returns:
            None
        """
        self.run = run
        self.nvidia_gpu_available = nvidia_gpu_available
        self.threshold = threshold

        self._asset_path = "/assets/"
        try:
            if not os.path.exists(self._asset_path):
                os.makedirs(self._asset_path)
        except Exception as e:
            print("Error in generating the folder to store assets.")
            traceback.print_exc()

    def log_ROC_image(self, actuals: list, probas: list) -> None:
        """
        This method is used to generate and log the ROC curve.

        Args:
            actuals (list): List of actual values.
            probas (list): List of predicted probabilities.

        Returns:
            None
        """
        try:
            preds = probas > self.threshold
            fpr, tpr, _ = metrics.roc_curve(actuals, preds)
            roc_auc = metrics.auc(fpr, tpr)

            plt.rcParams["figure.figsize"] = [10, 6]
            plt.title("Receiver Operating Characteristic")
            plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")

            fig_path = os.path.join(self._asset_path, "ROC.png")
            plt.savefig(fig_path, bbox_inches="tight")
            self.run.log_image(
                name="ROC",
                path=fig_path,
                plot=None,
                description="Receiver Operating Characteristics",
            )
        except Exception as e:
            print("Error in logging ROC curve")
            traceback.print_exc()

    @timeit
    def generate_and_log_shap_plot(
        self, model: object, X: object
    ) -> Tuple[shap.TreeExplainer.shap_values, shap.TreeExplainer]:
        """
        This method is used to generate and log the SHAP plot.

        Args:
            model (object): Model object.
            X (object): Dataframe containing features.

        Returns:
            Tuple[shap.TreeExplainer.shap_values, shap.TreeExplainer]: SHAP values and explainer object.
        """
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            print("Shape of SHAP values:", shap_values.shape)

            plt.figure()
            shap_plot = shap.summary_plot(shap_values, X, show=False)

            shap_plot_path = os.path.join(self._asset_path, "shap.png")
            plt.savefig(shap_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="SHAP", path=shap_plot_path, plot=None, description="SHAP"
            )
            return shap_values, explainer
        except Exception as e:
            print("Error in generating/logging SHAP plot/values.")
            traceback.print_exc()
            return (None, None)

    def log_shap_scatter_plot(self, shap_values: object, X: object) -> None:
        """
        This method is used to generate and log the SHAP scatter plot.

        Args:
            shap_values (object): SHAP values.
            X (object): Dataframe containing features.

        Returns:
            None
        """
        try:
            plt.figure()
            shap_scatter_plot = shap.dependence_plot(0, shap_values, X)
            shap_scatter_plot_path = os.path.join(self._asset_path, "shap_scatter.png")
            plt.savefig(shap_scatter_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="SHAP_Scatter",
                path=shap_scatter_plot_path,
                plot=None,
                description="SHAP Scatter Plot",
            )
        except Exception as e:
            print("Error in generating/logging SHAP scatter plot")
            traceback.print_exc()

    def log_importance_plot(self, model: object) -> None:
        """
        This method is used to generate and log the importance score plot.

        Args:
            model (object): Model object.

        Returns:
            None
        """
        try:
            importance_fig_path = os.path.join(self._asset_path, "importance.png")
            fig, ax = plt.subplots(nrows=1, ncols=1)
            x = plt.rcParams["figure.figsize"] = [10, 6]
            ax = plot_importance(model, max_num_features=10)
            ax.figure.tight_layout()
            ax.figure.savefig(importance_fig_path)
            self.run.log_image(
                name="Importance Scores",
                path=importance_fig_path,
                plot=None,
                description="XGBoost Importance Scores",
            )
        except Exception as e:
            print("Error in logging importance score plot")
            traceback.print_exc()

    def log_single_dependence_plots(
        self, shap_values: object, feature: str, index: int, X: object
    ) -> None:
        """
        This method is used to generate and log the SHAP dependence plot.

        Args:
            shap_values (object): SHAP values.
            feature (str): Feature name.
            index (int): Index of the feature.
            X (object): Dataframe containing features.

        Returns:
            None
        """
        try:
            start = time.time()
            plt.figure()
            shap.dependence_plot(feature, shap_values, X, interaction_index=index)
            shap_plot_path = os.path.join(
                self._asset_path, "shap_dependence_feature_{}.png".format(feature)
            )
            plt.savefig(shap_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="shap_dependence_{}_index_{}".format(feature, index),
                path=shap_plot_path,
                plot=None,
                description="SHAP shap dependence plot for feature {} with interaction index {}".format(
                    feature, index
                ),
            )
            print(round(time.time() - start, 3), "secs time for SHAP dependence plot")
        except Exception as e:
            print("Error in generating/logging SHAP dependence plot")

    def log_dependence_plots(
        self, shap_values: object, number_important_feat: int, X: object
    ) -> None:
        """
        This method is used to generate and log the SHAP dependence plots.

        Args:
            shap_values (object): SHAP values.
            number_important_feat (int): Number of important features.
            X (object): Dataframe containing features.

        Returns:
            None
        """
        try:
            # Compute absolute SHAP values
            abs_shap_values = np.abs(shap_values)

            # Compute feature importance scores by summing SHAP values over all samples
            feat_importance_scores = np.sum(abs_shap_values, axis=0)

            # Get the top "number_important_feat" features by importance
            top_feats = np.argsort(feat_importance_scores)[::-1][:number_important_feat]
            i = 0
            for feat in top_feats:
                start = time.time()
                plt.figure()
                shap.dependence_plot(feat, shap_values, X)
                shap_plot_path = os.path.join(
                    self._asset_path, "shap_dependence_imp_feature_{}.png".format(i)
                )
                plt.savefig(shap_plot_path, bbox_inches="tight")
                self.run.log_image(
                    name="shap_dependence_imp_feature_{}".format(feat),
                    path=shap_plot_path,
                    plot=None,
                    description="SHAP shap dependence plot for important feature number {}".format(
                        feat
                    ),
                )
                print(
                    round(time.time() - start, 3), "secs time for SHAP dependence plot"
                )
                i = i + 1
        except Exception as e:
            print("Error in generating/logging SHAP dependence plot")

    def log_force_plot(self, model: object, row: int, X: object) -> None:
        """
        This method is used to generate and log the SHAP force plot for a record.

        Args:
            model (object): Model object.
            row (int): Row number.
            X (object): Dataframe containing features.

        Returns:
            None
        """
        try:
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X)
            start = time.time()
            plt.figure()
            shap.force_plot(
                expl.expected_value,
                shap_values[row, :],
                X.iloc[row, :],
                show=False,
                matplotlib=True,
            )
            shap_plot_path = os.path.join(
                self._asset_path, "shapforce_row_{}.png".format(row)
            )
            plt.savefig(shap_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="SHAP Force row {}".format(row),
                path=shap_plot_path,
                plot=None,
                description="SHAP force plot for observation {}".format(row),
            )
            print(round(time.time() - start, 3), "secs time for SHAP force plot")

        except Exception as e:
            print("Error in generating/logging SHAP force plot")

    def log_decision_plot(self, model: object, X: object, n: int) -> None:
        """
        This method is used to generate and log the SHAP decision plot for first n observations.

        Args:
            model (object): Model object.
            X (object): Dataframe containing features.
            n (int): Number of observations.

        Returns:
            None
        """
        try:
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X)

            start = time.time()
            plt.figure()
            shap.initjs()
            shap.decision_plot(
                expl.expected_value,
                shap_values[0:n],
                feature_names=list(X.columns),
            )
            shap_plot_path = os.path.join(self._asset_path, "shapdecision.png")
            plt.savefig(shap_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="SHAP decision",
                path=shap_plot_path,
                plot=None,
                description="SHAP decision plot",
            )
            print(round(time.time() - start, 3), "secs time for SHAP decision plot")

        except Exception as e:
            print("Error in generating/logging SHAP decision plot")

    def log_waterfall_plot(
        self, model: object, shap_values: object, X: object, row: int
    ) -> None:
        """
        This method is used to generate and log the SHAP waterfall plot for a record.

        Args:
            model (object): Model object.
            shap_values (object): SHAP values.
            X (object): Dataframe containing features.
            row (int): Row number.

        Returns:
            None
        """
        try:
            start = time.time()
            plt.figure()
            shap_exp = shap.Explanation(
                values=shap_values,
                base_values=model.predict(xgb.DMatrix(X)),
                feature_names=X.columns,
            )
            shap.waterfall_plot(shap_exp[row], max_display=10, show=False)
            shap_plot_path = os.path.join(
                self._asset_path, "shap_waterfall_row_{}.png".format(row)
            )
            plt.savefig(shap_plot_path, bbox_inches="tight")
            self.run.log_image(
                name="SHAP_Waterfall {}".format(row),
                path=shap_plot_path,
                plot=None,
                description="SHAP Waterfall Plot for observation {}".format(row),
            )
            print(round(time.time() - start, 3), "secs time for SHAP waterfall plot")
        except Exception as e:
            print("Error in generating/logging SHAP waterfall plot")

    def _fetch_metrics(self, y_true: object, predictions: object) -> tuple:
        """
        This helper method is used to fetch the classification report, confusion matrix and ROC AUC score.

        Args:
            y_true (object): True labels.
            predictions (object): Predicted labels.

        Returns:
            tuple: classification report, confusion matrix and ROC AUC score.
        """
        classification_report_ = classification_report(y_true, predictions)
        conf_matrix = confusion_matrix(y_true, predictions)

        fpr, tpr, _ = metrics.roc_curve(y_true, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        return classification_report_, conf_matrix, roc_auc

    def get_metrics(self, y_true: object, probas: object) -> tuple:
        """
        This method is used to fetch and print the classification report, confusion matrix and ROC AUC score.

        Args:
            y_true (object): True labels.
            probas (object): Predicted probabilities.

        Returns:
            tuple: classification report, confusion matrix and ROC AUC score.
        """
        try:
            predictions = probas > self.threshold
            classification_report_, conf_matrix, roc_auc = self._fetch_metrics(
                y_true, predictions
            )
        except Exception as e:
            print("Error in generating classification report")
            traceback.print_exc()
            return None, None, None
        try:
            print("\nTest Classification Report:\n", classification_report_)
        except Exception as e:
            print("Error in logging classification report")
            traceback.print_exc()
            return None, None, None

        try:
            print("\nTest Confusion Matrix:\n", conf_matrix)
        except Exception as e:
            print("Error in logging confusion matrix")
            traceback.print_exc()
            return None, None, None

        print("\nTest ROC AUC Score: ", roc_auc)

        return classification_report_, conf_matrix, roc_auc

    def generate_metrics_plots(
        self,
        X_test_df: object,
        y_test: object,
        model: object,
        probas: object,
        run_all: bool = False,
    ) -> None:
        """
        This method is used to generate metrics and log the plots including those related to explainability.

        Args:
            X_test_df (object): Dataframe containing features.
            y_test (object): True labels.
            model (object): Model object.
            probas (object): Predicted probabilities.
            run_all (bool): Flag to run all explainability plots.

        Returns:
            None
        """
        self.get_metrics(y_test, probas)
        self.log_ROC_image(y_test, probas)

        shap_values, explainer = self.generate_and_log_shap_plot(model, X_test_df)
        if run_all:
            self.log_dependence_plots(shap_values, 2, X_test_df)
            self.log_importance_plot(model)
            self.log_force_plot(model, 0, X_test_df)
            self.log_decision_plot(model, X_test_df, 10)
            self.log_waterfall_plot(model, shap_values, X_test_df, 0)
