import sys

from utils import is_nvidia_gpu_available

NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()

if NVIDIA_GPU_AVAILABILITY:
    import cupy
    from cudf import DataFrame, Series, concat, get_dummies, merge, read_csv
else:
    from numpy import concatenate, double, int8, zeros
    from pandas import (DataFrame, Series, concat, get_dummies, isnull, merge,
                        read_csv)
    from sklearn.preprocessing import OneHotEncoder

import traceback

import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import xgboost as xgb
from azureml.core.run import Run
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from constants_ import DATA_DOWNLOAD_URL, PERFORMANCE_COLS
from data_download import download_and_extract_data
from data_preparation.data_preparation import DataPreparation
from model.classsification_model import XGBClassificationModel
from performance_and_metrics.performance_and_metrics import \
    ClassificationReport
# import streamlit as st
# from streamlit_shap import st_shap
import shap
import joblib

run = Run.get_context()
data_folder_path = sys.argv[1]
years = sys.argv[2]
years = list(map(int, years.split(",")))

df = DataPreparation(data_folder_path, years).prepare_credit_risk_data()
n_rows, n_cols = df.shape

df = get_dummies(df)

y = df['Default'].values
X = df.drop(['Default','LoanID'], axis=1).values
Xcolumns = df.columns.tolist()
Xcolumns.remove('Default')
Xcolumns.remove('LoanID')

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
del X,y
print("Types of X and y:")
print(type(X_train),type(y_train))

print("X_train and y_train shapes:")
print(X_train.shape,y_train.shape)

if NVIDIA_GPU_AVAILABILITY:
    X_train, y_train = (cupy.asnumpy(X_train), cupy.asnumpy(y_train))
    X_test, y_test =  (cupy.asnumpy(X_test), cupy.asnumpy(y_test))


hyperparams = {"num_boost_round": 500, "learning_rate": 0.01, "max_depth":10, "scale_pos_weight":5.5}
xgb_model = XGBClassificationModel(NVIDIA_GPU_AVAILABILITY, hyperparams).fit(X_train,y_train)
classification_probas = xgb_model.predict(X_test)


classification_proba_threshold = .5
y_pred = (classification_probas > classification_proba_threshold)
try:
    print("\nTest Classification Report:\n", classification_report(y_test, y_pred))
except Exception as e:
    print("Error in logging classification report")
    traceback.print_exc()

try:
    print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
except Exception as e:
    print("Error in logging confusion matrix")
    traceback.print_exc()

reporter = ClassificationReport(run, classification_proba_threshold, 0.5)
reporter.log_ROC_image(y_test, classification_probas)


X_test_df = DataFrame(data=X_test,columns=Xcolumns)
y_test    = DataFrame(data=y_test,columns=['Default'])
print(X_test_df.shape,type(X_test_df))


# adding shap plots
shap_values, explainer = reporter.generate_and_log_shap_plot(xgb_model.model, X_test_df)
reporter.log_importance_plot(xgb_model.model, Xcolumns)
reporter.log_single_dependence_plots(shap_values ,"OrLoanTerm","CreditScore",X_test_df)
reporter.log_dependence_plots(shap_values ,5,X_test_df)
reporter.log_force_plot(xgb_model.model,100,X_test_df)
reporter.log_decision_plot(xgb_model.model,X_test_df,100)
reporter.log_waterfall_plot(xgb_model.model,shap_values,X_test_df,100)

