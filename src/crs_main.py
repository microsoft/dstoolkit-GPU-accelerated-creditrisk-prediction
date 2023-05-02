import sys
import os
import argparse
from utils import is_nvidia_gpu_available
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, dest="years", help="years", default="2007,2008,2009")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()
    print("NVIDIA GPU Availability: ", NVIDIA_GPU_AVAILABILITY)
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
    from azureml.core import Run, Dataset
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from constants_ import DATA_DOWNLOAD_URL, PERFORMANCE_COLS
    from data_preparation.data_preparation import DataPreparation
    from model.classsification_model import XGBClassificationModel
    from performance_and_metrics.performance_and_metrics import \
        ClassificationReport

    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'credit_risk_data'))
    data_folder_path = './data'
    # Note mount does not work here. Investigate
    os.makedirs(data_folder_path, exist_ok=True)
    dataset.download(target_path=data_folder_path, overwrite=True)
    print("Data downloaded to: ", data_folder_path)
    years = args.years
    years = list(map(int, years.split(",")))

    df = DataPreparation(data_folder_path, years).prepare_credit_risk_data()
    n_rows, n_cols = df.shape
    print(f"Number of rows: {n_rows}, Number of columns: {n_cols}")
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

    shap_values, explainer = reporter.generate_and_log_shap_plot(xgb_model.model, X_test_df)
    reporter.log_importance_plot(xgb_model.model, Xcolumns)
    run.complete()

if __name__ == "__main__":
    main()
