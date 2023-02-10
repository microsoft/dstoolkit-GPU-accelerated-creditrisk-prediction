import sys

from utils import is_nvidia_gpu_available

NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()

if NVIDIA_GPU_AVAILABILITY:
    from cudf import DataFrame, Series, concat, get_dummies, merge, read_csv
    from cuml.preprocessing import OneHotEncoder
    from cupy import concatenate, double, int8, zeros
else:
    from numpy import concatenate, double, int8, zeros
    from pandas import (DataFrame, Series, concat, get_dummies, isnull, merge,
                        read_csv)
    from sklearn.preprocessing import OneHotEncoder
    
data_folder_path = sys.argv[1]
mounted_output_path = sys.argv[2]
n_gpus = int(sys.argv[3])
n_years = int(sys.argv[4])

import os
import sys
import time

import numpy
import pandas as pd

if n_gpus == 0:
    from numpy import concatenate, double, int8, zeros
    from pandas import (DataFrame, Series, concat, get_dummies, isnull, merge,
                        read_csv)
    from sklearn.preprocessing import OneHotEncoder
else:
    print("Importing read_csv, concat, merge, DataFrame, get_dummies, Series using Rapid's Libraries")
    import cudf
    import cuml
    import cupy
    from cudf import DataFrame, Series, concat, get_dummies, merge, read_csv
    from cuml.preprocessing import OneHotEncoder
    from cupy import concatenate, double, int8, zeros

from performance_and_metrics.performance_and_metrics import \
    ClassificationReport

numpy.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import time
import traceback

import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.metrics as metrics
import xgboost as xgb
from azureml.core.run import Run
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from src.constants_ import DATA_DOWNLOAD_URL, PERFORMANCE_COLS
from src.data_download import download_and_extract_data
from src.utils import read_acquisition_files, read_performance_files

run = Run.get_context()    

sample_file_path = os.path.join(data_folder_path, "acq/Acquisition_2007Q1.txt")
if os.path.exists(sample_file_path):
    # if data is already downloaded to present folder use that
    print(f"Using the data already downloaded at {data_folder_path}. Skipping fresh download.")
else:
    print("Downloading and extracting data...")
    download_and_extract_data(DATA_DOWNLOAD_URL, data_folder_path)
    print("Downloading and extracting data completed")
from src.data_preparation.data_preparation import DataPreparation
from src.model.classsification_model import XGBClassificationModel

df = DataPreparation().prepare_credit_risk_data()
n_rows, n_cols = df.shape

y = df['Default'].values
X = df.drop(['Default','LoanID'], axis=1).values

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
del X,y
print("Types of X and y:")
print(type(X_train),type(y_train))

print("X_train and y_train shapes:")
print(X_train.shape,y_train.shape)

if n_gpus>=1:
    X_train, y_train = (cupy.asnumpy(X_train), cupy.asnumpy(y_train))
    X_test, y_test =  (cupy.asnumpy(X_test), cupy.asnumpy(y_test))

y = df['Default']
X = df.drop(['Default','LoanID'], axis=1)
Xcolumns = df.columns.tolist()
Xcolumns.remove('Default')
Xcolumns.remove('LoanID')

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

reporter = ClassificationReport(run, classification_proba_threshold)
reporter.log_ROC_image(y_test, classification_probas)


X_test_df = DataFrame(data=X_test,columns=Xcolumns)
y_test    = DataFrame(data=y_test,columns=['Default'])
print(X_test_df.shape,type(X_test_df))

reporter.generate_and_log_shap_plot(xgb_model, X_test_df)
