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

isPlot = False
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
numpy.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

try:
    import shap
except Exception as e:
    print(e)
import tarfile
import time
import traceback
import urllib.request

import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from azureml.core.run import Run
from imblearn.combine import SMOTEENN
# %matplotlib inlineimport cupy
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

from constants_ import DATA_DOWNLOAD_URL, PERFORMANCE_COLS, col_acq, dtypesDict
from data_download import download_and_extract_data
from utils import read_acquisition_files, read_performance_files

# import pathlib
# sys.path.append(pathlib.Path().resolve())





run = Run.get_context()    

YEARS = (2007 + np.arange(n_years)).tolist()

dir_path = os.path.dirname(os.path.realpath(__file__))
# print(f"Directory path is {dir_path}")

dtype = list(dtypesDict.values())
cwd = os.getcwd()
# print(f"cwd:{cwd}")
# os.chdir('/rapids/data')

# print("Data Types:", dtypesDict)
run.log('col_acq list is len',len(col_acq))

sample_file_path = os.path.join(data_folder_path, "acq/Acquisition_2007Q1.txt")
# print(f"sample_file_path = {sample_file_path}")
if os.path.exists(sample_file_path):
    # if data is already downloaded to present folder use that
    print(f"Using the data already downloaded at {data_folder_path}. Skipping fresh download.")
else:
    print("Downloading and extracting data...")
    download_and_extract_data(DATA_DOWNLOAD_URL, data_folder_path)
    print("Downloading and extracting data completed")



y = df['Default'].values
X = df.drop(['Default','LoanID'], axis=1).values

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
del X,y
print("Types of X and y:")
print(type(X_train),type(y_train))

print("X_train and y_train shapes:")
print(X_train.shape,y_train.shape)

# print("Applying Random Oversampling...")
# ros = RandomOverSampler(sampling_strategy=.93) #Apply to only X_train, y_train
# X_train, y_train = ros.fit_resample(cupy.asnumpy(X_train), cupy.asnumpy(y_train))
if n_gpus>=1:
    X_train, y_train = (cupy.asnumpy(X_train), cupy.asnumpy(y_train))
    X_test, y_test =  (cupy.asnumpy(X_test), cupy.asnumpy(y_test))
#sm = SMOTEENN()
#X_train, y_train = sm.fit_sample(cupy.asnumpy(X_train), cupy.asnumpy(y_train))


y = df['Default'] #no .values
X = df.drop(['Default','LoanID'], axis=1) #no .values
Xcolumns = df.columns.tolist()
Xcolumns.remove('Default')
Xcolumns.remove('LoanID')

## XGB Training and Inference
print('Start Training')
start = time.time()
if n_gpus == 0:
    model = xgb.train({"num_boost_round": 500, "learning_rate": 0.01, "max_depth":10, "scale_pos_weight":5.5},
            xgb.DMatrix(X_train,label=y_train))#,num_boost_round=500)
else:
    model = xgb.train({"num_boost_round": 500, "learning_rate": 0.01, "max_depth":10, "tree_method":"gpu_hist", "scale_pos_weight":5.5},
            xgb.DMatrix(X_train,label=y_train))#,num_boost_round=500)
# print(type(model))
ttrain = time.time()-start
print(round(ttrain, 2),'secs time for training')


start = time.time()
#Alternative equivalent ways to predict()
#was: probs = model.predict_proba(X_test)[:,1]
xtest = xgb.DMatrix(X_test)
probs = model.predict(xtest, pred_contribs=False).round(2)
tpred = time.time()-start
print(round(tpred, 3),'secs time for Inferencing')



threshold = .5
y_pred = (probs > threshold) # <=================== settable was: .85
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

try:
    #was:probs = model.predict_proba(X_test)
    preds = probs
    #was:preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.rcParams["figure.figsize"] = [10,6]
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    asset_path = "/assets/"
    if not os.path.exists(asset_path):
        os.makedirs(asset_path)
    fig_path = os.path.join(asset_path, 'ROC.png')
    plt.savefig(fig_path, bbox_inches='tight')
    run.log_image(name='ROC', path=fig_path, plot=None, description='Receiver Operating Characteristics')
except Exception as e:
    print("Error in logging ROC curve")
    traceback.print_exc()


# try:
#     importance_fig_path = os.path.join(asset_path, 'importance.png')
#     # fig, ax = plt.subplots( nrows=1, ncols=1)
#     # x = plt.rcParams["figure.figsize"] = [10,6]
#     ax = plot_importance(model,max_num_features=10, feature_names=Xcolumns)
#     ax.figure.tight_layout()
#     ax.figure.savefig(importance_fig_path)

#     # ax.plot(importance_plot)
#     # fig.savefig(importance_fig_path)
#     # plt.close(fig)
#     run.log_image(name='Importance Scores', path=importance_fig_path, plot=None, description='XGBoost Importance Scores')
# except Exception as e:
#     print("Error in logging importance score plot")
#     traceback.print_exc()



X_test_df = DataFrame(data=X_test,columns=Xcolumns)
y_test    = DataFrame(data=y_test,columns=['Default'])
print(X_test_df.shape,type(X_test_df))

# print('n_gpus',n_gpus)

try:
    start = time.time()
    print("SHAP...")
    import shap
    if n_gpus == 0:
        model.set_param({"predictor": "cpu_predictor"})
    else:
        model.set_param({"predictor": "gpu_predictor"})
    expl = shap.TreeExplainer(model)
    shap_values = expl.shap_values(X_test_df)
    tshap = time.time()-start
    print("Time for calculating SHAP:", tshap,'secs time')

    print("Shape of SHAP values:", shap_values.shape)
    print("XGB:", xgb.__version__)
except Exception as e:
    print("Error in SHAP")
    traceback.print_exc()

try:
    start = time.time()
    plt.figure()
    if n_gpus == 0: shap_plot = shap.summary_plot(shap_values, X_test_df,show=False)
    else: shap_plot = shap.summary_plot(shap_values, X_test_df.to_pandas(),show=False)
    shap_plot_path = os.path.join(asset_path, 'shap.png')
    plt.savefig(shap_plot_path, bbox_inches='tight')
    run.log_image(name='SHAP', path=shap_plot_path, plot=None, description='SHAP')

    print(round(time.time()-start, 3),'secs time for SHAP plot')   
except Exception as e:
    print("Error in generating/logging SHAP plot")
    traceback.print_exc()

# try:
#     start = time.time()
#     plt.figure()
#     shap_scatter_plot = shap.plots.scatter(shap_values[:,0]) #, color=shap_values
    
#     shap_scatter_plot_path = os.path.join(asset_path, 'shap_scatter.png')
#     plt.savefig(shap_scatter_plot_path, bbox_inches='tight')
#     run.log_image(name='SHAP_Scatter', path=shap_scatter_plot_path, plot=None, description='SHAP Scatter Plot')

#     print(round(time.time()-start, 3),'secs time for SHAP scatter plot')   
# except Exception as e:
#     print("Error in generating/logging SHAP scatter plot")
#     traceback.print_exc()


# try:
#     start = time.time()
#     plt.figure()
#     shap_scatter_plot = shap.force_plot(expl.expected_value, shap_values[0],show=False)
#     # shap_scatter_plot = shap.plots.force(shap_values[0])
    
#     shap_force_plot_path = os.path.join(asset_path, 'shap_force.png')
#     plt.savefig(shap_force_plot_path, bbox_inches='tight')
#     run.log_image(name='SHAP_Force', path=shap_force_plot_path, plot=None, description='SHAP Force Plot')

#     print(round(time.time()-start, 3),'secs time for SHAP force plot')   
# except Exception as e:
#     print("Error in generating/logging SHAP force plot")
#     traceback.print_exc()

profiles = pd.DataFrame(
    {
        'Task':['Read & Append', 'Read & Append Acq', 'Read & Append Perf', 'Merging', 'Imputation', 'Dummies', 'Model Training', 'Inference', 'SHAP Calculation'],
        'Profile':[tread_acq+tread_per, tread_acq, tread_per, tmerg, timpute, tdummies, ttrain, tpred, tshap],
    },
)
profiles['Profile'] = profiles['Profile'].round(3)
profiles['n_gpus'] = n_gpus
profiles['n_rows'] = n_rows
profiles['n_cols'] = n_cols
print(profiles)
print(Xcolumns)
print(expl.expected_value)
# print(shap_values)

# compute = 'cpu' if n_gpus==0 else 'gpu'
# op_fname = f'compute_{compute}_years_{"-".join(map(str, YEARS))}.csv'
# op_full_path = os.path.join(mounted_output_path, op_fname)
# profiles.to_csv(op_full_path, index=False)
# print(f"Profiles has been written successfully to '{op_full_path}'")
