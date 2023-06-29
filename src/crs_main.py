import argparse
import sys

from model_config import CLASSIFICATION_PROBA_THRESHOLD, HYPERPARAMS, TEST_SIZE
from utils import download_from_datastore, is_nvidia_gpu_available


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=str, dest="years", help="years", default="2007,2008,2009"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function.

    Args:
        None

    Returns:
        None
    """
    try:
        args = parse_args()
        years = args.years
    except:
        years = sys.argv[1]
    years = list(map(int, years.split(",")))

    NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()
    print("NVIDIA GPU Availability: ", NVIDIA_GPU_AVAILABILITY)
    if NVIDIA_GPU_AVAILABILITY:
        import cupy
        from cudf import DataFrame, get_dummies
    else:
        from pandas import DataFrame, get_dummies
    import pandas as pd
    from azureml.core import Dataset, Run
    from sklearn.model_selection import train_test_split

    from constants_ import (
        BACKUP_DATA_FOLDER_NAME,
        DATASTORE_FOLDER_NAME,
        ID_COL_NAME,
        N_SAMPLES,
        TARGET_COL_NAME,
        CATEGORICAL_COLS,
    )
    from data_preparation.data_preparation import DataPreparation
    from model.classsification_model import XGBClassificationModel
    from performance_and_metrics.performance_and_metrics import ClassificationReport

    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = ws.get_default_datastore()

    try:
        data_folder_path = Dataset.File.from_files(
            datastore.path(DATASTORE_FOLDER_NAME)
        ).as_mount()
        df_raw = DataPreparation(data_folder_path, years).prepare_credit_risk_data()
    except:
        print("Downloading data from datastore...")
        download_from_datastore(datastore, BACKUP_DATA_FOLDER_NAME)
        df_raw = DataPreparation(BACKUP_DATA_FOLDER_NAME, years).prepare_credit_risk_data()

    df = get_dummies(df_raw)

    y = df[TARGET_COL_NAME].values
    X = df.drop([TARGET_COL_NAME, ID_COL_NAME], axis=1).values
    Xcolumns = df.drop(TARGET_COL_NAME, axis=1).columns.tolist()
    Xcolumns.remove(ID_COL_NAME)

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=0
    )
    del X, y
    print("Types of X and y:")
    print(type(X_train), type(y_train))

    print("X_train and y_train shapes:")
    print(X_train.shape, y_train.shape)

    if NVIDIA_GPU_AVAILABILITY:
        X_train, y_train = (cupy.asnumpy(X_train), cupy.asnumpy(y_train))
        X_test, y_test = (
            cupy.asnumpy(X_test[:N_SAMPLES]),
            cupy.asnumpy(y_test[:N_SAMPLES]),
        )
    X_test_df = pd.DataFrame(data=X_test, columns=Xcolumns)
    y_test = pd.DataFrame(data=y_test, columns=[TARGET_COL_NAME])

    xgb_model = XGBClassificationModel(NVIDIA_GPU_AVAILABILITY, HYPERPARAMS).fit(
        X_train, y_train
    )
    classification_probas = xgb_model.predict(X_test)

    reporter = ClassificationReport(
        run, NVIDIA_GPU_AVAILABILITY, CLASSIFICATION_PROBA_THRESHOLD
    )
    reporter.generate_metrics_plots(
        X_test_df, y_test, xgb_model.model, classification_probas
    )

    reporter.stack_powerBi_table(
        model=xgb_model.model,
        X_test_df=X_test_df,
        y_test=y_test,
        probas=classification_probas,
        X_raw=df_raw,
        categorical_cols=CATEGORICAL_COLS,
        save_Table=True
    )
    run.complete()

if __name__ == "__main__":
    main()
