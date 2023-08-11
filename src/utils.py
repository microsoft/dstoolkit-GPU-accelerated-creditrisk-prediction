import os
import subprocess
from functools import wraps
from time import time

from azureml.core import Dataset


def is_nvidia_gpu_available() -> bool:
    """Check if NVIDIA GPU is available.

    Returns:
        bool: True if NVIDIA GPU is available, else False.
    """
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False


def timeit(f: object) -> object:
    """Decorator to time a function.

    Args:
        f (function): Function to time.

    Returns:
        object: Decorated function.
    """

    @wraps(f)
    def timer(*args, **kw):
        start = time()
        result = f(*args, **kw)
        total_time = time() - start
        print(f"Function {f.__name__} took {total_time} seconds with args {args}")
        return result

    return timer


def get_dataframe_summary(f: object) -> object:
    """Decorator to print a summary of a dataframe.

    Args:
        f (function): Function that returns a dataframe.

    Returns:
        object: Decorated function.
    """

    @wraps(f)
    def summarizer(*args, **kw):
        df = f(*args, **kw)
        print("Dataframe shape: ", df.shape)
        print("\nSample DataFrame Rows:")
        print(df.sample(5, random_state=10))
        print(df.info())
        return df

    return summarizer


def download_from_datastore(datastore: str, data_folder_path: str) -> None:
    """Downloads data from a datastore to a directory.

    Args:
        datastore (str): Name of the datastore.
        data_folder_path (str): Path to the directory where the data will be downloaded.

    Returns:
        None
    """
    dataset = Dataset.File.from_files(path=(datastore, "credit_risk_data"))
    os.makedirs(data_folder_path, exist_ok=True)
    # Note mount does not work here. Investigate
    dataset.download(target_path=data_folder_path, overwrite=True)
    print("Data downloaded to: ", data_folder_path)
