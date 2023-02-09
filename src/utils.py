import os
import subprocess
from functools import wraps
from time import time


def is_nvidia_gpu_available():
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception:
        return False

def timeit(f):
    @wraps(f)
    def timer(*args, **kw):
        start = time()
        result = f(*args, **kw)
        total_time = time() - start
        print(f"Function {f.__name__} took {total_time} seconds with args {args}")
        return result
    return timer

def get_dataframe_summary(f):
    @wraps(f)
    def summarizer(*args, **kw):
        df = f(*args, **kw)
        print("Dataframe shape: ", df.shape)
        print("\nSample DataFrame Rows:")
        print(df.sample(5, random_state=10))
        print(df.info())
        return df
    return summarizer