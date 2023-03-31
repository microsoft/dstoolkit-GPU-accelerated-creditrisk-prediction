from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from utils import is_nvidia_gpu_available

NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()

if NVIDIA_GPU_AVAILABILITY:
    from cudf import DataFrame, Series
else:
    from pandas import DataFrame, Series

class Imputation:
    """Imputation base class. Other classes can be extended by implementing 'impute_column' method.
    To add a new class for imputation, just override the `impute_column` method."""
    def __init__(self, nvidia_gpu_available: bool) -> None:
        """Initializer for Imputaation class.

        Args:
            nvidia_gpu_available (bool): Boolean if nvidia gpu available.
        """
        self.nvidia_gpu_available = nvidia_gpu_available
        self.impute_info = {} # to store the values to impute for inference
    
    @abstractmethod
    def impute_column(self, col_to_impute: Series, mode: str='training') -> Series:
        """Abstract method for imputing a column or series. Override this method when implementing new imputation classes.
        
        Args:
            col_to_impute (Series): Column or series whose null values are to imputed.
            mode (str): if imputation is being done for 'training' or 'inference'.

        Returns:
            Series: Series or column with imputed values.
        """
        pass

    def impute(self, df: DataFrame, mode: str='training') -> DataFrame:
        """Impute all the columns of the dataframe.

        Args:
            df (DataFrame): Dataframe whose columns are to be imputed. 
                Utilizes the 'impute_column' method for imputing each of the columns.
            mode (str): if imputation is being done for 'training' or 'inference'.
        
        Returns:
            DataFrame: Dataframe with all columns imputes.
        """
        if not self.nvidia_gpu_available:
            columns = df.columns[df.isnull().any().tolist()]
        else: 
            columns = df.columns[df.isnull().any().to_arrow().to_pylist()]
        for col in columns:
            df[col] = self.impute_column(df[col])
        return df 
