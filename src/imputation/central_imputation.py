from typing import Any, Dict, List, Tuple

from imputation.imputation import Imputation
from utils import is_nvidia_gpu_available

NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()
if NVIDIA_GPU_AVAILABILITY:
    from cudf import DataFrame, Series
else:
    from pandas import DataFrame, Series

class CentralImputation(Imputation):
    """Central Imputation where missing values are imputed with mean/mode based on data types.
    """
    def __init__(self, nvidia_gpu_available: bool) -> None:
        """Initializer for CentralImputation.

        Args:
            nvidia_gpu_available (bool): Boolean if nvidia gpu available.
        """
        super().__init__(nvidia_gpu_available)

    def impute_column(self, col_to_impute: Series, mode: str='training') -> Series:
        """Impute a column/series.
        
        Args:
            col_to_impute (Series): Column/Series to impute.
            mode (str): if imputation is being done for 'training' or 'inference'.

        Returns:
            Series: Column/Series with missing values filled with mean/mode based on dtype.
        """
        if (mode=='training') or (col_to_impute.name not in self.impute_info.keys()):
            if col_to_impute.dtype in ['object', 'int']:
                val = col_to_impute.mode()    
            else:
                val = col_to_impute.median()
            self.impute_info[col_to_impute.name] = val
        else:
            val = self.impute_info[col_to_impute.name]
        col_to_impute[col_to_impute.isnull()] = val
        return col_to_impute