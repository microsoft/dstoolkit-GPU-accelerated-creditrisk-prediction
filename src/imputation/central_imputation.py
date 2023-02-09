from imputation import Imputation


class CentralImputation(Imputation):
    """Central Imputation where missing values are imputed with mean/mode based on data types.
    """
    def __init__(self) -> None:
        """Initializer for CentralImputation."""
        super().__init__()

    def impute_column(self, col_to_impute: Series) -> Series:
        """Impute a column/series.
        
        Args:
            col_to_impute (Series): Column/Series to impute.

        Returns:
            Series: Column/Series with missing values filled with mean/mode based of dtype.
        """
        if col_to_impute.dtype == 'object':
           col_to_impute[col_to_impute.isnull()] = col_to_impute.mode() 
        else:
            col_to_impute[col_to_impute.isnull()] = col_to_impute.median()
        return col_to_impute