from abc import abstractmethod


class Imputation:
    """Imputation base class. Other classes can be extended by implementing 'impute_column' method."""
    def __init__(self, nvidia_gpu_available: bool) -> None:
        """

        Args:
            nvidia_gpu_available (bool): Boolean if nvidia gpu available.
        """
        self.nvidia_gpu_available = nvidia_gpu_available
    
    @abstractmethod
    def impute_column(self, col_to_impute: Series) -> Series:
        """Abstract method for imputing a column or series. Override this method when implementing new imputation classes.
        
        Args:
            col_to_impute (Series): Column or series whose null values are to imputed.

        Returns:
            Series: Series or column with imputed values.
        """
        pass

    def impute(self, df: DataFrame) -> DataFrame:
        """Impute all the columns of the dataframe.

        Args:
            df (DataFrame): Dataframe whose columns are to be imputed. 
            Utilizes the 'impute_column' method for imputing each of the columns.
        
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
