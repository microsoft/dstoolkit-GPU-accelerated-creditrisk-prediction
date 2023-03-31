import os
from typing import Any, Dict, List, Tuple

from constants_ import ACQUISITION_COLS, PERFORMANCE_COLS
from imputation.central_imputation import CentralImputation
from utils import get_dataframe_summary, is_nvidia_gpu_available, timeit

NVIDIA_GPU_AVAILABILITY = is_nvidia_gpu_available()

if NVIDIA_GPU_AVAILABILITY:
    from cudf import DataFrame, concat, merge, read_csv
else:
    from pandas import DataFrame, concat, merge, read_csv


class DataPreparation:
    """Prepare the credit risk data from Acquisition and Performance files."""
    def __init__(self, data_folder_path: str, years: List[int]) -> None:
        """Initialize the Data Preparation object.

        Args:
            data_folder_path (str): Path where Acquisition(acq) and Performance(perf) files reside.
            years (List[int]): Years to include for data preparation.
        """
        self.data_folder_path = data_folder_path 
        self.years = years
    
    @timeit
    @get_dataframe_summary
    def read_acquisition_files(self) -> DataFrame:
        """Read and concatenate the acquisition files.
        
        Returns:
            DataFrame: Concatenated Acquisition data for all the years.
        """
        df_acq = DataFrame()
        for yr in self.years:
            for qtr in [1,2,3,4]:
                print('Reading the file Acquisition_'+str(yr)+'Q'+str(qtr)+'.txt')
                file_path = os.path.join(self.data_folder_path, 'acq/Acquisition_'+str(yr)+'Q'+str(qtr)+'.txt')
                df_acq_qtr = read_csv(file_path, sep='|', names=ACQUISITION_COLS, index_col=False) #dtype=dtype)
                df_acq = concat([df_acq,df_acq_qtr],axis=0)
                del df_acq_qtr
        return df_acq

    @timeit
    @get_dataframe_summary
    def read_performance_files(self) -> DataFrame:
        """Read and concatenate the performance files.
        
        Returns:
            DataFrame: Concatenated Performance data for all the years.
        """
        df_per = DataFrame()
        for yr in self.years:
            for qtr in [1,2,3,4]:
                if yr==2009 and qtr in [1,2,3]: #special for 2009 Q1-Q3
                    suffix = '_0'
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                    suffix = '_1'
                elif yr==2010 and qtr in [4]: #special for 2010 Q4
                    suffix = '_0'
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                    suffix = '_1'   
                elif yr==2011 and qtr in [4]: #special for 2011 Q4
                    suffix = '_0'
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                    suffix = '_1' 
                elif yr==2012 and qtr in [1,2,3,4]: #special for 2012 Q1-Q4
                    suffix = '_0'
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                    suffix = '_1'    
                elif yr==2013 and qtr in [1,2,3]: #special for 2012 Q1-Q4
                    suffix = '_0'
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                    suffix = '_1'                    
                else:
                    suffix = ''
                    # fn = '../data/perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix
                    fn = os.path.join(self.data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                    print(fn)
                    df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                df_per_qtr['CLDS'] = df_per_qtr['CLDS'].astype(str)
                df_per = concat([df_per,df_per_qtr],axis=0)
                del df_per_qtr
        return df_per

    @timeit
    def merge_acq_per_files(self, df_acq: DataFrame, df_delinq4: DataFrame) -> DataFrame:
        """Outer join Acquisition and Performance files. 'CLDS' column in renamed to 'Default'

        Args:
            df_acq (DataFrame): Acquisition Dataframe.
            df_delinq4 (DataFrame): Dataframe with delinquency values.

        Returns:
            DataFrame: Merged Dataframe with 'CLDS' renamed to 'Default'.
        """
        print("Merging Acquisition and Performace files...")
        df = merge(df_acq, df_delinq4, on='LoanID', how='outer')
        print("Shape of dataframe after merging:", df.shape)
        df = df.reset_index().rename(columns={'CLDS': 'Default'})
        print("'CLDS' column has been renamed to 'Default'")
        return df

    @timeit
    def create_target(self, df: DataFrame) -> DataFrame:
        """Transforms target for Credit Risk dataset such that default==4 is mapped to 1, others to 0.

        Args:
            df (DataFrame): Dataframe with Default values unmapped.

        Returns:
            DataFrame: "Default" values mapped to 1 and 0.
        """
        df.loc[df['Default'] == '4', 'Default'] = 1
        df.loc[df['Default'].isnull(), 'Default'] = 0
        df['Default'] = df['Default'].astype(int)
        print("Target Column value counts:")
        print(df['Default'].value_counts())
        return df

    def prepare_credit_risk_data(self) -> DataFrame:
        df_acq = self.read_acquisition_files()

        print('Performance col list is len',len(PERFORMANCE_COLS),'but using only 2!')
        df_per = self.read_performance_files()

        df_per = df_per.dropna(subset=['CLDS'])
        df_per['CLDS'] = df_per['CLDS'].astype(str)
        print("Performance file shape after dropping nulls: ", df_per.shape)

        df_delinq4 = df_per.loc[df_per['CLDS']=='4']
        print("Performance (Delinquency=4) shape: ", df_delinq4.shape)

        df_delinq4.drop_duplicates(subset='LoanID', keep='last', inplace=True)
        print("Performance after dropping duplicates shape: ",df_delinq4.shape)

        df = self.merge_acq_per_files(df_acq, df_delinq4)
        del df_acq
        del df_per
        del df_delinq4

        df = self.create_target(df)

        print("Dropping the columns - 'index','OrDate','OrLTV','MortInsPerc','RelMortInd','FirstPayment', 'Zip','PropertyState'...")
        df.drop(['index','OrDate','OrLTV','MortInsPerc','RelMortInd','FirstPayment', 'Zip','PropertyState'], axis=1, inplace=True)
        print("Shape:", df.shape)
        df = CentralImputation(NVIDIA_GPU_AVAILABILITY).impute(df)
        return df
