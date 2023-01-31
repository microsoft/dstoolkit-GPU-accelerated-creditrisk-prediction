import os
n_gpus = 1
if n_gpus == 0:
    from pandas import read_csv,concat,merge,DataFrame,get_dummies,Series,isnull
    from sklearn.preprocessing import OneHotEncoder
    from numpy import zeros,double,int8,concatenate
else:
    from cudf import read_csv,concat,merge,DataFrame,get_dummies,Series
    from cuml.preprocessing import OneHotEncoder
    from cupy import zeros,double,int8,concatenate
from constants_ import PERFORMANCE_COLS, col_acq, dtypesDict

def read_acquisition_files(data_folder_path, years):
    df_acq = DataFrame()
    for yr in years:
        for qtr in [1,2,3,4]:
            print('Reading the file Acquisition_'+str(yr)+'Q'+str(qtr)+'.txt')
            file_path = os.path.join(data_folder_path, 'acq/Acquisition_'+str(yr)+'Q'+str(qtr)+'.txt')
            df_acq_qtr = read_csv(file_path, sep='|', names=col_acq, index_col=False) #dtype=dtype)
            df_acq = concat([df_acq,df_acq_qtr],axis=0)
            del df_acq_qtr
    return df_acq

def read_performance_files(data_folder_path, years):
    df_per = DataFrame()
    for yr in years:
        for qtr in [1,2,3,4]:
            if yr==2009 and qtr in [1,2,3]: #special for 2009 Q1-Q3
                suffix = '_0'
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                suffix = '_1'
            elif yr==2010 and qtr in [4]: #special for 2010 Q4
                suffix = '_0'
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                suffix = '_1'   
            elif yr==2011 and qtr in [4]: #special for 2011 Q4
                suffix = '_0'
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                suffix = '_1' 
            elif yr==2012 and qtr in [1,2,3,4]: #special for 2012 Q1-Q4
                suffix = '_0'
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                suffix = '_1'    
            elif yr==2013 and qtr in [1,2,3]: #special for 2012 Q1-Q4
                suffix = '_0'
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
                suffix = '_1'                    
            else:
                suffix = ''
                # fn = '../data/perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix
                fn = os.path.join(data_folder_path, 'perf/Performance_'+str(yr)+'Q'+str(qtr)+'.txt'+suffix)
                print(fn)
                df_per_qtr = read_csv(fn, sep='|',names=PERFORMANCE_COLS, usecols=[0,10],index_col=False)
            df_per_qtr['CLDS'] = df_per_qtr['CLDS'].astype(str)
            df_per = concat([df_per,df_per_qtr],axis=0)
            del df_per_qtr
    return df_per
    