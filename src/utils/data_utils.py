import os
import numpy as np
import pandas as pd

def read_purchase_data(file_name, index_col='MEMBER_ID'):
    purchase_df = pd.read_csv(file_name, index_col=index_col)
    return purchase_df, purchase_df.values

def read_csv(file_name, encoding='utf-8'):
    df = pd.read_csv(file_name, encoding=encoding)
    return df

def save_temp_data(file_name, data_dict):
    if os.path.exists(file_name):
        print('[info] {} is exist, will override the temporary data.'.format(file_name))
    np.savez(file_name, **data_dict)
    print('[info] temporary data already saved in {}.'.format(file_name))