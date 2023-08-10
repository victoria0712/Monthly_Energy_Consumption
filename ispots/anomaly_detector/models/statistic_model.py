'''
    This python file is for statistic model used to make anomaly detection.
    The model compares the detection date consumption with the consumption 
    within the lookback period to detect anomaly concumption patterns.
'''

import datetime as dt
import numpy as np
import os
from configparser import ConfigParser

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file
file_path = 'config/config.ini' # path of the config file relative to this py file
abs_path = os.path.join(dir_path, '..', file_path) # absolute config file path

config_object = ConfigParser()
config_object.read(abs_path)
stat_params = config_object['statistic_params']

min_stdev_width = stat_params['min_stdev_width'] # 0.01

class Statistic_Model():
    '''
    Statistic Model which generates expected values of time series data by statistic method

    Parameters:
    model_params (dict): model parameters default as {}
    '''
    def __init__(self,  
                 model_params = {}):
        
        self.name = "statistic"
        self.model_params = model_params
    
    def fit_transform(self, temp_full_df):
        '''
        fit_transform
            use data from normal consumption period to derive the metrics needed for anomaly detection
            values on weekdays and weekends may be treated separately
            standard deviation should be minimally 1% of the mean
        input:
            df (pandas.DataFrame): dataframe (index: DateTime, col: Value)
        output:
            test_df_1d (pandas.DataFrame): dataframe of the last date (detection date) with derived 
            columns (median, standard deviation, mean)
        '''
        
        # retrieve data for detection date
        test_df_1d = temp_full_df.loc[temp_full_df.index > max(temp_full_df.index) - dt.timedelta(days=1)].copy()
        test_df_1d['date'] = test_df_1d.index.date
        test_df_1d['time'] = test_df_1d.index.time

        # retrieve data from look back period
        full_df = temp_full_df.loc[temp_full_df.index <= max(temp_full_df.index) - dt.timedelta(days=1)].copy()

        # if sep_weekend is true
        if 'weekend' in full_df.columns.values:
            wd_df = full_df.loc[full_df.weekend != True].copy()
            we_df = full_df.loc[full_df.weekend == True].copy()
   
            # insert reference values into the detection day dataframe  
            if (test_df_1d.weekend != True).all():
                # weekday
                test_df_1d = insert_reference_values(test_df_1d, wd_df)
            else:
                # weekend
                test_df_1d = insert_reference_values(test_df_1d, we_df)
        
        # if sep_weekend is false
        else:
            test_df_1d = insert_reference_values(test_df_1d, temp_full_df) 

        test_df_1d['std'] = test_df_1d[['actual_std', 'mean']].apply(lambda x: max(x['actual_std'], x['mean'] * 0.01), axis =1)  
        test_df_1d['iqr'] = test_df_1d[['q3', 'q1']].apply(lambda x: x['q3'] - x['q1'], axis = 1)

        # create columns to align naming convention with other models
        test_df_1d['pred'] = test_df_1d['mean']
        test_df_1d['deviation'] = np.abs(test_df_1d['Value'] - test_df_1d['mean'])
        test_df_1d['pred_low'] = test_df_1d['mean'] - test_df_1d['std']
        test_df_1d['pred_high'] = test_df_1d['mean'] + test_df_1d['std']

        return test_df_1d

def insert_reference_values(df, full_df):
    '''
    insert_reference_values
        insert reference values (median, std, mean) to dataframe for selected rows based on condition
    input:
        df (pandas.DataFrame): dataframe for detection date (index: Datetime col: Value)
        full_df (pandas.DataFrame): full dataframe for data from look back period (index: Datetime col: Value)
    output:
        df (pandas.DataFrame): dataframe with reference value columns (median, std, mean)
    '''
    # use only data from normal consumption dates    
    temp_df = full_df[full_df['normal'] == True].copy()

    # create reference values dataframe
    def q1(x):
        return x.quantile(0.25)
    def q3(x):
        return x.quantile(0.75)

    f = {'Value': ['median', 'std', q1, q3, 'mean']}
    ref_df = temp_df.groupby('time').agg(f)
    ref_df = ref_df.droplevel(level = 0, axis = 1)

    for column in ['median', 'std', 'q1', 'q3', 'mean']:
        df[column] = list(map(lambda x: ref_df.loc[x, column], df.time))

    df.rename(columns={'std': "actual_std"}, inplace = True)

    return df