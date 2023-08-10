'''
    This python file is for level shift model used to make anomaly detection.
    The model compares the detection date consumption with the consumption 
    value within the lookback period to detect anomaly concumption patterns.
    This model is more sensitive to level shifts.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from adtk.detector import LevelShiftAD
import matplotlib.pyplot as plt
import datetime as dt
from adtk.visualization import plot
import os

from ispots.anomaly_detector.utils.feature_engineering import remove_level_shift, remove_extreme_values, insert_reference_values

# from configparser import ConfigParser
# dir_path = os.path.dirname(os.path.realpath(__file__)) 
# file_path = 'config/config.ini' 
# abs_path = os.path.join(dir_path, '..', file_path) 
# config_object = ConfigParser()
# config_object.read(abs_path)
# lvl_params = config_object['level_shift_params']

# model_lookback_period = eval(lvl_params['model_lookback_period']) # 30 days
# model_agg_period = lvl_params['model_agg_period'] # 2H

class LevelShift_Model():
    '''
    LevelShift Model which generates expected values of time series data by levelshift method

    Parameters:
    lookback_period (int): lookback period (days) as normal consumption (eg 30) default as 30 (days)

    scaler (str): scaler method ('minmax' or 'standard') default as 'minmax'
    
    sep_weekend (boolean): whether to separate weekend effect (eg True) default as True
    
    agg_period (str): aggregate period for anomaly detection (eg '30min') default as '2H'

    model_params (dict): model parameters default as {}
    '''
    def __init__(self,
                 lookback_period,
                 agg_period, 
                 scaler = 'minmax', 
                 sep_weekend = True, 
                 model_params = {}):
        if pd.Timedelta(agg_period) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(agg_period) != pd.Timedelta(0):
            raise ValueError(f'{agg_period} is not daily divisable')
        elif pd.Timedelta(agg_period) > pd.Timedelta('1d'):
            raise ValueError(f'{agg_period} frequency not suppoeted. Only support daily or daily divisable frequency')
            
        self.lookback_period = lookback_period
        self.scaler = scaler
        self.sep_weekend = sep_weekend
        self.agg_period = agg_period
        self.name = "level_shift"
        self.model_params = model_params
        
    def fit_transform(self, df):
        '''
        fit_transform
            fit the normal consumption period data and derive the metrics needed for anomaly detection
        input:
            df (pandas.DataFrame): dataframe (index: DateTime, col: Value)
        output:
            test_df_1d (pandas.DataFrame): dataframe of the last date (detection date) with derived 
            columns (median, standard deviation, mean)
        '''
        temp_df = df.resample(self.agg_period).mean().copy()
        
        test_start_index = min(temp_df.index)
        temp_df = temp_df.loc[temp_df.index < test_start_index + dt.timedelta(days=self.lookback_period + 1)].copy()
        
        # standardize and/or normalize consunmption value
        if self.scaler == 'standard':
            scaler = StandardScaler()
            temp_df['Value'] = scaler.fit_transform(temp_df[['Value']].values)
    
        if self.scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            temp_df['Value'] = scaler.fit_transform(temp_df[['Value']].values)
        
        # get the latest (1 month) data starting from the start index
        test_df_norm = temp_df.loc[temp_df.index < test_start_index + dt.timedelta(days=self.lookback_period+1)].copy()
        
        # build adtk level shift model and detect level shifts
        if len(self.model_params) > 0:
            level_shift_ad = LevelShiftAD(**self.model_params) # add params
        else:
            level_shift_ad = LevelShiftAD(window = 1, c = 12, side = 'both')
        anomalies = level_shift_ad.fit_detect(test_df_norm)
        level_shifts = list(anomalies.loc[anomalies.Value == True].index.values)
    
        if len(level_shifts) >= 1:
            end_ds = test_df_norm.index.values[-1]
            if (end_ds - level_shifts[-1]) >= np.timedelta64(7,'D'):
                temp_lvl_df = test_df_norm.loc[test_df_norm.index >= level_shifts[-1]].copy()
            elif len(level_shifts) >= 2:
                end_ds = level_shifts[-1]
                level_shifts.pop(-1)
                level_shifts = [test_df_norm.index.values[0],] + level_shifts 
                while (end_ds - level_shifts[-1]) < np.timedelta64(7,'D'):
                    level_shifts.pop(-1)
                temp_lvl_df = test_df_norm.loc[(test_df_norm.index >= level_shifts[-1])&(test_df_norm.index <= end_ds)].copy()
            else:
                temp_lvl_df = test_df_norm.loc[test_df_norm.index <= level_shifts[-1]].copy()
        else:
            temp_lvl_df = test_df_norm.copy()

        test_df_norm = temp_df.loc[(temp_df.index >= min(temp_lvl_df.index)) & (temp_df.index <= max(temp_lvl_df.index) - dt.timedelta(days=1))].copy()
        test_df_1d = temp_df.loc[(temp_df.index > max(temp_df.index) - dt.timedelta(days=1))].copy()
        
        temp_full_df = test_df_norm.copy()
        temp_full_df['time'] = temp_full_df.index.time
        # remove the top 5% and substitute with median value
        temp_full_df = remove_extreme_values(temp_full_df, ['Value'], 0.05)
        test_df_1d = insert_reference_values(test_df_1d, temp_full_df, test_df_1d.index != None)
        
        return test_df_1d