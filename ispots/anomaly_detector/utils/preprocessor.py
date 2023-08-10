'''
    This python file is for preprocessing the data for anomaly detection.

    The function preprocess the csv files data by removing data with missing
    values over a certain threshold and applying interpolation over remaining
    data. The output will be a data dictionary containing a dataframe (value) 
    for each measuring point name (key). The dataframe has DatetimeIndex and 
    one column (Value).
'''

import numpy as np
import pandas as pd
import os

from configparser import ConfigParser

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file
file_path = 'config/config.ini' # path of the config file relative to this py file
abs_path = os.path.join(dir_path, '..', file_path) # absolute config file path

config_object = ConfigParser()
config_object.read(abs_path)
prep_params = config_object['preprocess_params']

miss_val_threshold = eval(prep_params['missing_val_threshold']) # 0.01 as default

class Preprocessor():
    '''
    preprocess
        First, identify the frequency of the records
        Apply preprocessing to the data by removing measuring points with more than 1% missing values (default is 1%)
        Use interpolation to fill in missing values for remaining data (i.e. the less than 1% missing values)
        Create data dictionary for these measuring points 
    input: 
        data_dict (dict): data dictionary (key: measuring point, value: dataframe (column: Time, Value)). Preprocessor() assumes all 
            data in the data_dict to have the same time frequency.
        miss_val_threshold (float): proportion of missing values in the dataframe. If missing values exceed proportion, a printout 
            will be generated to mention the number of missing values for the measuring point and measuring point will be removed from 
            data dictionary.
    output:
        preprocessed_data_dict (dict): preprocessed data dictionary (key: measuring point, value: dataframe (index: Datetime, column: Value))
    '''
    def __init__(self, data_dict, 
                        miss_val_threshold = miss_val_threshold):
        self.data_dict = data_dict
        self.miss_val_threshold = miss_val_threshold
    
    def preprocess(self):         
        # identify frequency of records
        for mp in self.data_dict:
            df = self.data_dict[mp]
            df.Time = pd.to_datetime(df.Time)
            self.data_frequency = df.Time.diff().mode()[0]               
            if self.data_frequency is not None:
                break

        # pre-process and store results in a dictionary
        preprocessed_data_dict = {}
        for mp in self.data_dict:
            df = self.data_dict[mp]

            # retrieve the start and end timestamp
            start_ts = min(df.Time)
            end_ts = max(df.Time)
            temp_df = df.copy()
            # set index to Time
            temp_df['Time'] = pd.to_datetime(temp_df['Time'])
            temp_df = temp_df.set_index('Time')
            
            # create new timestamp values based on the start and end timestamps and frequency
            new_ts = pd.date_range(start=start_ts, end=end_ts, freq=self.data_frequency)

            temp_df = temp_df.reindex(index=new_ts)
            missing_rows = temp_df.loc[np.isnan(temp_df.Value)]

            # print number of missing values if greater then the threshold (1%)
            if len(missing_rows) > self.miss_val_threshold*len(df):
                print(f'{mp}: missing values {len(missing_rows)}')

            else:
                # interpolate the missing values
                temp_df = temp_df.interpolate(method='linear',limit_direction = 'both')
                # store the dataframe into the data dictionary
                preprocessed_data_dict[mp] = temp_df  

        return preprocessed_data_dict