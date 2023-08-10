'''
    This python file is for prophet model used to make anomaly detection.
    The model compares the detection date consumption with the consumption
    value forecasted based on lookback period to detect anomaly concumption 
    patterns. The model might be useful when there is a trend in the
    consumption data.
'''

import pandas as pd
import numpy as np
import os
import datetime as dt
from prophet import Prophet

# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class Prophet_Model():
    '''
    Prophet Model which generates expected values of time series data by prediction-based method

    Paramters:

    interval_width (float): % uncertainty interval for predicted value
    
    model_params (dict): model parameters (for Prophet model) default as {}
    '''
    def __init__(self, 
                 interval_width,
                 model_params = {}):
        
        # if pd.Timedelta(agg_period) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(agg_period) != pd.Timedelta(0):
        #     raise ValueError(f'{agg_period} is not daily divisable')
        # elif pd.Timedelta(agg_period) > pd.Timedelta('1d'):
        #     raise ValueError(f'{agg_period} frequency not supported. Only support daily or daily divisable frequency')
            
        self.name = "fb_prophet"
        self.interval_width = interval_width
        self.model_params = model_params
    
    def fit_transform(self, temp_full_df):
        '''
        fit_transform
            use data from normal consumption period to derive the metrics needed for anomaly detection
        input:
            df (pandas.DataFrame): dataframe (index: pandas.DateTime, col: Value)
        output:
            test_df_1d (pandas.DataFrame): dataframe of the last date (detection date) with fb prophet
            predicted value columns: pred, pred_low, pred_high
        '''
        # retrieve data from look back period
        lookback_df = temp_full_df.loc[temp_full_df.index < min(temp_full_df.index) + dt.timedelta(days=30)].copy()
        lookback_df_helper = pd.DataFrame()
        lookback_df_helper['ds'] = lookback_df.index.values
        lookback_df_helper['y'] = lookback_df.Value.values        

        # retrieve data for detection date
        test_df_1d = temp_full_df.loc[temp_full_df.index > max(temp_full_df.index) - dt.timedelta(days=1)].copy()
        test_df_1d_helper = pd.DataFrame()
        test_df_1d_helper['ds'] = test_df_1d.index.values
        test_df_1d_helper['y'] = test_df_1d.Value.values        
        
        prophet = Prophet(interval_width = self.interval_width,**self.model_params) # add params
        with suppress_stdout_stderr():
            prophet.fit(lookback_df_helper)
    
        test_df_1d['pred'] = prophet.predict(test_df_1d_helper).yhat.values
        test_df_1d['pred_low'] = prophet.predict(test_df_1d_helper).yhat_lower.values
        test_df_1d['pred_high'] = prophet.predict(test_df_1d_helper).yhat_upper.values

        test_df_1d['deviation'] = np.abs(test_df_1d['Value'] - test_df_1d['pred'])

        return test_df_1d