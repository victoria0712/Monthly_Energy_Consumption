import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class BaselineModel(BaseEstimator, RegressorMixin):
    """ Baseline Model which uses aggregation of each time index in the look back cycle to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    look_back_dur: str, default='7d'
        Duration of look back cycle
    
    method: {'median', 'mean', 'drift', 'ewm'}, default='median'
        Aggregation function used to generate forecast prediction

    alpha: float, default=0.1
        Smoothing factor for exponentially weighted calculations. Only applicable when method='ewm'.
    """
    def __init__(self, freq='30min', horizon='1d', look_back_dur='7d', method='median', alpha=0.1):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not supported. Only support daily or daily divisable frequency')

        self.freq = freq
        self.horizon = horizon
        self.look_back_dur = look_back_dur
        self.method = method
        self.alpha = alpha

        #self.look_back_dur_default = '4w' if pd.Timedelta(self.freq) == pd.Timedelta('1d') else '7d'
        self.look_back_cycle = int(self.look_back_dur[0])
        #self.look_back_cycle_default = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        #self.look_back_periods_default = int(pd.Timedelta(self.look_back_dur_default) / pd.Timedelta(self.freq))
        self.look_back_periods = int(pd.Timedelta(self.look_back_dur) / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'Baseline'

    def fit(self, X):
        """ Fit model with input data

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Univariate time series dataframe from preprocessor

        Returns
        -------
        self: object
            Fitted model

        """
        return self

    def predict(self, X):
        """ Generate forecast predictions using fitted model

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, 1)
            The data used to generate forecast predictions.

        Returns
        -------
        pandas.DataFrame
            Time series dataframe containing predictions for the forecast horizon

        """
        if pd.Timedelta(self.freq) == pd.Timedelta('1d'):
            if pd.Timedelta(self.look_back_dur) < pd.Timedelta('1w') or pd.Timedelta(self.look_back_dur) > pd.Timedelta('4w'):
                raise ValueError(f'look_back_dur should be in the range of 1w to 4w')
        else:
            if pd.Timedelta(self.look_back_dur) < pd.Timedelta('1d') or pd.Timedelta(self.look_back_dur) > pd.Timedelta('7d'):
                raise ValueError(f'look_back_dur should be in the range of 1d to 7d')
        
        if len(X) < self.look_back_periods:
            raise ValueError(f'At least {self.look_back_dur} needs to be provided for forecasting')


        #print(f'look_back_cycle: {self.look_back_cycle}')
        #print(f'cycle_periods: {self.cycle_periods}')
        #print(f'look_back_periods: {self.look_back_periods}')
        #print(f'fcst_periods: {self.fcst_periods}')

        X_temp = X.iloc[-self.look_back_periods:].copy()
        #print("X_temp: ")
        #print(X_temp)

        for i in range(self.fcst_periods):
            start_time = X_temp.index[i]
            end_time = X_temp.index[-1] + pd.Timedelta(self.freq)
            idx_curr = pd.date_range(start=start_time, end=end_time, freq=self.freq)
            X_curr = X_temp.reindex(index=idx_curr)

            if self.method == 'median':
                y_fcst = np.median(X_curr.iloc[-self.cycle_periods-1:-self.cycle_periods*self.look_back_cycle-2:-self.cycle_periods].values)
            elif self.method == 'mean':
                y_fcst = np.mean(X_curr.iloc[-self.cycle_periods-1:-self.cycle_periods*self.look_back_cycle-2:-self.cycle_periods].values)
            elif self.method == 'drift': # last observation + average change
                X_temp2 = X_curr.iloc[-self.cycle_periods-1:-self.cycle_periods*self.look_back_cycle-2:-self.cycle_periods].sort_index().values.flatten()
                if len(X_temp2) == 1:
                    y_fcst = X_temp2[-1] 
                else:
                    y_fcst = X_temp2[-1] + ( (X_temp2[-1]-X_temp2[0]) / (len(X_temp2)-1) )
            elif self.method == 'ewm':
                X_temp2 = X_curr.iloc[-self.cycle_periods-1:-self.cycle_periods*self.look_back_cycle-2:-self.cycle_periods].sort_index()
                if len(X_temp2) == 1:
                    y_fcst = X_temp2.values.flatten()[-1]
                else:
                    y_fcst = X_temp2.ewm(alpha=self.alpha).mean().values[-1][0]
            else:
                raise ValueError("Supported methods are: ['median', 'mean', 'drift', 'ewm']")

            idx_new = pd.date_range(start=X_temp.index[0], end=X_temp.index[-1]+pd.Timedelta(self.freq), freq=self.freq)
            X_temp = X_temp.reindex(idx_new)
            X_temp.iloc[-1] = y_fcst

        fcst = X_temp.iloc[-self.fcst_periods:]
        fcst.index.name = 'ds'
        fcst.columns = ['y']

        return fcst