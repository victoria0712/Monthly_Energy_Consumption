import pandas as pd
import statsmodels.api as sm

from ispots.forecaster.utils.feature_builder import feature_builder

class SarimaxModel():
    """ Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX) model

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    p: int, default=0
        Number of AR parameters

    d: int, default=1
        Number of differences

    q: int, default=0
        Number of MA parameters

    seasonal_p: int, default=1
        Number of AR parameters for the seasonal component of the model

    seasonal_d: int, default=1
        Number of differences for the seasonal component of the model

    seasonal_q: int, default=1
        Number of MA parameters for the seasonal component of the model
    
    cyclic_feature_encoding: {'sincos', 'onehot'}, default='onehot'
        Cyclic feature encoding method
    """
    def __init__(self, freq='30min', horizon='1d', p=0, d=1, q=0, seasonal_p=1, seasonal_d=1, seasonal_q=1, cyclic_feature_encoding='onehot'):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not supported. Only support daily or daily divisable frequency')

        if cyclic_feature_encoding not in ['sincos', 'onehot']:
            raise ValueError("Supported cyclic_feature_encoding methods are: ['sincos', 'onehot']")

        self.freq = freq
        self.horizon = horizon
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.order = (p,d,q) 
        self.seasonal_order = (seasonal_p, seasonal_d, seasonal_q, self.cycle_periods)
        self.name = 'Sarimax'

    def fit(self, X):
        """ Generate features for data and fit model with input data and features

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Univariate time series dataframe from preprocessor

        Returns
        -------
        self: object
            Fitted model

        """
        X = X.copy()
        X_regressors = feature_builder(X, self.freq, self.name, self.cyclic_feature_encoding)
        self.model = sm.tsa.statespace.SARIMAX(endog=X, 
                                                exog=X_regressors, 
                                                order=self.order, 
                                                seasonal_order=self.seasonal_order,
                                                enforce_stationarity=False)            
        self.model = self.model.fit()

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
        X = X.copy()
        start_time = X.index[-1] + pd.Timedelta(self.freq)
        fcst_regressors = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=self.fcst_periods, freq=self.freq)})
        fcst_regressors.set_index('ds', inplace=True)
        fcst_regressors = feature_builder(fcst_regressors, self.freq, self.name, self.cyclic_feature_encoding)
        
        pred = self.model.forecast(self.fcst_periods, exog=fcst_regressors)
        fcst = pd.DataFrame(pred)
        fcst = fcst.rename_axis('ds')
        fcst.columns = ['y']

        return fcst
