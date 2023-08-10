import pandas as pd
from prophet import Prophet

from ispots.forecaster.utils.feature_builder import feature_builder

class ProphetModel():
    """ Prophet Model which uses FB Prophet to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    growth: {'linear', 'logistic'}, default='linear'
        The growth of the trend

    seasonality_mode: {'additive', 'multiplicative'}, default='additive'
        The type of seasonality

    n_changepoints: int, default=25
        Number of changepoints to be automatically included

    changepoint_range: float, default=0.8
        Affects how close the changepoints can go to the end of the time series.
        The larger the value, the more flexible the trend.

    seasonality_prior_scale: float, default=10.0
        Strength of seasonality model

    changepoint_prior_scale: float, default=0.05
        Flexibility of automatic changepoint selection

    cyclic_feature_encoding: {'sincos', 'onehot'}, default='onehot'
        Cyclic feature encoding method
    """
    def __init__(
        self, 
        freq='30min', 
        horizon='1d', 
        growth='linear', 
        seasonality_mode='additive', 
        n_changepoints=25, 
        changepoint_range=0.8, 
        seasonality_prior_scale=10.0, 
        changepoint_prior_scale=0.05, 
        cyclic_feature_encoding='onehot'
    ):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not suppoeted. Only support daily or daily divisable frequency')

        if cyclic_feature_encoding not in ['sincos', 'onehot']:
            raise ValueError("Supported cyclic_feature_encoding methods are: ['sincos', 'onehot']")

        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = Prophet(
            growth=self.growth, 
            seasonality_mode=self.seasonality_mode, 
            n_changepoints=self.n_changepoints, 
            changepoint_range=self.changepoint_range, 
            seasonality_prior_scale=self.seasonality_prior_scale, 
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        self.freq = freq
        self.horizon = horizon
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'Prophet'

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
        # X = X.iloc[-self.look_back_cycle*self.cycle_periods:].copy()
        X = X.copy()

        if self.growth == 'logistic':
            self.cap = X.values.max() * 1.2
            self.floor = max(0, X.values.min() * 0.8)
            X['cap'] = self.cap
            X['floor'] = self.floor

        X_regressors = feature_builder(X, self.freq, self.name, self.cyclic_feature_encoding)
        regressor_names = X_regressors.columns
        X = pd.concat([X, X_regressors], axis=1)
        X.reset_index(inplace=True)

        for n in regressor_names:
            self.model.add_regressor(n)
        self.model.fit(X)

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
        df_fcst = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=self.fcst_periods, freq=self.freq)})
        df_fcst.set_index('ds', inplace=True)
        df_fcst = feature_builder(df_fcst, self.freq, self.name, self.cyclic_feature_encoding)
        df_fcst.reset_index(inplace=True)
        if self.growth == 'logistic':
            df_fcst['cap'] = self.cap
            df_fcst['floor'] = self.floor
        fcst = self.model.predict(df_fcst)[['ds', 'yhat']]
        fcst.set_index('ds', inplace=True)
        fcst.columns = ['y']

        return fcst

