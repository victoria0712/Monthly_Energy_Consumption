import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

from ispots.forecaster.utils.feature_builder import feature_builder

class LinearModel(BaseEstimator, RegressorMixin):
    """ Linear Model which uses Lasso to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    alpha: float, default=0.1
        Constant that multiplies the L1 term
    
    cyclic_feature_encoding: {'sincos', 'onehot'}, default='onehot'
        Cyclic feature encoding method
    """
    def __init__(self, freq='30min', horizon='1d', alpha=0.1, cyclic_feature_encoding='onehot'):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not suppoeted. Only support daily or daily divisable frequency')

        if cyclic_feature_encoding not in ['sincos', 'onehot']:
            raise ValueError("Supported cyclic_feature_encoding methods are: ['sincos', 'onehot']")

        self.alpha = alpha
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.scaler = MinMaxScaler()
        self.model = Lasso(alpha=alpha, max_iter=10000) # tunable
        self.freq = freq
        self.horizon = horizon
        self.look_back_dur = '4w' if pd.Timedelta(self.freq) == pd.Timedelta('1d') else '7d'
        self.look_back_periods = int(pd.Timedelta(self.look_back_dur) / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'Linear'

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
        X, y, features = feature_builder(X, self.freq, self.name, self.cyclic_feature_encoding)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

        self.features = features
        self.best_features = pd.DataFrame({'feature': features, 'coef': self.model.coef_}).sort_values(by='coef', ascending=False).head(5)['feature'].values

        
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

        if len(X) < self.look_back_periods:
            raise ValueError(f'At least {self.look_back_dur} needs to be provided for forecasting')

        X_temp = X.iloc[-self.look_back_periods:].copy()

        for i in range(self.fcst_periods):
            start_time = X_temp.index[i]
            end_time = X_temp.index[-1] + pd.Timedelta(self.freq)
            idx_curr = pd.date_range(start=start_time, end=end_time, freq=self.freq)
            X_curr = X_temp.reindex(index=idx_curr)
            X_feature, _, _ = feature_builder(X_curr, self.freq, self.name, self.cyclic_feature_encoding)
            X_feature = X_feature[[-1]]
            X_feature = self.scaler.transform(X_feature)
            y_fcst = self.model.predict(X_feature)
            idx_new = pd.date_range(start=X_temp.index[0], end=X_temp.index[-1]+pd.Timedelta(self.freq), freq=self.freq)
            X_temp = X_temp.reindex(idx_new)
            X_temp.iloc[-1] = y_fcst

        fcst = X_temp.iloc[-self.fcst_periods:]
        fcst.index.name = 'ds'
        fcst.columns = ['y']

        return fcst
