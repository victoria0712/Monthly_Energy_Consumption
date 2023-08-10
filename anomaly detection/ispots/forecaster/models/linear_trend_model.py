import pandas as pd
from sklearn.linear_model import Lasso
from statsmodels.tsa.deterministic import DeterministicProcess

class LinearTrendModel():
    def __init__(self, freq='30min', horizon='1d', alpha=0.1, trend_order=3):
        self.freq = freq
        self.horizon = horizon
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.alpha = alpha
        self.trend_order = trend_order
        self.name = 'LinearTrend'

    def fit(self, df):
        y = df.copy()
        self.first_timestamp = y.index[0]
        self.model = Lasso(alpha=self.alpha)
        dp = DeterministicProcess(index=y.index, order=self.trend_order)
        X = dp.in_sample()
        self.model.fit(X, y)

        return self

    def predict(self, df):
        y = df.copy()
        idx = pd.date_range(start=self.first_timestamp, end=y.index[-1], freq=self.freq)
        dp = DeterministicProcess(index=idx, order=self.trend_order)
        dp.in_sample()
        trend_features_fcst = dp.out_of_sample(self.fcst_periods)
        y_fit_fcst = pd.DataFrame(self.model.predict(trend_features_fcst), index=trend_features_fcst.index, columns=y.columns)
        y_fit_fcst.index.name = 'ds'

        return y_fit_fcst