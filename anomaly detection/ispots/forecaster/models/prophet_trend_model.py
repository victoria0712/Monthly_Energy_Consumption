import pandas as pd
from prophet import Prophet

from ispots.forecaster.utils.feature_builder import feature_builder

class ProphetTrendModel():
    def __init__(self, freq='30min', horizon='1d', cyclic_feature_encoding='onehot',
        growth='logistic', n_changepoints=20, changepoint_range=0.5, changepoint_prior_scale=0.005, 
        seasonality_mode='additive', seasonality_prior_scale=10.0):
        self.freq = freq
        self.horizon = horizon
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.name = 'Prophet'

    def fit(self, df):
        y = df.copy()
        self.model = Prophet(
            growth=self.growth, 
            seasonality_mode=self.seasonality_mode, 
            n_changepoints=self.n_changepoints, 
            changepoint_range=self.changepoint_range, 
            seasonality_prior_scale=self.seasonality_prior_scale, 
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        if self.growth == 'logistic':
            self.cap = y.values.max() * 1.2
            self.floor = max(0, y.values.min() * 0.8)
            y['cap'] = self.cap
            y['floor'] = self.floor
        
        X_regressors = feature_builder(y, self.freq, self.name, self.cyclic_feature_encoding)
        regressor_names = X_regressors.columns
        X = pd.concat([y, X_regressors], axis=1)
        X.reset_index(inplace=True)

        for n in regressor_names:
            self.model.add_regressor(n)
        
        self.model.fit(X)

        return self

    def predict(self, df):
        y = df.copy()
        start_time = y.index[-1] + pd.Timedelta(self.freq)
        df_fcst = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=self.fcst_periods, freq=self.freq)})
        df_fcst.set_index('ds', inplace=True)
        df_fcst = feature_builder(df_fcst, self.freq, self.name, self.cyclic_feature_encoding)
        df_fcst.reset_index(inplace=True)
        if self.growth == 'logistic':
            df_fcst['cap'] = self.cap
            df_fcst['floor'] = self.floor
        y_fit_fcst = self.model.predict(df_fcst)[['ds', 'trend']]
        y_fit_fcst = y_fit_fcst.set_index('ds')
        y_fit_fcst.columns = ['y']

        return y_fit_fcst
