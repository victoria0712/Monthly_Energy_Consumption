import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess

from ispots.forecaster.models.linear_trend_model import LinearTrendModel
from ispots.forecaster.models.prophet_trend_model import ProphetTrendModel
from ispots.forecaster.models.linear_model import LinearModel
from ispots.forecaster.models.lgb_model import LGBModel
from ispots.forecaster.models.lstm_model import LSTMModel
from ispots.forecaster.models.prophet_model import ProphetModel
from ispots.forecaster.models.sarimax_model import SarimaxModel
from ispots.forecaster.models.xgb_model import XGBModel
from ispots.forecaster.utils.feature_builder import feature_builder

MODEL_1 = ['Prophet', 'LinearTrend']
MODEL_2 = ['Linear', 'XGB', 'LGB', 'Sarimax', 'LSTM']

class HybridModel():
    def __init__(
        self, 
        freq='30min', 
        horizon='1d', 
        model1='Prophet', 
        model2='XGB', 
        model1_params={},
        model2_params={},
        ):
        if model1 not in MODEL_1:
            raise ValueError(f'Supported ensemble methods are {MODEL_1}')
        if model2 not in MODEL_2:
            raise ValueError(f'Supported ensemble methods are {MODEL_2}')

        self.freq = freq
        self.horizon = horizon
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.model1 = model1
        self.model2 = model2
        self.model1_params = model1_params
        self.model2_params = model2_params
        self.name = 'Hybrid'

    def fit(self, df):
        ## Model 1 ##
        y = df.copy()
        if self.model1 == 'Prophet':
            self.prophet_model = ProphetTrendModel(
                freq=self.freq, 
                    horizon=self.horizon, 
                    **self.model1_params
            )
            self.prophet_model.fit(y)
            if self.prophet_model.growth == 'logistic':
                y['cap'] = self.prophet_model.cap
                y['floor'] = self.prophet_model.floor
        
            X_regressors = feature_builder(y, self.prophet_model.freq, self.prophet_model.name, self.prophet_model.cyclic_feature_encoding)
            X = pd.concat([y, X_regressors], axis=1)
            X.reset_index(inplace=True)
            y_fit = self.prophet_model.model.predict(X)[['ds', 'trend']]
            y_fit = y_fit.set_index('ds')

        elif self.model1 == 'LinearTrend':
            self.linear_model = LinearTrendModel(
                    freq=self.freq, 
                    horizon=self.horizon, 
                    **self.model1_params
                )
            self.linear_model.fit(y)
            dp = DeterministicProcess(index=df.index, order=self.linear_model.trend_order)
            X = dp.in_sample()
            y_fit = self.linear_model.model.predict(X)
            y_fit = pd.DataFrame(y_fit, index=X.index)

        y_fit.columns = ['y']

        ## Model 2 ##
        y = df.copy()
        y_detrended = y - y_fit

        cyclic_feature_encoding = self.model2_params.pop('cyclic_feature_encoding', 'onehot')

        if self.model2 == 'Linear':
            self.detrended_model = LinearModel(freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=cyclic_feature_encoding, **self.model2_params)
        elif self.model2 == 'LGB':
            lgb_params = None if self.model2_params=={} else self.model2_params
            self.detrended_model = LGBModel(freq=self.freq, horizon=self.horizon, params=lgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
        elif self.model2 == 'XGB':
            xgb_params = None if self.model2_params=={} else self.model2_params
            self.detrended_model = XGBModel(freq=self.freq, horizon=self.horizon, params=xgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
        elif self.model2 == 'Prophet':
            self.detrended_model = ProphetModel(freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=cyclic_feature_encoding, **self.model2_params)
        elif self.model2 == 'Sarimax':
            self.detrended_model = SarimaxModel(freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=cyclic_feature_encoding, **self.model2_params)
        elif self.model2 == 'LSTM':
            self.detrended_model = LSTMModel(freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=cyclic_feature_encoding, **self.model2_params)
        
        self.detrended_model.fit(y_detrended)
        return self

    def predict(self, df):
        y = df.copy()
        ## Model 1 ##
        if self.model1 == 'Prophet':
            y_fit_fcst = self.prophet_model.predict(y)

            y = df.iloc[-self.look_back_cycle*self.cycle_periods:].copy()
            if self.prophet_model.growth == 'logistic':
                y['cap'] = self.prophet_model.cap
                y['floor'] = self.prophet_model.floor

            X_regressors = feature_builder(y, self.prophet_model.freq, self.prophet_model.name, self.prophet_model.cyclic_feature_encoding)
            X = pd.concat([y, X_regressors], axis=1)
            X.reset_index(inplace=True)
            y_fit_look_back = self.prophet_model.model.predict(X)[['ds', 'trend']]
            y_fit_look_back = y_fit_look_back.set_index('ds')

        elif self.model1 == 'LinearTrend':
            y_fit_fcst = self.linear_model.predict(y)

            idx = pd.date_range(start=self.linear_model.first_timestamp, end=y.index[-1], freq=self.freq)
            dp = DeterministicProcess(index=idx, order=self.linear_model.trend_order)
            trend_features_look_back = dp.in_sample()[-self.look_back_cycle*self.cycle_periods:]
            y_fit_look_back = self.linear_model.model.predict(trend_features_look_back)
            y_fit_look_back = pd.DataFrame(y_fit_look_back, index=trend_features_look_back.index)

        y_fit_look_back.columns = ['y']

        ## Model 2 ##
        df2 = df.iloc[-self.look_back_cycle*self.cycle_periods:].copy()
        y_look_back = df2.copy()
        y_detrended_look_back = y_look_back - y_fit_look_back
        y_detrended_fcst = self.detrended_model.predict(y_detrended_look_back)
        fcst = y_fit_fcst + y_detrended_fcst
        fcst.index.name = 'ds'

        return fcst
