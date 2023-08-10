import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from ispots.forecaster.models.linear_model import LinearModel
from ispots.forecaster.models.prophet_model import ProphetModel
from ispots.forecaster.models.lgb_model import LGBModel
from ispots.forecaster.models.xgb_model import XGBModel
from ispots.forecaster.models.sarimax_model import SarimaxModel
from ispots.forecaster.models.lstm_model import LSTMModel
from ispots.forecaster.utils.feature_builder import feature_builder, generate_sequence

MODELS = {
    'Linear': LinearModel,
    'Prophet': ProphetModel,
    'LGB': LGBModel,
    'XGB': XGBModel,
    'Sarimax': SarimaxModel,
    'LSTM': LSTMModel,
}

class StackingModel():
    def __init__(
        self,
        freq='30min', 
        horizon='1d',
        base_models=['Linear', 'Prophet', 'LGB', 'XGB', 'Sarimax', 'LSTM'],
        lgb_params=None, 
        xgb_params=None, 
        prophet_params={}, 
        linear_params={}, 
        sarimax_params={}, 
        lstm_params={}, 
        cyclic_feature_encoding='sincos'):

        self.base_models = {}
        for m in base_models:
            if m in MODELS.keys() and m == 'Linear':
                self.base_models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **linear_params)
            elif m in MODELS.keys() and m == 'Prophet':
                self.base_models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **prophet_params)
            elif m in MODELS.keys() and m == 'LGB':
                self.base_models[m] = MODELS[m](freq=freq, horizon=horizon, params=lgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
            elif m in MODELS.keys() and m == 'XGB':
                self.base_models[m] = MODELS[m](freq=freq, horizon=horizon, params=xgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
            elif m in MODELS.keys() and m == 'Sarimax':
                self.base_models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **sarimax_params)
            elif m in MODELS.keys() and m == 'LSTM':
                self.base_models[m] = MODELS[m](freq=freq, horizon=freq, cyclic_feature_encoding=cyclic_feature_encoding, **lstm_params)
            else:
                raise ValueError(f'{m} model not supported')
        
        self.meta_learner = LinearRegression()
        self.freq = freq
        self.horizon = horizon
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 7
        self.look_back_dur = '4w' if pd.Timedelta(self.freq) == pd.Timedelta('1d') else '7d'
        self.look_back_periods = int(pd.Timedelta(self.look_back_dur) / pd.Timedelta(self.freq))
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.name = 'Stacking'

    def fit(self, X):
        for _, m in self.base_models.items():
            m.fit(X)
        
        base_model_res = []

        if 'Linear' in self.base_models.keys():
            X_linear = X.copy()
            X_linear, _, _ = feature_builder(X_linear, self.freq, 'Linear', self.cyclic_feature_encoding)
            X_linear = self.base_models['Linear'].scaler.transform(X_linear)
            linear_res = self.base_models['Linear'].model.predict(X_linear)
            linear_res = linear_res[..., None]
            base_model_res.append(linear_res)

        if 'Prophet' in self.base_models.keys():
            X_prophet = X.copy()
            if self.base_models['Prophet'].growth == 'logistic':
                X_prophet['cap'] = self.base_models['Prophet'].cap
                X_prophet['floor'] = self.base_models['Prophet'].floor
            X_regressors = feature_builder(X_prophet, self.freq, 'Prophet', self.cyclic_feature_encoding)
            X_prophet = pd.concat([X_prophet, X_regressors], axis=1)
            X_prophet.reset_index(inplace=True)
            prophet_res = self.base_models['Prophet'].model.predict(X_prophet)[self.look_back_cycle*self.cycle_periods:].yhat.values
            prophet_res = prophet_res[..., None]
            base_model_res.append(prophet_res)

        if 'LGB' in self.base_models.keys():
            X_lgb = X.copy()
            X_lgb, _, _ = feature_builder(X_lgb, self.freq, 'LGB', self.cyclic_feature_encoding)
            lgb_res = self.base_models['LGB'].model.predict(X_lgb)
            lgb_res = lgb_res[..., None]
            base_model_res.append(lgb_res)

        if 'XGB' in self.base_models.keys():
            X_xgb = X.copy()
            X_xgb, _, _ = feature_builder(X_xgb, self.freq, 'XGB', self.cyclic_feature_encoding)
            X_xgb = xgb.DMatrix(X_xgb)
            xgb_res = self.base_models['XGB'].model.predict(X_xgb)
            xgb_res = xgb_res[..., None]
            base_model_res.append(xgb_res)

        if 'Sarimax' in self.base_models.keys():
            X_sarimax = X.copy()
            sarimax_res = np.array(self.base_models['Sarimax'].model.predict(start=X_sarimax.index[0], end=X_sarimax.index[-1])[self.cycle_periods*self.look_back_cycle:])
            sarimax_res = sarimax_res[..., None]
            base_model_res.append(sarimax_res)

        if 'LSTM' in self.base_models.keys():
            X_lstm = X.copy()
            date_range = X_lstm[:self.look_back_periods].index
            X_samples, _ = generate_sequence(X_lstm, self.look_back_periods, 1)
            lstm_res = []
            for x in X_samples:
                df_sample = pd.DataFrame(x, index=date_range)
                res = self.base_models['LSTM'].predict(df_sample).values[0][0]
                lstm_res.append(res)
                date_range = pd.date_range(start=date_range[1], end=date_range[-1]+pd.Timedelta(self.freq), freq=self.freq)
            lstm_res = np.array(lstm_res)
            lstm_res = lstm_res[..., None]
            base_model_res.append(lstm_res)

        meta_learner_features = np.concatenate(base_model_res, axis=1)
        self.meta_learner.fit(meta_learner_features, X[self.look_back_cycle*self.cycle_periods:].values)

        return self

    def predict(self, X):

        if len(X) < self.look_back_periods:
            raise ValueError(f'At least {self.look_back_dur} needs to be provided for forecasting')

        X_temp = X.iloc[-self.look_back_periods:].copy()

        for i in range(self.fcst_periods):
            start_time = X_temp.index[i]
            end_time = X_temp.index[-1] + pd.Timedelta(self.freq)
            idx_curr = pd.date_range(start=start_time, end=end_time, freq=self.freq)
            X_curr = X_temp.reindex(index=idx_curr)

            base_model_res = []

            if 'Linear' in self.base_models.keys():
                X_linear = X_curr.copy()
                X_linear, _, _ = feature_builder(X_linear, self.freq, 'Linear', self.cyclic_feature_encoding)
                X_linear = X_linear[[-1]]
                X_linear = self.base_models['Linear'].scaler.transform(X_linear)
                linear_res = self.base_models['Linear'].model.predict(X_linear)
                base_model_res.append(linear_res[0])

            if 'Prophet' in self.base_models.keys():
                X_prophet = X_curr.copy()
                start_time = X_prophet.index[-1]
                df_fcst = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=1, freq=self.freq)})
                df_fcst.set_index('ds', inplace=True)
                df_fcst = feature_builder(df_fcst, self.freq, 'Prophet', self.cyclic_feature_encoding)
                df_fcst.reset_index(inplace=True)
                if self.base_models['Prophet'].growth == 'logistic':
                    df_fcst['cap'] = self.base_models['Prophet'].cap
                    df_fcst['floor'] = self.base_models['Prophet'].floor
                prophet_res = self.base_models['Prophet'].model.predict(df_fcst)['yhat']
                base_model_res.append(prophet_res[0])

            if 'LGB' in self.base_models.keys():
                X_lgb = X_curr.copy()
                X_lgb, _, _ = feature_builder(X_lgb, self.freq, 'LGB', self.cyclic_feature_encoding)
                X_lgb = X_lgb[[-1]]
                lgb_res = self.base_models['LGB'].model.predict(X_lgb)
                base_model_res.append(lgb_res[0])

            if 'XGB' in self.base_models.keys():
                X_xgb = X_curr.copy()
                X_xgb, _, _ = feature_builder(X_xgb, self.freq, 'XGB', self.cyclic_feature_encoding)
                X_xgb = X_xgb[[-1]]
                X_xgb = xgb.DMatrix(X_xgb)
                xgb_res = self.base_models['XGB'].model.predict(X_xgb)
                base_model_res.append(xgb_res[0])

            if 'Sarimax' in self.base_models.keys():
                start_time = X_curr.index[-1]
                fcst_regressors = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=1, freq=self.freq)})
                fcst_regressors.set_index('ds', inplace=True)
                fcst_regressors = feature_builder(fcst_regressors, self.freq, 'Sarimax', self.cyclic_feature_encoding)
                sarimax_res = self.base_models['Sarimax'].model.forecast(1, exog=fcst_regressors).values
                base_model_res.append(sarimax_res[0])

            if 'LSTM' in self.base_models.keys():
                X_lstm = X_temp.copy()
                lstm_res = self.base_models['LSTM'].predict(X_lstm).values[0]
                base_model_res.append(lstm_res[0])

            meta_learner_features = np.array(base_model_res).reshape(1, -1)
            fcst = self.meta_learner.predict(meta_learner_features)
            idx_new = pd.date_range(start=X_temp.index[0], end=X_temp.index[-1]+pd.Timedelta(self.freq), freq=self.freq)
            X_temp = X_temp.reindex(idx_new)
            X_temp.iloc[-1] = fcst
            # X_temp = X_temp.iloc[1:]

        df_fcst = X_temp.iloc[-self.fcst_periods:]
        df_fcst.index.name = 'ds'
        df_fcst.columns = ['y']

        return df_fcst
