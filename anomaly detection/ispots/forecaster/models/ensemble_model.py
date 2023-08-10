import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ispots.forecaster.models.linear_model import LinearModel
from ispots.forecaster.models.prophet_model import ProphetModel
from ispots.forecaster.models.lgb_model import LGBModel
from ispots.forecaster.models.xgb_model import XGBModel
from ispots.forecaster.models.sarimax_model import SarimaxModel
from ispots.forecaster.models.lstm_model import LSTMModel

MODELS = {
    'Linear': LinearModel,
    'Prophet': ProphetModel,
    'LGB': LGBModel,
    'XGB': XGBModel,
    'Sarimax': SarimaxModel,
    'LSTM': LSTMModel
}

ENSEMBLE_METHODS = ['median', 'weighted']

class EnsembleModel(BaseEstimator, RegressorMixin):
    """ Ensemble Model which aggregates the prediction of different models to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    models: list, default=['Linear', 'Prophet', 'LGB', 'XGB', 'Sarimax', 'LSTM']
        Models to be used in the ensemble model

    lgb_params: dict, default=None
        Parameters for LGBModel

    xgb_params: dict, default=None
        Parameters for XGBModel
        
    prophet_params: dict, default={}
        Parameters for ProphetModel
        
    linear_params: dict, default={}
        Parameters for LinearModel
        
    sarimax_params: dict, default={}
        Parameters for SarimaxModel
        
    lstm_params: dict, default={}
        Parameters for LSTMModel
    
    ensemble_method: {'median', 'weighted'}, default='median'
        Method for aggregating the predictions of different models
    
    weights: list, default=None
        Weights for aggregation of model predictions. Only applicable when ensemble_method='weighted'.

    cyclic_feature_encoding: {'sincos', 'onehot'}, default='sincos'
        Cyclic feature encoding method
    """
    def __init__(self, freq='30min', horizon='1d', models=['Linear', 'Prophet', 'LGB', 'XGB', 'Sarimax', 'LSTM'], lgb_params=None, xgb_params=None, prophet_params={}, linear_params={}, sarimax_params={}, lstm_params={}, ensemble_method='median', weights=None, cyclic_feature_encoding='sincos'):
        if ensemble_method not in ENSEMBLE_METHODS:
            raise ValueError(f'Supported ensemble methods are {ENSEMBLE_METHODS}')
        self.ensemble_method = ensemble_method

        if not isinstance(models, list) or (isinstance(models, list) and len(models) <=1):
            raise ValueError('Parameter models is a list with at lease two elements')

        if self.ensemble_method == 'weighted' and (not isinstance(weights, list) or (isinstance(weights, list) and len(weights) != len(models))):
            raise ValueError("weights much be a list of the same length as models when ensemble_method is 'weighted'")

        if self.ensemble_method == 'weighted' and (np.sum(weights) != 1 or any(np.array(weights) < 0)):
            raise ValueError("All weights must be non-negative and add up to 1")
        self.weights = weights
        
        self.models = {}

        for m in models:
            if m in MODELS.keys() and m == 'Linear':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **linear_params)
            elif m in MODELS.keys() and m == 'Prophet':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **prophet_params)
            elif m in MODELS.keys() and m == 'LGB':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, params=lgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
            elif m in MODELS.keys() and m == 'XGB':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, params=xgb_params, cyclic_feature_encoding=cyclic_feature_encoding)
            elif m in MODELS.keys() and m == 'Sarimax':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **sarimax_params)
            elif m in MODELS.keys() and m == 'LSTM':
                self.models[m] = MODELS[m](freq=freq, horizon=horizon, cyclic_feature_encoding=cyclic_feature_encoding, **lstm_params)
            else:
                raise ValueError(f'{m} not supported')
        self.freq = freq
        self.horizon = horizon
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'Ensemble'

    def fit(self, X):
        """ Fit each model with input data

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Univariate time series dataframe from preprocessor

        Returns
        -------
        self: object
            Fitted ensemble model

        """
        for n, m in self.models.items():
            m.fit(X)
            print(f'====={n} model training completed=====')

        return self

    def predict(self, X):
        """ Generate predictions using each fitted model and aggregate them to produce forecasts

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, 1)
            The data used to generate forecast predictions.

        Returns
        -------
        pandas.DataFrame
            Time series dataframe containing predictions for the forecast horizon
        """
        fcst = []
        for n, m in self.models.items():
            fcst.append(m.predict(X))
            print(f'====={n} model forecasting completed=====')
        fcst = pd.concat(fcst, axis=1)

        if self.ensemble_method == 'median':
            fcst = pd.DataFrame(index=fcst.index, data=fcst.median(axis=1).values)
        elif self.ensemble_method == 'weighted':
            fcst = pd.DataFrame(index=fcst.index, data=fcst.multiply(self.weights, axis=1).sum(axis=1).values)

        fcst.index.name = 'ds'
        fcst.columns = ['y']
        
        return fcst