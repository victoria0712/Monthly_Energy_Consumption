import itertools
import copy
import random

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from ispots.forecaster.models.baseline_model import BaselineModel
from ispots.forecaster.models.linear_model import LinearModel
from ispots.forecaster.models.prophet_model import ProphetModel
from ispots.forecaster.models.sarimax_model import SarimaxModel
from ispots.forecaster.models.lstm_model import LSTMModel
from ispots.forecaster.models.hybrid_model import HybridModel

LARGE_ERROR = np.Inf

LARGE_ERROR_DICT = {'rmse_mean': LARGE_ERROR,
                    'mse_mean': LARGE_ERROR,
                    'mae_mean': LARGE_ERROR,
                    'mape_mean': LARGE_ERROR,
                    'rmse_std': LARGE_ERROR,
                    'mse_std': LARGE_ERROR,
                    'mae_std': LARGE_ERROR,
                    'mape_std': LARGE_ERROR}

def _evaluate(actual, fcst, metric):
    """ Returns error between actual and forecast based on specified evaluation metric
    """
    if metric == 'rmse':
        error = mean_squared_error(actual, fcst) ** 0.5
    elif metric == 'mse':
        error = mean_squared_error(actual, fcst)
    elif metric == 'mae':
        error = mean_absolute_error(actual, fcst)
    elif metric == 'mape':
        error = mean_absolute_percentage_error(actual, fcst)
    else:
        raise ValueError("Supported metrics are 'rmse', 'mse', 'mae', 'mape'")

    return error

def back_test(model, X, test_size=0.2, stride='1d', retrain_stride=False, retrain=False, window='expanding'):
    """ Conducts backtesting and returns error between actual and forecast

    Parameters
    ----------
    model: {'Linear','LGB','XGB','Prophet','Sarimax','LSTM'}
        Model that will be used to produce forecast

    X: pandas.DataFrame
        Proprocessed time series dataframe with shape (n_samples, 1)
    
    test_size: float, default=0.2
        Proportion of dataset to be used as initial test set
    
    stride: str, default='1d'
        Stride ahead period

    retrain_stride: bool, default=False
        If True, with every stride ahead, model will be retrained

    retrain: bool, default=False
        If True, model will be retrained at the end of backtesting with the full dataset

    window: {'expanding', 'rolling'}, default='expanding'
        If 'expanding' window is used, training data will increase with every stride ahead.
        If 'rolling' window is used, training data will move forward at a fixed length with every stride ahead.

    Returns
    -------
    pandas.DataFrame: dataframe containing model name and error values

    """
    if window == 'rolling' and retrain_stride == False:
        raise ValueError(f"retrain_stride needs to be True for window = 'rolling'")
    if window not in ['expanding', 'rolling']:
        raise ValueError("Supported windows are 'expanding', 'rolling'")

    freq = model.freq
    horizon = model.horizon
    fcst_periods = model.fcst_periods
    stride_periods = int(pd.Timedelta(stride) / pd.Timedelta(freq))
    split_loc = int(len(X) * test_size)
    X_train = X.iloc[:-split_loc].copy()
    X_test = X.iloc[-split_loc:].copy()
    error = {
        'rmse': [],
        'mse': [],
        'mae': [],
        'mape': []
    }

    if model.name == 'Prophet':
        growth = model.growth
        seasonality_mode = model.seasonality_mode
        n_changepoints=model.n_changepoints
        changepoint_range=model.changepoint_range
        seasonality_prior_scale=model.seasonality_prior_scale
        changepoint_prior_scale=model.changepoint_prior_scale

    
    if model.name == 'Ensemble' and 'Prophet' in model.models.keys():
        growth = model.models['Prophet'].growth
        seasonality_mode = model.models['Prophet'].seasonality_mode
        n_changepoints=model.models['Prophet'].n_changepoints
        changepoint_range=model.models['Prophet'].changepoint_range
        seasonality_prior_scale=model.models['Prophet'].seasonality_prior_scale
        changepoint_prior_scale=model.models['Prophet'].changepoint_prior_scale

    if retrain_stride:
        for i in range(0, len(X_test), stride_periods):
            if i <= len(X_test) - fcst_periods:
                if model.name == 'Prophet':
                    model.model = Prophet(
                        growth=growth, 
                        seasonality_mode=seasonality_mode, 
                        n_changepoints=n_changepoints, 
                        changepoint_range=changepoint_range, 
                        seasonality_prior_scale=seasonality_prior_scale, 
                        changepoint_prior_scale=changepoint_prior_scale
                    )
                elif model.name == 'Ensemble' and 'Prophet' in model.models.keys():
                    model.models['Prophet'].model = Prophet(
                        growth=growth, 
                        seasonality_mode=seasonality_mode, 
                        n_changepoints=n_changepoints, 
                        changepoint_range=changepoint_range, 
                        seasonality_prior_scale=seasonality_prior_scale, 
                        changepoint_prior_scale=changepoint_prior_scale
                    )

                fcst = model.fit(X_train).predict(X_train).values
                actual = X_test.iloc[i:i+model.fcst_periods].values
                for e in ['rmse', 'mse', 'mae', 'mape']:
                    error[e].append(_evaluate(actual, fcst, e))
                if window == 'expanding':
                    X_train = pd.concat([X_train, X_test.iloc[i:i+stride_periods]])
                else:
                    X_train = pd.concat([X_train.iloc[stride_periods:], X_test.iloc[i:i+stride_periods]])
    else:
        # if model.name == 'Prophet':
        #     model.model = Prophet(growth=growth, seasonality_mode=seasonality_mode)
        # elif model.name == 'Ensemble' and 'Prophet' in model.models.keys():
        #     model.models['Prophet'].model = Prophet(growth=growth, seasonality_mode=seasonality_mode)

        model.fit(X_train)
        for i in range(0, len(X_test), stride_periods):
            if i <= len(X_test) - fcst_periods:
                fcst = model.predict(X_train).values
                actual = X_test.iloc[i:i+fcst_periods].values
                for e in error.keys():
                    error[e].append(_evaluate(actual, fcst, e))
                X_train = pd.concat([X_train, X_test.iloc[i:i+stride_periods]])

    error_mean = pd.DataFrame(error).mean().add_suffix('_mean').to_dict()
    error_std = pd.DataFrame(error).std().add_suffix('_std').to_dict()
    error = {**error_mean, **error_std}

    if retrain:
        if model.name == 'Prophet':
            model.model = Prophet(
                growth=growth, 
                seasonality_mode=seasonality_mode, 
                n_changepoints=n_changepoints, 
                changepoint_range=changepoint_range, 
                seasonality_prior_scale=seasonality_prior_scale, 
                changepoint_prior_scale=changepoint_prior_scale
            )
        elif model.name == 'Ensemble' and 'Prophet' in model.models.keys():
            model.models['Prophet'].model = Prophet(
                growth=growth, 
                seasonality_mode=seasonality_mode, 
                n_changepoints=n_changepoints, 
                changepoint_range=changepoint_range, 
                seasonality_prior_scale=seasonality_prior_scale, 
                changepoint_prior_scale=changepoint_prior_scale
            )
        model.fit(X)
        
    return error


def grid_search(model, X, parameters=None, model1_parameters=None, model2_parameters=None, test_size=0.2, stride='1d', retrain_stride=False, retrain=False):
    """ Conducts hyperparameter grid search and returns backtesting error for each combination of parameters

    Parameters
    ----------
    model: {'Linear','LGB','XGB','Prophet','Sarimax','LSTM','Hybrid'}
        Model that will be used to produce forecast

    X: pandas.DataFrame
        Proprocessed time series dataframe with shape (n_samples, 1)
    
    parameters: dict, default=None
        Dictionary with parameter names as keys; list of parameter values as values.
        Only applicable when model is not Hybrid.

    model1_parameters: dict, default=None
        Dictionary with stage 1 model parameter names as keys; list of parameter values as values.
        Only applicable when model is Hybrid.
    
    model2_parameters: dict, default=None
        Dictionary with stage 2 model parameter names as keys; list of parameter values as values.
        Only applicable when model is Hybrid.

    test_size: float, default=0.2
        Proportion of dataset to be used as initial test set

    stride: str, default='1d'
        Stride ahead period

    retrain_stride: bool, default=False
        If True, with every stride ahead, model will be retrained

    retrain: bool, default=False
        If True, model will be retrained at the end of backtesting with the full dataset

    window: {'expanding', 'rolling'}, default='expanding'
        If 'expanding' window is used, training data will increase with every stride ahead.
        If 'rolling' window is used, training data will move forward at a fixed length with every stride ahead.

    Returns
    -------
    pandas.DataFrame: dataframe containing parameters and error values
    """
    errors = []
    if model.name == 'Hybrid':
        model1 = model.model1
        model2 = model.model2
        if model1_parameters == None:
            raise ValueError("model1_parameters should be a dict with param name as keys and list of parameter values as values")
        elif model2_parameters == None:
            raise ValueError("model2_parameters should be a dict with param name as keys and list of parameter values as values")
        else:
            model1_params = [dict(zip(model1_parameters.keys(), v)) for v in itertools.product(*model1_parameters.values())]
            model2_params = [dict(zip(model2_parameters.keys(), v)) for v in itertools.product(*model2_parameters.values())]
        df_model1_params = []
        df_model2_params = []
        for param1 in model1_params:
            for param2 in model2_params:
                df_model1_params.append(param1)
                df_model2_params.append(param2)
                model = HybridModel(model1=model1, model2=model2, model1_params=param1, model2_params=param2, freq=model.freq, horizon=model.horizon)
                try:
                    error = back_test(model, X, test_size=test_size, stride=stride, retrain_stride=retrain_stride, retrain=False)
                except:
                    error = LARGE_ERROR_DICT
                errors.append(error)
        df_errors = pd.DataFrame(errors)
        print(df_model1_params)
        print(df_model2_params)
        df_params = pd.DataFrame({'model1_params': df_model1_params, 'model2_params': df_model2_params})
        df_res = pd.concat([df_params, df_errors], axis=1)
        df_res['model'] = model.name
        df_res['model1'] = model1
        df_res['model2'] = model2
    else:
        if model.name == 'Baseline':
            alpha_values = parameters.pop('alpha')
            params = [dict(zip(parameters.keys(), v)) for v in itertools.product(*parameters.values())]
            params_cp = []
            for curr in params:
                # alpha only applicable for ewm
                if curr.get('method') == 'ewm':
                    for alpha in alpha_values:
                        temp = copy.deepcopy(curr)
                        temp['alpha'] = alpha
                        params_cp.append(temp)
                else:
                    params_cp.append(curr)
            df_params = pd.DataFrame(columns=['parameter'], data=[[param] for param in params_cp])
        else:
            params = [dict(zip(parameters.keys(), v)) for v in itertools.product(*parameters.values())]
            params_cp = copy.deepcopy(params)
            df_params = pd.DataFrame(columns=['parameter'], data=[[param] for param in params])

        for param in tqdm(params_cp):
            if model.name == 'Baseline':
                model = BaselineModel(**param)
            elif model.name == 'Linear':
                model = LinearModel(**param)
            elif model.name == 'Prophet':
                model = ProphetModel(**param)
            elif model.name in ['XGB', 'LGB']:
                model.cyclic_feature_encoding = param.pop('cyclic_feature_encoding', 'onehot')
                model.params = {**model.params, **param}
            elif model.name == 'Sarimax':
                model = SarimaxModel(**param)
            elif model.name == 'LSTM':
                model = LSTMModel(**param)
            try:
                error = back_test(model, X, test_size=test_size, stride=stride, retrain_stride=retrain_stride, retrain=retrain)
            except:
                error = LARGE_ERROR_DICT
            errors.append(error)

        df_errors = pd.DataFrame(errors)
        df_res = pd.concat([df_params, df_errors], axis=1)
        df_res['model'] = model.name
    
    return df_res

def randomized_search(model, X, parameters=None, hybrid_model1=['LinearTrend', 'Prophet'], hybrid_model2=['Linear', 'LGB', 'XGB', 'LSTM', 'Sarimax'], model1_parameters=None, model2_parameters=None, n_iter=50, test_size=0.2, stride='1d', retrain_stride=False, random_state=None):
    """ Conducts hyperparameter randomized search and returns backtesting error for each combination of parameters

    Parameters
    ----------
    model: {'Linear','LGB','XGB','Prophet','Sarimax','LSTM','Hybrid'}
        Model that will be used to produce forecast

    X: pandas.DataFrame
        Proprocessed time series dataframe with shape (n_samples, 1)

    parameters: dict, default=None
        Dictionary with parameter names as keys; list or distribution of parameter values as values.
        Only applicable when model is not Hybrid.
    
    hybrid_model1: list, default=['LinearTrend', 'Prophet']
        Models included in the search space for stage 1 of the Hybrid model.
        Only applicable when model is Hybrid.
    
    hybrid_model2: list, default=['Linear', 'LGB', 'XGB', 'LSTM', 'Sarimax']
        Models included in the search space for stage 2 of the Hybrid model.
        Only applicable when model is Hybrid.

    model1_parameters: dict, default=None
        Dictionary with stage 1 model parameter names as keys; list or distribution of parameter values as values.
        Only applicable when model is Hybrid.
    
    model2_parameters: dict, default=None
        Dictionary with stage 2 model parameter names as keys; list or distribution of parameter values as values.
        Only applicable when model is Hybrid.
    
    n_inter: int, default=50
        Number of parameter settings that are produced
    
    test_size: float, default=0.2
        Proportion of dataset to be used as initial test set

    stride: str, default='1d'
        Stride ahead period

    retrain_stride: bool, default=False
        If True, with every stride ahead, model will be retrained

    window: {'expanding', 'rolling'}, default='expanding'
        If 'expanding' window is used, training data will increase with every stride ahead.
        If 'rolling' window is used, training data will move forward at a fixed length with every stride ahead.

    random_state: int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling from lists of possible values 
        instead of scipy.stats distributions. Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    pandas.DataFrame: dataframe containing model name, parameters and error values
    """
    errors = []
    if model.name == 'Hybrid':
        # param distribution for model 1
        param_grid_dict1 = {}
        for m in model1_parameters:
            model_name = m['model_name']
            hyperparams = m['params']
            param_grid_dict1[model_name] = hyperparams

        # param distribution for model 2
        param_grid_dict2 = {}
        for m in model2_parameters:
            model_name = m['model_name']
            hyperparams = m['params']
            param_grid_dict2[model_name] = hyperparams

        errors = []
        df_params = []
        df_model1 = []
        df_model2 = []

        for _ in range(n_iter):
            # randomly select model 1 and a set of parameters
            model1 = random.sample(hybrid_model1, 1)[0]
            df_model1.append(model1)
            params_model1 = list(ParameterSampler(param_grid_dict1[model1], n_iter=1, random_state=random_state))[0]

            # randomly select model 2 and a set of parameters
            model2 = random.sample(hybrid_model2, 1)[0]
            df_model2.append(model2)
            params_model2 = list(ParameterSampler(param_grid_dict2[model2], n_iter=1, random_state=random_state))[0]

            df_params.append({'parameter': {'model1_params': params_model1, 'model2_params': params_model2}})

            # print(f'{model.name}: {model1}, {model2}')
            # print('model1_params: ', params_model1)
            # print('model2_params: ', params_model2)
            model = HybridModel(model1=model1, model2=model2, model1_params=params_model1, model2_params=params_model2, freq=model.freq, horizon=model.horizon)
            try:
                error = back_test(model, X, test_size=test_size, stride=stride, retrain_stride=retrain_stride, retrain=False)
            except:
                error = LARGE_ERROR_DICT             
            errors.append(error)
        df_errors = pd.DataFrame(errors)
        df_params = pd.DataFrame(df_params)
        df_res = pd.concat([df_params, df_errors], axis=1)
        df_res['model'] = model.name
        df_res['model1'] = df_model1
        df_res['model2'] = df_model2

    else:
        params = list(ParameterSampler(parameters, n_iter=n_iter, random_state=random_state))
        params_cp = copy.deepcopy(params)
        df_params = pd.DataFrame(columns=['parameter'], data=[[param] for param in params])

        for param in tqdm(params_cp):
            if model.name == 'Baseline':
                model = BaselineModel(**param)
            elif model.name == 'Linear':
                model = LinearModel(**param)
            elif model.name == 'Prophet':
                model = ProphetModel(**param)
            elif model.name in ['XGB', 'LGB']:
                model.cyclic_feature_encoding = param.pop('cyclic_feature_encoding', 'onehot')
                model.params = {**model.params, **param}
            elif model.name == 'Sarimax':
                model = SarimaxModel(**param)
            elif model.name == 'LSTM':
                model = LSTMModel(**param)
            try:
                error = back_test(model, X, test_size=test_size, stride=stride, retrain_stride=retrain_stride, retrain=False)
            except:
                error = LARGE_ERROR_DICT             
            errors.append(error)
        df_errors = pd.DataFrame(errors)
        df_res = pd.concat([df_params, df_errors], axis=1)
        df_res['model'] = model.name
    
    return df_res
