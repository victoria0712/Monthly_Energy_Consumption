import warnings
import logging
import os
import time
import copy

import pandas as pd
import numpy as np
import yaml
from yaml.loader import SafeLoader
from sklearn.base import BaseEstimator, RegressorMixin
from hyperopt import hp, fmin, Trials, anneal
from hyperopt.early_stop import no_progress_loss
from scipy.stats import loguniform, uniform
from sklearn.utils import check_random_state

from ispots.forecaster.models.linear_model import LinearModel
from ispots.forecaster.models.prophet_model import ProphetModel
from ispots.forecaster.models.xgb_model import XGBModel
from ispots.forecaster.models.lgb_model import LGBModel
from ispots.forecaster.models.sarimax_model import SarimaxModel
from ispots.forecaster.models.lstm_model import LSTMModel
from ispots.forecaster.models.hybrid_model import HybridModel
from ispots.forecaster.utils.model_selection import back_test, randomized_search
from ispots.utils.logger import get_logger

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

logger = get_logger('forecasting-engine')

# quniform distribution for randomized search
class quniform():
    def __init__(self, low, high, q):
        self.low = low
        self.high = high
        self.q = q

    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        upper = np.ceil((self.high-self.low) / self.q) * self.q
        return int(self.low + rng.uniform(0, upper, size) // self.q * self.q)

# convert strings to distributions for yaml files
def convert_str_to_dist(params):
    params_new = copy.deepcopy(params)
    for m in params:
        for n in params[m]:
            if type(params[m][n]) == str:
                params_new[m][n] = eval(params[m][n])
            else:
                params_new[m][n] = params[m][n]
    return params_new

MODELS = {
    'Linear': LinearModel,
    'Prophet': ProphetModel,
    'LGB': LGBModel,
    'XGB': XGBModel,
    'Sarimax': SarimaxModel,
    'LSTM': LSTMModel,
    'Hybrid': HybridModel,
}

LARGE_ERROR = np.Inf

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file

file_path1 = 'config/cat_params.yaml' # path of the config file relative to this py file
abs_path1 = os.path.join(dir_path, file_path1) # absolute config file path
CATEGORICAL_PARAMS = yaml.load(open(abs_path1), Loader=SafeLoader)

file_path2 = 'config/params_bayesian.yaml' # path of the config file relative to this py file
abs_path2 = os.path.join(dir_path, file_path2) # absolute config file path
PARAMS_RAW = yaml.load(open(abs_path2), Loader=SafeLoader)
PARAMS = convert_str_to_dist(PARAMS_RAW)

file_path3 = 'config/params_randomized.yaml' # path of the config file relative to this py file
abs_path3 = os.path.join(dir_path, file_path3) # absolute config file path
PARAMS_RANDOMIZED_SEARCH_RAW = yaml.load(open(abs_path3), Loader=SafeLoader)
PARAMS_RANDOMIZED_SEARCH = convert_str_to_dist(PARAMS_RANDOMIZED_SEARCH_RAW)

class AutoForecasting(BaseEstimator, RegressorMixin):
    """ Conducts Automated Forecasting which finds the best combination of model and hyperparameters

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of time series
    
    horizon: str, default='1d'
        Forecast horizon

    models: list
        Models included in the search space

    hybrid_stage1: list, default=['LinearTrend', 'Prophet']
        Models included in the search space for stage 1 of the Hybrid model.
        Only applicable when model is Hybrid.
    
    hybrid_stage2: list, default=['Linear', 'LGB', 'XGB', 'LSTM', 'Sarimax']
        Models included in the search space for stage 2 of the Hybrid model.
        Only applicable when model is Hybrid.

    test_size: float, default=0.2
        Proportion of dataset to be used as initial test set
    
    stride: str, default='1d'
        Stride ahead period

    retrain_stride: bool, default=False
        If True, with every stride ahead, model will be retrained

    metric: {'rmse', 'mse', 'mae', 'mape'}, default='rmse'
        Evaluation metric for backtesting

    method: {'bayesian', 'randomized_search'}, default='bayesian'
        Hyperparameter search method

    early_stopping_steps: int, default=10
        Stops bayesian optimization early if best loss has not improved in early_stopping_steps trials

    Attributes
    ----------
    best_model: Model with the best performance and fitted with the input time series

    """

    def __init__(self, freq='30min', horizon='1d', models=['Linear', 'Prophet', 'LGB', 'XGB', 'Sarimax', 'LSTM', 'Hybrid'], hybrid_stage1=['LinearTrend', 'Prophet'], hybrid_stage2=['Linear', 'LGB', 'XGB', 'LSTM', 'Sarimax'], test_size=0.2, stride='1d', retrain_stride=False, metric='rmse', iteration=50, method='bayesian', early_stopping_steps=10):
        if set(models) - set(MODELS.keys()):
            raise ValueError(f'Not all models in {models} are supported')

        if method not in ['bayesian', 'randomized_search']:
            raise ValueError(f"Supported methods are ['bayesian', 'randomized_search']")

        self.freq = freq
        self.horizon = horizon
        self.models = models
        self.test_size = test_size
        self.stride = stride
        self.retrain_stride = retrain_stride
        self.metric = metric
        self.iteration = iteration
        self.method = method
        self.hybrid_stage1 = hybrid_stage1
        self.hybrid_stage2 = hybrid_stage2
        self.early_stopping_steps = early_stopping_steps

    def fit(self, X, y=None):
        """ Finds the best model and parameters combination

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The data used to conduct backtesting to find the best model and parameters.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted forecaster

        """
        if self.method == 'bayesian':
            param_grid = []

            for m in self.models:
                if m != 'Hybrid':
                    params = {
                        'model': MODELS[m], 
                        'model_name': m, 
                        'params': PARAMS[m]
                    }
                else:
                    params = {
                        'model': MODELS[m],
                        'model_name': m, 
                        'params': {
                            'stage1': hp.choice('stage1', [(model, PARAMS[model]) for model in self.hybrid_stage1]), 
                            'stage2': hp.choice('stage2', [(model, PARAMS[model]) for model in self.hybrid_stage2])
                        }
                    }
                param_grid.append(params)

            param_grid = hp.choice('regressor', param_grid)

            models = [] # for tracking
            models1 = [] # for tracking
            models2 = [] # for tracking
            parameters = [] # for tracking
            errors = [] # for tracking
            times = [] # for tracking

            def objective(params):
                start = time.time()
                model_name = params['model_name']
                if model_name == 'Hybrid':
                    model1_name, model1_params = params['params']['stage1']
                    model2_name, model2_params = params['params']['stage2']
                    
                    logger.debug(f"""
                    Model type: {model_name}
                    Stage1 model: {model1_name}
                    Stage1 model params: {model1_params}
                    Stage2 model: {model2_name}
                    Stage2 model params: {model2_params}
                    """)
    
                    # model 1 integer param handling
                    if model1_name == 'LinearTrend':
                        model1_params['trend_order'] = int(model1_params['trend_order'])
                    elif model1_name == 'Prophet':
                        model1_params['n_changepoints'] = int(model1_params['n_changepoints'])
                    
                    # model 2 integer param handling
                    if model2_name == 'LGB':
                        model2_params['max_depth'] = int(model2_params['max_depth'])
                        model2_params['num_leaves'] = int(model2_params['num_leaves'])
                    elif model2_name == 'XGB':
                        model2_params['max_depth'] = int(model2_params['max_depth'])
                    elif model2_name == 'Sarimax':
                        model2_params['p'] = int(model2_params['p'])
                        model2_params['d'] = int(model2_params['d'])
                        model2_params['q'] = int(model2_params['q'])
                        model2_params['seasonal_p'] = int(model2_params['seasonal_p'])
                        model2_params['seasonal_d'] = int(model2_params['seasonal_d'])
                        model2_params['seasonal_q'] = int(model2_params['seasonal_q'])
                    elif model2_name == 'LSTM':
                        model2_params['batch_size'] = int(model2_params['batch_size'])
                        model2_params['lstm_units'] = int(model2_params['lstm_units'])
                        model2_params['lstm_layers'] = int(model2_params['lstm_layers'])
                        model2_params['dense_units'] = int(model2_params['dense_units'])
                        model2_params['dense_layers'] = int(model2_params['dense_layers'])

                    models.append(model_name)
                    models1.append(model1_name)
                    models2.append(model2_name)
                    parameters.append({'parameter': {'model1_params': model1_params, 'model2_params': model2_params}})

                    model = params['model'](freq=self.freq, horizon=self.horizon, model1=model1_name, model2=model2_name,
                                            model1_params=model1_params, model2_params=model2_params)

                else: # base models
                    models.append(model_name)
                    models1.append(None)
                    models2.append(None)

                    hyperparams = params['params']

                    logger.debug(f"""
                    Model type: {model_name}
                    Model params: {hyperparams}
                    """)

                    if model_name == 'Linear':
                        model = params['model'](freq=self.freq, horizon=self.horizon, **hyperparams)
                        parameters.append({'parameter': hyperparams})
                    elif model_name == 'LGB':
                        hyperparams['max_depth'] = int(hyperparams['max_depth'])
                        hyperparams['num_leaves'] = int(hyperparams['num_leaves'])
                        model = params['model'](freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=hyperparams.pop('cyclic_feature_encoding'))
                        model.params = {**model.params, **hyperparams}
                        parameters.append({'parameter': model.params})
                    elif model_name == 'XGB':
                        hyperparams['max_depth'] = int(hyperparams['max_depth'])
                        model = params['model'](freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=hyperparams.pop('cyclic_feature_encoding'))
                        model.params = {**model.params, **hyperparams}
                        parameters.append({'parameter': model.params})
                    elif model_name == 'Prophet':
                        hyperparams['n_changepoints'] = int(hyperparams['n_changepoints'])
                        model = params['model'](freq=self.freq, horizon=self.horizon, **hyperparams)
                        parameters.append({'parameter': hyperparams})
                    elif model_name == 'Sarimax':
                        hyperparams['p'] = int(hyperparams['p'])
                        hyperparams['d'] = int(hyperparams['d'])
                        hyperparams['q'] = int(hyperparams['q'])
                        hyperparams['seasonal_p'] = int(hyperparams['seasonal_p'])
                        hyperparams['seasonal_d'] = int(hyperparams['seasonal_d'])
                        hyperparams['seasonal_q'] = int(hyperparams['seasonal_q'])
                        model = params['model'](freq=self.freq, horizon=self.horizon, **hyperparams)
                        parameters.append({'parameter': hyperparams})
                    elif model_name == 'LSTM':
                        hyperparams['batch_size'] = int(hyperparams['batch_size'])
                        hyperparams['lstm_units'] = int(hyperparams['lstm_units'])
                        hyperparams['lstm_layers'] = int(hyperparams['lstm_layers'])
                        hyperparams['dense_units'] = int(hyperparams['dense_units'])
                        hyperparams['dense_layers'] = int(hyperparams['dense_layers'])
                        model = params['model'](freq=self.freq, horizon=self.horizon, **hyperparams)
                        parameters.append({'parameter': hyperparams})

                try:
                    error = back_test(model, X, test_size=self.test_size, stride=self.stride, retrain_stride=self.retrain_stride)[f'{self.metric}_mean']
                except:
                    error = LARGE_ERROR

                errors.append(error)
                end = time.time()
                times.append(end-start)

                return error

            logger.info(f'Start hyperparameter tuning ({self.method})')

            trials = Trials()
            anneal_search = fmin(
                fn=objective, 
                space=param_grid, 
                max_evals=self.iteration, 
                algo=anneal.suggest, 
                rstate=np.random.default_rng(2022),
                trials=trials,
                early_stop_fn=no_progress_loss(self.early_stopping_steps)
            )

            logger.info('Hyperparameter tuning completed')

            # for tracking
            df_errors = pd.DataFrame(errors)
            df_params = pd.DataFrame(parameters) 
            df_models = pd.DataFrame(models) 
            df_models1 = pd.DataFrame(models1) 
            df_models2 = pd.DataFrame(models2) 
            df_res = pd.concat([df_params, df_errors, df_models, df_models1, df_models2], axis=1)
            df_time = pd.concat([pd.DataFrame(times), df_models, df_models1, df_models2], axis=1)

            model_idx = anneal_search.pop('regressor')
            model_name = self.models[model_idx]

            self.best_model_metric = {
                self.metric: min(trials.losses())
            }

            if model_name == 'Hybrid':
                model1_idx = anneal_search.pop('stage1')
                model1_name = self.hybrid_stage1[model1_idx]
                model2_idx = anneal_search.pop('stage2')
                model2_name = self.hybrid_stage2[model2_idx]

                model1_params = {}
                model2_params = {}

                for k, v in anneal_search.items():
                    m = k.split('_')[-1]
                    param_name = '_'.join(k.split('_')[:-1])
                    if m in ['linearTrend', 'prophet']:
                        model1_params[param_name] = v
                    else:
                        model2_params[param_name] = v

                if model1_name != 'LinearTrend': #linear_trend has no cat params
                    cat_params = CATEGORICAL_PARAMS[model1_name]
                    for p in model1_params.keys():
                        if p in cat_params.keys():
                            model1_params[p] = cat_params[p][model1_params[p]]

                cat_params = CATEGORICAL_PARAMS[model2_name]
                for p in model2_params.keys():
                    if p in cat_params.keys():
                        model2_params[p] = cat_params[p][model2_params[p]]

                # model 1 integer param handling
                if model1_name == 'LinearTrend':
                    model1_params['trend_order'] = int(model1_params['trend_order'])
                elif model1_name == 'Prophet':
                    model1_params['n_changepoints'] = int(model1_params['n_changepoints'])
                
                # model 2 integer param handling
                if model2_name == 'LGB':
                    model2_params['max_depth'] = int(model2_params['max_depth'])
                    model2_params['num_leaves'] = int(model2_params['num_leaves'])
                elif model2_name == 'XGB':
                    model2_params['max_depth'] = int(model2_params['max_depth'])
                elif model2_name == 'Sarimax':
                    model2_params['p'] = int(model2_params['p'])
                    model2_params['d'] = int(model2_params['d'])
                    model2_params['q'] = int(model2_params['q'])
                    model2_params['seasonal_p'] = int(model2_params['seasonal_p'])
                    model2_params['seasonal_d'] = int(model2_params['seasonal_d'])
                    model2_params['seasonal_q'] = int(model2_params['seasonal_q'])
                elif model2_name == 'LSTM':
                    model2_params['batch_size'] = int(model2_params['batch_size'])
                    model2_params['lstm_units'] = int(model2_params['lstm_units'])
                    model2_params['lstm_layers'] = int(model2_params['lstm_layers'])
                    model2_params['dense_units'] = int(model2_params['dense_units'])
                    model2_params['dense_layers'] = int(model2_params['dense_layers'])

                best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, 
                                                model1=model1_name, model2=model2_name,
                                                model1_params=model1_params, model2_params=model2_params)

                self.best_model_params = {
                    model_name: {
                        model1_name: model1_params,
                        model2_name: model2_params
                    }
                }

            else:
                hyperparams = {'_'.join(k.split('_')[:-1]):v for (k, v) in anneal_search.items()}
                cat_params = CATEGORICAL_PARAMS[model_name]

                for p in hyperparams.keys():
                    if p in cat_params.keys():
                        hyperparams[p] = cat_params[p][hyperparams[p]]

                if model_name == 'Linear':
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, **hyperparams)
                elif model_name == 'LGB':
                    hyperparams['max_depth'] = int(hyperparams['max_depth'])
                    hyperparams['num_leaves'] = int(hyperparams['num_leaves'])
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=hyperparams.pop('cyclic_feature_encoding'))
                    best_model.params = {**best_model.params, **hyperparams}
                elif model_name == 'XGB':
                    hyperparams['max_depth'] = int(hyperparams['max_depth'])
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=hyperparams.pop('cyclic_feature_encoding'))
                    best_model.params = {**best_model.params, **hyperparams}
                elif model_name == 'Prophet':
                    hyperparams['n_changepoints'] = int(hyperparams['n_changepoints'])
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, **hyperparams)
                elif model_name == 'Sarimax':
                    hyperparams['p'] = int(hyperparams['p'])
                    hyperparams['d'] = int(hyperparams['d'])
                    hyperparams['q'] = int(hyperparams['q'])
                    hyperparams['seasonal_p'] = int(hyperparams['seasonal_p'])
                    hyperparams['seasonal_d'] = int(hyperparams['seasonal_d'])
                    hyperparams['seasonal_q'] = int(hyperparams['seasonal_q'])
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, **hyperparams)
                elif model_name == 'LSTM':
                    hyperparams['batch_size'] = int(hyperparams['batch_size'])
                    hyperparams['lstm_units'] = int(hyperparams['lstm_units'])
                    hyperparams['lstm_layers'] = int(hyperparams['lstm_layers'])
                    hyperparams['dense_units'] = int(hyperparams['dense_units'])
                    hyperparams['dense_layers'] = int(hyperparams['dense_layers'])
                    self.best_model_params = {model_name: hyperparams}
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, **hyperparams)

            best_model.fit(X)

            logger.info('Forceaster training completed')
            logger.info(f'Best model: {self.best_model_params}')

            self.best_model = best_model
            df_res.columns = ['parameter', 'error', 'model', 'model1', 'model2']
            self.df_res = df_res
            df_time.columns = ['time', 'model', 'model1', 'model2']
            self.df_time = df_time
        
        else: # randomized search
            logger.info(f'Start hyperparameter tuning ({self.method})')

            df_res = pd.DataFrame() # for tracking
            df_time = pd.DataFrame() # for tracking

            if 'Hybrid' in self.models:
                self.models.remove('Hybrid')
                param_grid1 = [{'model_name': m, 'params': PARAMS_RANDOMIZED_SEARCH[m]} for m in self.hybrid_stage1]
                param_grid2 = [{'model_name': m, 'params': PARAMS_RANDOMIZED_SEARCH[m]} for m in self.hybrid_stage2]
                model = HybridModel(freq=self.freq, horizon=self.horizon)
                start = time.time()
                temp = randomized_search(model, X, hybrid_model1=self.hybrid_stage1, hybrid_model2=self.hybrid_stage2, model1_parameters=param_grid1, model2_parameters=param_grid2, n_iter=self.iteration, test_size=self.test_size, stride=self.stride, retrain_stride=self.retrain_stride)
                df_res = df_res.append(temp)
                end = time.time()
                curr_time = {'model': 'Hybrid', 'time': end-start}
                df_time = df_time.append(curr_time, ignore_index=True)

            param_grid = [{'model': MODELS[m], 'model_name': m, 'params': PARAMS_RANDOMIZED_SEARCH[m]} for m in self.models]

            for m in param_grid:
                model_name = m['model_name']
                hyperparams = m['params']
                model = m['model'](freq=self.freq, horizon=self.horizon)
                start = time.time()
                temp = randomized_search(model, X, parameters=hyperparams, n_iter=self.iteration, test_size=self.test_size, stride=self.stride, retrain_stride=self.retrain_stride)
                df_res = df_res.append(temp)
                end = time.time()
                curr_time = {'model': model_name, 'time': end-start}
                df_time = df_time.append(curr_time, ignore_index=True)
            
            logger.info('Hyperparameter tuning completed')

            self.df_time = df_time
            self.df_res = df_res
            df_res_sorted = df_res.sort_values(by=[f'{self.metric}_mean', f'{self.metric}_std'], ascending=True)
            best_res = df_res_sorted.iloc[0]
            model_name = best_res['model']

            self.best_model_metric = {
                self.metric: best_res[f'{self.metric}_mean']
            }

            if model_name == 'Hybrid':
                model1_name = best_res['model1']
                model2_name = best_res['model2']
                model1_params = best_res['parameter']['model1_params']
                model2_params = best_res['parameter']['model2_params']
                best_model = HybridModel(freq=self.freq, horizon=self.horizon, model1=model1_name, model2=model2_name, model1_params=model1_params, model2_params=model2_params)
                
                self.best_model_params = {
                    model_name: {
                        model1_name: model1_params,
                        model2_name: model2_params
                    }
                }
            
            else: # base models
                hyperparams = best_res['parameter']
                self.best_model_params = {model_name: hyperparams}

                if model_name in ['Linear', 'Prophet', 'Sarimax', 'LSTM']:
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, **hyperparams)
                elif model_name in ['LGB', 'XGB']:
                    best_model = MODELS[model_name](freq=self.freq, horizon=self.horizon, cyclic_feature_encoding=hyperparams.pop('cyclic_feature_encoding'))
                    best_model.params = {**best_model.params, **hyperparams}

            best_model.fit(X)

            logger.info('Forecaster training completed')
            logger.info(f'Best model: {self.best_model_params}')

            self.best_model = best_model

        return self

    def predict(self, X):
        """ Generate forecast predictions using the best model and parameters

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The data used to generate forecast predictions.

        Returns
        -------
        pandas.DataFrame
            Time series dataframe containing predictions for the forecast horizon

        """
        return self.best_model.predict(X)
    
