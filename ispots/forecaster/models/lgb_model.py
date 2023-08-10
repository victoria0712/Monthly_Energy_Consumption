import pandas as pd
import lightgbm as lgb

from ispots.forecaster.utils.feature_builder import feature_builder

class LGBModel():
    """ LGB Model which uses light gradient boosting to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    params: dict
        Parameters for LightGBM

    num_boost_round: int, default=5000
        Number of boosting iterations

    val_size: float, default=0.1
        Proportion of dataset to be used as validation set

    early_stopping_rounds: int, default=50
        Number of rounds with no improvement in validation data metric before training is stopped

    cyclic_feature_encoding: {'sincos', 'onehot'}, default='onehot'
        Cyclic feature encoding method
    """
    def __init__(self, freq='30min', horizon='1d', params=None, num_boost_round=5000, val_size=0.1, early_stopping_rounds=20, cyclic_feature_encoding='onehot'):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not suppoeted. Only support daily or daily divisable frequency')

        if cyclic_feature_encoding not in ['sincos', 'onehot']:
            raise ValueError("Supported cyclic_feature_encoding methods are: ['sincos', 'onehot']")

        if params is None:
            self.params = {
                'boosting':'gbdt',
                'objective': 'rmse', 
                'max_depth': 6,
                'min_child_weight': 10, 
                'learning_rate': 0.1,
                'num_leaves': 80,
                'min_gain_to_split': 0.1,
                'subsample': 0.5, 
                'colsample_bytree': 0.5, 
                'colsample_bynode': 0.5, 
                'reg_lambda': 1, 
                'reg_alpha': 0, 
                'bagging_freq': 1,
                'force_col_wise': True, 
                'metric' : 'rmse',
                'verbose': -1
            }
        else:
            self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.freq = freq
        self.horizon = horizon
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.look_back_dur = '4w' if pd.Timedelta(self.freq) == pd.Timedelta('1d') else '7d'
        self.look_back_periods = int(pd.Timedelta(self.look_back_dur) / pd.Timedelta(self.freq))
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'LGB'

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
        X, y, _ = feature_builder(X, self.freq, self.name, self.cyclic_feature_encoding)
        split_loc = int(len(X) * self.val_size)
        X_train, X_val = X[:-split_loc], X[-split_loc:]
        y_train, y_val = y[:-split_loc], y[-split_loc:]
        train_matrix = lgb.Dataset(X_train, label=y_train)
        val_matrix = lgb.Dataset(X_val, label=y_val)
        self.model = lgb.train(
            self.params,
            train_matrix,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_matrix, val_matrix],
            valid_names=['train','validation'],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
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
            raise ValueError(f'At least {self.look_back_dur} data needs to be provided for forecasting')

        X_temp = X.iloc[-self.look_back_periods:].copy()

        for i in range(self.fcst_periods):
            start_time = X_temp.index[i]
            end_time = X_temp.index[-1] + pd.Timedelta(self.freq)
            idx_curr = pd.date_range(start=start_time, end=end_time, freq=self.freq)
            X_curr = X_temp.reindex(index=idx_curr)
            X_feature, _, _ = feature_builder(X_curr, self.freq, self.name, self.cyclic_feature_encoding)
            X_feature = X_feature[[-1]]
            y_fcst = self.model.predict(X_feature, num_iteration=self.model.best_iteration)

            idx_new = pd.date_range(start=X_temp.index[0], end=X_temp.index[-1]+pd.Timedelta(self.freq), freq=self.freq)
            X_temp = X_temp.reindex(idx_new)
            X_temp.iloc[-1] = y_fcst

        fcst = X_temp.iloc[-self.fcst_periods:]
        fcst.index.name = 'ds'
        fcst.columns = ['y']

        return fcst

