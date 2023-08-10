from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf

def feature_builder(df, freq, model_name, cyclic_feature_encoding):
    """ Creates features for time series dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 1)
        Univariate time series dataframe that has been preprocessed
    
    freq: str
        Frequency of the time series data
    
    model_name: {'Linear','LGB','XGB','Prophet','Sarimax','LSTM'}
        Model that will be fitted with the new features to do forecasting

    cyclic_feature_encoding: {'sincos','onehot'}
         Cyclic feature encoding method

    Returns
    -------
    if model_name in ['Linear', 'LGB', 'XGB'],
        (X, y, features): (numpy.ndarray, numpy.ndarray, list)
            X: nested array of feature values with shape (n_samples, n_features)
            y: array of ycol values with shape (n_samples, 1)
            features: list of feature names

    if model_name in ['Prophet', 'Sarimax', 'LSTM'],
        pandas.DataFrame: dataframe of shape (n_samples, n_features)
        
    """
    # df is dataframe
    df_new = df.copy()

    if model_name in ['Linear', 'LGB', 'XGB']:
        if pd.Timedelta(freq) < pd.Timedelta('1d'):
            periods = int(pd.Timedelta('1d') / pd.Timedelta(freq))
        elif pd.Timedelta(freq) == pd.Timedelta('1d'):
            periods = 7

        look_back_cycle = 4 if pd.Timedelta(freq) == pd.Timedelta('1d') else 7

        X = []
        X_temp = []
        feature_names = []

        # lag feature
        for i in range(look_back_cycle*periods):
            X.append(df_new.shift(periods=i+1).values)
            feature_names.append(f'lag_{i+1}')

        # median in look_back_cycle
        for i in range(look_back_cycle):
            X_temp.append(df_new.shift(periods=periods*(i+1)).values)

        X_temp = np.concatenate(X_temp, axis=1)
        X.append(np.median(X_temp, axis=1, keepdims=True))
        feature_names.append('median_look_back')

        # std in look_back_cycle
        X.append(np.std(X_temp, axis=1, keepdims=True))
        feature_names.append('std_look_back')

        # ratio in look_back_cycle
        idx_pair = combinations(range(look_back_cycle), 2)
        for idx_1, idx_2 in idx_pair:
            X.append(X_temp[:,[idx_1]] / (X_temp[:,[idx_2]] + 0.01))
            feature_names.append(f'ratio_{idx_1+1}d_{idx_2+1}d')

        # lag ratio
        for n in range(periods):
            #lag_n ratio
            X_lag_ratio = df_new.shift(periods=n+1) / (df_new.shift(periods=n+2) + 0.01)
            # X.append(X_lag_1_ratio.values)
            X_temp = []
            #median lag_n ratio in look_back_ratio
            look_back_cycle_temp = look_back_cycle if n != periods - 1 else look_back_cycle - 1

            for i in range(look_back_cycle_temp):
                X.append(X_lag_ratio.shift(periods=periods*i).values)
                feature_names.append(f'lag_{n+1}_ratio_{i+1}d')
                X_temp.append(X_lag_ratio.shift(periods=periods*i).values)

            X_temp = np.concatenate(X_temp, axis=1)
            X.append(np.median(X_temp, axis=1, keepdims=True))
            feature_names.append(f'median_lag_{n+1}_ratio')

            #std lag_n ratio in look_back_ratio
            X.append(np.std(X_temp, axis=1, keepdims=True))
            feature_names.append(f'std_lag_{n+1}_ratio')

            # ratio of ratio in look_back_cycle
            idx_pair = combinations(range(look_back_cycle_temp), 2)
            for idx_1, idx_2 in idx_pair:
                X.append(X_temp[:,[idx_1]] / (X_temp[:, [idx_2]] + 0.01))
                feature_names.append(f'lag_{n+1}_ratio_of_ratio_{idx_1+1}d_{idx_2+1}d')

        if cyclic_feature_encoding == 'onehot':
            # hour features
            if pd.Timedelta(freq) < pd.Timedelta('1d'):
                X.append(pd.get_dummies(df_new.index.hour).values)
                for i in range(1,25):
                    feature_names.append(f'hour_{i}')

            # dayofweek features
            X.append(pd.get_dummies(df_new.index.dayofweek).values)
            for i in range(1,8):
                feature_names.append(f'day_{i}_of_week')
        
        elif cyclic_feature_encoding == 'sincos':
            # hour features
            if pd.Timedelta(freq) < pd.Timedelta('1d'):
                X.append(np.sin(df_new.index.hour.values * (2 * np.pi / 24))[...,None])
                feature_names.append(f'sin_hour')
                X.append(np.cos(df_new.index.hour.values * (2 * np.pi / 24))[...,None])
                feature_names.append(f'cos_hour')

            # dayofweek features
            X.append(np.sin(df_new.index.dayofweek.values * (2 * np.pi / 7))[...,None])
            feature_names.append(f'sin_day_of_week')
            X.append(np.cos(df_new.index.dayofweek.values * (2 * np.pi / 7))[...,None])
            feature_names.append(f'cos_day_of_week')

        # weekend feature
        if df_new.index.dayofweek[0] >= 5:
            X.append(pd.get_dummies(df_new.index.dayofweek >= 5).values[:,[0]])
        else:
            X.append(pd.get_dummies(df_new.index.dayofweek >= 5).values[:,[-1]])
        feature_names.append('weekend_true')

        # # dayofmonth features
        # X.append(pd.get_dummies(df_new.index.day).values)

        X = np.concatenate(X, axis=1)

        return X[look_back_cycle*periods:], df_new.iloc[look_back_cycle*periods:].values, feature_names
    
    elif model_name in ['Prophet', 'Sarimax', 'LSTM']:
        regressors = {}
        if cyclic_feature_encoding == 'onehot':
        # hour features
            if pd.Timedelta(freq) < pd.Timedelta('1d'):
                for i in range(24):
                    regressors[f'hr_{i}'] = (df_new.index.hour == i).astype(int)

            # dayofweek features
            for i in range(7):
                regressors[f'dayofweek_{i}'] = (df_new.index.dayofweek == i).astype(int)
        elif cyclic_feature_encoding == 'sincos':
            # hour features
            if pd.Timedelta(freq) < pd.Timedelta('1d'):
                regressors['hr_sin'] = np.sin(df_new.index.hour * (2 * np.pi / 24))
                regressors['hr_cos'] = np.cos(df_new.index.hour * (2 * np.pi / 24))
            # dayofweek features
            regressors['dayofweek_sin'] = np.sin(df_new.index.dayofweek * (2 * np.pi / 7))
            regressors['dayofweek_cos'] = np.cos(df_new.index.dayofweek * (2 * np.pi / 7))

        # weekend feature
        regressors['weekend'] = (df_new.index.dayofweek >= 5).astype(int)

        # # dayofmonth features

        df_regressors = pd.DataFrame(index=df_new.index, data=regressors)

        return df_regressors


def generate_sequence(X, look_back_periods, fcst_periods):
    """ Generate sliding window X and y sequences based on look back and forecast periods

    Parameters
    ----------
    X: pandas.DataFrame or np.ndarray
        Univariate time series dataframe or array that has been preprocessed
    
    look_back_periods: int
        Number of periods in look back cycle
        
    fcst_periods: int
        Number of periods in forecast

    Returns
    -------
    X: np.ndarray of shape (num_sequences, look_back_periods, num_features)
        Array containing sequences of input data

    y: np.ndarray of shape (num_sequences, fcst_periods, 1)
        Array containing sequences of output data

    """
    if isinstance(X, pd.DataFrame):
        sequence = X.values
    elif isinstance(X, np.ndarray) and X.ndim > 1:
        sequence = X
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        sequence = X[..., None]
    else:
        raise TypeError('X is either pd.DataFrame or np.ndarray')

    starting_idx_range = range(len(sequence)-look_back_periods-fcst_periods+1)
    X = np.stack([sequence[idx:idx+look_back_periods] for idx in starting_idx_range])
    y = np.stack([sequence[idx+look_back_periods:idx+look_back_periods+fcst_periods] for idx in starting_idx_range])

    return X, y


def convert_to_tf_dataset(x, y, batch_size):
    """ Converts a pair of np.ndarray to tf.data.Dataset
    Parameters
    ----------
    x: np.ndarray of shape (num_sequences, look_back_periods, num_features)
        Array containing sequences of input data
    
    y: np.ndarray of shape (num_sequences, fcst_periods, 1)
        Array containing sequences of output data
        
    batch_size: int
        Number of consecutive elements of this dataset to combine in a single batch

    Returns
    -------
    tf.data.Dataset
        Dataset whose elements are slices from the given x and y

    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
