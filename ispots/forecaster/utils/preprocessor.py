import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ispots.utils.logger import get_logger

logger = get_logger('forecasting-preprocessor')

def remove_duplicate_rows(df):
    """Returns df with duplicates removed 
    """    
    idx_name = df.index.name

    return df.reset_index().drop_duplicates().set_index(idx_name)

def get_sampling_frequencies(df, freq_th=0.75):
    """Returns list of sampling frequencies in data
    """
    ts = df.index.to_series()
    freq_count = ts.diff().value_counts()
    freq_count = freq_count / len(freq_count)

    if freq_count.iloc[0] >= freq_th:
        return freq_count.index[0]
    else:
        raise RuntimeError('No dominant sampling frequency detected')

def resample_df(df, resample_freq, agg, fullday):
    agg_type = ['mean', 'median', 'max', 'min', 'std']

    if agg not in agg_type:
        raise ValueError(f'{agg} is not a supported aggregation type. Supported aggregation types are {agg_type}')
    
    df = df.resample(resample_freq).agg(agg)
    original_ts = df.index
    start_ts, end_ts = original_ts[0], original_ts[-1]
    start_ts = pd.Timestamp(start_ts.date())

    if fullday:
        end_ts = pd.Timestamp(end_ts.date()) + pd.Timedelta('1day') - pd.Timedelta(resample_freq)

    new_ts = pd.date_range(start=start_ts, end=end_ts, freq=resample_freq)
    df = df.reindex(index=new_ts)
    df.index.name = 'ts'

    return df

def median_imputation(df, **kwargs):
    ycol = kwargs['ycol']
    median_profile = kwargs['median_profile']
    freq = kwargs['freq']
    na_th = kwargs['na_th']

    if df['ts'].max() - df['ts'].min() + pd.Timedelta(freq) >= pd.Timedelta(na_th):
        df[ycol] = df['ts'].dt.time.map(median_profile)

    return df[[ycol]]

def fill_na(df, ycol, median_profile, freq, na_th):
    df.reset_index(inplace=True)
    na_idx = df[ycol].isnull()
    df['count'] = df[ycol].notnull().cumsum()
    df1 = df[na_idx].copy()
    df.loc[na_idx, ycol] = df1.groupby('count').apply(median_imputation, ycol=ycol, median_profile=median_profile, freq=freq, na_th=na_th)
    df.set_index('ts', inplace=True)
    df[ycol] = pd.to_numeric(df[ycol], errors='coerce')
    df[ycol] = df[ycol].interpolate(method='linear',limit_direction = 'both')
    df.drop(columns='count', inplace=True)

    return df

def get_median_profile(df, ycol):
    """Calculates the median profiles for each time index of the day
    """
    df['time'] = df.index.time
    median_profile = df.groupby('time')[ycol].median().to_dict()

    return median_profile

def get_outlier_profile(df, ycol, iqr_coeff):
    """Calculates the outlier profiles for each time index of the day for each month"""
    df['time'] = df.index.time
    df['month'] = df.index.to_period('M')
    
    q1 = df.groupby(['month', 'time'])[ycol].apply(lambda x: np.quantile(x, 0.25))
    q3 = df.groupby(['month', 'time'])[ycol].apply(lambda x: np.quantile(x, 0.75))
    iqr = q3 - q1
    df_stats = df.groupby(['month', 'time']).agg(['median'])[ycol]
    df_stats['lower_bound'] = q1 - iqr_coeff * iqr
    df_stats['upper_bound'] = q3 + iqr_coeff * iqr

    return df_stats[['median', 'lower_bound', 'upper_bound']].to_dict('index')

def is_outlier_helper(y, stats):
    """Helper function for is_outlier function"""
    if y < stats['lower_bound']:
        return True
    elif y > stats['upper_bound']:
        return True
    else:
        return False

def is_outlier(group, ycol, outlier_profile, min_month, max_month):
    """Labels data points lower than lower_bound or higher than upper_bound as outliers"""
    ts = group['time'].values[0]
    m = group['month'].values[0]

    if m < min_month:
        raise Exception("Input data contains data earlier than training data")
    elif m > max_month:
        stats = outlier_profile[max_month, ts]
    else:
        stats = outlier_profile[m, ts]
    
    y = group[ycol].values
    outlier = list(map(lambda x: is_outlier_helper(x, stats), y))

    return outlier

def replace_outlier(x, ycol, outlier_profile, min_month, max_month):
    """Median imputation for 0 values; Capping for other values"""
    ts = x['time'].values[0]
    m = x['month'].values[0]

    if m < min_month:
        raise Exception("Input data contains data earlier than training data")
    elif m > max_month:
        stats = outlier_profile[(max_month, ts)]
    else:
        stats = outlier_profile[(m, ts)]
    
    val = x[ycol].values # array of ycol values
    x['new_y'] = np.where(val>stats['upper_bound'], stats['upper_bound'], np.where(val<=0, stats['median'], np.where(val<stats['lower_bound'], stats['lower_bound'], val)))

    return x[['new_y']].copy()

def handle_outlier(df, outlier_profile, ycol):
    df['time'] = df.index.time
    df['month'] = df.index.to_period('M')
    min_month = min(outlier_profile.keys())[0]
    max_month = max(outlier_profile.keys())[0]
    
    df1 = df.groupby(['month', 'time']).apply(lambda x: replace_outlier(x, ycol, outlier_profile, min_month, max_month))
    df1 = df1.rename(columns={'new_y': ycol})
    outliers = df1[df[ycol] != df1[ycol]].index

    return df1, outliers

class Preprocessor(BaseEstimator, TransformerMixin):
    """ Preprocesses a univariate time series dataframe into a time series dataframe of the specified frequency

    1) Removes duplicated rows
    2) Raise exception if rows with identical datetime but different values are found
    3) Check if one sampling frequency present in the value column
    4) Resample data using specified frequency
    5) Find and replace outliers:
        - For extremely large/ small values, replace them with upper/ lower bound profiles of time index
        - For invalid values (<= 0), replace them with median profiles of time index
    6) Find gap size:
        - For large gaps, fill missing values with median profiles of time index
        - For small gaps, fill missing values with linear interpolation

    Parameters
    ----------
    resample_freq: str, default='30min'
        Resample frequency of time series. If None, raw majority frequency is used.

    na_th: str, default='2h'
        Threshold of data length for median imputation. 

    agg: {'mean', 'median', 'std', 'min', or 'max'), default='mean'
        Aggregation function for resampling
    
    remove_outlier: bool, default=True
        If True, outliers will be removed. 
    
    iqr_coeff: float, default=1.7
        Interquartile range used to obtain the upper and lower bound profiles of each time index.
        Values beyond these bounds are considered outliers.

    Attributes
    ----------
    ycol: str
        Column name of the original dataframe
    
    freq: pd.Timedelta
        Sampling frequency of the dataframe

    median_profile: dict
        Median profiles of each time index

    outlier_profile: dict
        Upper and lower bound profiles of each time index
    """
    def __init__(self, resample_freq='30min', na_th='2h', agg='mean', fullday=False, remove_outlier=True, iqr_coeff=1.7):
        self.resample_freq = resample_freq
        self.na_th = na_th
        self.agg = agg
        self.fullday = fullday
        self.remove_outlier = remove_outlier
        self.iqr_coeff = iqr_coeff

    def fit(self, X):
        """ Finds the ycol, freq, outlier profiles and median profiles of the training data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The data used to compute frequency, outlier profiles and median profiles for later processing.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted proccessor

        """
        if len(X.columns) > 1:
            raise ValueError("Input data is not univariate")

        ## Obtain ycol
        self.ycol = X.columns[0]

        X = remove_duplicate_rows(X.copy(deep=True))
        
        if sum(X.index.duplicated()) != 0:
            raise Exception("Rows with duplicate datetimes detected")

        self.freq = get_sampling_frequencies(X)

        if self.resample_freq is not None:
            self.freq = self.resample_freq

        X = resample_df(X, self.freq, self.agg, self.fullday)

        ## Get Outlier Profiles
        if self.remove_outlier:
            self.outlier_profile = get_outlier_profile(X, self.ycol, self.iqr_coeff)

        ## Get Median Profiles
        self.median_profile = get_median_profile(X, self.ycol)

        return self

    def transform(self, X):
        """ Pre-processes the dataframe using the fitted frequency, outlier profiles and median profiles. 
        
        Parameters
        ----------
        X :  pd.DataFrame of shape (n_samples, 1)
            The univariate data to process and convert into time series of specified freq.

        Returns
        -------
        X_tr : pd.DataFrame shape (n_samples, n_features)
            Time Series Dataframe of specified frequency

        """
        if len(X.columns) > 1:
            raise ValueError("Input data is not univariate")

        ycol = X.columns[0]
        
        if ycol != self.ycol:
             raise ValueError(f'Test data uses ycol=({ycol}) which does not match training data ycol=({self.ycol})')

        X = remove_duplicate_rows(X.copy(deep=True))
        
        if sum(X.index.duplicated()) != 0:
            raise Exception("Rows with duplicate datetimes detected")
        
        X = resample_df(X, self.freq, self.agg, self.fullday)

        ## Replace Outliers
        if self.remove_outlier:
            X, self.outliers = handle_outlier(X, self.outlier_profile, self.ycol)
            self.outlier_pct = round(len(self.outliers) / len(X) * 100, 2)
            logger.info(f"Percentage of outlier: {self.outlier_pct}%")

        ## Find Gap Sizes
        X = fill_na(X, self.ycol, self.median_profile, self.freq, self.na_th)

        X.index.name = 'ds'
        X.columns = ['y']

        return X