import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

def remove_duplicate_rows(df):
    """Returns df with duplicates removed 
    """    
    idx_name = df.index.name

    return df.reset_index().drop_duplicates().set_index(idx_name)

def get_sampling_frequencies(df, freq_th=0.75):
    """Returns list of sampling frequencies in data
    """
    ts = pd.Series(df.index)
    freq_count = ts.diff().value_counts()
    freq_count = freq_count / len(freq_count)

    if freq_count.iloc[0] >= freq_th:
        return freq_count.index[0]
    else:
        raise RuntimeError('No dominant sampling frequency detected')

def resample_df(df, resample_freq, agg):
    agg_type = ['mean', 'median', 'max', 'min', 'std']

    if agg not in agg_type:
        raise ValueError(f'{agg} is not a supported aggregation type. Supported aggregation types are {agg_type}')
    
    df = df.resample(resample_freq).agg(agg)
    original_ts = df.index
    start_ts, end_ts = original_ts[0], original_ts[-1]
    start_ts = pd.Timestamp(start_ts.date())
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

    return df

def get_median_profile(df, ycol):
    """Calculates the median profiles for each time index of the day
    """
    df['time'] = df.index.time
    median_profile = df.groupby('time')[ycol].median().to_dict()

    return median_profile

class Preprocessor(TransformerMixin, BaseEstimator):
    """ Preprocesses a univariate dataframe into a daily time series dataframe

    1) removes duplicated rows
    2) raise exception if we find rows with identical datetime but different values
    3) check if one sampling frequency present in the value column
    4) check Sparse Columns 
    5) Align Data to at 00:00:00 and resample using frequency detected
    6) Find Gap Sizes 
    7) For Large Gaps - fills missing values with median profiles of time index
    8) For Small Gaps - fills missing values with linear interpolation
    9) Convert into daily time series

    Parameters
    ----------
    resample_freq: DataOffset, Timedelta or str, default None
        Resample freqency of time series. If None, raw majority frequency is used

    na_th: DataOffset, Timedelta or str, default '1h'
        Threshold of gas length for median imputation

    agg: {'mean', 'median', 'std', 'min', or 'max'}, default 'mean'
        Aggregation function for resampling

    Attributes
    ----------
    ycol : str
        the column name of the dataframe converted into daily time series 

    freq : pd.Timedelta
        the sampling frequency of the dataframe

    median_profile : pd.DataFrame
        the median profiles of each time index
    """

    def __init__(self, resample_freq=None, na_th='1h', agg='mean'):
        self.resample_freq = resample_freq
        self.na_th = na_th
        self.agg = agg

    def fit(self, X):
        """ Finds the ycol, freq and median profiles of the training data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The data used to compute frequency and median profiles for later processing.
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

        X = resample_df(X, self.freq, self.agg)

        ## Get Median Profiles
        self.median_profile = get_median_profile(X, self.ycol)

        return self

    def transform(self, X):
        """ Pre-processes the dataframe using the fitted frequency and median profiles. 
        
        Parameters
        ----------
        X :  pd.DataFrame of shape (n_samples, 1)
            The univariate data to process and convert into daily time series.

        Returns
        -------
        X_tr : pd.DataFrame shape (n_samples, n_features)
            Daily Time Series Dataframe

        """
        if len(X.columns) > 1:
            raise ValueError("Input data is not univariate")

        ycol = X.columns[0]
        
        if ycol != self.ycol:
             raise ValueError(f'Test data uses ycol=({ycol}) which does not match training data ycol=({self.ycol})')

        X = remove_duplicate_rows(X.copy(deep=True))
        
        if sum(X.index.duplicated()) != 0:
            raise Exception("Rows with duplicate datetimes detected")

        X = resample_df(X, self.freq, self.agg)

        period = int(pd.Timedelta('1d') / self.freq)
        
        if len(X) < (period*2):
            raise Exception("Time Series has less than two days")

        ## Find Gap Sizes
        X = fill_na(X, self.ycol, self.median_profile, self.freq, self.na_th)

        ## Convert into daily time series
        X['date'] = X.index.date
        X['time'] = X.index.time
        X = X.pivot(index = 'time', columns = 'date', values = self.ycol)

        return X