'''
    This python file contains methods for feature engineering for anomaly detection.
'''

import datetime as dt
import os

import numpy as np
from adtk.detector import LevelShiftAD
from configparser import ConfigParser
from scipy import stats
from adtk.visualization import plot

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file
file_path = 'config/config.ini' # path of the config file relative to this py file
abs_path = os.path.join(dir_path, '..', file_path) # absolute config file path

config_object = ConfigParser()
config_object.read(abs_path)
feature_params = config_object['feature_params']

lvl_agg_period = feature_params['lvl_agg_period'] # 1 day
lvl_shift_window_size_mean_wkday_sep = eval(feature_params['lvl_shift_window_size_mean_wkday_sep']) # (7,7)
lvl_shift_window_size_mean_no_sep = eval(feature_params['lvl_shift_window_size_mean_no_sep']) # (5,5)
lvl_shift_c_mean = eval(feature_params['lvl_shift_c_mean']) # 3
lvl_shift_side_mean = feature_params['lvl_shift_side_mean'] # both

lvl_shift_window_size_30min = eval(feature_params['lvl_shift_window_size_30min']) # (7*48,7*48)
lvl_shift_c_30min = eval(feature_params['lvl_shift_c_30min']) # 6
lvl_shift_side_30min = feature_params['lvl_shift_side_30min'] # both

min_lvl_period = eval(feature_params['min_lvl_period']) # 7 days
pvalue_criteria = eval(feature_params['pvalue_criteria']) # 0.01
iqr_factor = eval(feature_params['iqr_factor']) #3

def remove_level_shift(df, sep_weekend, additional_days):
    '''
    remove_level_shift
        * remove level shift effect from the consumption data using 90 days prior to detection date.
        * Level shift identification is performed using LevelShiftAD from adtk.detector package.
        * For sep_weekend = True, presence of level shift is evaluated using daily mean over (7 days, 7 days) rolling window. 
        * For sep_weekend = False, presence of level shift is evaluated using daily mean over (5 days, 5 days) rolling window. 
        * Because level shift evaluation is determined using a rolling window, additional 7 days prior or 5 days prior to earliest date in 90 days 
            is required to cater for the first rolling window (i.e. earliest date in 90 days look back).
        * After level shift determination, the additional 7 or 5 days are dropped.
        * If level shift is present, t-test will be performed to check for comparable means between most recent period and past periods. 
        * Past periods will similar means as the most recent period will be regarded as normal consumption period.
        * t-test is performed using 30 min timestamp data and daily mean data.
        * Most recent period regarded to have normal consumption should be over at least 7 days.
    input:
        df (pandas.DataFrame): dataframe (index: Datetime col: Value) containing look back period
        sep_weekend: boolean indicating whether weekends have different consumption levels compared to weekdays
        additional_days: integer (days) indicating number of days added to df to calculate the first window for level shift evaluation 
    output:
        norm_df (pandas.DataFrame): creates a 'date' column and a 'level_shift' column which takes boolean values to indicate if the timestamp has a level shift consumption.
    '''

    if sep_weekend:
        temp_mean_df = df.resample(lvl_agg_period).mean().copy()
        level_shift_ad = LevelShiftAD(window = lvl_shift_window_size_mean_wkday_sep, c=lvl_shift_c_mean, side=lvl_shift_side_mean)
        anomalies = level_shift_ad.fit_detect(temp_mean_df)
        anomalies_list = list(anomalies.loc[anomalies.Value == True].index.date)

    else:
        temp_mean_df = df.resample(lvl_agg_period).mean().copy()
        level_shift_ad = LevelShiftAD(window = lvl_shift_window_size_mean_no_sep, c=lvl_shift_c_mean, side=lvl_shift_side_mean)
        anomalies = level_shift_ad.fit_detect(temp_mean_df)
        anomalies_list = list(anomalies.loc[anomalies.Value == True].index.date)     

    anomalies_list.sort()
    # plot(temp_mean_df, anomaly=anomalies, anomaly_color='red',figsize=(20,3))       # only for visualisation, not required for analysis
    
    # removing additional days from df after level shift has been determined
    df['date'] = df.index.date
    df = df[df.date >= df.date[0] + dt.timedelta(days=additional_days)].copy()
    temp_mean_df = temp_mean_df[temp_mean_df.index.date >= temp_mean_df.index.date[0] + dt.timedelta(days=additional_days)].copy()

    level_shifts = [df.index[0].date(),] + anomalies_list
    end_date = df.index[-1].date()

    # scenario 1: no anomaly detected
    if len(anomalies_list) < 1:
        df['normal_level'] = True
        return df

    # scenario 2: all anomalies are within min_lvl_period
    # if so, all dates are considered normal consumption
    if (end_date - anomalies_list[0]).days < min_lvl_period:
        df['normal_level'] = True
        return df
    
    # scenario 3: anomalies are detected beyond and within min_lvl_period
    # scenario has spikes in consumption and are not true level shifts 
    # objective here is to identify past normal consumption and keep them
    # firstly, anomalies detected within min_lvl_period will not be considered as normal consumption   
    if (end_date - anomalies_list[-1]).days < min_lvl_period:
        while (end_date - level_shifts[-1]).days < min_lvl_period:
            level_shifts.pop(-1)
    level_shifts = level_shifts + [end_date + dt.timedelta(days =1)]

    # keep df with interval of min_lvl_period days or more to check if the consumption level is comparable with
    # the most recent consumption
    subset_df_list = []
    subset_df_list_30min = []
    for date in range(len(level_shifts)-1):
        if date == 0:
            if (level_shifts[date+1] - level_shifts[date]).days >= min_lvl_period:
                subset_df = temp_mean_df[temp_mean_df.index.date < level_shifts[date+1]].copy()
                subset_df_list.append([subset_df])
                subset_df_30min = df[df.index.date < level_shifts[date+1]].copy()
                subset_df_list_30min.append([subset_df_30min])  

        elif (level_shifts[date+1] - level_shifts[date]).days > min_lvl_period:
            subset_df = temp_mean_df[(temp_mean_df.index.date < level_shifts[date+1]) & (temp_mean_df.index.date > level_shifts[date])].copy()
            subset_df_list.append([subset_df])
            subset_df_30min = df[(df.index.date < level_shifts[date+1]) & (df.index.date > level_shifts[date])].copy()
            subset_df_list_30min.append([subset_df_30min])

    # compare each subset df with the most recent df to see if consumption levels are similiar
    normal_consump_dates_list = []
    if len(subset_df_list) > 1:
        most_recent_df = subset_df_list[-1]
        most_recent_df_30min = subset_df_list_30min[-1]                      
        for subset in range(len(subset_df_list)-1):
            past_df = subset_df_list[subset]

            pvalue = stats.ttest_ind(most_recent_df[0].Value, past_df[0].Value).pvalue
            # print('mean-pvalue:', pvalue)
  
            # pvalues obtained by using 30minute interval data                 
            past_df_30min = subset_df_list_30min[subset]
            pvalue_30min = stats.ttest_ind(most_recent_df_30min[0].Value, past_df_30min[0].Value).pvalue
            # print('30min interval pvalue:', pvalue_30min)
                
            if pvalue > pvalue_criteria or pvalue_30min > pvalue_criteria:
                normal_consump_dates_list.extend(past_df[0].index.date)

        normal_consump_dates_list.extend(most_recent_df[0].index.date)
        df['normal_level'] = df['date'].apply(lambda x: True if x in(normal_consump_dates_list) else False)         
        return df

    else:
        df['normal_level'] = df['date'].apply(lambda x: True if x in(subset_df_list[0][0].index.date) else False)
        return df

def remove_extreme_values(df, columns, factor = iqr_factor):
    '''
    remove_extreme_values
        remove outlier values from look back period that exceeds q1 - factor * IQR or q3 + factor * IQR and replace with median values
        In the df, for these outliers, the corresponding 'normal' column will be updated to False.
    input:
        df (pandas.DataFrame): dataframe (index: Datetime col: Value, normal, time)
        columns (list of str): columns selected to remove extreme values with median values
        factor (float): factor to be multiplied to interquartile range. values that exceed q1 - factor * IQR or 
                        q3 + factor * IQR will be removed
    output:
        output_df (pandas.DataFrame): dataframe with extreme values replaced by median values
    '''
    temp_df = df[df['normal_level'] == True]
    if len(temp_df) > 15 * 48:
        for col in columns:
            q1_dict = temp_df.groupby('time')[col].quantile(.25).to_dict()
            q3_dict = temp_df.groupby('time')[col].quantile(.75).to_dict()
            median_dict = temp_df.groupby('time')[col].quantile(.5).to_dict()

        lower_bound_dict = {}
        upper_bound_dict = {}
        for time in np.unique(df.index.time):
            lower_bound_dict[time] = q1_dict[time] - factor * (q3_dict[time] - q1_dict[time])
            upper_bound_dict[time] = q3_dict[time] + factor * (q3_dict[time] - q1_dict[time])

        for time in np.unique(temp_df.index.time):
            upper_bound = upper_bound_dict[time]
            lower_bound = lower_bound_dict[time]
            median_value = median_dict[time]
            df.loc[df['time'] == time, 'not_outlier'] = df.loc[df['time'] == time, 'Value'].apply(lambda x: False if x > upper_bound else True)
            df.loc[df['time'] == time, 'not_outlier'] = df.loc[df['time'] == time, 'Value'].apply(lambda x: False if x < lower_bound else True)
            df.loc[df['time'] == time, 'Value'] = df.loc[df['time'] == time, 'Value'].apply(lambda x: median_value if x > upper_bound else x)
            df.loc[df['time'] == time, 'Value'] = df.loc[df['time'] == time, 'Value'].apply(lambda x: median_value if x < lower_bound else x)
 
        df['normal'] = df[['normal_level', 'not_outlier']].apply(lambda row: True if (row['normal_level'] == True and row['not_outlier'] == True) else False, axis = 1)
        return df
    
    else:
        df['normal'] = df['normal_level']
        return df