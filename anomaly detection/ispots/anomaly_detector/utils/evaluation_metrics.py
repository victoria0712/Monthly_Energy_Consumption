'''
    This python file contains methods for evaluation metrics for anomaly
    detection.
'''

import numpy as np
import os

from configparser import ConfigParser

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file
file_path = 'config/config.ini' # path of the config file relative to this py file
abs_path = os.path.join(dir_path, '..', file_path) # absolute config file path

config_object = ConfigParser()
config_object.read(abs_path)
eval_params = config_object['evaluation_params']

lvl_con_number = eval(eval_params['consecutive_number_level_shift']) # 16
dev_scr_threshold = eval(eval_params['deviation_score_threshold']) # 3 # only for norm_metric.

#for level shift
std_min_divide_amount = eval(eval_params['std_min_divide_amount']) # 0.01
lvl_anomaly_threshold = eval(eval_params['lvl_anomaly_threshold']) # 2
sequence_anomaly_threshold = eval(eval_params['sequence_anomaly_threshold']) # 3
point_anomaly_threshold = eval(eval_params['point_anomaly_threshold']) # 4
consecutive_number = eval(eval_params['consecutive_number']) # 4

def identify_anomaly(df):
    '''
    <identify_anomaly> (default)
        classifies the anomalies into point, sequence, or level shift
        calulate the total deviation and % deviation for the detection date
    input: 
        dataframe (pandas.dataFrame): input dataframe with predicted, stdev columns
    output:
        temp_df (pandas.dataFrame): output dataframe with weekday, weekend, med, std, mean, deviation, 
        anomaly, lvl_anomaly, p_anomaly, and con_seq_anomaly columns
        deviation_value (float): total deviation based on anomalies identified
        dev_per (float): percentage deviation based on anomalies identified against the total expected consumption 
    '''
    temp_df = df.copy()

    # label anomaly type
    temp_df = classify_anomaly(temp_df)

    # column to specify that anomaly is detected 
    conditions = [temp_df['lvl_anomaly'].eq(True), temp_df['p_anomaly'].eq(True), temp_df['con_seq_anomaly'].eq(True)]
    choices = [1,1,1]
    temp_df['anomaly'] = np.select(conditions,choices,default=0)

    # calulate the total deviation and % deviation 
    anomaly_df = temp_df.loc[temp_df.anomaly == 1].copy()
    expected_value = round(sum(temp_df.pred),2)
    deviation_value = round(sum(anomaly_df.deviation),2)
    dev_per = round(100 * deviation_value / expected_value if expected_value != 0 else 0,2)

    return temp_df, deviation_value, dev_per

def classify_anomaly(df):
    '''
    <classify_anomaly>
        helper function to classify anomaly as point, sequence, or level shift for statistical model
    input:
        df (pd.DataFrame): input dataframe with deviation values
    output:
        df (pd.DataFrame): output dataframe with point, sequence, or level shift label
    '''
    temp_df = df.copy()
    std_dev = temp_df['std']            
    std_dev = std_dev.apply(lambda x: x if x > std_min_divide_amount else std_min_divide_amount)
    temp_df['lvl_anomaly'] = (temp_df['deviation'] - lvl_anomaly_threshold*std_dev).apply(lambda x: x > 0)
    # check whether level shift is observed
    if sum(temp_df['lvl_anomaly']) == len(temp_df):
        temp_df['p_anomaly'] = False
        temp_df['con_seq_anomaly'] = False
        return temp_df
    else:
        temp_df['lvl_anomaly'] = False
        # check whether criteria for point anomaly is met
        temp_df['p_anomaly'] = (temp_df['deviation'] - point_anomaly_threshold*std_dev).apply(lambda x: x > 0)
        # check whether criteria for sequence anomaly is met
        temp_df['seq_anomaly'] = (temp_df['deviation'] - sequence_anomaly_threshold*std_dev).apply(lambda x: x > 0)
        temp_df['con_seq_anomaly'] = False
            
        # label consecutive sequence anomaly
        for i in range(consecutive_number, len(temp_df)+1): 
            if temp_df.iloc[i-consecutive_number:].head(4)['seq_anomaly'].sum() == consecutive_number:
                for j in range(consecutive_number):
                    temp_df.loc[temp_df.index[i-j-1],'con_seq_anomaly'] = True
        # label consecutive point anomaly as sequence anomaly
        for i in range(consecutive_number, len(temp_df)+1): 
            if temp_df.iloc[i-consecutive_number:].head(consecutive_number)['p_anomaly'].sum() >= 2:
                for j in range(consecutive_number):
                    if temp_df.loc[temp_df.index[i-j-1],'p_anomaly'] == True:
                        temp_df.loc[temp_df.index[i-j-1],'con_seq_anomaly'] = True
    return temp_df

def norm_dev_metric(df, min_div_amt, agg_method = lambda x: np.mean(x)):
    '''
    <norm_dev_metric>
        calculate the anomaly score based on deviation score of each time point
        anomaly score is calculated by deviation as difference between actual and median consumption value 
        and then standardize the deviation score based on the median and standard deviation value of deviation values
        classifies the anomalies into point, sequence, or level shift
    input: 
        dataframe (pandas.dataFrame): input dataframe with weekday, weekend, med, std, mean columns
        min_div_amt (float): minimum divisor value
        aggregate method (function): aggregate the anomaly_score of each time point (default as mean)
    output:
        score (float): anomaly score aggregated for one day
        score_df (pandas.dataFrame): output dataframe with weekday, weekend, med, std, mean, deviation, 
        anomaly classification, deviation_score, anomaly, con_anomaly, anomaly_score columns
    '''
    temp_df = df.copy()
    temp_df['deviation'] = np.abs((temp_df['Value'] - temp_df['med']))
    med_dev = np.median(temp_df['deviation'])
    std_dev = np.std(temp_df['deviation'])
    # adjust the standard deviation to be greater than minimum divisor value
    std_dev = std_dev if std_dev > min_div_amt else min_div_amt
    temp_df['deviation_score'] = (temp_df['deviation'] - med_dev) / std_dev
    # label anomalies (deviation score higher then threshold)
    temp_df['anomaly'] = temp_df['deviation_score'] > dev_scr_threshold
    # label anomaly type
    temp_df = classify_anomaly(temp_df)
    temp_df['anomaly_score'] = temp_df['anomaly'] * temp_df['deviation_score']
    return {'score': agg_method(temp_df['anomaly_score']), 'score_df': temp_df}

def fb_pred_metric(df):
    '''
    <fb_pred_metric>
        classifies the anomalies into point, sequence, or level shift
        calculate the anomaly score based on deviation score of each time point
        anomaly score is calculated by deviation as difference between actual and predicted value
        and then divide the deviation by standard deviation of consumption
        (aggregate deviation score of consecutive anomalies only)
    input: 
        dataframe (pandas.dataFrame): input dataframe with weekday, weekend, med, std, mean columns
        min_div_amt (float): minimum divisor value
        aggregate method (function): aggregate the anomaly_score of each time point (default as mean)
    output:
        score (float): anomaly score aggregated for one day
        score_df (pandas.dataFrame): output dataframe with weekday, weekend, med, std, mean, deviation, 
        anomaly classification, deviation_score, anomaly, con_anomaly, anomaly_score pred, pred_low, pred_high columns
    '''
    temp_df = df.copy()
    temp_df['deviation'] = np.abs(temp_df['Value'] - temp_df['pred'])
    # label anomalies (actual value higher then pred_high or lower than pred_low)
    temp_df['anomaly'] = (temp_df['Value'] > temp_df['pred_high']) + (temp_df['Value'] < temp_df['pred_low'])
    # label anomaly type
    temp_df = classify_anomaly_prophet(temp_df)
    return temp_df

def classify_anomaly_prophet(df):
    '''
    <classify_anomaly_prophet>
        helper function to classify anomaly as point, sequence, or level shift for prophet model
    input:
        df (pd.DataFrame): input dataframe with deviation values
    output:
        df (pd.DataFrame): output dataframe with point, sequence, or level shift label
    '''
    temp_df = df.copy()
    # check whether level shift is observed
    if sum(temp_df['anomaly']) == len(temp_df):
        temp_df['lvl_anomaly'] = True
        temp_df['p_anomaly'] = False
        temp_df['con_seq_anomaly'] = False
        return temp_df
    else:
        temp_df['lvl_anomaly'] = False
        temp_df['con_seq_anomaly'] = False
            
        # label consecutive sequence anomaly
        for i in range(consecutive_number, len(temp_df)+1): 
            if temp_df.iloc[i-consecutive_number:].head(4)['anomaly'].sum() == consecutive_number:
                for j in range(consecutive_number):
                    temp_df.loc[temp_df.index[i-j-1],'con_seq_anomaly'] = True
        # label consecutive point anomaly as sequence anomaly
        for i in range(consecutive_number, len(temp_df)+1): 
            if temp_df.iloc[i-consecutive_number:].head(consecutive_number)['anomaly'].sum() >= 2:
                for j in range(consecutive_number):
                    temp_df.loc[temp_df.index[i-j-1],'con_seq_anomaly'] = True
        # label remaining anomalies as point anomalies
        temp_df['p_anomaly'] = False
        temp_df.loc[(df['anomaly'] == True) & (temp_df['con_seq_anomaly'] == False), 'p_anomaly'] = True
    return temp_df