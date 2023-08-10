'''
    This python file is for automatic anomaly detection model. 
    The framework detects and reports top anomalous consumption patterns
    and classifies the anomalies detected into point, sequence
    anomalies with description messages.
'''
import datetime as dt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from configparser import ConfigParser

from ispots.anomaly_detector.utils.evaluation_metrics import identify_anomaly
from ispots.anomaly_detector.models.statistic_model import Statistic_Model
from ispots.anomaly_detector.models.prophet_model import Prophet_Model
from ispots.anomaly_detector.utils.feature_engineering import remove_level_shift, remove_extreme_values

dir_path = os.path.dirname(os.path.realpath(__file__)) # directory path of this py file
file_path = 'config/config.ini' # path of the config file relative to this py file
abs_path = os.path.join(dir_path, file_path) # absolute config file path

config_object = ConfigParser()
config_object.read(abs_path)
ad_params = config_object['ad_params']

model_lookback_period = eval(ad_params['model_lookback_period']) # 90 days
separate_weekend = ad_params['separate_weekend'] # True
pvalue_criteria_weekend = eval(ad_params['pvalue_criteria_weekend']) # 0.01
aggregate_method = ad_params['aggregate_method'] # mean 
dev_per_threshold = eval(ad_params['deviation_percentage_threshold']) # 5
std_min_divide_amount = eval(ad_params['std_min_divide_amount']) # 0.01
prophet_interval_width = eval(ad_params['prophet_interval_width']) # 0.80
additional_days_before_wkday_sep = eval(ad_params['additional_days_before_wkday_sep']) # 7 days
additional_days_before_no_sep = eval(ad_params['additional_days_before_no_sep']) # 5 days

class AutoAD():
    '''
    Auto_AnomalyDetection which detects and reports the measuring points with top anomalous consumption patterns 

    Parameters:
    namespace (str): namespace for anomaly detection

    data_frequency (str): frequency of records. 
    
    method (str): anomaly detection method ('statistic', 'fb-prophet') default as statistic

    method_params (dict): model parameters default as {}

    lookback_period (int): lookback period (days) as normal consumption. default is 90 (days) 

    interval_width (float): % uncertainty interval for predicted value. only valid for fb-prophet method.

    unit_of_measurement (str): unit of measurement for the records
    
        '''
    def __init__(self, namespace, 
                        data_frequency,
                        method = 'statistic',
                        method_params = {},
                        lookback_period = model_lookback_period,
                        interval_width = prophet_interval_width,
                        unit_of_measurement = ''):
        self.namespace = namespace
        self.freq = data_frequency
        self.method = method
        self.method_params = method_params
        self.lookback_period = lookback_period
        self.interval_width = interval_width
        self.uom = unit_of_measurement
    
    def detect(self, data_dict, detect_date):
        '''
        detect
            detect anomalous patterns on detection date and generate a dictionary to report anomaly detection.
            looks at past 90 days to determine the expected consumption on the detection date.
            auto detect if weekday and weekend consumption levels are different.
            this auto weekday-weekend-separation detection is determined using t-test on daily mean values.
            model checks for level shift. if observed, only values recorded after level shift is regarded as normal consumption
            these values are then evaluated at timestamp level to remove outliers 
        input:
            data_dict (dict): data dictionary (key: measuring point, value: dataframe)
            detect_date (pandas.DateTime): anomaly detection date
        output:
            score_dict (dict): (key: measuring point, value: [measuring point name, score dataframe, deviation, deviation percentage])
            score dataframe refers to the df for detection date containing expected mean, median, standard deviation, and classification 
            of anomalies
        '''

        self.data_dict = data_dict
        
        # check anomaly detection method
        if self.method == 'statistic':
            model = Statistic_Model(model_params = self.method_params)          
        elif self.method == 'fb_prophet':
            model = Prophet_Model(interval_width = self.interval_width, model_params = self.method_params)
        else:
            raise ValueError(f'{self.method} is not supported')

        score_dict = {}
        for mp in self.data_dict:

            full_df = self.data_dict[mp].copy()
            # retrieve the data corresponding to lookback period
            df = full_df.loc[(full_df.index >= detect_date - dt.timedelta(days=self.lookback_period + additional_days_before_wkday_sep)) &     
                            (full_df.index < detect_date)].copy()


            # check if look back data is sufficient
            if (df.index[-1].date() - df.index[0].date()).days < self.lookback_period -1:                       
                raise IndexError('Not enough data for look back period')           

             ## evaluate whether weekend separation is required by using daily mean as aggregate
            test_df_norm_daily = df.resample('1D').mean().copy()

            # create weekday and weekend column
            test_df_norm_daily["weekday"] = pd.to_datetime(test_df_norm_daily.index.values).dayofweek
            test_df_norm_daily["weekend"] = test_df_norm_daily['weekday']>=5
            weekday_df = test_df_norm_daily[(test_df_norm_daily['weekend'] != True)].copy()
            weekend_df = test_df_norm_daily[(test_df_norm_daily['weekend'] == True)].copy()

            # calculate pvalue
            pvalue = stats.ttest_ind(weekday_df.Value, weekend_df.Value).pvalue
            if pvalue < pvalue_criteria_weekend:
                self.sep_weekend = True
            else:
                self.sep_weekend = False   

            if self.sep_weekend:
                additional_days = additional_days_before_wkday_sep
            else: 
                additional_days = additional_days_before_no_sep

            test_df_norm = full_df.loc[(full_df.index >= detect_date - dt.timedelta(days=(self.lookback_period + additional_days))) &  
                                        (full_df.index < detect_date)].copy()

            # get the data from lookback period and remove level shift observed
            # if level shift observed in last 7 days (minimum), dates prior to level shift will be dropped
            test_df_norm = remove_level_shift(test_df_norm, self.sep_weekend, additional_days)

            # get data for the detection date
            test_df_1d = full_df.loc[(full_df.index < detect_date + dt.timedelta(days=1)) &
                        (full_df.index >= detect_date)].copy()

            if self.sep_weekend:
            # create weekday and weekend column
                test_df_norm["weekday"] = pd.to_datetime(test_df_norm.index.values).dayofweek
                test_df_norm["weekend"] = test_df_norm['weekday']>=5
                self.test_df_norm = test_df_norm
                test_df_1d["weekday"] = pd.to_datetime(test_df_1d.index.values).dayofweek
                test_df_1d["weekend"] = test_df_1d['weekday']>=5
            
                # separate the weekday and weekend to remove the top 5% and substitute with median value
                # weekday
                wd_df = test_df_norm.loc[test_df_norm.weekend != True].copy()
                wd_df['time'] = wd_df.index.time
                wd_df = remove_extreme_values(wd_df, ['Value'])           
                # weekend
                we_df = test_df_norm.loc[test_df_norm.weekend == True].copy()
                self.before_rm_ex = we_df
                we_df['time'] = we_df.index.time            
                we_df = remove_extreme_values(we_df, ['Value'])

                self.after_rm_ex = we_df           
                # combine weekday and weekend into 1 dataframe
                temp_full_df = pd.concat([wd_df, we_df], ignore_index = False)
                temp_full_df.sort_index(inplace = True)

            else:
                # no separate treatment for weekday and weekend
                temp_full_df = test_df_norm.copy()
                temp_full_df['time'] = temp_full_df.index.time
                # remove the top 5% and substitute with median value
                temp_full_df = remove_extreme_values(temp_full_df, ['Value'])

            # add the data for detection date into dataframe
            temp_full_df= pd.concat([temp_full_df, test_df_1d], ignore_index = False)

            # fit the statistical model to derive median, standard deviation and mean reference values 
            # fit the prophet model to derive the predicted, predicted_high and predicted_low values 
            if self.method == 'statistic':
                output_df = model.fit_transform(temp_full_df)
            elif self.method == 'fb_prophet':
                output_df = model.fit_transform(temp_full_df)

            # identify anomalies and label anomaly type in output_df
            df, deviation_val, deviation_per = identify_anomaly(output_df)

            score_dict[mp] = {'measuring point':mp, 'score dataframe':df, \
                              'deviation':deviation_val, 'deviation_percentage': deviation_per} 

        self.score_dict = score_dict
        
        return score_dict

    def anomaly_message(self, score_df):
        '''
        message
            generate deviation message for each type of anomaly detected
        input: 
            score_df (pandas.DataFrame): score dataframe
        output:
            msg (str): anomaly deviation message for each type of anomaly detected
        '''
        msg = ''
        temp_df = score_df.copy()
        # check for level shift anomaly
        if sum(temp_df['lvl_anomaly']) > 0:
            # message to mention detection of level shift anomaly 
            msg += 'Level Shift Anomaly observed\n'
            level_shift_df = temp_df.loc[temp_df.lvl_anomaly == True].copy()
            msg += f'Level shift observed from {min(level_shift_df.index)} to {max(level_shift_df.index)} \n'
            msg += f'Deviation: {round(sum(level_shift_df.deviation),2)} {self.uom}, Deviation % = {round(100*sum(level_shift_df.deviation)/sum(temp_df.pred),2)}% \n'
        else:
            # check for sequence anomaly
            if sum(temp_df['con_seq_anomaly']) > 0:
                # message to mention detection of sequence anomaly
                msg += 'Sequence Anomaly observed\n' 
                con_anomaly_df = temp_df.loc[temp_df.con_seq_anomaly == True].copy()
                msg += f'Sequence Anomaly observed from {min(con_anomaly_df.index)} to {max(con_anomaly_df.index)} \n'
                msg += f'Deviation: {round(sum(con_anomaly_df.deviation),2)} {self.uom}, Deviation % = {round(100*sum(con_anomaly_df.deviation)/sum(temp_df.pred),2)}% \n'
            
            # check for point anomaly
            point_anomaly_df = temp_df.loc[(temp_df.p_anomaly == True) & (temp_df.con_seq_anomaly == False)].copy()
            if len(point_anomaly_df) != 0:
                # message to mention detection of point anomaly
                msg += f'Number of Point Anomaly observed: {len(point_anomaly_df)} \n'
                for i in range(len(point_anomaly_df)):
                    msg += f'Point Anomaly observed at {point_anomaly_df.index[i]} \n'
                    msg += f'Deviation: {round(point_anomaly_df.deviation[i],2)} {self.uom}, Deviation % = {100*round(point_anomaly_df.deviation[i]/sum(temp_df.pred),2)}% \n'
        return msg
    
    def top_k(self, k = 5, sort_by = 'deviation_percentage', report = True, plot = False, dev_perc_threshold = dev_per_threshold):
        '''
        top_k
            reports k measuring points with top anomalies for the detection date, prints anomaly message and / or
            prints corresponding plots for detection date and for 90 days look back + detection date
        input: 
            k (int): number of anomalies displayed for detection date (default is 5)
            sort_by (str): method used for ranking measuring points. sorting options are: 'deviation_percentage'
                and 'deviation'. default: 'deviation_percentage'  
            report (boolean): whether to display anomaly description messages
            plot (boolean): whether to display consumption plots
            dev_perc_threshold (float): deviation percentage threshold to report measuring point. default = 5.
        output: 
            None
        '''
        if self.data_dict:
            # retrive the score list (score, measuring point name, score dataframe)
            score_lst = list(self.score_dict.values())
            score_lst.sort(reverse = True, key = lambda x:x[sort_by]) # use deviation to sort
            # report the top k anomalies 
            count = 0
            for i in range(min(k, len(self.data_dict))):
                if count == min(k, len(self.data_dict)): 
                    break
                # measuring point is reported only if total percentage deviation exceeds threshold
                if np.abs(score_lst[i]['deviation_percentage']) < dev_perc_threshold: 
                    continue

                else:
                    # retrieve values on measuring point
                    mp = score_lst[i]['measuring point']
                    score_df = score_lst[i]['score dataframe']
                    deviation_val = score_lst[i]['deviation']
                    deviation_per = score_lst[i]['deviation_percentage']
                    date = score_df.index[0].date()
                    
                    count += 1
                    # print overall anomaly message
                    print(f'Namespace: {self.namespace}')
                    print(f'NO {count} anomalous measuring point: {mp}')
                    print(f'Date of detection: {date}')
                    print(f'Deviation: {deviation_val} {self.uom}, Deviation %: {deviation_per}%')

                    # retrieve data for detection date + look back period
                    full_df = self.data_dict[mp]
                    long_df = full_df.loc[(full_df.index <= score_df.index[-1]) & 
                                        (full_df.index >= score_df.index[0] - dt.timedelta(days=self.lookback_period))].copy()

                    # plot the consumption data for reference
                    if plot:
                        # plot for detection date
                        fig, ax = plt.subplots(figsize=(17, 3))
                        ax.plot(score_df.index, score_df.Value)
                        ax.plot(score_df.index, score_df.pred)
                        ax.scatter(score_df[score_df.anomaly == True].index, score_df[score_df.anomaly == True].Value)
                        ax.fill_between(score_df.index, score_df.pred_low, score_df.pred_high, alpha = 0.1)
                        plt.show()

                        if report:
                            # generate anomaly message
                            anomaly_message = self.anomaly_message(score_df)
                            print(anomaly_message)

                        # plot detection date + lookback period
                        fig, ax = plt.subplots(figsize=(17, 3))
                        ax.plot(long_df.index, long_df.Value)
                        ax.plot(score_df.index, score_df.pred)
                        ax.axvspan(score_df.index[0], score_df.index[-1], alpha=0.2, color='r')
                        ax.title.set_text(mp)
                        plt.show()

            self.count = count

        else:
            raise ValueError(f'No data available')

    def kafka_message(self, score_df):
        
        msg = {}
        msg["anomaly"] = {}
        temp_df = score_df.copy()

        # check for level shift anomaly
        if sum(temp_df['lvl_anomaly']) == len(temp_df):  
            msg["anomaly"]["lvl_shift_anomaly"] = {}
            msg["anomaly"]["lvl_shift_anomaly"]["deviation"] = round(sum(temp_df.deviation),2)
            msg["anomaly"]["lvl_shift_anomaly"]["deviation_%"] = round(100*sum(temp_df.deviation)/sum(temp_df.pred),2)
            
        else:
            # check for sequence anomaly
            if sum(temp_df['con_seq_anomaly']) > 0:        
                con_anomaly_df = temp_df.loc[temp_df.con_seq_anomaly == True].copy()
                msg["anomaly"]["seq_anomaly"] = {}
                msg["anomaly"]["seq_anomaly"]["deviation"] = round(sum(con_anomaly_df.deviation),2)
                msg["anomaly"]["seq_anomaly"]["deviation_%"] = round(100*sum(con_anomaly_df.deviation)/sum(temp_df.pred),2)
                msg["anomaly"]["seq_anomaly"]["from"] = min(con_anomaly_df.index.strftime('%Y-%m-%d %H:%M:%S'))
                msg["anomaly"]["seq_anomaly"]["to"] = max(con_anomaly_df.index.strftime('%Y-%m-%d %H:%M:%S'))
                
            # check for point anomaly
            point_anomaly_df = temp_df.loc[(temp_df.p_anomaly == True) & (temp_df.con_seq_anomaly == False)].copy()
            if len(point_anomaly_df) != 0:    
                points = []
                for i in range(len(point_anomaly_df)):
                    abnormal_point = {}
                    abnormal_point["point_anomaly_point"] = point_anomaly_df.index[i].strftime('%Y-%m-%d %H:%M:%S')
                    abnormal_point["point_anomaly_deviation"] = round(point_anomaly_df.deviation[i],2)
                    abnormal_point["point_anomaly_deviation_%"] = 100*round(point_anomaly_df.deviation[i]/sum(temp_df.pred),2)
                    points.append(abnormal_point)
                msg["anomaly"]["point_anomaly"] = points

        return msg