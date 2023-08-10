import os
import pandas as pd
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from ispots.anomaly_detector import AutoAD, Preprocessor

# load data into a dictionary

#path = os.path.abspath('JC/electrical')

path = os.path.abspath('inputdata/JC/electrical/monthly')
csv_files = glob.glob(os.path.join(path, "*.csv"))
pre_data_dict = {}
for f in csv_files:
    df = pd.read_csv(f)
    pre_data_dict[f.split("/")[-1][:-4]] = df

    
# preprocess data
preprocessor = Preprocessor(pre_data_dict, miss_val_threshold = 0.01)
data_dict = preprocessor.preprocess()

# statistic method
anomalies = {}
lookback_period = 30
sort_by = 'deviation_percentage'
dev_perc_threshold = 5
namespace = 'JC'
unit_of_measurement = 'Wh'

ad_detector = AutoAD(namespace = 'JC', data_frequency = preprocessor.data_frequency, method = 'statistic', unit_of_measurement = unit_of_measurement, lookback_period = lookback_period)

start_date = pd.to_datetime('2023-2-1') 
end_date = pd.to_datetime('2023-2-28')    
score_dict = ad_detector.detect(data_dict = data_dict, detect_date = start_date)
score_dict_copy = score_dict.copy()
for mp in score_dict.keys():
    mp_new = str(mp) + str(start_date.date())
    score_dict_copy[mp_new] = score_dict_copy.pop(mp)
score_dict = score_dict_copy

detection_date = start_date + pd.Timedelta(1, unit='d')
while detection_date <= end_date:
    score_dict_append = ad_detector.detect(data_dict = data_dict, detect_date = detection_date)
    score_dict_copy = score_dict_append.copy()
    for mp in score_dict_append.keys():
        mp_new = str(mp) + str(detection_date.date())
        score_dict_copy[mp_new] = score_dict_copy.pop(mp)
    score_dict_append = score_dict_copy
    score_dict.update(score_dict_append)
    detection_date += pd.Timedelta(1, unit='d')

score_lst = list(score_dict.values())
score_lst.sort(reverse = True, key = lambda x:x[sort_by]) # use deviation to sort
# report the top 5 anomalies 
count = 0
for i in range(min(5, len(data_dict))):
    if count == min(5, len(data_dict)): 
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
        print(f'Namespace: {namespace}')
        print(f'NO {count} anomalous measuring point: {mp}')
        print(f'Date of detection: {date}')
        print(f'Deviation: {deviation_val} {unit_of_measurement}, Deviation %: {deviation_per}%')

        # retrieve data for detection date + look back period
        full_df = data_dict[mp]
        long_df = full_df.loc[(full_df.index <= score_df.index[-1]) & 
                            (full_df.index >= score_df.index[0] - dt.timedelta(days=lookback_period))].copy()
        
        # plot the consumption data for reference
        #if plot:
            # plot for detection date
        fig, ax = plt.subplots(figsize=(17, 3))
        ax.plot(score_df.index, score_df.Value)
        ax.plot(score_df.index, score_df.pred)
        ax.scatter(score_df[score_df.anomaly == True].index, score_df[score_df.anomaly == True].Value)
        ax.fill_between(score_df.index, score_df.pred_low, score_df.pred_high, alpha = 0.1)
        ax.title.set_text(mp[5:12] +  ' Anomaly in ' + str(date)) #blk-218 anomaly in 2023-02-01
        #plt.show()
        ax.figure.savefig('./output/JC/anomaly_output_img/' + mp[5:12] +  '_' + str(date) +'_day.png') #,  bbox_inches = 'tight') #30SCE_monthly.png

        #if report:
        # generate anomaly message
        anomaly_message = ad_detector.anomaly_message(score_df)
        print(anomaly_message)

        # plot detection date + lookback period
        fig, ax = plt.subplots(figsize=(17, 3))
        ax.plot(long_df.index, long_df.Value)
        ax.plot(score_df.index, score_df.pred)
        ax.axvspan(score_df.index[0], score_df.index[-1], alpha=0.2, color='r')
        ax.title.set_text(mp[5:12] +  ' Look Back Period - ' + str(lookback_period) + ' days') #blk-218 look back period - 30 days
        ax.figure.savefig('./output/JC/anomaly_output_img/' + mp[5:12] +  '_' + str(date) +'_period.png')

        anomalies[count] = {'blk':mp, 'date':date, 'deviation':deviation_val, 'unit_of_measurement': unit_of_measurement, 'deviation_percentage': deviation_per} 
    
print(anomalies)
#if count > 0:
#    anomalies_detected_dates.append(detection_date)
    #detection_date += pd.Timedelta(1, unit='d')
