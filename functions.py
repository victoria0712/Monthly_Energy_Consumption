import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.cm import get_cmap
from datetime import datetime
import datetime as dt
import calendar
import os
from ispots.anomaly_detector import AutoAD, Preprocessor
import re

def truncate_string(string):
    """
    Extract the desired substring using regex

    Args:
        string: blk name. e.g.'dsta_blk-5-water'

    Returns:
        truncated string. e.g. 'blk-5'
    """
    match = re.search(r'blk-\d+', string)
    if match:
        desired_substring = match.group()
    else:
        desired_substring = None

    return desired_substring


def extract_numbers(string):
    """
    Extract the desired substring using regex

    Args:
        string: blk name. e.g.'dsta_blk-5-water'

    Returns:
        truncated string. e.g. '5'
    """

    match = re.search(r'\d+', string)
    if match:
        number_as_string = match.group()
    else:
        number_as_string = None

    # Return the number as a string
    return number_as_string


def read_raw(path):
    """
    :param path: raw file path
    :return: raw file name list
    """
    fine_lst = []
    for filename in glob.glob(path):
        fine_lst.append(filename)
    fine_lst = list(set(fine_lst))
    new_filename_lst = []
    for i in fine_lst:
        new_filename = i.split("/")[-1]
        new_filename = new_filename.split(".")[:-1]
        new_filename = '.'.join(map(str, new_filename))
        new_filename_lst.append(new_filename)
    new_filename_lst = set(new_filename_lst)
    new_filename_lst = list(new_filename_lst)
    return new_filename_lst


def millions_formatter(x, pos):
    return f'{x / 1000}'


def clean_column(block_elec):
    """
        :param path: each raw file
        :return: cleaned raw file with new date column
    """
    block_elec['Time'] = block_elec['Time'].astype(str)
    time_lst = []
    date_lst = []
    for item in block_elec['Time']:
        new_date = item.split(" ")[0]
        new_date = new_date.split("-")[0:3]
        new_date = ''.join(map(str, new_date))
        new_time = item.split(" ")[1]
        new_time = new_time.split(":")[0:2]
        new_time = ''.join(map(str, new_time))
        time_lst.append(new_time)
        date_lst.append(new_date)
    block_elec['new_date'] = date_lst
    block_elec['new_time'] = time_lst
    for index, row in block_elec.iterrows():
        if len(row['new_time']) == 3:
            row['new_time'] = '0'+row['new_time']
            block_elec.at[index, 'new_time'] = row['new_time']
    return block_elec


def combine_monthly_file_jc(root_path, path, month, inputpath, separate_file):
    """
    Combine data in last month, this month, and next month in Jurong Camp.

    Args:
        root_path: raw file root path
        path: raw file path
        month: select month
        inputpath: input file path
        separate_file (list): separate file names. e.g. ['dsta_blk-102-1-elect' , 'dsta_blk-102-2-elect']

    Returns:
        None
    """
    # get month variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = str(last_month.strftime("%Y-%m"))
    next_month = first + dt.timedelta(days=32)
    next_month = str(next_month.strftime("%Y-%m"))
    selected_path = path + month
    
    # get available file in last month, this month, and next month
    #file_list = read_raw(selected_path)
    #last_list = read_raw(path + last_month + "/*.csv")
    #next_list = read_raw(path + next_month + "/*.csv")
    file_list = os.listdir(selected_path)
    last_list = os.listdir(path + last_month)
    next_list = os.listdir(path + next_month)

    # combine the file in last month, this month, and next month if file in this month exists in last and next month at the same time
    for i in file_list:
        if (i in last_list) & (i in next_list):
            selected = path + month + "/" + i
            selected_file = pd.read_csv(selected)
            last = path + last_month + "/" + i
            last_file = pd.read_csv(last)
            next = path + next_month + "/" + i
            next_file = pd.read_csv(next)
            df_all = pd.concat([last_file, selected_file,  next_file])
            df_all = df_all.reset_index(drop = True)
            #df_all = df_all.drop_duplicates()
            df_all = df_all.drop_duplicates(subset=['Time'], keep='first')
            #separate
            if i in separate_file:
                df_all.to_csv(root_path + inputpath + 'separate/' + i, index=False )
            #multifloor
            elif "lvl" in i:
                df_all.to_csv(root_path  + inputpath + 'multifloor/' + i , index=False )
            elif "washing-bay" in i:
                df_all.to_csv(root_path + inputpath + 'multifloor/' + i, index=False )          
            elif "toilet" in i:
                df_all.to_csv(root_path + inputpath + 'multifloor/' + i, index=False ) 
            #others
            else:
                df_all.to_csv(root_path + inputpath + i, index=False )
    if "elect" in file_list[0]:
        os.rename(root_path + inputpath  + 'dsta_blk-207-main-elect' + '.csv', root_path  + inputpath + 'dsta_blk-207-elect' + '.csv')
    else:
        os.rename(root_path + inputpath  + 'dsta_blk-207a-water' + '.csv', root_path  + inputpath + 'dsta_blk-207-water' + '.csv')


def combine_monthly_file_plab(root_path, path, month, inputpath):
    """
    Combine data in last month, this month, and next month in Paya Lebar Camp.

    Args:
        root_path: raw file root path
        path: raw file path
        month: select month
        inputpath: input file path

    Returns:
        None
    """
    # get month variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = str(last_month.strftime("%Y-%m"))
    next_month = first + dt.timedelta(days=32)
    next_month = str(next_month.strftime("%Y-%m"))
    selected_path = path + month

    # get available file in last month, this month, and next month
    #file_list = read_raw(selected_path)
    #last_list = read_raw(path + last_month + "/*.csv")
    #next_list = read_raw(path + next_month + "/*.csv")
    file_list = os.listdir(selected_path)
    last_list = os.listdir(path + last_month)
    next_list = os.listdir(path + next_month)

    # combine the file in last month, this month, and next month if file in this month exists in last and next month at the same time
    for i in file_list:
        if (i in last_list) & (i in next_list):
            selected = path + month + "/" + i 
            selected_file = pd.read_csv(selected)
            last = path + last_month + "/" + i 
            last_file = pd.read_csv(last)
            next = path + next_month + "/" + i 
            next_file = pd.read_csv(next)
            df_all = pd.concat([last_file, selected_file,  next_file])
            df_all = df_all.reset_index(drop = True)
            df_all = df_all.drop_duplicates()
            df_all.to_csv(root_path + inputpath + i , index=False )


def processing_separate_file(root_path, input_folder, file_1path, file_2path):
    """
    blk 105 = blk 105 incoming 1 + blk 105 incoming 2
    """
    file_1 = root_path + input_folder + file_1path
    file_2 = root_path + input_folder + file_2path
    df_file_1 = pd.read_csv(file_1)
    df_file_2 = pd.read_csv(file_2)
    df_file_contact = pd.concat([df_file_1, df_file_2])
    df_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
    return df_file


def processing_separate_file_plab(root_path, input_folder, file_1path, file_2path, file_3path, file_4path, file_5path,
                                  file_6path):
    """
    blk 503 = blk 503_1+2+3+4
    """
    file_1 = root_path + input_folder + file_1path
    file_2 = root_path + input_folder + file_2path
    file_3 = root_path + input_folder + file_3path
    file_4 = root_path + input_folder + file_4path
    file_5 = root_path + input_folder + file_5path
    file_6 = root_path + input_folder + file_6path
    df_file_1 = pd.read_csv(file_1)
    df_file_2 = pd.read_csv(file_2)
    df_file_3 = pd.read_csv(file_3)
    df_file_4 = pd.read_csv(file_4)
    df_file_5 = pd.read_csv(file_5)
    df_file_6 = pd.read_csv(file_6)
    df_file_contact = pd.concat([df_file_1, df_file_2, df_file_3, df_file_4, df_file_5, df_file_6])
    df_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
    return df_file


def process_blk101(path_100, path_101, path_102):
    df_raw_101 = pd.read_csv(path_101)
    df_raw_101 = df_raw_101.add_suffix('_101')
    df_raw_100 = pd.read_csv(path_100)
    df_raw_100 = df_raw_100.add_suffix('_100')
    df_raw_102 = pd.read_csv(path_102)
    df_raw_102 = df_raw_102.add_suffix('_102')

    df_clean_temp = pd.merge(df_raw_101,df_raw_100, how='left', left_on='Time_101', right_on='Time_100')
    df_clean_combine = pd.merge(df_clean_temp, df_raw_102, how='left', left_on='Time_101', right_on='Time_102')
    df_clean_combine = df_clean_combine.fillna(0)
    df_clean_combine['Value'] = df_clean_combine['Value_101']-df_clean_combine['Value_100']-df_clean_combine['Value_102']
    df_clean_combine = df_clean_combine.loc[:,['Time_101','Value']]
    df_clean_combine.rename(columns={'Time_101': 'Time'}, inplace=True)
    return df_clean_combine


def separate_file_jc(blk_lst, root_path, elec_input_folder, water_input_folder):
    """
    Data processing for blk with separate files in Jurong Camp

    Args:
        blk_lst (list): blk name with separate files. e.g. ['dsta_blk-102', 'dsta_blk-105', 'dsta_blk-210']
        root_path: raw file root path
        elec_input_folder: elec file input path
        water_input_folder: water file input path

    Returns:
        None    
    """
    # process electricity
    for i in blk_lst:
        clean_file = processing_separate_file(root_path, elec_input_folder,'separate/'+i+'-1-elect.csv',
                                 'separate/'+i+'-2-elect.csv')
        clean_file.to_csv(elec_input_folder+i+'-elect.csv')
    # process water
    blk_lst_water = ['dsta_blk-102', 'dsta_blk-105']
    for i in blk_lst_water:
        clean_file = processing_separate_file(root_path, water_input_folder,'separate/' + i + '-1-water.csv',
                                              'separate/' + i + '-2-water.csv')
        clean_file.to_csv(water_input_folder + i + '-water.csv')

    # data processing for blk 101 electricity
    # minus blk 102 and blk 100 electricity
    path_101 = root_path+ elec_input_folder+'101/dsta_blk-101-elect.csv'
    path_100 = root_path+ elec_input_folder+'dsta_blk-100-elect.csv'
    path_102 = root_path+ elec_input_folder+'dsta_blk-102-elect.csv'
    df_clean_combine = process_blk101(path_100, path_101,path_102)
    df_clean_combine.to_csv(elec_input_folder +'dsta_blk-101-elect.csv')


def separate_file_cnb(root_path, elec_input_folder, water_input_folder):
    """
    Data processing for blk with separate files in Changi Naval Base Camp

    Args:
        root_path: raw file root path
        elec_input_folder: elec file input path
        water_input_folder: water file input path

    Returns:
        None    
    """
    # process electricity
    separate_file_dic = {'204': ['204-2-elect', '204-1-elect'], '207': ['207-1-elect', '207-2-elect'],
                         '317': ['317-msb-2-elect', '317-msb-1-elect'],
                         '117': ['117-1-elect', '117-2-elect'],
                         '118': ['118-east-wing-elect', '118-west-wing-elect']}
    for blk, speratemeter in separate_file_dic.items():
        df_concact = pd.DataFrame()
        for i in speratemeter:
            file = root_path + elec_input_folder + 'separate/dsta_blk-' + i + '.csv'
            df_file = pd.read_csv(file)
            df_concact = pd.concat([df_concact, df_file])
        clean_file = df_concact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(elec_input_folder + 'dsta_blk-' + blk + '-elect.csv')

    # process 210
    speratemeter = ['210-ac-l1-3-elect', '210-l6-elect', '210-l1-elect', '210-ac-l4-8-elect',
                    '210-l7-elect', '210-l5-elect', '210-l3-1-elect', '210-l3-2-elect',
                    '210-l2-elect', '210-l4-elect', '210-l8-elect']
    df_concact = pd.DataFrame()
    for i in speratemeter:
        file = root_path + elec_input_folder + 'multifloor/dsta_blk-' + i + '.csv'
        df_file = pd.read_csv(file)
        df_concact = pd.concat([df_concact, df_file])
    clean_file = df_concact.groupby(['Time'])['Value'].sum().reset_index()
    clean_file.to_csv(elec_input_folder + 'dsta_blk-210-elect.csv')

    # process water
    separate_file_dic_water = {'100': ['blk-100-water'], '103': ['blk-103-water'],
                               '116': ['blk-116-water'], '117': ['blk-117-water'],
                               '136': ['blk-140-water'], '203': ['blk-203-water'], '204': ['blk-204-water'],
                               '207': ['blk-207-water'], '213': ['blk-213-water'], '317': ['blk-317-water'],
                               '319': ['blk-319-water'], '321': ['blk-321-water'], '210': ['blk-210-l5-1-water']}
    for blk, speratemeter in separate_file_dic_water.items():
        df_concact = pd.DataFrame()
        for i in speratemeter:
            file = root_path + water_input_folder + 'separate/dsta_' + i + '.csv'
            df_file = pd.read_csv(file)
            df_concact = pd.concat([df_concact, df_file])
        clean_file = df_concact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(water_input_folder + 'dsta_blk-' + blk + '-water.csv')

    # process 210
    speratemeter = [ '210-l4-2-water', '210-l1-main-incoming-water',
                    '210-l3-2-water', '210-l3-5-water', '210-l3-4-water',
                    '210-l3-1-water', '210-l8-water', '210-l5-2-water', '210-l7-water',
                     '210-l4-1-water', '210-l6-water', '210-l2-2-water',
                    '210-l2-1-water']#'210-l3-6-water','210-l3-3-water',
    df_concact = pd.DataFrame()
    for i in speratemeter:
        file = root_path + water_input_folder + 'multifloor/dsta_blk-' + i + '.csv'
        df_file = pd.read_csv(file)
        df_concact = pd.concat([df_concact, df_file])
    clean_file = df_concact.groupby(['Time'])['Value'].sum().reset_index()
    clean_file.to_csv(water_input_folder + 'dsta_blk-210-water.csv')


def separate_file_plab_elec(root_path, elec_input_folder):
    """
    Data processing for blk with separate elec files in Paya Lebar Camp
    Blocks include blk-503, blk-553, blk-502, blk-53.

    Args:
        root_path: raw file root path
        elec_input_folder: elec file input path

    Returns:
        None    
    """

    # blk-503: blk-503-main-db-elect, blk-503-hangar-sb-h-m-elect, blk-503-hangar-sb-h1-2-elect, 
    #          blk-503-hangar-elect, blk-503-eboard-elect, blk-503-hangar-sb-h1-1-elect
    blk_lst = ['dsta_blk-503']
    for i in blk_lst:
        file_1 = root_path + elec_input_folder + i + '-eboard-elect.csv'
        file_2 = root_path + elec_input_folder + i + '-hangar-elect.csv'
        file_3 = root_path + elec_input_folder + i + '-hangar-sb-h-m-elect.csv'
        file_4 = root_path + elec_input_folder + i + '-hangar-sb-h1-1-elect.csv'
        file_5 = root_path + elec_input_folder + i + '-hangar-sb-h1-2-elect.csv'
        file_6 = root_path + elec_input_folder + i + '-main-db-elect.csv'

        df_file_1 = pd.read_csv(file_1)
        df_file_2 = pd.read_csv(file_2)
        df_file_3 = pd.read_csv(file_3)
        df_file_4 = pd.read_csv(file_4)
        df_file_5 = pd.read_csv(file_5)
        df_file_6 = pd.read_csv(file_6)
        df_file_contact = pd.concat([df_file_1, df_file_2, df_file_3, df_file_4, df_file_5,df_file_6])
        clean_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()

        clean_file.to_csv(elec_input_folder+i+'-elect.csv')

    #blk-553: blk-553-msb553-h1-elect, blk-553-msb553-h2-elect
    blk_lst = ['dsta_blk-553']
    for i in blk_lst:
        file_1 = root_path + elec_input_folder + i + '-msb553-h2-elect.csv'
        file_2 = root_path + elec_input_folder + i + '-msb553-h1-elect.csv'
        df_file_1 = pd.read_csv(file_1)
        df_file_2 = pd.read_csv(file_2)
        df_file_contact = pd.concat([df_file_1, df_file_2])
        clean_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(elec_input_folder + i + '-elect.csv')

    # blk-502: blk-502-l1-accommodation-elect, blk-502-l3-elect
    blk_lst = ['dsta_blk-502']
    for i in blk_lst:
        file_1 = root_path + elec_input_folder + i + '-l1-accommodation-elect.csv'
        file_2 = root_path + elec_input_folder + i + '-l3-elect.csv'
        df_file_1 = pd.read_csv(file_1)
        df_file_2 = pd.read_csv(file_2)
        df_file_contact = pd.concat([df_file_1, df_file_2])
        clean_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(elec_input_folder + i + '-elect.csv')

    # blk-53: blk-53-db-2c-ac-elect, blk-53-db2-1-elect, blk-53-db-1a-elect, blk-53-db-4a-elect,
    #         blk-53-db-2b-elect, blk-53-db-3a-elect, blk-53-db-6a-elect, blk-53-db1-1-elect
    #         blk-53-db-5a-elect, blk-53-ac-sbm-l3-elect
    blk_lst = ['dsta_blk-53']
    for i in blk_lst:
        file_1 = root_path + elec_input_folder +  i + '-db2-1-elect.csv'
        file_2 = root_path + elec_input_folder + i + '-db1-1-elect.csv'
        file_3 = root_path + elec_input_folder + i + '-ac-sbm-l3-elect.csv'
        file_4 = root_path + elec_input_folder + i + '-db-6a-elect.csv'
        file_5 = root_path + elec_input_folder + i + '-db-2c-ac-elect.csv'
        file_6 = root_path + elec_input_folder + i + '-db-5a-elect.csv'
        file_7 = root_path + elec_input_folder + i + '-db-2b-elect.csv'
        file_8 = root_path + elec_input_folder + i + '-db-3a-elect.csv'
        file_9 = root_path + elec_input_folder + i + '-db-4a-elect.csv'
        file_10 = root_path + elec_input_folder + i + '-db-1a-elect.csv'

        df_file_1 = pd.read_csv(file_1)
        df_file_2 = pd.read_csv(file_2)
        df_file_3 = pd.read_csv(file_3)
        df_file_4 = pd.read_csv(file_4)
        df_file_5 = pd.read_csv(file_5)
        df_file_6 = pd.read_csv(file_6)
        df_file_7 = pd.read_csv(file_7)
        df_file_8 = pd.read_csv(file_8)
        df_file_9 = pd.read_csv(file_9)
        df_file_10 = pd.read_csv(file_10)

        df_file_contact = pd.concat([df_file_1, df_file_2, df_file_3, df_file_4, df_file_5,
                                     df_file_6, df_file_7, df_file_8, df_file_9, df_file_10])
        clean_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(elec_input_folder + i + '-elect.csv')


def separate_file_plab_water(root_path, water_input_folder):
    """
    Data processing for blk with separate water files in Paya Lebar Camp
    Blocks include blk-95

    Args:
        root_path: raw file root path
        water_input_folder: water file input path

    Returns:
        None    
    """
    # blk-95: blk-95-2-water, blk-95-1-water
    blk_lst = ['dsta_blk-95']
    for i in blk_lst:
        file_1 = root_path + water_input_folder + i + '-1-water.csv'
        file_2 = root_path + water_input_folder +  i + '-2-water.csv'

        df_file_1 = pd.read_csv(file_1)
        df_file_2 = pd.read_csv(file_2)

        df_file_contact = pd.concat([df_file_1, df_file_2])
        clean_file = df_file_contact.groupby(['Time'])['Value'].sum().reset_index()
        clean_file.to_csv(water_input_folder + i + '-water.csv')


def get_dataframe(func_dic_usage, func_dic_unit, root_path, input_folder, gfa_file_path_name, output_csv_folder, month):
    """
    Import and combine consumption data, building characteristic, and unit / function information into dataframe.

    Args:
        func_dic_usage (dictionary): building function list
        func_dic_unit (dictionary): building unit list
        root_path: raw file root path
        input_folder:
        gfa_file_path_name: building characteristic with gfa information file path
        output_csv_folder: combined csv file path
        month (str): selected month 

    Returns:
        total (dataframe): dataframe with combined information 
        blklst (list): blk (buiding name) list in the camp
        missing_dict (dictionary): missing data in ratio in each blk
    """
    total = pd.DataFrame(columns=['Time', 'year_month', 'week', 'hour', 'unit','block', 'gfa','usage','Value'])
    missing_dict = {}
    
    for key, value in func_dic_unit.items():
        unit_group = pd.DataFrame(columns=['Time','year_month', 'week', 'hour','unit','block','gfa','usage','Value'])
        
        for itemname in value:
            file_path_name = root_path + input_folder + itemname + '.csv'
            block_elec = pd.read_csv(file_path_name)
            block_elec['block'] = itemname
            block_elec['unit'] = key

            # generate year_month, week, hour, usage, gfa columns
            block_elec = clean_column(block_elec)
            block_elec['year_month'] = block_elec['Time'].str[0:7]
            n = 0
            for each in block_elec['Value']:
                if each ==0:
                    n += 1
            missing_dict[itemname] = n/len(block_elec)
            block_elec['Value'].replace(to_replace=0, method='ffill')
            block_elec['hour'] = block_elec['new_time'].str[0:2]
            block_elec['NewTime'] = pd.to_datetime(block_elec['Time'], errors='coerce')
            block_elec['date'] = block_elec['Time'].str[0:10]
            block_elec['week'] = block_elec['NewTime'].dt.isocalendar()['week']            
            for function, blks in func_dic_usage.items():
                for n in blks:
                    if n == itemname:
                        usage = function
            block_elec['usage'] = usage

            # get building gfa data
            gfa_df = pd.read_csv(gfa_file_path_name)
            for i in range(gfa_df.shape[0]):
                if gfa_df['Building'][i] == itemname:
                    gfa = gfa_df['GFA'][i]
            block_elec['gfa'] =gfa
            unit_group = pd.concat([unit_group, block_elec])      
        total = pd.concat([total, unit_group], join = 'inner', ignore_index=True)
        total.to_csv(output_csv_folder + 'all_data_cleaned.csv')
        blklst = list(total.loc[total['year_month']==month]['block'].unique())

    return total, blklst, missing_dict


def monthly_consump_unit(func_dic, root_path, total, output_img_folder, type, camp_fullname, month):
    """
    Generate monthly consumption graphs by unit.

    Args: 
        func_dic (dictionary): building unit list
        root_path: raw file root path
        total: the combined dataframe provided by get_datafram function
        output_img_folder: output image folder path
        type (str): type of consumption - 'electricity' or 'water'
        camp_fullname (str): camp full name 
        month (str): selected month

    Returns:
        Graphs:
            Camp EUI by Unit                   e.g. Jurong Camp EUI by Unit in 2023-02
            Camp EUI by Unit for each Unit     e.g. Jurong Camp EUI by Unit in 2023-02
            Camp EUI change by Unit            e.g. Jurong Camp Changes in EUI by Unit in 2023-02

        Variables:
            val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low, med
    """
    # how many unit is shared in one building - {building: num of unit using it}
    building = {}
    for key, value in func_dic.items():
        for itemname in value:
            if itemname not in building.keys():
                building[itemname] = 1
            else:
                building[itemname] += 1 

    # get variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    total['val_avg'] = np.nan
    total['num'] = np.nan
    for i in range(total.shape[0]):
        for key, value in building.items():
            if total['block'][i] == key:
                total['num'][i] = value
    total['val_avg'] = total['Value']/total["num"]
    total['gfa'] = total['gfa']/total["num"]
    total = total.groupby(['Time', 'year_month', 'week','hour', 'unit'], as_index=False).agg({'val_avg':'sum','gfa':'sum'})
    total = total.groupby(['year_month', 'unit'], as_index=False).agg({'val_avg':'sum','gfa':'first'})
    total['EUI'] = total['val_avg']/total['gfa']

    # get improvement data
    grouped = total.groupby(total.year_month)
    total_new = grouped.get_group(month)
    total_new2 = grouped.get_group(last_month)
    total_change = total_new.merge(total_new2, how = 'inner', on = ['unit'])
    total_change['change'] = total_change['EUI_x']/total_change['EUI_y']-1
    total_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # get returned variables
    val_high = total.loc[total['year_month']==month,['EUI']].max().values[0]
    val_low = total.loc[total['year_month']==month,['EUI']].min().values[0]
    unit_high = total.loc[(total['year_month']==month) & (total['EUI'] == val_high), ['unit']].values[0][0]
    unit_low = total.loc[(total['year_month']==month) & (total['EUI'] == val_low), ['unit']].values[0][0]
    change_high = total_change['change'].max()
    change_low = total_change['change'].min()
    change_unit_high = total_change.loc[total_change['change'] == change_high, ['unit']].values[0][0]
    change_unit_low = total_change.loc[total_change['change'] == change_low, ['unit']].values[0][0]
    camp = total.groupby(['year_month'], as_index=False).agg({'val_avg':'sum','gfa':'sum'})
    camp['EUI'] = camp['val_avg']/camp['gfa']
    val_camp = camp.loc[camp.year_month==month]['EUI'].values[0]
    change_camp = camp.loc[camp.year_month==month]['EUI'].values[0]/camp.loc[camp.year_month==last_month]['EUI'].values[0]-1
    total = total.loc[total['year_month']==month]
    
    # plot all units - Camp EUI by Unit
    if type == 'electricity':
        title_name = 'EUI'
        unit_name = '(Wh/m2)'
    else:
        title_name = 'WEI'
        unit_name = '(m3/m2)'
    total = total.sort_values('EUI', ascending = False)
    name = "Set3"
    cmap = get_cmap(name)
    colors = cmap.colors
    fig1 = total.plot(x='unit', y = 'EUI', kind='bar', color=colors, figsize=(24, 12), label = None)
    plt.axhline(y=val_camp, label ='Camp ' + title_name, color='dimgrey', linestyle =":")
    plt.ylabel(title_name + '-' + unit_name, fontsize=22)
    plt.xticks(rotation=90, fontsize=22)
    plt.yticks(rotation=0, fontsize=22)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim(0, 1.40 * val_high)
    current_values = plt.gca().get_yticks()
    if type == 'electricity':
        plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    else:
        plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
    plt.legend(bbox_to_anchor=(1,1), prop={'size': 20})
    plt.title(camp_fullname + ': ' + title_name + ' by Units in ' + month, fontsize=26) #Jurong Camp EUI by Unit in 2023-02
    fig1.figure.savefig(root_path + output_img_folder + 'units_' + title_name + '.png', bbox_inches = 'tight') #units_EUI.png
    
    # plot all units - Camp EUI by Unit for each unit
    for key, value in func_dic.items():
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        colors = []
        for each in total['unit']:
            if each == key:
                colors.append('tomato')
            else:
                colors.append('silver')
        med = total['EUI'].median()
        fig = total.plot(x='unit', y = 'EUI', kind='bar', color=colors, figsize=(24, 12), label = None)
        plt.axhline(y=med, label ='Median ' + title_name, color='dimgrey', linestyle =":")
        plt.ylabel(title_name + '-' + unit_name, fontsize=22)
        plt.xticks(rotation=90, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.ticklabel_format(style='plain', axis='y')
        plt.xlabel('', fontsize=22)
        plt.tick_params(labelsize=22)
        plt.ylim(0, 1.40 * val_high)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.legend()  
        plt.title(camp_fullname + ': ' + title_name + ' by Units in ' + month, fontsize=26)
        fig.figure.savefig(root_path + output_img_folder + key + '_' + title_name + '.png', bbox_inches = 'tight')

    # plot improvement - Camp EUI change by Unit 
    if type == 'electricity':
        title_name = 'EUI'
        unit_name = '%'
    else:
        title_name = 'WEI'
        unit_name = '%'
    total_change = total_change.sort_values('change', ascending = False)
    colors = ['lightgreen' if y<0 else 'orangered' for y in total_change['change']]
    fig2 = total_change.plot(x='unit', y = 'change', kind='bar', color=colors, figsize=(24, 12))
    plt.axhline(y=change_camp, label ='Camp ' + title_name, color='dimgrey', linestyle =":")
    plt.axhline(y=0, color = 'black', linewidth = 0.8)
    plt.ylabel(title_name + '-' + unit_name, fontsize=22)
    plt.xticks(rotation=90, fontsize=22)
    plt.yticks(rotation=0, fontsize=22)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim(change_low, 1.60 * change_high)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(["{:.0%}".format(x) for x in current_values])
    color_dict = {'increase':'orangered', 'decrease':'lightgreen'}
    color_dict['Camp Month to Month ' + title_name]=matplotlib.colors.to_rgb('dimgrey')
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:-1]]
    markers.append([plt.Line2D([0,0],[0,0],color=color,  linestyle=':') for color in color_dict.values()][-1])
    plt.legend(markers, color_dict.keys(), numpoints=1, bbox_to_anchor=(1,1), prop={'size': 20})
    plt.title(camp_fullname + ': Month to Month ' + title_name + ' by Units in ' + month, fontsize=26) #Jurong Camp Changes in EUI by Unit in 2023-02
    fig2.figure.savefig(root_path + output_img_folder +'units_' + title_name + '_changes' + '.png', bbox_inches = 'tight') #units_EUI_changes.png

    return val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low, med


def monthly_consump_building(func_dic_usage, root_path, total, output_img_folder, type, camp_fullname, month):
    """
    Generate monthly consumption graphs by building.

    Args: 
        func_dic_usage (dictionary): building function list
        root_path: file root path
        total: the combined dataframe provided by get_datafram function
        output_img_folder: output image folder path
        type (str): type of consumption - 'electricity' or 'water'
        camp_fullname (str): camp full name 
        month (str): selected month

    Returns:
        Graphs:
            Camp EUI by Building                          e.g. Jurong Camp EUI by Building in 2023-02
            Camp EUI by Building for Each Peer            e.g. Training Buildings EUI in 2023-02
            Camp EUI Changes by Building                  e.g. Training Buildings EUI Changes in 2023-02
            Camp EUI Changes by Building for Each Peer    e.g. Jurong Camp Changes in EUI by Building in 2023-02

        Variables:
            val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low
    """
    total = total.groupby(['year_month','usage','block','unit'], as_index=False).agg({'Value':'sum', 'gfa':'first', 'unit':'first'})
    total = total.drop_duplicates(subset=['year_month','usage','block','Value','gfa'], keep='first')
    
    # get variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    total = total.loc[(total['year_month']==month) | (total['year_month']==last_month)]

    # get EUI
    total['EUI'] = total['Value']/total['gfa']

    # get improvement data
    grouped = total.groupby(total.year_month)
    total_new = grouped.get_group(month)
    total_new2 = grouped.get_group(last_month)
    total_change = total_new.merge(total_new2, how = 'inner', on = ['block'])
    total_change['change'] = total_change['EUI_x']/total_change['EUI_y']-1
    total_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # get variables
    val_high = total.loc[total['year_month']==month,['EUI']].max().values[0]
    val_low = total.loc[total['year_month']==month,['EUI']].min().values[0]
    unit_high = total.loc[(total['year_month']==month) & (total['EUI'] == val_high), ['block']].values[0][0]
    unit_low = total.loc[(total['year_month']==month) & (total['EUI'] == val_low), ['block']].values[0][0]
    change_high = total_change['change'].max()
    change_low = total_change['change'].min()
    change_unit_high = total_change.loc[total_change['change'] == change_high, ['block']].values[0][0]
    change_unit_low = total_change.loc[total_change['change'] == change_low, ['block']].values[0][0]
    camp = total.groupby(['year_month'], as_index=False).agg({'Value':'sum','gfa':'sum'})
    camp['EUI'] = camp['Value']/camp['gfa']
    val_camp = camp.loc[camp.year_month==month]['EUI'].values[0]
    change_camp = (camp.loc[camp.year_month==month]['EUI'].values[0])/(camp.loc[camp.year_month==last_month]['EUI'].values[0])-1
    total = total.loc[total['year_month']==month]
    
    # plot all buildings
    if type == 'electricity':
        title_name = 'EUI'
        unit_name = '(Wh/m2)'
    else:
        title_name = 'WEI'
        unit_name = '(m3/m2)'
    total = total.sort_values(['usage','EUI'], ascending = [True, False])
    categories = total['usage'].unique()
    color_map = plt.cm.get_cmap('Set3', len(categories))
    color_dict = {cat: color_map(i) for i, cat in enumerate(categories)}
    colors = [color_dict[total.loc[total['EUI']==y]['usage'].values[0]] for y in total['EUI']]
    fig1 = total.loc[total['year_month']==month].plot(kind='bar', x = 'block',y = 'EUI' ,color=colors, figsize=(24, 12))
    plt.axhline(y=val_camp, label ='Camp' + title_name, color='dimgrey', linestyle =":")
    plt.ylabel(title_name + '-' + unit_name, fontsize=22)
    plt.xticks(rotation=90, fontsize=22)
    
    plt.gca().set_xticklabels([f'{truncate_string(n)}' for n in total.loc[total['year_month']==month]['block'].tolist()])
    plt.yticks(rotation=0, fontsize=22)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim(0, 1.40 * val_high)
    current_values = plt.gca().get_yticks()
    if type == 'electricity':
        plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    else:
        plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
    color_dict['Camp ' + title_name]=matplotlib.colors.to_rgb('dimgrey')
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:-1]]
    markers.append([plt.Line2D([0,0],[0,0],color=color,  linestyle=':') for color in color_dict.values()][-1])
    plt.legend(markers, color_dict.keys(), numpoints=1, bbox_to_anchor=(1,1), prop={'size': 20})
    plt.title(camp_fullname + ': ' +title_name + ' by Buildings in ' + month, fontsize=26) 
    fig1.figure.savefig(root_path + output_img_folder + 'buildings_' + title_name + '.png', bbox_inches = 'tight') #buildings_EUI.png
    
    # plot building in peer group
    for key, value in func_dic_usage.items():
        for blk in value:
            total_group = total.loc[total['usage']==key]
            med = total_group['EUI'].median()
            if type == 'electricity':
                title_name = 'EUI'
                unit_name = '(Wh/m2)'
            else:
                title_name = 'WEI'
                unit_name = '(m3/m2)'
            colors = []
            for each in total_group['block']:
                if each == blk:
                    colors.append('tomato')
                else:
                    colors.append('silver')
            fig = total_group.loc[total['year_month']==month].plot(kind='bar', x = 'block',y = 'EUI' ,color=colors, figsize=(24, 12))
            plt.axhline(y=med, label ='Peer Group Median ' + title_name, color='dimgrey', linestyle =":")
            plt.ylabel(title_name + '-' + unit_name, fontsize=22)
            plt.xticks(rotation=90, fontsize=22)
            plt.gca().set_xticklabels([f'{truncate_string(n)}' for n in total_group.loc[total['year_month']==month]['block'].tolist()])
            plt.yticks(rotation=0, fontsize=22)
            plt.ticklabel_format(style='plain', axis='y')
            plt.xlabel('', fontsize=22)
            plt.tick_params(labelsize=22)
            current_values = plt.gca().get_yticks()
            if type == 'electricity':
                plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
            else:
                plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
            plt.legend()            
            plt.title(key + " Buildings: " + title_name  + ' by Buildings in ' + month, fontsize=26)
            fig.figure.savefig(root_path + output_img_folder + truncate_string(blk) + '_peer_' + title_name + '.png', bbox_inches = 'tight') # blk-213_peer_EUI.png
    
    # plot improvement
    if type == 'electricity':
        title_name = 'EUI Change'
        unit_name = '%'
    else:
        title_name = 'WEI Change'
        unit_name = '%'
    total_change = total_change.sort_values(['change'], ascending = False)
    colors = ['lightgreen' if y<0 else 'orangered' for y in total_change['change']]
    fig2 = total_change.plot(x='block', y = 'change', kind='bar', color=colors, figsize=(24,12))
    plt.axhline(y=change_camp, label ='Camp' + title_name, color='dimgrey', linestyle =":")
    plt.axhline(y=0, color = 'black', linewidth = 0.8)
    plt.ylabel(title_name + '-' + unit_name, fontsize=22)
    plt.xticks(rotation=90, fontsize=22)
    plt.gca().set_xticklabels([f'{truncate_string(n)}' for n in total_change['block'].tolist()])
    plt.yticks(rotation=0, fontsize=22)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim(1.10 * change_low, 1.40 * change_high)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(["{:.0%}".format(x) for x in current_values])
    color_dict = {'increase':'orangered', 'decrease':'lightgreen'}
    color_dict['Camp ' + title_name]=matplotlib.colors.to_rgb('dimgrey')
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:-1]]
    markers.append([plt.Line2D([0,0],[0,0],color=color,  linestyle=':') for color in color_dict.values()][-1])
    plt.legend(markers, color_dict.keys(), numpoints=1, bbox_to_anchor=(1,1), prop={'size': 20})
    plt.title(camp_fullname + ': Month to Month '+ title_name + ' by Buildings in ' + month, fontsize=26)
    fig2.figure.savefig(root_path + output_img_folder + 'buildings_' + title_name[0:3] + '_changes' + '.png', bbox_inches = 'tight') #buildings_EUI_changes.png
    
    # plot improvement in peer group
    for key, value in func_dic_usage.items():
        total_change_group =  total_change.loc[total_change['usage_x']==key]
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '%'
        else:
            title_name = 'WEI'
            unit_name = '%'
        colors = ['lightgreen' if y<0 else 'orangered' for y in total_change_group['change']]
        fig2 = total_change_group.plot(x='block', y = 'change', kind='bar', color=colors, figsize=(24,12))
        plt.axhline(y=0, color = 'black', linewidth = 0.8)
        plt.ylabel(title_name + '-' + unit_name, fontsize=22)
        plt.xticks(rotation=90, fontsize=22)
        plt.gca().set_xticklabels([f'{truncate_string(n)}' for n in total_change_group['block'].tolist()])
        plt.yticks(rotation=0, fontsize=22)
        plt.ticklabel_format(style='plain', axis='y')
        plt.xlabel('', fontsize=22)
        plt.tick_params(labelsize=22)
        plt.ylim(1.10 * change_low, 1.10 * change_high)
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(["{:.0%}".format(x) for x in current_values])
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:-1]]
        plt.legend(markers, color_dict.keys(), numpoints=1, bbox_to_anchor=(1,1), prop={'size': 20})
        plt.title(key + " Buildings:  Month to Month " + title_name  + ' Change by Buildings in ' + month, fontsize=26) #Training Buildings EUI Changes in 2023-02
        fig2.figure.savefig(root_path + output_img_folder + key + '_buildings_' + title_name + '_changes' + '.png', bbox_inches = 'tight') #Training_buildings_EUI_changes.png

    return val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low


def overall_consump_monthly_elec(func_dic, root_path, input_folder, gfa_file_path_name, output_img_folder, type, camp_fullname, month):
    """
    Generate camp overall elec monthly consumption graphs.

    Args: 
        func_dic (dictionary): building unit list
        root_path: file root path
        input_folder: input file path
        gfa_file_path_name: building characteristic file path
        output_img_folder: output image folder path
        type (str): type of measurement - 'Electricity Consumption' or 'EUI'
        camp_fullname (str): camp full name 
        month (str): selected month

    Returns:
        Graphs:
            Camp overall consumption                e.g. Jurong Camp Consumption
            Camp overall EUI                        e.g. Jurong Camp EUI

        Variables:
            value_0: variable in last month
            value_1: variable in this month
    """
    total_elec = pd.DataFrame(columns=['Time','Electricity Consumption','new_date','new_time','year_month'])
    elec_value = 0
    for key, value in func_dic.items():
        for itemname in value:
            file_path_name = root_path + input_folder + itemname + '.csv'
            block_elec = pd.read_csv(file_path_name)
            # seperate time and date
            block_elec = clean_column(block_elec)
            block_elec['Value'].replace(to_replace=0, method='ffill')
            # date/time unique value list
            block_elec['year_month'] = block_elec['Time'].str[0:7]
            total_elec = pd.concat([total_elec, block_elec])
            elec_value += block_elec['Value'].sum()
    all_elec = total_elec.groupby(['year_month'])['Value'].sum().reset_index()

    # get GFA
    gfa_df = pd.read_csv(gfa_file_path_name)
    gfa_value = gfa_df['GFA'].sum()/2
    all_elec['gfa_value'] = gfa_value
    all_elec['EUI'] = all_elec['Value']/all_elec['gfa_value']
    all_elec.rename(columns={'Value':'Electricity Consumption'}, inplace=True)

    # get variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    value_0 = all_elec.loc[all_elec.year_month==last_month][type]
    value_1 = all_elec.loc[all_elec.year_month==month][type]
    all_elec = all_elec.loc[(all_elec['year_month']==month) | (all_elec['year_month']==last_month)]
    
    # plot overall consumption / EUI
    if type =="Electricity Consumption":
        unit_name = '(Wh)'
    else:
        unit_name = '(Wh/m2)'
    colors = 'orange'
    fig = all_elec.plot.bar(x = 'year_month', y = type, color=colors, width=0.15, figsize=(24, 9))
    plt.ylabel(type+unit_name, fontsize=22)
    plt.xticks(rotation=0, fontsize=22)
    plt.tick_params(labelsize=22)
    plt.locator_params(axis='x', nbins=20)
    plt.title(camp_fullname + ' ' + type, fontsize=26)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("Month",fontsize=22)
    plt.legend(bbox_to_anchor=(1.0, 1.0),prop={'size': 22})
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    fig.figure.savefig(root_path + output_img_folder + 'overall_' + type + '.png', bbox_inches = 'tight')

    return value_0, value_1
    

def overall_consump_monthly_water(func_dic, root_path, input_folder, gfa_file_path_name, output_img_folder, type, camp_fullname, month):
    """
    Generate camp overall water monthly consumption graphs.

    Args: 
        func_dic (dictionary): building unit list
        root_path: file root path
        input_folder: input file path
        gfa_file_path_name: building characteristic file path
        output_img_folder: output image folder path
        type (str): type of measurement - 'Water Consumption' or 'WEI'
        camp_fullname (str): camp full name 
        month (str): selected month

    Returns:
        Graphs:
            Camp overall consumption                e.g. Jurong Camp Consumption
            Camp overall WEI                        e.g. Jurong Camp WEI

        Variables:
            value_0: variable in last month
            value_1: variable in this month
    """
    total_elec = pd.DataFrame(columns=['Time','Water Consumption','new_date','new_time','year_month'])
    elec_value = 0
    for key, value in func_dic.items():
        for itemname in value:
            file_path_name = root_path + input_folder + itemname + '.csv'
            block_elec = pd.read_csv(file_path_name)
            # seperate time and date
            block_elec = clean_column(block_elec)
            block_elec['Value'].replace(to_replace=0, method='ffill')
            # date/time unique value list
            block_elec['year_month'] = block_elec['Time'].str[0:7]
            total_elec = pd.concat([total_elec, block_elec])
            elec_value += block_elec['Value'].sum()
    all_elec = total_elec.groupby(['year_month'])['Value'].sum().reset_index()

    # get GFA
    gfa_df = pd.read_csv(gfa_file_path_name)
    gfa_value = gfa_df['GFA'].sum()/2
    all_elec['gfa_value'] = gfa_value
    all_elec['WEI'] = all_elec['Value']/all_elec['gfa_value']
    all_elec.rename(columns={'Value':'Water Consumption'}, inplace=True)

    # get variable
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    value_0 = all_elec.loc[all_elec.year_month==last_month][type]
    value_1 = all_elec.loc[all_elec.year_month==month][type]
    all_elec = all_elec.loc[(all_elec['year_month']==month) | (all_elec['year_month']==last_month)]
    
    # plot overall consumption / WEI
    if type =="Water Consumption":
        unit_name = '(m3)'
    else:
        unit_name = '(m3/m2)'
    colors = 'lightskyblue'
    fig = all_elec.plot.bar(x = 'year_month', y = type, color=colors, width=0.15, figsize=(24, 9), title='Monthly ' + type +' in '+camp_fullname)
    plt.ylabel(type+unit_name, fontsize=22)
    plt.xticks(rotation=0, fontsize=22)
    plt.tick_params(labelsize=22)
    plt.locator_params(axis='x', nbins=20)
    plt.title(camp_fullname + ' ' + type, fontsize=26)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("Month",fontsize=22)
    plt.legend(bbox_to_anchor=(1.0, 1.0),prop={'size': 22}) #'best'right
    if type == "Water Consumption":
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    else:
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{0:.3f}'.format(x) for x in current_values])
    fig.figure.savefig(root_path + output_img_folder + 'overall_' + type + '.png', bbox_inches = 'tight')

    return value_0, value_1


def each_building(root_path, type, output_img_path, month, total):
    """
    Generate graphs and variables for each building

    Args:
        root_path: file root path
        type (str): type of consumption - 'electricity' or 'Water'
        output_img_path: output image folder path
        month: selected month
        total: the combined dataframe provided by get_datafram function

    Returns:
        Graphs:
            Monthly Performance
            Weekly Performance
            Average Hourly Performance
        Variables:
            description (dataframe): building information with columns 'block', 'EUI', 'EUI_peer', 'EUI_last', 'wk', 'wk_val', 'hr', 'hr_val', 'median'
    
    """
    description = pd.DataFrame(columns=['block', 'EUI', 'EUI_peer', 'EUI_last', 'wk', 'wk_val', 'hr', 'hr_val', 'median'])
    # get month variables
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    
    # get weeks in selected month

    weeks = []
    firstdate = pd.to_datetime(pd.Series([month + '-01']))
    res = calendar.monthrange(firstdate[0].year, firstdate[0].month)
    day = res[1]
    lastdate = str(firstdate[0].year) + '-' + str(firstdate[0].month) + '-' + str(day)
    lastdate = pd.to_datetime(pd.Series([lastdate]))
    maxwk = lastdate.dt.isocalendar()['week'][0]
    if firstdate.dt.isocalendar()['day'][0] != 1:
        week = firstdate.dt.isocalendar()['week'][0]
        for i in range(week,week+6):
            weeks.append(i)
        for w in range(0,6):
            if weeks[w]>52:
                weeks[w] = weeks[w]-52
    if len(weeks) != 0:
        if weeks[-1] > maxwk:   
            weeks.pop(-1)

    elif len(weeks) == 0: 
        week = firstdate.dt.isocalendar()['week'][0]
        for i in range(week,week+5):
            weeks.append(i)
    
    total = total.drop_duplicates(subset=['Time', 'year_month','week','hour','usage','block','Value','gfa'], keep='first')
    blklst = list(total.loc[total['year_month']==month]['block'].unique())

    for itemname in blklst:
        # get best performing peer and median peer data
        total0 = total.loc[(total['year_month']==month)]
        total0 = total0.groupby(['year_month','usage','block','gfa'])['Value'].sum().reset_index()
        total0['EUI'] = total0['Value']/total0['gfa']
        usage = total0.loc[total0['block'] == itemname]['usage'].values[0]
        row_self = total0.loc[(total0['usage'] == usage) & (total0['block'] == itemname), ['year_month','usage','block','EUI']]
        EUI = row_self.iloc[0][3]
        good = total0.loc[total0['usage'] == usage,['EUI']].min().values[0]
        med_value = total0.loc[total0['usage'] == usage,['EUI']].median().values[0]

        # plot monthly performance
        total_last = total.groupby(['year_month','usage','block','gfa'])['Value'].sum().reset_index()
        total_last = total_last.loc[(total_last['year_month']==month) | (total_last['year_month']==last_month)]
        total_last['EUI'] = total_last['Value']/total_last['gfa']
        val = total_last.loc[(total_last['block'] == itemname) & (total_last['year_month'] == month),['EUI']].iloc[0]
        val_last = total_last.loc[(total_last['block'] == itemname) & (total_last['year_month'] == last_month),['EUI']].values[0]
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        if total_last.loc[(total_last['block'] == itemname) & (total_last['year_month'] == month),['EUI']].values[0] <= val_last:
            self_color = 'lightgreen'
        else:
            self_color = 'tomato'
        colors = ['silver', self_color]
        fig = total_last.loc[total_last['block'] == itemname].plot(x = 'year_month', y = 'EUI', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=0, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title('Monthly Performance', fontsize=26)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Month", fontsize=22)
        plt.legend([title_name], loc='upper left', prop={'size': 20}) 
        fig.figure.savefig(root_path + output_img_path + truncate_string(itemname) + '_monthly' + '.png',  bbox_inches = 'tight') #blk-213_monthly.png
        
        # plot weekly performance    
        total_week1 = total.loc[total['week'].isin(weeks)]
        gfa = total_week1.loc[total_week1['block'] == itemname].iloc[0]
        gfa = gfa['gfa']
        total_week1 = total_week1.loc[total_week1['block'] == itemname]
        total_week1['date'] = total_week1['Time'].str[0:10]
        total_week2 = total_week1.groupby('week').agg({'Value': sum}).reset_index()
        total_week2 = total_week2.rename(columns={'Value': 'Sum_value'})       
        total_week = pd.merge(total_week1, total_week2, left_on='week', right_on='week', how='right')
        total_week = total_week.drop_duplicates(subset=['week'], keep='first')
        total_week = total_week.loc[:,['date','Sum_value']].reset_index(drop=True)
        total_week = total_week.rename(columns={'Sum_value': 'Weekly Consumption'})
        total_week = total_week.sort_values('date')
        total_week['Weekly Consumption'] = total_week['Weekly Consumption']/gfa
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        colors = []
        val_max = total_week['Weekly Consumption'].max()
        idx_max = total_week['Weekly Consumption'].astype('float64').idxmax()
        week_max = total_week.iloc[idx_max]['date']
        for each in total_week['Weekly Consumption']:
            if each == val_max:
                colors.append('tomato')
            else:
                colors.append('silver')
        fig = total_week.plot(x = 'date', y = 'Weekly Consumption', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=0, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title('Weekly Performance' , fontsize=26) #Weekly Performance
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Weekly", fontsize=22)
        plt.legend([title_name], loc='upper left', prop={'size': 20}) 
        fig.figure.savefig(root_path + output_img_path + truncate_string(itemname) + '_weekly' + '.png',  bbox_inches = 'tight') #blk-213_weekly.png

        # plot average hourly performance
        total_hour = total.loc[total['block'] == itemname]
        gfa = total_hour.loc[total_hour['block'] == itemname].iloc[0]
        gfa = gfa['gfa']
        total_hour = total_hour.loc[(total_hour['year_month']==month)]
        total_hour['date'] = total_hour['Time'].str[0:10]
        total_hour = total_hour.groupby(['date','hour']).agg({'Value': sum}).reset_index()
        total_hour = total_hour.groupby(['hour']).agg({'Value': 'mean'}).reset_index()
        total_hour = total_hour.rename(columns={'Value': 'Average Hourly Consumption'})
        total_hour['Average Hourly Consumption'] = total_hour['Average Hourly Consumption']/gfa
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        colors = []
        hour_max = total_hour['Average Hourly Consumption'].max()
        idx_max = total_hour['Average Hourly Consumption'].astype('float64').idxmax()
        peak = total_hour.iloc[idx_max]['hour']
        for each in total_hour['Average Hourly Consumption']:
            if each == hour_max:
                colors.append('tomato')
            else:
                colors.append('silver')
        fig = total_hour.plot(x = 'hour', y = 'Average Hourly Consumption', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=0, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title( 'Average Hourly Performance', fontsize=26) #Average Hourly Performance
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Hourly", fontsize=22)
        plt.legend([title_name],loc='upper left', prop={'size': 20})
        fig.figure.savefig(root_path + output_img_path + truncate_string(itemname) + '_hourly' + '.png',  bbox_inches = 'tight') #blk-213_hourly.png
        
        #store variables into description dataframe
        new = {'block':itemname, 
               'EUI': EUI, 
               'EUI_peer': good, 
               'EUI_last': val_last[0],
               'wk': week_max, 
               'wk_val': val_max, 
               'hr': peak,
               'hr_val': hour_max,
               'median' : med_value
              }
        description.loc[len(description)] = new
        
    return description


def each_unit(root_path, type, output_img_path, month, total, func_dic):
    """
    Generate graphs and variables for each unit

    Args:
        root_path: file root path
        type (str): type of consumption - 'electricity' or 'Water'
        output_img_path: output image folder path
        month: selected month
        total: the combined dataframe provided by get_datafram function
        func_dic (dictionary): building unit list

    Returns:
        Graphs:
            Monthly Performance
            Weekly Performance
            Average Hourly Performance
        Variables:
            description (dataframe): building information with columns 'block','val', 'val_last', 'wk', 'val_wk', 'hr', 'val_hr'
    
    """
    description = pd.DataFrame(columns=['block','val', 'val_last', 'wk', 'val_wk', 'hr', 'val_hr'])
    
    # get unit data
    building = {}
    for key, value in func_dic.items():
        for itemname in value:
            if itemname not in building.keys():
                building[itemname] = 1
            else:
                building[itemname] += 1 
    
    total['val_avg'] = np.nan
    total['num'] = np.nan
    for i in range(total.shape[0]):
        for key, value in building.items():
            if total['block'][i] == key:
                total['num'][i] = value
    total['Value'] = total['Value']/total["num"]
    total['gfa'] = total['gfa']/total["num"]
    total = total.groupby(['Time', 'year_month', 'week','hour', 'unit'], as_index=False).agg({'Value':'sum','gfa':'sum'})

    # get data last month
    date_object = datetime.strptime(month, '%Y-%m').date()
    first = date_object.replace(day=1)
    last_month = first + dt.timedelta(days=-1)
    last_month = last_month.strftime("%Y-%m")
    
    # get weeks in selected month

    weeks = []
    firstdate = pd.to_datetime(pd.Series([month + '-01']))
    res = calendar.monthrange(firstdate[0].year, firstdate[0].month)
    day = res[1]
    lastdate = str(firstdate[0].year) + '-' + str(firstdate[0].month) + '-' + str(day)
    lastdate = pd.to_datetime(pd.Series([lastdate]))
    maxwk = lastdate.dt.isocalendar()['week'][0]
    if firstdate.dt.isocalendar()['day'][0] != 1:
        week = firstdate.dt.isocalendar()['week'][0]
        for i in range(week,week+6):
            weeks.append(i)
        for w in range(0,6):
            if weeks[w]>52:
                weeks[w] = weeks[w]-52
    if len(weeks) != 0:
        if weeks[-1] > maxwk:   
            weeks.pop(-1)

    elif len(weeks) == 0: 
        week = firstdate.dt.isocalendar()['week'][0]
        for i in range(week,week+5):
            weeks.append(i)
    
    unitlst = list(total.loc[total['year_month']==month]['unit'].unique())

    for itemname in unitlst:
        # plot monthly performance
        total_last = total.loc[total['unit']==itemname]
        total_last = total_last.groupby(['year_month','unit'], as_index = False).agg({'Value':'sum','gfa':'first'})
        total_last['EUI'] = total_last['Value']/total_last['gfa']
        total_last = total_last.loc[(total_last['year_month']==month) | (total_last['year_month']==last_month)]
        val = total_last.loc[(total_last['unit'] == itemname) & (total_last['year_month'] == month),['EUI']].values[0]
        val_last = total_last.loc[(total_last['unit'] == itemname) & (total_last['year_month'] == last_month),['EUI']].values[0]
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        if total_last.loc[(total_last['unit'] == itemname) & (total_last['year_month'] == month),['EUI']].values[0] <= val_last:
            self_color = 'lightgreen'
        else:
            self_color = 'tomato'
        colors = ['silver', self_color]
        fig = total_last.loc[total_last['unit'] == itemname].plot(x = 'year_month', y = 'EUI', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=90, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title('Monthly Performance', fontsize=26) #Monthly Performance
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Month", fontsize=22)
        plt.legend(loc='upper left', prop={'size': 20})  # 'best'
        fig.figure.savefig(root_path + output_img_path + itemname + '_monthly.png',  bbox_inches = 'tight') #30SCE_monthly.png
        
        # plot weekly performance        
        total_week1 = total.loc[total['week'].isin(weeks)]
        gfa = total_week1.loc[total_week1['unit'] == itemname].iloc[0]
        gfa = gfa['gfa']
        total_week1 = total_week1.loc[total_week1['unit'] == itemname]
        total_week1['date'] = total_week1['Time'].str[0:10]
        total_week2 = total_week1.groupby('week').agg({'Value': sum}).reset_index()
        total_week2 = total_week2.rename(columns={'Value': 'Sum_value'})       
        total_week = pd.merge(total_week1, total_week2, left_on='week', right_on='week', how='right')
        total_week = total_week.drop_duplicates(subset=['week'], keep='first')
        total_week = total_week.loc[:,['date','Sum_value']].reset_index(drop=True)
        total_week = total_week.rename(columns={'Sum_value': 'Weekly Consumption'})
        total_week = total_week.sort_values('date')
        total_week['Weekly Consumption'] = total_week['Weekly Consumption']/gfa
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        colors = []
        val_max = total_week['Weekly Consumption'].max()
        idx_max = total_week['Weekly Consumption'].astype('float64').idxmax()
        week_max = total_week.iloc[idx_max]['date']
        for each in total_week['Weekly Consumption']:
            if each == val_max:
                colors.append('tomato')
            else:
                colors.append('silver')
        fig = total_week.plot(x = 'date', y = 'Weekly Consumption', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=90, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title('Weekly Performance' , fontsize=26) #Weekly Performance
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Weekly", fontsize=22)
        plt.legend(loc='upper left', prop={'size': 20})
        fig.figure.savefig(root_path + output_img_path + itemname + '_weekly.png',  bbox_inches = 'tight')
        
        # plot average hourly performance  
        total_hour = total.loc[total['unit'] == itemname]
        gfa = total_hour.loc[total_hour['unit'] == itemname].iloc[0]
        gfa = gfa['gfa']
        total_hour = total_hour.loc[(total_hour['year_month']==month)]
        total_hour['date'] = total_hour['Time'].str[0:10]
        total_hour = total_hour.groupby(['date','hour']).agg({'Value': sum}).reset_index()
        total_hour = total_hour.groupby(['hour']).agg({'Value': 'mean'}).reset_index()
        total_hour = total_hour.rename(columns={'Value': 'Average Hourly Consumption'})
        total_hour['Average Hourly Consumption'] = total_hour['Average Hourly Consumption']/gfa
        if type == 'electricity':
            title_name = 'EUI'
            unit_name = '(Wh/m2)'
        else:
            title_name = 'WEI'
            unit_name = '(m3/m2)'
        colors = []
        hour_max = total_hour['Average Hourly Consumption'].max()
        idx_max = total_hour['Average Hourly Consumption'].astype('float64').idxmax()
        peak = total_hour.iloc[idx_max]['hour']
        for each in total_hour['Average Hourly Consumption']:
            if each == hour_max:
                colors.append('tomato')
            else:
                colors.append('silver')
        fig = total_hour.plot(x = 'hour', y = 'Average Hourly Consumption', kind='bar', color=colors, figsize=(24, 12))
        plt.ylabel(title_name + ' ' + unit_name, fontsize=22)
        plt.xticks(rotation=90, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.locator_params(axis='x', nbins=10)
        plt.title('Average Hourly Performance', fontsize=26)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(labelsize=22)
        current_values = plt.gca().get_yticks()
        if type == 'electricity':
            plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        else:
            plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plt.xlabel("Hourly", fontsize=22)
        plt.legend(loc='upper left', prop={'size': 20})
        fig.figure.savefig(root_path + output_img_path + itemname + '_hourly.png',  bbox_inches = 'tight') #30SCE_hourly.png
    
        # store variables into description dataframe
        new = {'block':itemname, 
               'val': val[0] , 
               'val_last': val_last[0], 
               'wk': week_max, 
               'val_wk': val_max, 
               'hr':peak,
               'val_hr': hour_max
               }
        description.loc[len(description)] = new
     
    return description
               

def anomaly_detection(root_path, input_path, output_img_folder, start_date, end_date, lookback_period, top_k ,camp_fullname, unit_of_measurement, miss_val_threshold, blk = None):
    """
    Detect and graph anomaly
    
    Args:
        root_path: file root path
        input_path: input file path
        output_img_folder: output image folder path
        start_date (str): the first date of the selected month
        end_date (str): the last date of the selected month
        lookback_period (int): the data that are used to detect anomaly. Make sure the data is included. 
                               For example, if the start date is 01.04.2023 and the earliest data available is 01.03.2023, 
                               then the lookback period can not be longer than 30 days.
        top_k (int): the number of the top highest anomalies
        camp_fullname (str): camp full name 
        unit_of_measurement (str): unit used for measurement
        miss_val_threshold: drop the detection if the threshold is reached
        blk: Generate the anomaly ranking for certain building if specified. Otherwise, the result ranks among buildings. 
             Default is None.
    
    Returns:
        Graphs:
            top k anomaly graphs_day
            top k anomaly graphs_period
        
        Variables:
            anomalies (dictionary): top anomaly building with information including date, deviation, unit of measurement, deviation percentage
    """
    # load data into a dictionary
    path = os.path.abspath(input_path[:-1])
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    pre_data_dict = {}
    for f in csv_files:
        df = pd.read_csv(f)
        pre_data_dict[f.split("/")[-1][:-4]] = df
        
    # preprocess data
    preprocessor = Preprocessor(pre_data_dict, miss_val_threshold = miss_val_threshold)
    data_dict = preprocessor.preprocess()

    # statistic method
    anomalies = {}
    lookback_period = lookback_period
    sort_by = 'deviation_percentage'
    dev_perc_threshold = 5
    namespace = camp_fullname
    unit_of_measurement = unit_of_measurement

    ad_detector = AutoAD(namespace = camp_fullname, data_frequency = preprocessor.data_frequency, method = 'statistic', unit_of_measurement = unit_of_measurement, lookback_period = lookback_period)

    start_date = pd.to_datetime(start_date) 
    end_date = pd.to_datetime(end_date)    
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

    if blk == None:
        score_lst = list(score_dict.values())
        type = 'all'
    else:
        score_dict_copy = {}
        type = ''
        for each in score_dict.keys():
            if blk in each:
                score_dict_copy[each] = score_dict[each]
        score_dict = score_dict_copy
        score_lst = list(score_dict.values())

    score_lst.sort(reverse = True, key = lambda x:x[sort_by]) # use deviation to sort
    
    # report the top k anomalies 
    count = 0
    for i in range(min(top_k, len(data_dict))):
        if count == min(top_k, len(data_dict)) | bool(score_lst) == False: 
            break
        if bool(score_lst) == False: 
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
            ax.plot(score_df.index, score_df.Value, label = 'Actual Consumption')
            ax.plot(score_df.index, score_df.pred, label = 'Expected Consumption')
            ax.scatter(score_df[score_df.anomaly == True].index, score_df[score_df.anomaly == True].Value, label = 'Estimated Anomaly')
            ax.fill_between(score_df.index, score_df.pred_low, score_df.pred_high, alpha = 0.1)
            ax.title.set_text(truncate_string(mp) +  ' Daily Anomalous Consumption on ' + str(date)) #blk-218 daily anomalous consumption on 2023-02-03
            ax.set_ylim(0, score_df.Value.max()*1.4)
            ax.legend(loc = 'upper left', fontsize = 10)
            ax.figure.savefig(root_path + output_img_folder + type + '_' + truncate_string(mp) +  '_' + str(date) +'_day.png') #,  bbox_inches = 'tight') #30SCE_monthly.png

            #if report:
            # generate anomaly message
            anomaly_message = ad_detector.anomaly_message(score_df)
            print(anomaly_message)

            # plot detection date + lookback period
            fig, ax = plt.subplots(figsize=(17, 3))
            ax.plot(long_df.index, long_df.Value, label = 'Actual Consumption')
            ax.plot(score_df.index, score_df.pred, label = 'Expected Consumption')
            ax.axvspan(score_df.index[0], score_df.index[-1], alpha=0.2, color='r')
            ax.title.set_text('Consumption Comparison with Past ' + str(lookback_period) + ' days') #Consumption comparison with past 30 days
            ax.set_ylim(0, long_df.Value.max())
            ax.legend(loc = 'upper left', fontsize = 10)
            ax.figure.savefig(root_path + output_img_folder + type + '_' + truncate_string(mp) +  '_' + str(date) +'_period.png')
            anomalies[count] = {'blk':mp, 'date':date, 'deviation':deviation_val, 'unit_of_measurement': unit_of_measurement, 'deviation_percentage': deviation_per} 

    return anomalies


def takeDeviation(elem):
    """
    Select the 'deviation_percentnage'
    """
    return elem['deviation_percentage']

