from functions import *
from fpdf import FPDF
from ispots.anomaly_detector import AutoAD, Preprocessor

if __name__ == "__main__":

    # Define path
    root_path = "./"
    # Define selected month
    select_month = '2023-05'
    end_date = '2023-5-31'
    start_date = '2023-5-1'
    lookback_period = 30
    miss_val_threshold = 0.01
    # Camp name
    camp = 'JC'
    camp_fullname = 'AC$B Camp'
    gfa_file_path_name = root_path+"inputdata/" + camp + "/chateristic/GFA_JC.csv"
    
    # Electricity files path
    elec_input_folder = "inputdata/" + camp + "/electrical/monthly/"
    elec_output_img_folder = "output/" + camp + "/elec_output_img/"
    elec_output_csv_folder = "output/" + camp + "/elec_output_csv/"
    elec_raw_path = root_path + elec_input_folder + "*.csv"
    elec_multifloor_input_folder = "inputdata/" + camp + "/electrical/multifloor/"
    elec_floor_output_folder = "output/" + camp + '/floorlevel/'
    elec_rawpath = "./rawdata/"+camp+"_elect/"
    elec_anomaly_output_folder = 'output/' + camp + '/anomaly_output_img/'
    
    # Water files path
    water_input_folder = "inputdata/" + camp + "/water/monthly/"
    water_multifloor_input_folder = "inputdata/" + camp + "/water/multifloor/monthly/"
    water_output_img_folder = "output/" + camp + "/water_output_img/"
    water_output_csv_folder = "output/" + camp + "/water_output_csv/"
    water_raw_path = root_path + water_input_folder + "*.csv"
    water_floor_output_folder = "output/" + camp + 'floorlevel_water'
    water_rawpath = "./rawdata/"+camp+"_water/"
    
    # Monthly report path
    report_output_folder = 'monthly_report_pdf/' + camp

    # Define time list and function list
    time_lst_elec = ['0000', '0030', '0100', '0130', '0200', '0230', '0300', '0330', '0400', '0430', '0500',
                     '0530', '0600', '0630', '0700', '0730', '0800', '0830', '0900', '0930', '1000',
                     '1030', '1100', '1130', '1200', '1230', '1300', '1330', '1400', '1430', '1500',
                     '1530', '1600', '1630', '1700', '1730', '1800', '1830', '1900', '1930', '2000',
                     '2030', '2100', '2130', '2200', '2230', '2300', '2330']
    time_lst_water = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100',
                      '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']

    func_dic_water_usage =  {'Admin Office': ['dsta_blk-700-water',  'dsta_blk-704-water',  'dsta_blk-723-water',
                                            'dsta_blk-724-water',  'dsta_blk-725-water',  'dsta_blk-726-water',
                                            'dsta_blk-902-water',  'dsta_blk-900-water',  'dsta_blk-901-water'],
                            'Shared Facilities': ['dsta_blk-701-water',  'dsta_blk-800-water',  'dsta_blk-801-water',
                                                'dsta_blk-808-water',  'dsta_blk-820-water',  'dsta_blk-822-water',
                                                'dsta_blk-806-water',  'dsta_blk-809-water'],
                            'Operation Maintenance': ['dsta_blk-921-water'],
                            'Training': ['dsta_blk-812-water',      'dsta_blk-815-water',     'dsta_blk-817-water'],
                            'Non-discretionary Usage': ['dsta_blk-699-water']}

    func_dic_elec_usage = {'Admin Office': ['dsta_blk-700-elect',  'dsta_blk-704-elect',  'dsta_blk-723-elect',
                                            'dsta_blk-724-elect',  'dsta_blk-725-elect',  'dsta_blk-726-elect',
                                            'dsta_blk-902-elect',  'dsta_blk-900-elect',  'dsta_blk-901-elect'],
                            'Shared Facilities': ['dsta_blk-701-elect',  'dsta_blk-800-elect',  'dsta_blk-801-elect',
                                                    'dsta_blk-808-elect',  'dsta_blk-820-elect',  'dsta_blk-822-elect',
                                                    'dsta_blk-806-elect',  'dsta_blk-809-elect'],
                            'Operation Maintenance': ['dsta_blk-921-elect'],
                            'Training': ['dsta_blk-812-elect',  'dsta_blk-815-elect',  'dsta_blk-817-elect'],
                            'Non-discretionary Usage': ['dsta_blk-699-elect']}


    # When creating dict by unit, put all the building used by unit in the same key. For example, blk-102 is in the lists belongs to key '3DIV' and '3AMB'
    func_dic_elec_unit = {
                        'A7Rf': ['dsta_blk-900-elect', 'dsta_blk-901-elect', 'dsta_blk-902-elect'],
                        '9RbW': ['dsta_blk-723-elect',  'dsta_blk-724-elect',  'dsta_blk-725-elect',  'dsta_blk-726-elect',  'dsta_blk-704-elect',  'dsta_blk-701-elect'],
                        '2Qx': ['dsta_blk-921-elect'],
                        'P9Tk': ['dsta_blk-700-elect', 'dsta_blk-699-elect'],
                        '5YjK': ['dsta_blk-801-elect',  'dsta_blk-808-elect',  'dsta_blk-820-elect',  'dsta_blk-822-elect',  'dsta_blk-806-elect',  'dsta_blk-809-elect'],
                        'B3Z': ['dsta_blk-800-elect'],
                        'M8Fq': ['dsta_blk-815-elect'],
                        '6VhT': ['dsta_blk-817-elect'],
                        'C4N': ['dsta_blk-812-elect'],
                        '1LpK': ['dsta_blk-701-elect'],
                        'X5Gv': ['dsta_blk-701-elect', 'dsta_blk-704-elect']}
    
    func_dic_water_unit = {
                        'A7Rf': ['dsta_blk-900-water', 'dsta_blk-901-water', 'dsta_blk-902-water'],
                        '9RbW': ['dsta_blk-723-water',  'dsta_blk-724-water',  'dsta_blk-725-water',  'dsta_blk-726-water',  'dsta_blk-704-water',  'dsta_blk-701-water'],
                        '2Qx': ['dsta_blk-921-water'],
                        'P9Tk': ['dsta_blk-700-water', 'dsta_blk-699-water'],
                        '5YjK': ['dsta_blk-801-water',  'dsta_blk-808-water',  'dsta_blk-820-water',  'dsta_blk-822-water',  'dsta_blk-806-water',  'dsta_blk-809-water'],
                        'B3Z': ['dsta_blk-800-water'],
                        'M8Fq': ['dsta_blk-815-water'],
                        '6VhT': ['dsta_blk-817-water'],
                        'C4N': ['dsta_blk-812-water'],
                        '1LpK': ['dsta_blk-701-water'],
                        'X5Gv': ['dsta_blk-701-water', 'dsta_blk-704-water']}
    
    ########################################################
    # Step 1. Data processing and import data as dataframe #
    ########################################################

    blk_lst = ['dsta_blk-701', 'dsta_blk-704', 'dsta_blk-809']
    separate_file_elec = ['dsta_blk-701-1-elect' , 'dsta_blk-701-2-elect', 
                          'dsta_blk-704-1-elect', 'dsta_blk-704-2-elect', 
                          'dsta_blk-809-1-elect', 'dsta_blk-809-2-elect']
    separate_file_water = ['dsta_blk-701-1-water' , 'dsta_blk-701-2-water', 
                           'dsta_blk-704-1-water', 'dsta_blk-704-2-water']
    
    # (only need to run at the first time)
    #combine_monthly_file_jc(root_path , elec_rawpath, select_month, elec_input_folder, separate_file_elec)
    #combine_monthly_file_jc(root_path, water_rawpath, select_month, water_input_folder, separate_file_water)
    #separate_file_jc(blk_lst, root_path, elec_input_folder, water_input_folder)
    
    total_elec, blklst_elec, missing_dict_elec = get_dataframe(func_dic_elec_usage, func_dic_elec_unit, root_path, elec_input_folder, gfa_file_path_name, 
                                                               elec_output_csv_folder, select_month)
    total_water, blklst_water, missing_dict_water = get_dataframe(func_dic_water_usage, func_dic_water_unit, root_path, water_input_folder, gfa_file_path_name, 
                                                                  water_output_csv_folder, select_month)

    ###################################################
    # Step 2. Create pdf class and report information #
    ###################################################

    WIDTH = 210
    HEIGHT = 297

    class PDF(FPDF):
        def header(self):
            # Logo
            self.image('company-logo.png', 5, 5, 10)
        def footer(self):
            # Page footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
    
    
    
    ##################################################
    # Step 3. Generate pdf report for Camp Commander #
    ##################################################
    
    # Instantiation of inherited class
    pdf = PDF()
    pdf.set_top_margin(20)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Cover page
    pdf.set_x(WIDTH/2)
    pdf.set_y(HEIGHT/2)
    pdf.set_font("Arial", "B", 16)
    pdf.cell((WIDTH-20), 10, camp_fullname + ' Monthly Report - Camp Commander', 0, 0, 'C')
    pdf.ln(10)
    pdf.cell((WIDTH-20), 10, select_month , 0, 0, 'C')
    pdf.add_page()

    # Table of Contents
    pdf.set_font("Arial", "B", 16)
    pdf.set_x(10)
    pdf.cell(0, 10, "Table of Contents", 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    toc = [("1. Overall Consumption", 3), ("2. Building Consumption", 5), ("3. Unit Consumption", 7),  ("4. Anomaly Detection", 10)]

    # Loop through the sections and add their titles and page numbers to the list
    for title, page in toc:
        pdf.set_x(50)
        pdf.cell(0, 10, title + '......................................', 0, 0)
        pdf.set_x(90)
        pdf.cell(0, 10, 'page ' + str(page), 0, 1, "C")
        pdf.ln(5)
    
    # 1. Overall consumption
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, f'1. Overall Consumption')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'1.1. Electricity')
    pdf.ln(10)
    type = 'Electricity Consumption'
    value_0, value_1 = overall_consump_monthly_elec(func_dic_elec_usage, root_path, elec_input_folder, gfa_file_path_name, elec_output_img_folder,
                                                    type, camp_fullname,select_month)
    pdf.image(root_path + elec_output_img_folder + 'overall' + '_' + type + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    change = ['increase' if value_1.iloc[0]-value_0.iloc[0] >= 0 else 'decrease'][0]
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, {camp_fullname} consumes {value_1.iloc[0]:,.0f} Wh electricity in total. Compared with last month, the total consumption {change} {value_1.iloc[0]-value_0.iloc[0]:,.0f} Wh.',align = 'L')
    
    pdf.ln(10)
    type = 'EUI'
    value_0, value_1 = overall_consump_monthly_elec(func_dic_elec_usage, root_path, elec_input_folder, gfa_file_path_name, elec_output_img_folder,
                                                    type, camp_fullname,select_month)
    pdf.image(root_path + elec_output_img_folder + 'overall' + '_' + type + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    change = ['increase' if value_1.iloc[0]-value_0.iloc[0] >= 0 else 'decrease'][0]
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, {camp_fullname} consumes {value_1.iloc[0]:,.0f} Wh/m2 EUI in total. Compared with last month, the total EUI {change} {value_1.iloc[0]-value_0.iloc[0]:,.0f} Wh/m2.',align = 'L')
    pdf.add_page()
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'1.2. Water')
    pdf.ln(10)
    type = 'Water Consumption'
    value_0, value_1 = overall_consump_monthly_water(func_dic_water_usage, root_path, water_input_folder, gfa_file_path_name, water_output_img_folder,
                                                     type, camp_fullname,select_month)
    pdf.image(root_path + water_output_img_folder + 'overall' + '_' + type + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    pdf.set_font('Arial', '', 10)
    change = ['increase' if value_1.iloc[0]-value_0.iloc[0] >= 0 else 'decrease'][0]
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, {camp_fullname} consumes {value_1.iloc[0]:,.0f} m3 water in total. Compared with last month, the total consumption {change} {value_1.iloc[0]-value_0.iloc[0]:,.0f} m3.',align = 'L')
    pdf.ln(10)
    type = 'WEI'
    value_0, value_1 = overall_consump_monthly_water(func_dic_water_usage, root_path, water_input_folder, gfa_file_path_name, water_output_img_folder,
                                                     type, camp_fullname,select_month)
    pdf.image(root_path + water_output_img_folder + 'overall' + '_' + type + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    change = ['increase' if value_1.iloc[0]-value_0.iloc[0] >= 0 else 'decrease'][0]
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, {camp_fullname} consumes {value_1.iloc[0]:,.3f} m3/m2 WEI in total. Compared with last month, the total WEI {change} {value_1.iloc[0]-value_0.iloc[0]:,.3f} m3/m2.',align = 'L')
    pdf.ln(10)
    
    # 2. Building Consumption
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, f'2. Building Consumption')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'2.1. Electricity')
    pdf.ln(10)
    type = 'electricity'
    val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low = monthly_consump_building(func_dic_elec_usage, root_path, total_elec, elec_output_img_folder,
                                                                                                                                  type, camp_fullname, select_month)
    pdf.image(root_path + elec_output_img_folder + 'buildings_EUI' + '.png', w = WIDTH-40, h = 0) # buildings_EUI.png
    pdf.ln(5)
    pdf.set_x(20)
    pdf.set_font('Arial', '', 10)
    # In 2023-02, among all the buildings, blk 100 has the highest EUI at XX Wh/m2, while blk 221 has the lowest at YY Wh/m2
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the buildings, {truncate_string(unit_high)} has the highest EUI at {val_high:,.0f} Wh/m2, while {truncate_string(unit_low)} has the lowest at {val_low:,.0f} Wh/m2.',align = 'L')
    pdf.ln(10)
    pdf.image(root_path + elec_output_img_folder + 'buildings_EUI_changes' + '.png', w = WIDTH-40, h = 0) # buildings_EUI_changes.png
    pdf.ln(5)
    pdf.set_x(20)
    # In 2023-02, among all the buildings, blk 125 has the highest month to month EUI decrease of -47%, while blk 218 has the highest month to month EUI increase of 372%
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the buildings, {truncate_string(change_unit_low)} has the highest month to month EUI decrease of {change_low*100:,.0f}%, while {truncate_string(change_unit_high)} has the highest month to month EUI increase of {change_high*100:,.0f}%.',align = 'L')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'2.2. Water')
    pdf.ln(10)
    type = 'water'
    val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low = monthly_consump_building(func_dic_water_usage, root_path, total_water, water_output_img_folder,
                                                                                                                                  type, camp_fullname, select_month)
    pdf.image(root_path + water_output_img_folder + 'buildings_WEI' + '.png', w = WIDTH-40, h = 0) # buildings_WEI.png
    pdf.ln(5)
    pdf.set_x(20)
    pdf.set_font('Arial', '', 10)
    # In 2023-02, among all the buildings, blk 100 has the highest EUI at XX Wh/m2, while blk 221 has the lowest at YY Wh/m2
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the buildings, {truncate_string(unit_high)} has the highest WEI at {val_high:,.3f} m3/m2, while {truncate_string(unit_low)} has the lowest at {val_low:,.3f} m3/m2.',align = 'L')
    pdf.ln(10)
    pdf.image(root_path + water_output_img_folder + 'buildings_WEI_changes' + '.png', w = WIDTH-40, h = 0) # buildings_WEI_changes.png
    pdf.ln(5)
    pdf.set_x(20)
    # In 2023-02, among all the buildings, blk 125 has the highest month to month EUI decrease of -47%, while blk 218 has the highest month to month EUI increase of 372%
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the buildings, {truncate_string(change_unit_low)} has the highest month to month WEI decrease of {change_low*100:,.0f}%, while {truncate_string(change_unit_high)} has the highest month to month WEI increase of {change_high*100:,.0f}%.',align = 'L')
    pdf.ln(10)
    
    # 3. Unit Consumption
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, f'3. Unit Consumption')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'3.1. Electricity')
    pdf.ln(10)
    type = 'electricity'
    val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low, med = monthly_consump_unit(func_dic_elec_unit, root_path, total_elec, elec_output_img_folder,
                                                                                                                                   type, camp_fullname,select_month)
    pdf.image(root_path + elec_output_img_folder + 'units_EUI' + '.png', w = WIDTH-40, h = 0) # units_EUI.png
    pdf.ln(5)
    pdf.set_x(20)
    pdf.set_font('Arial', '', 10)
    # In 2023-02, among all the buildings, blk 100 has the highest EUI at XX Wh/m2, while blk 221 has the lowest at YY Wh/m2
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the units, {unit_high} has the highest EUI at {val_high:,.0f} Wh/m2, while {unit_low} has the lowest at {val_low:,.0f} Wh/m2.',align = 'L')
    pdf.ln(10)
    pdf.image(root_path + elec_output_img_folder + 'units_EUI_changes' + '.png', w = WIDTH-40, h = 0) # units_EUI_changes.png 
    pdf.ln(5)
    pdf.set_x(20)
    # In 2023-02, among all the buildings, blk 125 has the highest month to month EUI decrease of -47%, while blk 218 has the highest month to month EUI increase of 372%
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the units, {change_unit_low} has the highest month to month EUI decrease of {change_low*100:,.0f}%, while {change_unit_high} has the highest month to month EUI increase of {change_high*100:,.0f}%.',align = 'L')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, f'3.2. Water')
    pdf.ln(10)
    type = 'water'
    val_high, unit_high, val_low, unit_low, change_high, change_unit_high, change_low, change_unit_low, med = monthly_consump_unit(func_dic_water_unit, root_path, total_water, water_output_img_folder,
                                                                                                                                   type, camp_fullname,select_month)
    pdf.image(root_path + water_output_img_folder + 'units_WEI' + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    pdf.set_font('Arial', '', 10)
    # In 2023-02, among all the buildings, blk 100 has the highest EUI at XX Wh/m2, while blk 221 has the lowest at YY Wh/m2
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the units, {unit_high} has the highest WEI at {val_high:,.3f} m3/m2, while {unit_low} has the lowest at {val_low:,.3f} m3/m2.',align = 'L')
    pdf.ln(10)
    pdf.image(root_path + water_output_img_folder + 'units_WEI_changes' + '.png', w = WIDTH-40, h = 0)
    pdf.ln(5)
    pdf.set_x(20)
    # In 2023-02, among all the buildings, blk 125 has the highest month to month EUI decrease of -47%, while blk 218 has the highest month to month EUI increase of 372%
    pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, among all the units, {change_unit_low} has the highest month to month WEI decrease of {change_low*100:,.0f}%, while {change_unit_high} has the highest month to month WEI increase of{change_high*100:,.0f}%.',align = 'L')
    
    
    # 4. Anomaly Detection
    anomalies = anomaly_detection(root_path = root_path, input_path = elec_input_folder, output_img_folder= elec_anomaly_output_folder, 
                                  start_date = start_date, end_date = end_date ,lookback_period = lookback_period, top_k = 5, 
                                  camp_fullname = camp_fullname, unit_of_measurement = 'Wh', miss_val_threshold = miss_val_threshold)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, f'4. Anomaly Detection: Top 5 Daily Anomalies in {select_month}')
    pdf.ln(10)
    m = 0
    for key in anomalies.keys():
        blk = anomalies[key]['blk']
        date = anomalies[key]['date']
        deviation = anomalies[key]['deviation']
        uom = anomalies[key]['unit_of_measurement']
        deviation_percentage = anomalies[key]['deviation_percentage']
        m += 1
        pdf.set_font('Arial', '', 12)
        pdf.cell(10, 10, f'4.{m}. Top {key} highest daily anomaly - Block {extract_numbers(blk)} on {date}')
        pdf.ln(10)
        pdf.image(root_path + elec_anomaly_output_folder + 'all_' + truncate_string(blk) + '_' + str(date) + '_day' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_day.png
        pdf.ln(5)
        pdf.image(root_path + elec_anomaly_output_folder +  'all_' + truncate_string(blk) + '_' + str(date) + '_period' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_period.png
        pdf.ln(5)
        pdf.set_x(20)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, the top {key} highest daily anomaly is observed in {truncate_string(blk)} on {date} with total anomalous consumption of {deviation} {uom} {deviation_percentage}% deviation from expected consumption.', align = 'L')
        pdf.ln(10)
    
    pdf.output(root_path + report_output_folder + '/camp commander report.pdf', 'F')
    print('Camp Commander Report Done')
    
    
    
    ##################################################
    # Step 4. Generate pdf report for Building Owner #
    ##################################################

    descrip_elec = each_building(root_path, 'electricity', elec_output_img_folder, select_month, total_elec)       
    descrip_water = each_building(root_path, 'Water', water_output_img_folder, select_month, total_water)        

    for key, value in func_dic_water_usage.items():

        for blk in value:
            
            # Instantiation of inherited class
            pdf = PDF()
            pdf.set_top_margin(20)
            pdf.alias_nb_pages()
            pdf.add_page()
            
            # Cover page
            pdf.set_x(WIDTH/2)
            pdf.set_y(HEIGHT/2)
            pdf.set_font("Arial", "B", 16)
            pdf.cell((WIDTH-20), 10, camp_fullname + ' Block ' + extract_numbers(blk) + ' Monthly Report - Building Owner', 0, 0, 'C')
            pdf.ln(10)
            pdf.cell((WIDTH-20), 10, select_month , 0, 0, 'C')
            pdf.add_page()

            # Table of Contents
            pdf.set_font("Arial", "B", 16)
            pdf.set_x(10)
            pdf.cell(0, 10, "Table of Contents", 0, 1, "C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            toc_anomaly = [("1. Electricity Consumption", 3), ("2. Water Consumption", 5), ("3. Anomaly Detection", 7)]
            for title, page in toc_anomaly:
                pdf.set_x(50)
                pdf.cell(0, 10, title + '......................................', 0, 0)
                pdf.set_x(90)
                pdf.cell(0, 10, 'page ' + str(page), 0, 1, "C")
                pdf.ln(5)
            pdf.add_page()

            # Get electricity variables
            if blk[:-5]+'elect' in descrip_elec['block'].tolist():
                idx = descrip_elec[descrip_elec['block'] == blk[:-5] + 'elect'].index[0]
                block = truncate_string(blk)
                EUI = descrip_elec.iloc[idx]['EUI']
                EUI_peer = descrip_elec.iloc[idx]['EUI_peer']
                peer = EUI/EUI_peer-1
                med_peer = descrip_elec.iloc[idx]['median']
                med = EUI/med_peer-1
                EUI_last = descrip_elec.iloc[idx]['EUI_last']
                monthly_change = EUI/EUI_last-1
                change = ['increase' if monthly_change >= 0 else 'decrease'][0]
                wk = descrip_elec.iloc[idx]['wk']
                wk_val = descrip_elec.iloc[idx]['wk_val']
                hr = int(descrip_elec.iloc[idx]['hr'])
                hr_val = descrip_elec.iloc[idx]['hr_val']

                
                # 1. Electricity Consumption - peer comparison
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'1. Electricity Consumption')
                pdf.ln(10)
                pdf.image(root_path + elec_output_img_folder + truncate_string(blk) + '_peer_EUI.png', x = 25, w=150) 
                pdf.ln(3)
                pdf.image(root_path + elec_output_img_folder + key + '_buildings_EUI_changes.png', x = 25 ,w=150)
                pdf.ln(5)

                pdf.set_x(30)
                pdf.set_font('Arial', '', 10)
                pdf.set_fill_color(200, 200, 200) # set fill color to red
                pdf.cell(150, 5, f'Block {extract_numbers(blk)} Peer Comparison in {select_month}', border = 1, align = 'C',fill=True, ln = 1)
                pdf.set_fill_color(255, 255, 255) # set fill color back to white
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI for {truncate_string(blk)}', border = 1, align = 'C')
                pdf.cell(75, 5, f'{EUI:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI for Best Performing Peer ', border = 1, align = 'C')
                pdf.cell(75, 5, f'{EUI_peer:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI % Against Best Performing Peer', border = 1, align = 'C')
                pdf.cell(75, 5, f'{peer*100:,.0f}%', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI % Against Median Peer', border = 1, align = 'C')
                pdf.cell(75, 5, f'{med*100:,.0f}%', border = 1, align = 'C', ln = 1)
                pdf.add_page()

                # 1. Electricity Consumption - self comparison
                pdf.image(root_path + elec_output_img_folder + truncate_string(blk) + '_monthly.png', x = 40 ,w = 130, h = 0) #blk-213_monthly.png
                pdf.ln(3)
                pdf.image(root_path + elec_output_img_folder + truncate_string(blk) + '_weekly.png', x = 40, w = 130, h = 0) #blk-213_weekly.png
                pdf.ln(3)
                pdf.image(root_path + elec_output_img_folder + truncate_string(blk) + '_hourly.png', x = 40 ,w = 130, h = 0) #blk-213_hourly.png
                pdf.ln(5)

                pdf.set_x(30)
                pdf.set_font('Arial', '', 10)
                pdf.set_fill_color(200, 200, 200)
                pdf.cell(150, 5, f'Block {extract_numbers(blk)} Self Comparison in {select_month}', border = 1, align = 'C', fill=True, ln = 1)
                pdf.set_fill_color(255, 255, 255)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI This Month ', border = 1, align = 'C')
                pdf.cell(75, 5, f'{EUI:,.0f} Wh/m2', border = 1, align = 'C',  ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI Last Month ', border = 1, align = 'C')
                pdf.cell(75, 5, f'{EUI_last:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'EUI Month to Month Comparison', border = 1, align = 'C')
                pdf.cell(75, 5, f'{change} {monthly_change*100:,.0f}%', border = 1, align = 'C',ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'Highest EUI Week', border = 1, align = 'C')
                pdf.cell(75, 5, f'{wk_val:,.0f} Wh/m2, Week Starting From {wk}', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'Highest EUI Hour', border = 1, align = 'C')
                pdf.cell(75, 5, f'{hr_val:,.0f} Wh/m2 in {hr}:00-{hr+1}:00 hr', border = 1, align = 'C')
                pdf.add_page()

            if blk[:-5]+'elect' not in descrip_elec['block'].tolist():
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'1. Electricity Consumption')
                pdf.ln(10)
                pdf.set_x(20)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'Electricity data N/A.', align = 'L')
                pdf.ln(10)
                pdf.add_page()
            
            # 2. Water Consumption - peer comparison
            if blk in descrip_water['block'].tolist():
                # Get water variables
                idx = descrip_water[descrip_water['block'] == blk[:-5] + 'water'].index[0]
                block = truncate_string(blk)
                WEI = descrip_water.iloc[idx]['EUI']
                WEI_peer = descrip_water.iloc[idx]['EUI_peer']
                peer = WEI/WEI_peer-1
                med_peer = descrip_water.iloc[idx]['median']
                med = WEI/med_peer-1
                WEI_last = descrip_water.iloc[idx]['EUI_last']
                monthly_change = WEI/WEI_last-1
                change = ['increase' if monthly_change >= 0 else 'decrease'][0]
                wk = descrip_water.iloc[idx]['wk']
                wk_val = descrip_water.iloc[idx]['wk_val']
                hr = int(descrip_water.iloc[idx]['hr'])
                hr_val = descrip_water.iloc[idx]['hr_val']
                
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'2. Water Consumption')
                pdf.ln(10)

                pdf.image(root_path + water_output_img_folder + truncate_string(blk) + '_peer_WEI.png' , x = 25, w = 150, h = 0) # blk-213_peer_WEI.png
                pdf.ln(3)
                pdf.image(root_path + water_output_img_folder + key + '_buildings_WEI_changes.png' , x = 25, w = 150, h = 0) # Training_buildings_EUI_changes.png
                pdf.ln(5)

                pdf.set_x(30)
                pdf.set_font('Arial', '', 10)
                pdf.set_fill_color(200, 200, 200) # set fill color to red
                pdf.cell(150, 5, f'Block {extract_numbers(blk)} Peer Comparison in {select_month}', border = 1, align = 'C',fill=True, ln = 1)
                pdf.set_fill_color(255, 255, 255) # set fill color back to white
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI for {truncate_string(blk)}', border = 1, align = 'C')
                pdf.cell(75, 5, f'{WEI:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI for Best Performing Peer', border = 1, align = 'C')
                pdf.cell(75, 5, f'{WEI_peer:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI % Against Best Performing Peer', border = 1, align = 'C')
                pdf.cell(75, 5, f'{peer*100:,.0f}%', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI % Against Median Peer', border = 1, align = 'C')
                pdf.cell(75, 5, f'{med*100:,.0f}%', border = 1, align = 'C', ln = 1)
                pdf.add_page()
                
                # 2. Water Consumption - self comparison
                pdf.image(root_path + water_output_img_folder + truncate_string(blk) + '_monthly.png' , x = 40,  w = 130, h = 0) # blk-213_monthly.png
                pdf.ln(3)
                pdf.image(root_path + water_output_img_folder + truncate_string(blk) + '_weekly.png' , x = 40, w = 130, h = 0) # blk-213_weekly.png
                pdf.ln(3)
                pdf.image(root_path + water_output_img_folder + truncate_string(blk) + '_hourly.png' , x = 40, w = 130, h = 0) # blk-213_hourly.png
                pdf.ln(5)

                pdf.set_x(30)
                pdf.set_font('Arial', '', 10)
                pdf.set_fill_color(200, 200, 200) # set fill color to red
                pdf.cell(150, 5, f'Block {extract_numbers(blk)} Self Comparison in {select_month}', border = 1, align = 'C',fill=True, ln = 1)
                pdf.set_fill_color(255, 255, 255) # set fill color back to white
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI This Month', border = 1, align = 'C')
                pdf.cell(75, 5, f'{WEI:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI Last Month', border = 1, align = 'C')
                pdf.cell(75, 5, f'{WEI_last:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'WEI Month to Month Comparison', border = 1, align = 'C')
                pdf.cell(75, 5, f'{change} {monthly_change*100:,.0f}%', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'Highest WEI Week', border = 1, align = 'C')
                pdf.cell(75, 5, f'{wk_val:,.3f} m3/m2, Week Starting From {wk}', border = 1, align = 'C', ln = 1)
                pdf.set_x(30)
                pdf.cell(75, 5, f'Highest WEI Hour', border = 1, align = 'C')
                pdf.cell(75, 5, f'{hr_val:,.4f} m3/m2 in {hr}:00-{hr+1}:00 hr', border = 1, align = 'C')

            if blk not in descrip_water['block'].tolist():
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'1. Water Consumption')
                pdf.ln(10)
                pdf.set_x(20)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'Water data N/A.', align = 'L')
                pdf.ln(10)
            
            # 3. Anomaly Detection
            # result from anomaly_detection function : /{1: {'blk': 'dsta_blk-301-elect', 'date': datetime.date(2023, 2, 25), 'deviation': 147530.0, 'unit_of_measurement': 'Wh', 'deviation_percentage': 37.22}, 2: {'blk': 'dsta_blk-301-elect', 'date': datetime.date(2023, 2, 13), 'deviation': 99170.71, 'unit_of_measurement': 'Wh', 'deviation_percentage': 16.6}}
            anomalies = anomaly_detection(root_path = root_path, input_path = elec_input_folder, output_img_folder= elec_anomaly_output_folder, 
                                          start_date = start_date, end_date = end_date ,lookback_period = lookback_period, top_k = 3,
                                          camp_fullname = camp_fullname, unit_of_measurement = 'Wh', miss_val_threshold = miss_val_threshold, blk = truncate_string(blk))
            if bool(anomalies):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'3. Anomaly Detection')
                pdf.ln(10)
                m = 0
                for count in anomalies.keys():
                    if blk[:-5]+'elect' == anomalies[count]['blk']:
                        blk_anomaly = anomalies[count]['blk']
                        date = anomalies[count]['date']
                        deviation = anomalies[count]['deviation']
                        uom = anomalies[count]['unit_of_measurement']
                        deviation_percentage = anomalies[count]['deviation_percentage']
                        m += 1
                        pdf.set_font('Arial', '', 12)
                        pdf.cell(10, 10, f'3.{m}. Top {count} Highest Daily Anomaly in Block {extract_numbers(blk)} - {date}')
                        pdf.ln(10)
                        pdf.image(root_path + elec_anomaly_output_folder + '_' + truncate_string(blk_anomaly) + '_' + str(date) + '_day' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_day.png
                        pdf.ln(5)
                        pdf.image(root_path + elec_anomaly_output_folder + '_' + truncate_string(blk_anomaly) + '_' + str(date) + '_period' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_period.png
                        pdf.ln(5)
                        pdf.set_x(20)
                        pdf.set_font('Arial', '', 10)
                        pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, the top {count} highest daily anomaly in {truncate_string(blk_anomaly)} is observed  on {date} with total anomalous consumption of {deviation} {uom} {deviation_percentage}% deviation from expected consumption.', align = 'L')
                        pdf.ln(10)
            else:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(10, 10, f'3. Anomaly Detection')
                pdf.ln(10)
                pdf.set_x(20)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'No anomalies detected or too much missing data.', align = 'L')
                pdf.ln(10)
                        
            pdf.output(root_path + report_output_folder + '/' + truncate_string(blk) + ' report.pdf', 'F')
            print(truncate_string(blk) + ' Report Done')
        
    
    ##################################################
    # Step 5. Generate pdf report for Unit Commander #
    ##################################################
    
    descrip_elec = each_unit(root_path, 'electricity', elec_output_img_folder, select_month, total_elec, func_dic_elec_unit)      
    descrip_water = each_unit(root_path, 'Water', water_output_img_folder, select_month, total_water,func_dic_water_unit) 

    type = 'electricity'
    val_high, unit_high, val_low_elec, unit_low_elec, change_high, change_unit_high, change_low, change_unit_low, med_peer_elec = monthly_consump_unit(func_dic_elec_unit, root_path, total_elec, elec_output_img_folder,
                                                                                                                                                       type, camp_fullname,select_month)
    type = 'water'
    val_high, unit_high, val_low_water, unit_low_water, change_high, change_unit_high, change_low, change_unit_low, med_peer_water = monthly_consump_unit(func_dic_water_unit, root_path, total_water, water_output_img_folder,
                                                                                                                                                          type, camp_fullname,select_month)
    
    for key, value in func_dic_elec_unit.items():
        
        # Instantiation of inherited class
        pdf = PDF()
        pdf.set_top_margin(20)
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Cover
        pdf.set_x(WIDTH/2)
        pdf.set_y(HEIGHT/2)
        pdf.set_font("Arial", "B", 16)
        pdf.cell((WIDTH-20), 10, camp_fullname + ' Unit ' + key + ' Monthly Report - Unit Commander', 0, 0, 'C')
        pdf.ln(10)
        pdf.cell((WIDTH-20), 10, select_month , 0, 0, 'C')
        pdf.add_page()

        # Table of Contents
        pdf.set_font("Arial", "B", 16)
        pdf.set_x(10)
        pdf.cell(0, 10, "Table of Contents", 0, 1, "C")
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        toc_anomaly = [("1. Electricity Consumption", 3), ("2. Water Consumption", 5), ("3. Anomaly Detection", 7)]
        for title, page in toc_anomaly:
            pdf.set_x(50)
            pdf.cell(0, 10, title + '......................................', 0, 0)
            pdf.set_x(90)
            pdf.cell(0, 10, 'page ' + str(page), 0, 1, "C")
            pdf.ln(5)
        pdf.add_page()

        # Get electricity variable
        if key in descrip_elec['block'].tolist():
            idx = descrip_elec[descrip_elec['block'] == key].index[0]
            block = key
            val = descrip_elec.iloc[idx]['val']
            val_last = descrip_elec.iloc[idx]['val_last']
            peer = val/val_low_elec-1
            med_peer = med_peer_elec
            med = val/med_peer-1
            monthly_change = val/val_last-1
            change = ['increase' if monthly_change >= 0 else 'decrease'][0]
            wk = descrip_elec.iloc[idx]['wk']
            wk_val = descrip_elec.iloc[idx]['val_wk']
            hr = int(descrip_elec.iloc[idx]['hr'])
            hr_val = descrip_elec.iloc[idx]['val_hr']

            # 1. Electricity Consumption - peer comparison
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'1. Electricity Consumption')
            pdf.ln(10)

            pdf.image(root_path + elec_output_img_folder + key +'_EUI.png' , x = 25, w = 150, h = 0) # Training_buildings_EUI.png
            pdf.ln(3)
            pdf.image(root_path + elec_output_img_folder + 'units_EUI_changes.png', x = 25, w = 150, h = 0) # Training_buildings_EUI_changes.png
            pdf.ln(5)

            pdf.set_x(30)
            pdf.set_font('Arial', '', 10)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(150, 5, f'Unit {key} Peer Comparison in {select_month}', border = 1, align = 'C', fill=True, ln = 1)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI for {key}', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI for Best Performing Peer', border = 1, align = 'C') # val_low_elec, unit_low_elec,
            pdf.cell(75, 5, f'{val_low_elec:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI % Against Best Performing Peer', border = 1, align = 'C')
            pdf.cell(75, 5, f'{peer*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI % Against Median Peer', border = 1, align = 'C')
            pdf.cell(75, 5, f'{med*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.add_page()

            # 1. Electricity Consumption - self comparison
            pdf.image(root_path + elec_output_img_folder + key + '_monthly.png', x = 40 , w = 120, h = 0) # blk-213_monthly.png
            pdf.ln(3)
            pdf.image(root_path + elec_output_img_folder + key + '_weekly.png', x = 40 , w = 120, h = 0) # blk-213_weekly.png
            pdf.ln(3)
            pdf.image(root_path + elec_output_img_folder + key + '_hourly.png', x = 40 , w = 120, h = 0) # blk-213_hourly.png
            pdf.ln(5)

            pdf.set_x(30)
            pdf.set_font('Arial', '', 10)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(150, 5, f'Unit {key} Self Comparison in {select_month}', border = 1, align = 'C', fill=True, ln = 1)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI This Month', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI Last Month', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val_last:,.0f} Wh/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'EUI Month to Month Comparison', border = 1, align = 'C')
            pdf.cell(75, 5, f'{change} {monthly_change*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'Highest EUI Week', border = 1, align = 'C')
            pdf.cell(75, 5, f'{wk_val:,.0f} Wh/m2, Week Starting From {wk}', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'Highest EUI Hour', border = 1, align = 'C')
            pdf.cell(75, 5, f'{hr_val:,.0f} Wh/m2 in {hr}:00-{hr+1}:00 hr', border = 1, align = 'C')
            pdf.add_page()
        if key not in descrip_elec['block'].tolist():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'1. Electricity Consumption')
            pdf.ln(10)
            pdf.set_x(20)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'Electricity data N/A.', align = 'L')
            pdf.ln(10)
            pdf.add_page()

        # Get water variable
        if key in descrip_water['block'].tolist():
            idx = descrip_water[descrip_water['block'] == key].index[0]
            block = key
            val = descrip_water.iloc[idx]['val']
            val_last = descrip_water.iloc[idx]['val_last']
            peer = val/val_low_water-1
            med_peer = med_peer_water
            med = val/med_peer-1
            monthly_change = val/val_last-1
            change = ['increase' if monthly_change >= 0 else 'decrease'][0]
            wk = descrip_water.iloc[idx]['wk']
            wk_val = descrip_water.iloc[idx]['val_wk']
            hr = int(descrip_water.iloc[idx]['hr'])
            hr_val = descrip_water.iloc[idx]['val_hr']

            # 2. Water Consumption - peer comparison
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'2. Water Consumption')
            pdf.ln(10)
                
            pdf.image(root_path + water_output_img_folder + key + '_WEI.png' , x = 25, w = 150, h = 0) # Training_buildings_EUI.png
            pdf.ln(3)
            pdf.image(root_path + water_output_img_folder + 'units_WEI_changes.png', x = 25 , w = 150, h = 0) # Training_buildings_EUI_changes.png
            pdf.ln(5)

            pdf.set_x(30)
            pdf.set_font('Arial', '', 10)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(150, 5, f'Unit {key} Peer Comparison in {select_month}', border = 1, align = 'C', fill=True, ln = 1)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI for {key}', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI for Best Performing Peer', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val_low_water:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI % Against Best Performing Peer', border = 1, align = 'C')
            pdf.cell(75, 5, f'{peer*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI % Against Median Peer', border = 1, align = 'C')
            pdf.cell(75, 5, f'{med*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.add_page()

            # 2. Water Consumption - self comparison
            pdf.image(root_path + water_output_img_folder + key + '_monthly.png', x = 40 , w = 120, h = 0) # blk-213_monthly.png
            pdf.ln(3)
            pdf.image(root_path + water_output_img_folder + key + '_weekly.png', x = 40 , w = 120, h = 0) # blk-213_weekly.png
            pdf.ln(3)
            pdf.image(root_path + water_output_img_folder + key + '_hourly.png', x = 40 , w = 120, h = 0) # blk-213_hourly.png
            pdf.ln(5)

            pdf.set_x(30)
            pdf.set_font('Arial', '', 10)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(150, 5, f'Unit {key} Peer Comparison in {select_month}', border = 1, align = 'C', fill=True, ln = 1)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI This Month ', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI Last Month: ', border = 1, align = 'C')
            pdf.cell(75, 5, f'{val_last:,.3f} m3/m2', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'WEI Month to Month Comparison', border = 1, align = 'C')
            pdf.cell(75, 5, f'{change} {monthly_change*100:,.0f}%', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'Highest WEI Week', border = 1, align = 'C')
            pdf.cell(75, 5, f'{wk_val:,.3f} m3/m2, Week Starting From {wk}', border = 1, align = 'C', ln = 1)
            pdf.set_x(30)
            pdf.cell(75, 5, f'Highest WEI Hour', border = 1, align = 'C')
            pdf.cell(75, 5, f'{hr_val:,.4f} m3/m2 in {hr}:00-{hr+1}:00 hr', border = 1, align = 'C')
        
        if key not in descrip_water['block'].tolist():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'2. Water Consumption')
            pdf.ln(10)
            pdf.set_x(20)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'Water data N/A.', align = 'L')
            pdf.ln(10)

        # 3. Anomaly Detection
        anomaly_lst = []
        for blk in value:
            anomalies_blk = anomaly_detection(root_path = root_path, input_path = elec_input_folder, output_img_folder= elec_anomaly_output_folder, 
                                              start_date = start_date , end_date = end_date ,lookback_period = lookback_period, top_k = 5,
                                              camp_fullname = camp_fullname, unit_of_measurement = 'Wh', miss_val_threshold = miss_val_threshold, blk = truncate_string(blk))
            for m, n in anomalies_blk.items():
                anomaly_lst.append(n)

        anomaly_lst.sort(key=takeDeviation, reverse=True)
        anomaly_lst[0:3]
        anomalies = {}
        top = 0
        for each in anomaly_lst:
            top += 1
            anomalies[top]= each

        if bool(anomalies):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'3. Anomaly Detection')
            pdf.ln(10)
            m = 0
            for count in anomalies.keys():
                blk_anomaly = anomalies[count]['blk']
                date = anomalies[count]['date']
                deviation = anomalies[count]['deviation']
                uom = anomalies[count]['unit_of_measurement']
                deviation_percentage = anomalies[count]['deviation_percentage']
                m += 1
                pdf.set_font('Arial', '', 12)
                pdf.cell(10, 10, f'3.{m}. Top {count} Highest Daily Anomaly - Block {extract_numbers(blk_anomaly)} on {date}')
                pdf.ln(10)
                pdf.image(root_path + elec_anomaly_output_folder + '_' + truncate_string(blk_anomaly) + '_' + str(date) + '_day' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_day.png
                pdf.ln(5)
                pdf.image(root_path + elec_anomaly_output_folder + '_' + truncate_string(blk_anomaly) + '_' + str(date) + '_period' +'.png', w = WIDTH-30, h = 0) # blk-218_2023-02-01_period.png
                pdf.ln(5)
                pdf.set_x(20)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'In {select_month}, the top {count} highest daily anomaly in {truncate_string(blk_anomaly)} is observed  on {date} with total anomalous consumption of {deviation} {uom} {deviation_percentage}% deviation from expected consumption.', align = 'L')
                pdf.ln(10)
        else:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(10, 10, f'3. Anomaly Detection')
            pdf.ln(10)
            pdf.set_x(20)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(w = WIDTH -40, h = 5, txt = f'No anomalies detected or too much missing data.', align = 'L')
            pdf.ln(10)

        pdf.output(root_path + report_output_folder + '/' + 'unit ' + key + ' report.pdf', 'F')
        print(key + ' Unit Report Done')
    
