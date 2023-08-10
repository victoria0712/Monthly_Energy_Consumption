#Monthly Energy Consumption Report


##0. Confidentiality

Welcome to the project documentation. In accordance with the commitment to confidentiality, certain sensitive information, including the company name, camp name, building name, will not be disclosed within this documentation. The data presented here has been randomly generated for illustrative purposes, ensuring the protection of proprietary details. The documentation showcases the project's capabilities while upholding the principles of confidentiality and security.

##1. Raw Data

    Query electricity and water data for JC and PLAB.
    
    Folder: ./rawdata/

    Modification: 
        In 2nd and 3rd section, change start date (line 5th, 6th, 7th) to first date of the month that needs data
        In 2nd and 3rd section, change end date (line 21st) to last date of the month that needs data

    * Need to ensure result folder to place the consumption file is created
        
    Example:

        The parameters required to query 2023 Jan consumption data in PLAB
            year = 2023
            month = 1
            day = 1
            end = start + 31*24*60*60 (There are 31 days in Jan)
    
        Each raw data shoul be put under the folder: ./rawdata/PLAB_elect/2023-01. (need to ensure the path is created)

    1) Data Collection_elec-JC.ipynb
       This is the electricity data collection code of JC camp

    2) Data Collection_DSTA_water_JC.ipynb
       This is the water data collection code of JC camp
    
    3) Data Collection_elec-PLAB.ipynb
       This is the electricity data collection code of PLAB camp

    4) Data Collection_DSTA_water_PLAB.ipynb
       This is the water data collection code of PLAB camp

##2. Main Code:

    1) funtion.py
       This script contains all the functions needed for the main code
    
    2) monthly_report_JC.py
        This is the main code of JC camp

    3) monthly_report_plab.py
        This is the main code of PLAB camp

##3. Main Code Logic

    1) Define root path at line 8th, select_month at line 10th, start_date at line 12th, and end_date at line 11th
       Define time list and function list (If some meters dun have data, this function list needs change)

    2) Step 1. Data processing for buildings with separate files and import data as dataframe
        Some buildings have separate meters
    
    3) Step 2. Create pdf class and report information

    4) Step 3. Generate pdf report for Camp Commander

    5) Step 4. Generate pdf report for Building Owner

    6) Step 5. Generate pdf report for Unit Commander








