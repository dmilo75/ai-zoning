import pandas as pd
import os
import yaml
import helper_functions as hf
import requests
import numpy as np

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

# Read in sample data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"), index_col=0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])

#Draw in muni data
muni_data = pd.read_csv(os.path.join(raw_path,'Government Finances','Municipalities', "MunicipalData.csv"))
township_data = pd.read_csv(os.path.join(raw_path,'Government Finances','Townships', "TownshipData.csv"))

def clean_data(df):

    #Filter for Year4 is 2017 since that is the latest full survey
    df = df[df['Year4'] == 2017]

    # Add up the IGR variables to get one unified IGR variable
    df['Total_IGR_Hous_Com_Dev'] = df['Fed_IGR_Hous_Com_Dev'] + df['State_IGR_Hous_Com_Dev'] + df[
        'Local_IGR_Hous_Com_Dev']


    #Select the variables we want
    variables = [
        "Total_Revenue",
        "Total_Expenditure",
        "Property_Tax",
        "Total_IGR_Hous_Com_Dev",
        "Cen_Staff_Total_Expend",
        "Population",
    ]

    #Merge vars
    merge_vars = ['FIPS_Code_State','FIPS_County','FIPS_Place']

    #Select the variables we want
    df = df[merge_vars + variables]

    #Rename all variables to add _2017 at the end
    df = df.rename(columns = {col:col + '_2017' for col in variables})

    #Rename merge vars to FIPS_STATE, FIPS_COUNTY, FIPS_PLACE
    df = df.rename(columns = {'FIPS_Code_State':'FIPS_STATE','FIPS_County':'FIPS_COUNTY','FIPS_Place':'FIPS_PLACE'})

    return df

muni_cleaned = clean_data(muni_data)
township_cleaned = clean_data(township_data)

#Just keep merge vars and CENSUS_ID_PID6 from sample
sample = sample[['FIPS_STATE','FIPS_COUNTY','FIPS_PLACE','CENSUS_ID_PID6','UNIT_TYPE']]

#Merge in with sample data
sample_muni = sample[sample['UNIT_TYPE'] == '2 - MUNICIPAL']
sample_muni_merged = pd.merge(sample_muni,muni_cleaned, on = ['FIPS_STATE','FIPS_COUNTY','FIPS_PLACE'], how = 'inner')
#Calc and print percent merged
percent_merged = len(sample_muni_merged)/len(sample_muni)
print(f"Percent of Muni data merged: {percent_merged}")

#Merge in with sample data
sample_township = sample[sample['UNIT_TYPE'] == '3 - TOWNSHIP']
sample_township_merged = pd.merge(sample_township,township_cleaned, on = ['FIPS_STATE','FIPS_COUNTY','FIPS_PLACE'], how = 'inner')
#Calc and print percent merged
percent_merged = len(sample_township_merged)/len(sample_township)
print(f"Percent of Township data merged: {percent_merged}")

#Concat the two dataframes
sample_merged = pd.concat([sample_muni_merged,sample_township_merged])

#Drop UNIT_TYPE, FIPS_STATE, FIPS_COUNTY, FIPS_PLACE
sample_merged = sample_merged.drop(columns = ['UNIT_TYPE','FIPS_STATE','FIPS_COUNTY','FIPS_PLACE'])

#Export
sample_merged.to_excel(os.path.join(data_path, 'interim_data', 'Gov_Finance_Data.xlsx'))

