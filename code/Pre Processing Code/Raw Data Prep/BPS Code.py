import pandas as pd
import os
import yaml
import helper_functions as hf

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#Read in sample data
sample = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"),index_col  = 0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])

#%%Merge in building permits data

bps_data = pd.read_excel(os.path.join(raw_path,"bps_raw.xlsx"))

sample_perm = sample.copy()

#Drop townships in sample_perm since BPS data only available for places
sample_perm = sample_perm[sample_perm['UNIT_TYPE'] != '3 - TOWNSHIP']

sample_perm = pd.merge(sample_perm, bps_data,
                       left_on=['FIPS_PLACE', 'FIPS_STATE'],
                       right_on=['FIPS Place', 'FIPS State'],
                       how='left')

# Count the number of non-NaN entries in 'FIPS Place' after the merge
merged_count = sample_perm['Number of Months Reported'].notna().sum()

# Calculate the percentage merged
percentage_merged = (merged_count / len(sample_perm)) * 100

print(f"Percentage merged: {percentage_merged:.2f}%")

#%%Keep only relevant variables now

variables = [
    'Number of Months Reported',
    '1-unit Buildings (Estimates with Imputation)', '1-unit Units (Estimates with Imputation)', '1-unit Valuation (Estimates with Imputation)',
    '2-units Buildings (Estimates with Imputation)', '2-units Units (Estimates with Imputation)', '2-units Valuation (Estimates with Imputation)',
    '3-4 units Buildings (Estimates with Imputation)', '3-4 units Units (Estimates with Imputation)', '3-4 units Valuation (Estimates with Imputation)',
    '5+ units Buildings (Estimates with Imputation)', '5+ units Units (Estimates with Imputation)', '5+ units Valuation (Estimates with Imputation)',
    '1-unit Buildings (Reported Only)', '1-unit Units (Reported Only)', '1-unit Valuation (Reported Only)', '2-units Buildings (Reported Only)',
    '2-units Units (Reported Only)', '2-units Valuation (Reported Only)', '3-4 units Buildings (Reported Only)', '3-4 units Units (Reported Only)',
    '3-4 units Valuation (Reported Only)', '5+ units Buildings (Reported Only)', '5+ units Units (Reported Only)', '5+ units Valuation (Reported Only)'
]

# Create 'Single Unit Permits'
sample_perm['Single Unit Permits'] = sample_perm['1-unit Units (Estimates with Imputation)']

# Create 'Multi-Unit Permits'
sample_perm['Multi-Unit Permits'] = (sample_perm['2-units Units (Estimates with Imputation)'] +
                                     sample_perm['3-4 units Units (Estimates with Imputation)'] +
                                     sample_perm['5+ units Units (Estimates with Imputation)'])

# Create 'All Unit Permits'
sample_perm['All Unit Permits'] = sample_perm['Single Unit Permits'] + sample_perm['Multi-Unit Permits']

# Drop the original variables from the list `variables`
sample_perm = sample_perm.drop(columns=variables)

#Make single, multi, and all unit permits in per capita terms but dont rename
sample_perm['Single Unit Permits'] = sample_perm['Single Unit Permits'] / sample_perm['POPULATION']
sample_perm['Multi-Unit Permits'] = sample_perm['Multi-Unit Permits'] / sample_perm['POPULATION']
sample_perm['All Unit Permits'] = sample_perm['All Unit Permits'] / sample_perm['POPULATION']

#Only keep new variables and CENSUS_ID_PID6
sample_perm = sample_perm[['Single Unit Permits','Multi-Unit Permits','All Unit Permits','CENSUS_ID_PID6']]


#Save to 'interim_data' under processed data folder
sample_perm.to_excel(os.path.join(data_path,'interim_data','BPS Data.xlsx'))



