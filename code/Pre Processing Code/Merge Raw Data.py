import pandas as pd
import os
import yaml
import helper_functions as hf
import create_combo_vars as ccv

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']


# Read in sample data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"), index_col=0)

# Define filenames for interim data
file_names = {
    "neighbors": 'Neighbor_Counts.csv',
    "sample_msa": 'MSA Data.xlsx',
    "final_df": 'ACS Data.xlsx',
    "sample_perm": 'BPS Data.xlsx',
    "urban_data": 'urban_raw.xlsx',
    "elasticity": 'Housing Elasticity.csv',
    "voting": 'Voting Data.xlsx',
    "census": 'Census Data.xlsx',
    "chetty": 'Chetty_Data.xlsx',
    "stanford_educ":'Stanford Ed Data.xlsx',
    "gov_finance": 'Gov_Finance_Data.xlsx',
    'shape_vars':'Shape_data.xlsx',
    'year_inc':'Year_Incorporated.xlsx',
    'social_capital':'Social_Capital_Data.xlsx',
    'school_finance':'School_Finance_Data.xlsx',
}

# Read in data from interim_data
data_frames = {}
for key, filename in file_names.items():
    file_path = os.path.join(data_path, 'interim_data', filename)
    if filename.endswith('.xlsx'):
        data_frames[key] = pd.read_excel(file_path, index_col=0)
    elif filename.endswith('.csv'):
        data_frames[key] = pd.read_csv(file_path, index_col = 0)

# Merge the DataFrames using 'CENSUS_ID_PID6' as the key
merge_key = 'CENSUS_ID_PID6'
merged_df = sample.copy()
for key in data_frames.keys():
    # If merge_key not in data_frames[key], then reset the index and make the index be called the merge_key
    if merge_key not in data_frames[key].columns:
        data_frames[key] = data_frames[key].reset_index().rename(columns={'index': merge_key})
    merged_df = pd.merge(merged_df, data_frames[key], on=merge_key, how='left')



#Assert number of rows in merged_df is the same as in sample
if len(merged_df) != len(sample):
    raise ValueError('Number of rows in merged_df is not the same as in sample')

#Create population and housing unit densities and drop population and land area
merged_df['Population_Density'] = merged_df['Population_ACS_2022'] / merged_df['Land Area Acres']
merged_df['Housing_Unit_Density'] = merged_df['Housing Units Census 2020'] / merged_df['Land Area Acres']

#Call function to process government finance vars
merged_df = ccv.process_gov_fin_vars(merged_df)

# Create affordability measures
api_key = config['census_key']
merged_df = ccv.create_affordability_measures(merged_df, api_key)

#Make region
def fips_to_region(fips_state):
    northeast = {9, 23, 25, 33, 34, 36, 42, 44, 50}
    midwest = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
    south = {1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54}
    west = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}

    if fips_state in northeast:
        return "Northeast"
    elif fips_state in midwest:
        return "Midwest"
    elif fips_state in south:
        return "South"
    elif fips_state in west:
        return "West"
    else:
        return "Unknown"

merged_df['Region'] = merged_df['FIPS_STATE'].apply(fips_to_region)

# Export so we have a dataframe with all the data
merged_df.to_excel(os.path.join(data_path, "Sample_Enriched.xlsx"))


