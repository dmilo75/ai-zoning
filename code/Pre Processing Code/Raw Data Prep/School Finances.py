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


#Load in sample data
sample_data = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"))


##

#Draw in school finance data
df = pd.read_excel(os.path.join(raw_path,'Survey of School Finances','elsec22.xls'))


vars_to_use = [
    "TOTALREV",    # Total Elementary-Secondary Revenue
    "TLOCREV",     # Total Revenue from Local Sources
    "T06",         # Property taxes
    "TOTALEXP",    # Total Elementary-Secondary Expenditure
    "Z32",         # Total salaries and wages
    "Z34",         # Total employee benefit payments
    "V33",         # Total enrollment
]


#Drop rows with enrollment less than 500
df = df[df['V33'] > 500]

#Filter out UNIT_TYPE of 0 or 1, which are state and county dependent schools
df = df[df['UNIT_TYPE'].isin([2,3,5])]


#Filter for SCHLEV of 03
df = df[df['SCHLEV'] == 3]



##

import pandas as pd
import numpy as np

# Get geo bridges
place_bridge = pd.read_csv(os.path.join(raw_path, 'GEO Crosswalks', 'geocorr2022_district_place.csv'), encoding='latin1')

# Delete first row
place_bridge = place_bridge.iloc[1:]

# Ensure numeric columns are properly typed
place_bridge['afact'] = pd.to_numeric(place_bridge['afact'], errors='coerce')

#Drop rows with afact 0
place_bridge = place_bridge[place_bridge['afact'] != 0]

#Sort by state then place
place_bridge = place_bridge.sort_values(by = ['state','place'])

# Make column ncesid which concats state and sduni20
place_bridge['NCESID'] = place_bridge['state'].astype(str).str.zfill(2) + place_bridge['sduni20'].astype(str).str.zfill(5)

# Create a set of (state, place) tuples from sample_data
sample_state_place = set(zip(sample_data['FIPS_STATE'].astype(str), sample_data['FIPS_PLACE'].astype(str)))

# Filter place_bridge to only include state-place pairs from sample_data
place_bridge_filtered = place_bridge[place_bridge.apply(lambda row: (str(int(row['state'])), str(int(row['place']))) in sample_state_place, axis=1)]


# Ensure numeric columns in df are properly typed
for var in vars_to_use:
    df[var] = pd.to_numeric(df[var], errors='coerce')

# Merge filtered place_bridge with df
df_place = df.merge(place_bridge_filtered, on='NCESID', how='inner')

# Calculate weighted average directly
result = df_place.groupby(['state', 'place']).apply(lambda x: pd.Series({
    var: np.average(x[var], weights=x['afact']) for var in vars_to_use
})).reset_index()


##

# Get geo bridges for county subdivisions
cousub_bridge = pd.read_csv(os.path.join(raw_path, 'GEO Crosswalks', 'geocorr2022_district_cousub.csv'), encoding='latin1')

# Delete first row
cousub_bridge = cousub_bridge.iloc[1:]

# Ensure numeric columns are properly typed
cousub_bridge['afact'] = pd.to_numeric(cousub_bridge['afact'], errors='coerce')

#Drop rows with afact 0
place_bridge = place_bridge[place_bridge['afact'] != 0]

# Make column ncesid which concats state and sduni20
cousub_bridge['NCESID'] = cousub_bridge['state'].astype(str).str.zfill(2) + cousub_bridge['sduni20'].astype(str).str.zfill(5)

# Split the 'county' column into state and county
cousub_bridge['FIPS_STATE'] = cousub_bridge['county'].astype(str).str[:2]
cousub_bridge['FIPS_COUNTY'] = cousub_bridge['county'].astype(str).str[2:].str.zfill(3)

# Create a set of (state, county, cousub) tuples from sample_data
sample_state_county_cousub = set(zip(sample_data['FIPS_STATE'].astype(str).str.zfill(2),
                                     sample_data['FIPS_COUNTY'].astype(str).str.zfill(3),
                                     sample_data['FIPS_PLACE'].astype(str).str.zfill(5)))

# Filter cousub_bridge to only include state-county-cousub triples from sample_data
cousub_bridge_filtered = cousub_bridge[cousub_bridge.apply(lambda row: (row['FIPS_STATE'],
                                                                        row['FIPS_COUNTY'],
                                                                        str(int(row['cousub20'])).zfill(5)) in sample_state_county_cousub, axis=1)]

# Merge filtered cousub_bridge with df
df_cousub = df.merge(cousub_bridge_filtered, on='NCESID', how='inner')

# Calculate weighted average for county subdivisions
result_cousub = df_cousub.groupby(['FIPS_STATE', 'FIPS_COUNTY', 'cousub20']).apply(lambda x: pd.Series({
    var: np.average(x[var], weights=x['afact']) for var in vars_to_use
})).reset_index()

final_cousub = result_cousub.copy()

##

# Prepare place data
result['FIPS_STATE'] = result['state'].astype(str).str.zfill(2)
result['FIPS_PLACE'] = result['place'].astype(str).str.zfill(5)
result = result.drop(['state', 'place'], axis=1)


# Prepare county subdivision data
final_cousub['FIPS_PLACE'] = final_cousub['cousub20'].astype(str).str.zfill(5)
final_cousub = final_cousub.drop(['cousub20'], axis=1)

# Prepare sample data
sample_data['FIPS_STATE'] = sample_data['FIPS_STATE'].astype(str).str.zfill(2)
sample_data['FIPS_COUNTY'] = sample_data['FIPS_COUNTY'].astype(str).str.zfill(3)
sample_data['FIPS_PLACE'] = sample_data['FIPS_PLACE'].astype(str).str.zfill(5)

# Split sample data into municipalities and townships
municipalities = sample_data[sample_data['UNIT_TYPE'] == '2 - MUNICIPAL']
townships = sample_data[sample_data['UNIT_TYPE'] == '3 - TOWNSHIP']

# Merge place data with municipalities
muni_data = pd.merge(
    result,
    municipalities[['CENSUS_ID_PID6', 'FIPS_STATE', 'FIPS_PLACE']],
    on=['FIPS_STATE', 'FIPS_PLACE'],
    how='inner'
)

# Merge county subdivision data with townships
township_data = pd.merge(
    final_cousub,
    townships[['CENSUS_ID_PID6', 'FIPS_STATE', 'FIPS_COUNTY', 'FIPS_PLACE']],
    on=['FIPS_STATE', 'FIPS_COUNTY', 'FIPS_PLACE'],
    how='inner'
)

# Combine municipality and township data
final_data = pd.concat([muni_data, township_data], ignore_index=True)


# Reorder columns to have CENSUS_ID_PID6 first
cols = ['CENSUS_ID_PID6'] + [col for col in final_data.columns if col != 'CENSUS_ID_PID6']
final_data = final_data[cols]

# Define a dictionary for renaming
rename_dict = {
    "TOTALREV": "Total_Revenue_Per_Student",
    "TLOCREV": "Local_Revenue_Per_Student",
    "T06": "Property_Tax_Revenue_Per_Student",
    "TOTALEXP": "Total_Expenditure_Per_Student",
    "Z32": "Salaries_And_Wages_Per_Student",
    "Z34": "Employee_Benefits_Per_Student",
}

#Loop over each key in rename_dict and divide by V33
for key in rename_dict.keys():
    final_data[key] = final_data[key] / final_data['V33']

#Drop V33
final_data = final_data.drop(columns = ['V33'])

# Rename the columns in the final dataset
final_data = final_data.rename(columns=rename_dict)

#Drop FIPS_STATE, FIPS_PLACE, FIPS_COUNTY
final_data = final_data.drop(columns = ['FIPS_STATE','FIPS_PLACE','FIPS_COUNTY'], errors='ignore')

# Export to Excel
final_data.to_excel(os.path.join(data_path, 'interim_data', 'School_Finance_Data.xlsx'), index=False)

# Print some information about the merge
print(f"Total municipalities in sample: {len(municipalities)}")
print(f"Municipalities matched with place data: {len(muni_data)}")
print(f"Total townships in sample: {len(townships)}")
print(f"Townships matched with county subdivision data: {len(township_data)}")
print(f"Total rows in final data: {len(final_data)}")

'''
Analyze data
'''
##

