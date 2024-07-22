import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pandas as pd

os.chdir('../../')

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Load in 2012 cog
cog = pd.read_excel(os.path.join(config['raw_data'],"Census of Governments","census_of_gov_12.xlsx"))

'''
We need to preprocess the 2012 census of goverments to match the 2022 format

We need the following columns
- UNIT_TYPE
- CENSUS_ID_PID6
- UNIT_NAME
'''

def preprocess_cog(df, unit_type_dict):

    # Initialize UNIT_TYPE column with 'UNKNOWN'
    df['UNIT_TYPE'] = 'UNKNOWN'

    # Loop over the dictionary to map unit types
    for keyword, unit_type in unit_type_dict.items():
        mask = df['NAME'].str.startswith(keyword)
        df.loc[mask, 'UNIT_TYPE'] = unit_type

    # Print out the names that have not met the filter criteria
    remaining_rows = df[df['UNIT_TYPE'] == 'UNKNOWN']
    if not remaining_rows.empty:
        print("Rows that did not meet the filter criteria:")
        print(remaining_rows['NAME'])

    return df, remaining_rows

# Define the unit type dictionary
unit_type_dict = {
    'CITY': '2 - MUNICIPAL',
    'CONSOLIDATED GOVERNMENT OF COLUMBUS': '2 - MUNICIPAL',
    'CONSOLIDATED GOVERNMENT OF TERREBONNE': '2 - MUNICIPAL',
    'CORPORATION': '2 - MUNICIPAL',
    'BOROUGH': '2 - MUNICIPAL',
    'MUNICIPALITY': '2 - MUNICIPAL',
    'VILLAGE': '2 - MUNICIPAL',
    'TOWN': '3 - TOWNSHIP',
    'PLANTATION': '3 - TOWNSHIP',
    'CIVIL TOWNSHIP': '3 - TOWNSHIP',
    'CHARTER TOWNSHIP': '3 - TOWNSHIP',
    'TOWNSHIP': '3 - TOWNSHIP',
}

#Rename STATE_AB to STATE
cog = cog.rename(columns = {'STATE_AB':'STATE'})

#Ensure state is uppercase
cog['STATE'] = cog['STATE'].str.upper()

# Filter out rows with 'COUNTY'
cog = cog[~cog['NAME'].str.contains('COUNTY', case=False, na=False)]

# And with 'PARISH'
cog = cog[~cog['NAME'].str.contains('PARISH', case=False, na=False)]

# States that have townships
township_states = ["CT", "MA", "NH", "PA", "IL", "MI", "NJ", "RI", "IN", "MN", "NY", "SD", "KS", "MO", "ND", "VT", "ME", "NE", "OH", "WI"]

# And their FIPS state values
fips_state_values = [9, 25, 33, 42, 17, 26, 34, 44, 18, 27, 36, 46, 20, 29, 38, 50, 23, 31, 39, 55]

# Determine township rows and other rows
township_rows = cog[(cog['STATE'].isin(township_states)) | (cog['FIPS_STATE'].isin(fips_state_values))]
other_rows = cog[~((cog['STATE'].isin(township_states)) | (cog['FIPS_STATE'].isin(fips_state_values)))]

other_rows['UNIT_TYPE'] = '2 - MUNICIPAL'

# Preprocess the 2012 census of governments data
township_rows_processed, remaining_rows = preprocess_cog(township_rows, unit_type_dict)

cog_preprocessed = pd.concat([township_rows_processed, other_rows])

#Rename CENSUS_ID to CENSUS_ID_PID6
cog_preprocessed = cog_preprocessed.rename(columns = {'CENSUS_ID':'CENSUS_ID_PID6'})


#Rename NAME to UNIT_NAME
cog_preprocessed = cog_preprocessed.rename(columns = {'NAME':'UNIT_NAME'})

#Save preprocessed data
cog_preprocessed.to_excel(os.path.join(config['raw_data'],"Census of Governments","census_of_gov_12_preprocessed.xlsx"), index = False)

