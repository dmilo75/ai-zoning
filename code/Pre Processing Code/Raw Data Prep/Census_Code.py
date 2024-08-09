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
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"))

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])


'''
Variables are here
https://api.census.gov/data/2020/dec/dhc/variables.html

For now we just want housing unit counts which is H1_001N

'''

API_KEY = config['census_key']

state_fips = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56"
]

def pull_census(var_code, var_name, year, geography='place'):
    API_ENDPOINT = f"https://api.census.gov/data/{year}/dec/dhc"

    if geography == 'place':
        params = {
            "get": var_code,
            "for": "place:*",
            "key": API_KEY
        }
        response = requests.get(API_ENDPOINT, params=params)
        data = response.json()

        header = data[0]
        rows = data[1:]

        df = pd.DataFrame(rows, columns=header)
        df = df.rename(columns={var_code: var_name})

        return df

    elif geography == 'county subdivision':

        all_states_data = []
        header_saved = False  # Flag to check if the header is saved

        for state in state_fips:
            params = {
                "get": var_code,
                "for": "county subdivision:*",
                "in": f"state:{state}",
                "key": API_KEY
            }

            response = requests.get(API_ENDPOINT, params=params)
            data = response.json()

            # Save header only once
            if not header_saved:
                header = data[0]
                header_saved = True
                all_states_data.append(header)

            # Extend the main data list with data rows excluding the header
            all_states_data.extend(data[1:])

        df = pd.DataFrame(all_states_data[1:], columns=all_states_data[0])  # Use saved header for columns
        df = df.rename(columns={var_code: var_name})

        return df

    else:
        raise ValueError(f"Unsupported geography: {geography}")



#Pull in census data

year = '2020'
place_data = pull_census('H1_001N', 'Housing Units', year, 'place')
county_subdivision_data = pull_census('H1_001N', 'Housing Units', year, 'county subdivision')

#Export place and county subdivision data
place_data.to_excel(os.path.join(config['raw_data'], 'Census Data', f'Housing_Place_{year}.xlsx'))
county_subdivision_data.to_excel(os.path.join(config['raw_data'], 'Census Data', f'Housing_Cousub_{year}.xlsx'))

place_data = place_data.rename(columns = {'place':'FIPS_PLACE', 'state':'FIPS_STATE'})
#Make fips columns ints
for var in ['FIPS_PLACE', 'FIPS_STATE']:
    place_data[var] = place_data[var].astype(int)

county_subdivision_data = county_subdivision_data.rename(columns = {'county subdivision':'FIPS_PLACE', 'state':'FIPS_STATE', 'county':'FIPS_COUNTY_ADJ'})
#Make fips columns ints
for var in ['FIPS_PLACE', 'FIPS_STATE', 'FIPS_COUNTY_ADJ']:
    county_subdivision_data[var] = county_subdivision_data[var].astype(int)

#Merge in the data into sample
final_df = sample.copy()
merged1 = pd.merge(final_df, place_data, on = ['FIPS_PLACE', 'FIPS_STATE'], how = 'left')
merged2 = pd.merge(merged1, county_subdivision_data, on = ['FIPS_PLACE', 'FIPS_STATE', 'FIPS_COUNTY_ADJ'], how = 'left')

merged2['Housing Units'] = merged2['Housing Units_x'].fillna(merged2['Housing Units_y'])

complete_df = merged2[['CENSUS_ID_PID6', 'Housing Units']]

#Print percent of non null rows
print(complete_df['Housing Units'].count() / complete_df.shape[0])

#Rename Housing Units to 'Housing Units Census 2020'
complete_df = complete_df.rename(columns = {'Housing Units':'Housing Units Census 2020'})

#Save the data
complete_df.to_excel(os.path.join(data_path, 'interim_data', 'Census Data.xlsx'))
