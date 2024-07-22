import pandas as pd
import requests
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['raw_data']
API_KEY = config['census_key']

#Make 2000_census_block_data folder
os.makedirs(os.path.join(data_path,'2000_census_block_data'), exist_ok=True)

'''
2000 variables are below:

https://api.census.gov/data/2000/dec/sf1/variables.html
'''

def pull_counties(state_fips):
    API_ENDPOINT = "https://api.census.gov/data/2000/dec/sf1"

    params = {
        "get": "NAME",
        "for": "county:*",
        "in": f"state:{state_fips}",
        "key": API_KEY
    }

    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} while fetching counties for state {state_fips}")
        return None

    data = response.json()
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    return df


def pull_census_2000_block(var_codes, var_names, state_fips, county_fips):
    year = '2000'
    API_ENDPOINT = f"https://api.census.gov/data/{year}/dec/sf1"

    params = {
        "get": ','.join(var_codes),
        "for": "block:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": API_KEY
    }

    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} while fetching block data for state {state_fips}, county {county_fips}")
        return None

    data = response.json()
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    #Set first n columns to var_names
    df.columns = var_names + list(df.columns[len(var_names):])

    return df


# Get the list of all states
state_fips_list = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56"
]

var_codes = ['P001001','H001001' ] # Population variable
var_names = ['Population','Housing Units']

def format_tract(tract):
    tract_str = str(tract)
    if len(tract_str) == 4:
        return tract_str + '00'
    return tract_str.zfill(6)


for state_fips in state_fips_list:
    all_block_data = []
    counties_df = pull_counties(state_fips)
    if counties_df is not None:
        county_fips_list = counties_df['county'].unique()
        for county_fips in county_fips_list:
            block_data = pull_census_2000_block(var_codes, var_names, state_fips, county_fips)
            if block_data is not None:
                all_block_data.append(block_data)

    if all_block_data:
        combined_block_data = pd.concat(all_block_data, ignore_index=True)

    # Apply the function to the tract column
    combined_block_data['tract'] = combined_block_data['tract'].apply(format_tract)

    # Save the data
    combined_block_data.to_csv(os.path.join(data_path,'2000_census_block_data',f"census_block_data_{state_fips}.csv"), index=False)
