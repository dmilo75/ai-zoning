import pandas as pd
import requests
import yaml
import os
os.chdir('../../')
# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['raw_data']
API_KEY = config['census_key']

# Make 2022_acs_blockgroup_data folder
os.makedirs(os.path.join(data_path, '2022_acs_blockgroup_data'), exist_ok=True)

def pull_counties(state_fips):
    API_ENDPOINT = "https://api.census.gov/data/2022/acs/acs5"

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

def pull_acs_2022_blockgroup(var_codes, var_names, state_fips, county_fips, test_mode=False):
    API_ENDPOINT = "https://api.census.gov/data/2022/acs/acs5"

    params = {
        "get": ','.join(var_codes),
        "for": "block group:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": API_KEY
    }

    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} while fetching block group data for state {state_fips}, county {county_fips}")
        return None

    data = response.json()
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    # Set first n columns to var_names
    df.columns = var_names + list(df.columns[len(var_names):])

    if test_mode:
        return df.head()  # Return only the first 5 rows in test mode
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

var_codes = [
    'B25077_001E',  # Median home value
    'B25001_001E',  # Total housing units
    'B25002_003E',  # Vacant housing units
    'B25003_002E',  # Owner-occupied housing units
    'B25003_001E',  # Total occupied housing units
    'B25035_001E',  # Median year structure built
    'B08303_001E',  # Total workers 16 years and over who did not work at home
    'B08303_012E',  # Workers with travel time 60 to 89 minutes
    'B08303_013E',  # Workers with travel time 90 or more minutes
    'B01001_020E',  # Males 65 to 66 years
    'B01001_021E',  # Males 67 to 69 years
    'B01001_022E',  # Males 70 to 74 years
    'B01001_023E',  # Males 75 to 79 years
    'B01001_024E',  # Males 80 to 84 years
    'B01001_025E',  # Males 85 years and over
    'B01001_044E',  # Females 65 to 66 years
    'B01001_045E',  # Females 67 to 69 years
    'B01001_046E',  # Females 70 to 74 years
    'B01001_047E',  # Females 75 to 79 years
    'B01001_048E',  # Females 80 to 84 years
    'B01001_049E',  # Females 85 years and over
    'B19013_001E',  # Median household income in the past 12 months
    'B01003_001E',  # Total population
'B25064_001E',  # Median Gross Rent
]

var_names = [
    'Median_Home_Value',
    'Total_Housing_Units',
    'Vacant_Housing_Units',
    'Owner_Occupied_Units',
    'Total_Occupied_Units',
    'Median_Year_Built',
    'Total_Workers',
    'Workers_60_to_89_min_commute',
    'Workers_90_plus_min_commute',
    'Males_65_to_66',
    'Males_67_to_69',
    'Males_70_to_74',
    'Males_75_to_79',
    'Males_80_to_84',
    'Males_85_plus',
    'Females_65_to_66',
    'Females_67_to_69',
    'Females_70_to_74',
    'Females_75_to_79',
    'Females_80_to_84',
    'Females_85_plus',
    'Median_Household_Income',
    'Total_Population',
    'Median_Gross_Rent',
]
def process_state_data(state_fips, test_mode=False):
    all_blockgroup_data = []
    counties_df = pull_counties(state_fips)
    if counties_df is not None:
        county_fips_list = counties_df['county'].unique()
        for county_fips in county_fips_list:
            blockgroup_data = pull_acs_2022_blockgroup(var_codes, var_names, state_fips, county_fips, test_mode)
            if blockgroup_data is not None:
                all_blockgroup_data.append(blockgroup_data)
            if test_mode:
                break  # Process only one county in test mode

    if all_blockgroup_data:
        combined_blockgroup_data = pd.concat(all_blockgroup_data, ignore_index=True)
        return combined_blockgroup_data
    return None

# Main processing loop
all_data = []
for state_fips in state_fips_list:
    print(f"Processing state {state_fips}")
    state_data = process_state_data(state_fips)
    if state_data is not None:
        all_data.append(state_data)
    print(f"Finished processing state {state_fips}")

# Combine all data and export to a single file
combined_data = pd.concat(all_data, ignore_index=True)
output_file = os.path.join(data_path, '2022_acs_blockgroup_data', "acs_blockgroup_data_all_states.csv")
combined_data.to_csv(output_file, index=False)
print(f"Saved combined data for all processed states to {output_file}")


