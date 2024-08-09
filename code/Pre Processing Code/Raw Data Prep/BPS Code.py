import pandas as pd
import os
import yaml
import helper_functions as hf
import numpy as np
# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#Read in sample data
sample = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"))

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])


##Get buliding permit data

import pandas as pd
import requests
from io import StringIO

def process_data(url):
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request was unsuccessful

    # Use pandas to read the CSV data directly from the response text
    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Reset index
    df = df.reset_index()

    return df


def concatenate_headers(df):
    # Make any unnamed columns null
    df.columns = [col if 'Unnamed' not in col else None for col in df.columns]

    # Forward fill the null values in columns
    df.columns = df.columns.to_series().ffill()

    # Concatenate the first row with the existing column titles
    new_header = df.columns.astype(str) + ' ' + df.iloc[0].astype(str)
    new_header = new_header.str.strip()  # Remove any leading/trailing whitespace

    # Set the new header and drop the first row
    df.columns = new_header

    # Drop the first row
    df = df.drop(0).reset_index(drop=True)



    return df


def get_bps_data(years):
    base_url = "https://www2.census.gov/econ/bps/Place/"
    regions = {
        'ne': 'Northeast',
        'mw': 'Midwest',
        'so': 'South',
        'we': 'West'
    }
    all_data = []

    for year in years:
        for region_code, region_name in regions.items():
            file_name = f"{region_code}{year}a.txt"
            url = f"{base_url}{region_name}%20Region/{file_name}"
            try:
                df = process_data(url)
                df = concatenate_headers(df)
                all_data.append(df)
            except requests.exceptions.HTTPError as e:
                print(f"Failed to retrieve data for {year} in {region_name} region: {e}")

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    columns = [
        "Survey Date","State Code", "6-Digit ID", "County Code", "Census Place Code",
        "FIPS Place Code", "FIPS MCD Code", "Population", "CSA Code", "CBSA Code",
        "Footnote Code", "Central City Code", "Zip Code", "Census Region Code",
        "Census Division Code", "Number of Months Reported", "Place Name",
        "1-unit Bldgs", "1-unit Units", "1-unit Value", "2-units Bldgs",
        "2-units Units", "2-units Value", "3-4 units Bldgs", "3-4 units Units",
        "3-4 units Value", "5+ units Bldgs", "5+ units Units", "5+ units Value",
        "1-unit rep Bldgs", "1-unit rep Units", "1-unit rep Value", "2-units rep Bldgs",
        "2-units rep Units", "2-units rep Value", "3-4 units rep Bldgs",
        "3-4 units rep Units", "3-4 units rep Value", "5+ units rep Bldgs",
        "5+ units rep Units", "5+ units rep Value"
    ]

    combined_df.columns = columns

    return combined_df

years = [str(year) for year in range(2019, 2023+1)]
bps_data = get_bps_data(years)

##Aggregate

for col in list(bps_data.columns):
    if ('unit' in col) or (col in ['State Code','FIPS Place Code']):
        bps_data[col] = pd.to_numeric(bps_data[col], errors='coerce')

#Drop rows with missing state of place code
bps_data = bps_data.dropna(subset=['State Code', 'FIPS Place Code'])

#Drop fips place codes of 0 or 999990
bps_data = bps_data[(bps_data['FIPS Place Code'] != 0) & (bps_data['FIPS Place Code'] != 999990)]

#Only keep state code, fips place code and units columns
unit_columns = [col for col in bps_data.columns if 'Unit' in col]

bps_data  = bps_data[['State Code','FIPS Place Code'] + unit_columns]

#Group by state and fips place code and average
bps_data = bps_data.groupby(['State Code','FIPS Place Code']).mean().reset_index()


##Merge in building permits data


sample_perm = sample.copy()

#Drop townships in sample_perm since BPS data only available for places
sample_perm = sample_perm[sample_perm['UNIT_TYPE'] != '3 - TOWNSHIP']

sample_perm = pd.merge(sample_perm, bps_data,
                       left_on=['FIPS_PLACE', 'FIPS_STATE'],
                       right_on=['FIPS Place Code', 'State Code'],
                       how='left')

# Count the number of non-NaN entries in 'FIPS Place' after the merge
merged_count = sample_perm['FIPS Place Code'].notna().sum()

# Calculate the percentage merged
percentage_merged = (merged_count / len(sample_perm)) * 100

print(f"Percentage merged: {percentage_merged:.2f}%")

#%%Keep only relevant variables now

# Create 'Single Unit Permits'
sample_perm['Single Unit Permits'] = sample_perm['1-unit Units']

# Create 'Multi-Unit Permits'
sample_perm['Multi-Unit Permits'] = (sample_perm['2-units Units'] +
                                     sample_perm['3-4 units Units'] +
                                     sample_perm['5+ units Units'])

# Create 'All Unit Permits'
sample_perm['All Unit Permits'] = sample_perm['Single Unit Permits'] + sample_perm['Multi-Unit Permits']


#Make single, multi, and all unit permits in per capita terms but dont rename
sample_perm['Single Unit Permits'] = sample_perm['Single Unit Permits'] / sample_perm['POPULATION']
sample_perm['Multi-Unit Permits'] = sample_perm['Multi-Unit Permits'] / sample_perm['POPULATION']
sample_perm['All Unit Permits'] = sample_perm['All Unit Permits'] / sample_perm['POPULATION']

#Only keep new variables and CENSUS_ID_PID6
sample_perm = sample_perm[['Single Unit Permits','Multi-Unit Permits','All Unit Permits','CENSUS_ID_PID6']]


#Save to 'interim_data' under processed data folder
sample_perm.to_excel(os.path.join(data_path,'interim_data','BPS Data.xlsx'))



