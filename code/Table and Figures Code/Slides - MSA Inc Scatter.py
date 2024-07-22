"""
This code takes in a list of CENSUS PID6 IDs and returns the population of the group
"""
import geopandas as gpd
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from shapely.geometry import box
import pandas as pd
import pickle
import logging
import sys
import glob
from joblib import Parallel, delayed
os.chdir('../../')


#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)




var = 'Housing'
year = '2000'

cog_year = str(int(year)+2)

#Draw in all shape files for places and county subdivisions
base_shape_path = config['shape_files']
shape_path = os.path.join(base_shape_path,'2002')





##Main code

sample = pd.read_excel(os.path.join(config['raw_data'],'Census of Governments', 'census_of_gov_02_preprocessed.xlsx'))

all_states = sample['FIPS_STATE'].unique()


#Draw in county shape file
county_shape_path = os.path.join(shape_path,'Counties',f'tl_2010_us_county00.shp')
county_shapes = gpd.read_file(county_shape_path)


#Make a dataframe with CNTYIDFP00 as the index
df = pd.DataFrame(index = county_shapes['CNTYIDFP00'])

#Make column that is land area
df['Land Area'] = county_shapes.set_index('CNTYIDFP00')['ALAND00']




##MSA crossswalk

import requests
import pandas as pd

url = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/1999/historical-delineation-files/99mfips.txt"

# Download the text data from the URL
response = requests.get(url)
data = response.text

# Split the data into lines
lines = data.split("\n")

# Find the line number where the data starts
start_line = 0
for i, line in enumerate(lines):
    if line.startswith("MSA/"):
        start_line = i + 1
        break

# Extract the relevant data lines
data_lines = lines[start_line:]

# Initialize empty lists to store the data
msa_cmsa_fips = []
pmsa_fips = []
alt_cmsa_fips = []
state_county_fips = []
flag = []
city_town_fips = []
metro_area_component = []

# Process each data line
for line in data_lines:
    if line.strip():  # Skip empty lines
        msa_cmsa_fips.append(line[:4].strip())
        pmsa_fips.append(line[4:12].strip())
        alt_cmsa_fips.append(line[12:20].strip())
        state_county_fips.append(line[20:29].strip())
        flag.append(line[29:37].strip())
        city_town_fips.append(line[37:45].strip())
        metro_area_component.append(line[45:].strip())

# Create a DataFrame from the extracted data
data = pd.DataFrame({
    "MSA/CMSA FIPS CODE": msa_cmsa_fips,
    "PMSA FIPS CODE": pmsa_fips,
    "ALT. CMSA FIPS CODE": alt_cmsa_fips,
    "State/County FIPS CODE": state_county_fips,
    "F": flag,
    "City/Town FIPS CODE": city_town_fips,
    "Metropolitan Area and Component Names": metro_area_component
})

#Drop first 3 rows
data = data.iloc[3:]

#And the last 20 rows
data = data.iloc[:-20]


##

#Find rows with valid MSA/CMSA FIPS COde and Metropolitan Area and Component Names but missing PMSA FIPS and State/County FIPS
msa_names = data[(data['PMSA FIPS CODE'] == '') & (data['State/County FIPS CODE'] == '') & (data['Metropolitan Area and Component Names'] != '') & (data['MSA/CMSA FIPS CODE'] != '')]

#Now make a dictinoary mapping between MSA/CMSA FIPS CODE and Metropolitan Area and Component Names
msa_dict = dict(zip(msa_names['MSA/CMSA FIPS CODE'],msa_names['Metropolitan Area and Component Names']))

#Now drop rows in data that were in msa_names
data = data.drop(msa_names.index)

#Now map the MSA/CMSA FIPS CODE to Metropolitan Area and Component Names in new column called MSA Name
data['MSA Name'] = data['MSA/CMSA FIPS CODE'].map(msa_dict)

#Rename Metropolitan Area and Component Names to County Name
data = data.rename(columns = {'Metropolitan Area and Component Names':'County Name'})

#Drop rows with non-missing City/Town FIPS Code
data = data[data['City/Town FIPS CODE'] == '']

#Drop rows with missing State/County FIPS Code
data = data[data['State/County FIPS CODE'] != '']

#Make map from State/County FIPS CODE to MSA Name via dictionary
state_county_dict = dict(zip(data['State/County FIPS CODE'],data['MSA Name']))

#Make a map between MSA Name and MSA/CMSA FIPS CODE
msa_fips_dict = dict(zip(data['MSA Name'],data['MSA/CMSA FIPS CODE']))

##Get census data
import requests
import pandas as pd

API_KEY = config['census_key']

def county_census_data(year):
    if year == 2000:
        API_ENDPOINT = f"https://api.census.gov/data/{year}/dec/sf1"
        params = {
            "get": "P001001,H001001",
            "for": "county:*",
            "key": API_KEY
        }
    elif year == 2020:
        API_ENDPOINT = f"https://api.census.gov/data/{year}/dec/dhc"
        params = {
            "get": "P1_001N,H1_001N",
            "for": "county:*",
            "key": API_KEY
        }
    else:
        raise ValueError("Year must be 2000 or 2020")

    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} while fetching counties for year {year}")
        return None

    data = response.json()
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    return df


# Fetch and process 2000 data
data_2000 = county_census_data(2000)
data_2000.rename(columns={'P001001': 'Pop2000', 'H001001': 'Housing2000'}, inplace=True)
data_2000['CNTYIDFP00'] = data_2000['state'] + data_2000['county']
data_2000.set_index('CNTYIDFP00', inplace=True)

# Fetch and process 2020 data
data_2020 = county_census_data(2020)
data_2020.rename(columns={'P1_001N': 'Pop2020', 'H1_001N': 'Housing2020'}, inplace=True)
data_2020['CNTYIDFP20'] = data_2020['state'] + data_2020['county']
data_2020.set_index('CNTYIDFP20', inplace=True)

# Assuming df is already defined and has a suitable index to join on
# Add Pop2000 and Housing2000 to df
df = df.join(data_2000[['Pop2000', 'Housing2000']], how='left')

# Add Pop2020 and Housing2020 to df
df = df.join(data_2020[['Pop2020', 'Housing2020']], how='left')

#Find any rows with missing data
missing_data = df[df.isnull().any(axis=1)]

#Find the shape counties with mising data
missing_shapes = county_shapes[county_shapes['CNTYIDFP00'].isin(missing_data.index)]

#Fillna(0) for rows 51515
df = df.fillna(0)

#Make housing and population variables all ints
df['Pop2000'] = df['Pop2000'].astype(int)
df['Housing2000'] = df['Housing2000'].astype(int)
df['Pop2020'] = df['Pop2020'].astype(int)
df['Housing2020'] = df['Housing2020'].astype(int)



#Add row 51515 to 51019 then delete 51515
df.loc['51019'] += df.loc['51515']
df = df.drop('51515')




##Draw in the land incorporated file

#Load in the incorporated land data
inc = pd.read_pickle(os.path.join(config['processed_data'],'Incorporated Counts','area_incorporated_series.pkl'))

#Cap values at 100
inc = inc.apply(lambda x: min(x,100))

#Add to df as column of % of land incorporated in 2000
df['percent_incorporated'] = inc

##
#Now add MSA column to df with mapping
df['MSA'] = df.index.map(state_county_dict)


#make column which is land area times percent incorporated
df['Incorporated Land'] = df['Land Area']*df['percent_incorporated']

msa_df = df.groupby('MSA').sum()

#Make share incorproated
msa_df['Share Incorporated'] = msa_df['Incorporated Land']/msa_df['Land Area']

#Calculate percent change in housing stock and population
msa_df['Housing Change'] = 100*(msa_df['Housing2020']/msa_df['Housing2000']-1)
msa_df['Population Change'] = 100*(msa_df['Pop2020']/msa_df['Pop2000']-1)

##Merge in Saiz variables

#Map MSA name to msa fips
msa_df['MSA FIPS'] = msa_df.index.map(msa_fips_dict)

#Make it an int
msa_df['MSA FIPS'] = msa_df['MSA FIPS'].astype(int)

#Draw in Saiz data as .dta file in raw data Saiz Data HOUSING_SUPPLY.dta
saiz = pd.read_stata(os.path.join(config['raw_data'],'Saiz Data','HOUSING_SUPPLY.dta'))

#Merge msa_df on MSA FIPS and saiz on msanecma
msa_df = msa_df.merge(saiz,left_on='MSA FIPS',right_on='msanecma')



##


rename_map = {
    'Housing Change':'Percent Change Housing Units 2000 - 2020',
    'Population Change':'Percent Change Population 2000 - 2020',
    'Share Incorporated':'Percent of Land Area Incoporated in 2000',

}

def rename(str):
    try:
        return rename_map[str]
    except:
        return str

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binsreg import binsreg
from scipy.stats import pearsonr
import statsmodels.api as sm

def weighted_corr(x, y, weights):
    """Compute the weighted Pearson correlation coefficient."""
    x_w_mean = np.average(x, weights=weights)
    y_w_mean = np.average(y, weights=weights)
    cov_xy = np.sum(weights * (x - x_w_mean) * (y - y_w_mean)) / np.sum(weights)
    std_x = np.sqrt(np.sum(weights * (x - x_w_mean)**2) / np.sum(weights))
    std_y = np.sqrt(np.sum(weights * (y - y_w_mean)**2) / np.sum(weights))
    return cov_xy / (std_x * std_y)

def create_bin_scatter(df, x_var, y_var, size_var=None, weights_var=None):
    # Color variables
    raw_data_color = '#87CEEB'
    binned_fit_color = '#FF7F0E'
    linear_fit_color = '#FF7F0E'

    weights = df[weights_var] if weights_var is not None else np.ones(len(df))

    # Perform binscatter
    est = binsreg(y=df[y_var], x=df[x_var], w=None, data=df, weights=df[weights_var] if weights_var else None,
                  noplot=True, polyreg=1)

    # Extract the plotting information
    result = est.data_plot[0]

    # Extract the dots data
    dots = pd.DataFrame(result.dots)

    # Determine dot sizes
    dot_sizes = df[size_var] if size_var is not None else np.full(len(df), 10)

    # Create the plot using matplotlib
    plt.figure(figsize=(5.8, 3.3),dpi = 300)

    # Plot all raw data points
    plt.scatter(df[x_var], df[y_var], s=dot_sizes / 10000, color=raw_data_color, alpha=0.5, label='Raw Data')

    # Plot the bin scatter data points
    plt.scatter(dots['x'], dots['fit'], color=binned_fit_color, s=50, label='Binned Fit')

    # Plot the linear fit line using poly data
    poly = pd.DataFrame(result.poly)
    plt.plot(poly['x'], poly['fit'], color=linear_fit_color, label='Linear Fit', linestyle='dashed')


    # Calculate and display the (weighted) correlation
    if weights_var is not None:
        correlation = weighted_corr(df[x_var], df[y_var], weights)
    else:
        correlation, _ = pearsonr(df[x_var], df[y_var])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Add labels and title
    plt.xlabel(rename(x_var))
    plt.ylabel(rename(y_var))
    plt.title(f'Area Incorporated vs Housing Unit Growth by MSA')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

    #Save figure to config results
    plt.savefig(os.path.join(config['figures_path'],'Slides - MSA Inc Scatter'),dpi = 300)



# Example usage
# Create bin scatter between percent incorporated and housing change
# Optionally include a variable to control dot sizes
create_bin_scatter(msa_df, 'Share Incorporated', 'Housing Change', size_var='Pop2020', weights_var = 'Pop2020')


##

