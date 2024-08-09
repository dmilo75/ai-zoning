import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import requests
import us

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

API_KEY = config['census_key']

def pull_census(var_code, year):
    if year == '2020':
        API_ENDPOINT = "https://api.census.gov/data/2020/dec/dhc"
    elif year == '2000':
        API_ENDPOINT = "https://api.census.gov/data/2000/dec/sf1"
    else:
        raise ValueError(f"Invalid year: {year}")

    params = {
        "get": var_code,
        "for": "state:*",
        "key": API_KEY
    }
    response = requests.get(API_ENDPOINT, params=params)
    data = response.json()

    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    return df

# Pull in census data for 2000 and 2020
states_housing_2020 = pull_census('H1_001N', '2020')
states_housing_2000 = pull_census('H001001', '2000')

# Rename columns and add year
states_housing_2020.columns = ['housing', 'state_fips']
states_housing_2020['year'] = 2020

states_housing_2000.columns = ['housing', 'state_fips']
states_housing_2000['year'] = 2000

# Concatenate 2000 and 2020 data
full_df = pd.concat([states_housing_2020, states_housing_2000], ignore_index=True)

# Convert 'state_fips' to int
full_df['state_fips'] = full_df['state_fips'].astype(int)

# Change to the directory containing your data
os.chdir(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Incorporated Counts")

# Initialize dictionary to store the series from each folder
housing_data = {}

# Loop through all directories and files in the current directory
for root, dirs, files in os.walk("."):
    if "Housing" in root:
        for filename in files:
            if filename.endswith(".pkl"):
                # Extract the state FIPS code from the filename
                state_fips = filename.split('#')[1].split('.')[0]

                # Extract the year from the directory path
                year = root.split(os.path.sep)[-2]

                # Load the pickle file into a pandas series
                file_path = os.path.join(root, filename)
                series = pd.read_pickle(file_path)

                # Add the state FIPS code and year to the series
                series['state_fips'] = state_fips
                series['year'] = year

                # Add the series to the housing data dictionary
                housing_data[(state_fips, year)] = series

# Initialize lists to store data for the DataFrame
housing_values = []
state_fips_list = []
year_list = []

# Loop over all keys (state FIPS and year) and collect data
for key, series in housing_data.items():
    state_fips, year = key
    if 'Housing' in series.index:
        housing_values.append(series['Housing'])
        state_fips_list.append(state_fips)
        year_list.append(year)

# Create a DataFrame
df = pd.DataFrame({
    'state_fips': state_fips_list,
    'year': year_list,
    'housing': housing_values
})

# Make state_fips and year ints
df['state_fips'] = df['state_fips'].astype(int)
df['year'] = df['year'].astype(int)

# For full_df make housing ints
full_df['housing'] = full_df['housing'].astype(int)

# Set index as state_fips, year for both
df = df.set_index(['state_fips', 'year'])
full_df = full_df.set_index(['state_fips', 'year'])
unincorporated = full_df - df

# Reset indices
df = df.reset_index()
unincorporated = unincorporated.reset_index()
##



##

import pandas as pd
import matplotlib.pyplot as plt
import us
import numpy as np

# Assuming df and unincorporated are already defined from previous steps

data = {}
data['Incorporated'] = {
    '2000': df[df['year'] == 2000].drop(columns='year').copy().set_index('state_fips') / 1000000,
    '2020': df[df['year'] == 2020].drop(columns='year').copy().set_index('state_fips') / 1000000
}
data['Unincorporated'] = {
    '2000': unincorporated[unincorporated['year'] == 2000].drop(columns='year').copy().set_index(
        'state_fips') / 1000000,
    '2020': unincorporated[unincorporated['year'] == 2020].drop(columns='year').copy().set_index('state_fips') / 1000000
}

for key in data:
    for year in data[key]:
        # Add total for US
        data[key][year].loc['US'] = data[key][year].sum()

# Define states in the South and West regions
south_states = ['Delaware', 'Maryland', 'Virginia', 'West Virginia', 'Kentucky', 'North Carolina', 'South Carolina',
                'Tennessee', 'Georgia', 'Florida', 'Alabama', 'Mississippi', 'Arkansas', 'Louisiana', 'Texas',
                'Oklahoma']
west_states = ['Montana', 'Wyoming', 'Colorado', 'New Mexico', 'Idaho', 'Utah', 'Arizona', 'Nevada', 'California',
               'Oregon', 'Washington', 'Alaska', 'Hawaii']


# Function to get state name from FIPS code
def get_name(fips):
    if fips == 'US':
        return 'US'
    try:
        return us.states.lookup(f"{int(fips):02d}").name
    except:
        return 'Washington DC' if fips == '11' else 'Unknown'


# Updated function to calculate changes for regions
def calculate_changes(data, region_states=None):
    if region_states:
        data_2000 = data['2000'][data['2000'].index.map(get_name).isin(region_states)].sum()
        data_2020 = data['2020'][data['2020'].index.map(get_name).isin(region_states)].sum()
    else:
        data_2000 = data['2000'].loc['US']
        data_2020 = data['2020'].loc['US']

    growth_rate = (data_2020 - data_2000) / data_2000 * 100
    raw_change = data_2020 - data_2000

    return growth_rate['housing'], raw_change['housing']


# Calculate changes for US, South, and West
changes = {
    'US': {
        'Incorporated': calculate_changes(data['Incorporated']),
        'Unincorporated': calculate_changes(data['Unincorporated'])
    },
    'South': {
        'Incorporated': calculate_changes(data['Incorporated'], south_states),
        'Unincorporated': calculate_changes(data['Unincorporated'], south_states)
    },
    'West': {
        'Incorporated': calculate_changes(data['Incorporated'], west_states),
        'Unincorporated': calculate_changes(data['Unincorporated'], west_states)
    }
}

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.3))

regions = ['US', 'South', 'West']
x = np.arange(len(regions))
width = 0.35

# Raw number of housing units added
inc_raw = [changes[region]['Incorporated'][1] for region in regions]
uninc_raw = [changes[region]['Unincorporated'][1] for region in regions]

ax1.bar(x - width / 2, inc_raw, width, label='Incorporated', color='blue')
ax1.bar(x + width / 2, uninc_raw, width, label='Unincorporated', color='orange')
ax1.set_ylabel('Housing Units Added (Millions)')
ax1.set_title('Raw Number of Housing Units Added (2000-2020)')
ax1.set_xticks(x)
ax1.set_xticklabels(regions)
ax1.legend()

# Percent change
inc_pct = [changes[region]['Incorporated'][0] for region in regions]
uninc_pct = [changes[region]['Unincorporated'][0] for region in regions]

ax2.bar(x - width / 2, inc_pct, width, label='Incorporated', color='blue')
ax2.bar(x + width / 2, uninc_pct, width, label='Unincorporated', color='orange')
ax2.set_ylabel('Percent Change (%)')
ax2.set_title('Percent Change in Housing Units (2000-2020)')
ax2.set_xticks(x)
ax2.set_xticklabels(regions)
ax2.legend()

plt.tight_layout()
plt.show()

##Scatter plots


from adjustText import adjust_text

# Scatter plot for growth rates
fig, ax = plt.subplots(figsize=(.8*10, .8*6))  # Increase figure size

texts = []

for state in south_states + west_states:
    state_fips = next(
        (fips for fips, name in zip(data['Incorporated']['2020'].index, data['Incorporated']['2020'].index.map(get_name)) if name == state), None)
    if state_fips:
        inc_rate = ((data['Incorporated']['2020'].loc[state_fips, 'housing'] - data['Incorporated']['2000'].loc[
            state_fips, 'housing']) /
                    data['Incorporated']['2000'].loc[state_fips, 'housing'] * 100)
        uninc_rate = ((data['Unincorporated']['2020'].loc[state_fips, 'housing'] - data['Unincorporated']['2000'].loc[
            state_fips, 'housing']) /
                      data['Unincorporated']['2000'].loc[state_fips, 'housing'] * 100)
        color = 'green' if state in south_states else 'purple'
        ax.scatter(inc_rate, uninc_rate, color=color, alpha=0.7)
        texts.append(ax.text(inc_rate, uninc_rate, state, fontsize=8))

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

ax.set_title('Comparison of Housing Growth Rates (2000-2020)\nSouth and West Regions')
ax.set_xlabel('Incorporated Growth Rate (%)')
ax.set_ylabel('Unincorporated Growth Rate (%)')

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.show()

# Scatter plot for raw number of housing units
fig, ax = plt.subplots(figsize=(.8*10, .8*6))  # Increase figure size

texts = []

for state in south_states + west_states:
    state_fips = next(
        (fips for fips, name in zip(data['Incorporated']['2020'].index, data['Incorporated']['2020'].index.map(get_name)) if name == state), None)
    if state_fips:
        inc_units = data['Incorporated']['2020'].loc[state_fips, 'housing']
        uninc_units = data['Unincorporated']['2020'].loc[state_fips, 'housing']
        color = 'green' if state in south_states else 'purple'
        ax.scatter(inc_units, uninc_units, color=color, alpha=0.7)
        texts.append(ax.text(inc_units, uninc_units, state, fontsize=8))

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

ax.set_title('Comparison of Housing Units (2020)\nSouth and West Regions')
ax.set_xlabel('Incorporated Housing Units (Millions)')
ax.set_ylabel('Unincorporated Housing Units (Millions)')

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.show()

##

# Define the font size variable
font_size = 16

# Create the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left subplot: Bar chart
regions = ['US', 'South', 'West']
x = np.arange(len(regions))
width = 0.35

inc_raw = [changes[region]['Incorporated'][1] for region in regions]
uninc_raw = [changes[region]['Unincorporated'][1] for region in regions]

ax1.bar(x - width / 2, inc_raw, width, label='Incorporated', color='blue')
ax1.bar(x + width / 2, uninc_raw, width, label='Unincorporated', color='orange')
ax1.set_ylabel('Housing Units Added (Millions)', fontsize=font_size)
ax1.set_title('Housing Unit Growth By Incorporation Status in 2000', fontsize=font_size)
ax1.set_xticks(x)
ax1.set_xticklabels(regions, fontsize=font_size)
# Add a title to the legend
ax1.legend(fontsize=font_size)


ax1.tick_params(axis='both', which='major', labelsize=font_size)

# Right subplot: Scatter plot
texts = []

for state in south_states + west_states:
    state_fips = next(
        (fips for fips, name in zip(data['Incorporated']['2020'].index, data['Incorporated']['2020'].index.map(get_name)) if name == state), None)
    if state_fips:
        inc_rate = ((data['Incorporated']['2020'].loc[state_fips, 'housing'] - data['Incorporated']['2000'].loc[state_fips, 'housing']) /
                    data['Incorporated']['2000'].loc[state_fips, 'housing'] * 100)
        uninc_rate = ((data['Unincorporated']['2020'].loc[state_fips, 'housing'] - data['Unincorporated']['2000'].loc[state_fips, 'housing']) /
                      data['Unincorporated']['2000'].loc[state_fips, 'housing'] * 100)
        color = 'green' if state in south_states else 'purple'
        ax2.scatter(inc_rate, uninc_rate, color=color, alpha=0.7)
        texts.append(ax2.text(inc_rate, uninc_rate, state, fontsize=font_size - 4))

# Add US data point in red
us_inc_rate = changes['US']['Incorporated'][0]
us_uninc_rate = changes['US']['Unincorporated'][0]
ax2.scatter(us_inc_rate, us_uninc_rate, color='#FF6347', zorder=5, alpha = 0.7)
texts.append(ax2.text(us_inc_rate, us_uninc_rate, 'US', fontsize=font_size - 4))

lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

ax2.set_title('Comparison of Housing Unit Growth Rates (2000-2020)\nSouth and West Regions', fontsize=font_size)
ax2.set_xlabel('Incorporated in 2000 Growth Rate (%)', fontsize=font_size)
ax2.set_ylabel('Unincorporated in 2000 Growth Rate (%)', fontsize=font_size)

ax2.tick_params(axis='both', which='major', labelsize=font_size)

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()

# Change to the directory containing your data
os.chdir(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning")


# Save figure to figures folder
fig.savefig(os.path.join(config['figures_path'], 'Figure X - Growth in Housing Units Inc.png'))

plt.show()

