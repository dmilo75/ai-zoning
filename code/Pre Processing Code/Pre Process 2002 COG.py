import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pandas as pd

os.chdir('../../')

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##Load in munis text file

# Define column widths (adjust these values based on your actual data structure)
col_widths = [14, 60, 30, 40, 90, 40, 3, 10, 100, 10,2,3,5,35, 30]

# Define column names if known
column_names = [
    "CENSUS_ID_PID6", "UNIT_NAME", "Type", "Contact", "Address", "City", "STATE", "Zip",
    "Website", "Population",'FIPS_STATE','FIPS_COUNTY','FIPS_PLACE', "County", "Extra"
]

# Load the data
muni_units = pd.read_fwf(
    os.path.join(config['raw_data'], "Census of Governments", "2002GID_Cities.txt"),
    widths=col_widths,
    names=column_names,
    skiprows=0  # Adjust if there are more rows to skip
)

# Setting display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first few rows to verify
print(muni_units.head())

#Set unit type to muni
muni_units['UNIT_TYPE'] = '2 - MUNICIPAL'

##Load in townships text file

# Define column widths (adjust these values based on your actual data structure)
col_widths = [14, 60, 30, 40, 90, 40, 3, 10, 100, 10,2,3,5,35, 30]

# Define column names if known
column_names = [
    "CENSUS_ID_PID6", "UNIT_NAME", "Type", "Contact", "Address", "City", "STATE", "Zip",
    "Website", "Population",'FIPS_STATE','FIPS_COUNTY','FIPS_PLACE', "County", "Extra"
]

# Load the data
town_units = pd.read_fwf(
    os.path.join(config['raw_data'], "Census of Governments", "2002GID_Towns.txt"),
    widths=col_widths,
    names=column_names,
    skiprows=0  # Adjust if there are more rows to skip
)

# Setting display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first few rows to verify
print(town_units.head())

#Set unit type to township
town_units['UNIT_TYPE'] = '3 - TOWNSHIP'

##

#Concat munis and towns and make cog
cog = pd.concat([muni_units,town_units])

#Filter out Type of 'METROPOLITAN GOVERNMENT'
cog = cog[cog['Type'] != 'METROPOLITAN GOVERNMENT']

#Filter out when UNIT_NAME includes COUNTY or PARISH in it and isolate those rows
county_units = cog[cog['UNIT_NAME'].str.contains(' COUNTY| PARISH', case=False, na=False)]

#Remove them
cog = cog[~cog['UNIT_NAME'].str.contains(' COUNTY| PARISH', case=False, na=False)]

#Save preprocessed data
cog.to_excel(os.path.join(config['raw_data'],"Census of Governments","census_of_gov_02_preprocessed.xlsx"), index = False)

