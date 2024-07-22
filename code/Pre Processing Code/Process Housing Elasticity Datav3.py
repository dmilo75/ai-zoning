import pandas as pd
import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

raw_path = config['raw_data']

#Read in sample data
sample = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"),index_col  = 0)

#Read in NBSLH Housing Elasticity stata file
df = pd.read_stata(os.path.join(config['raw_data'],'NBSLH Housing Elasticity','gammas_hat_all.dta'))

#Make tract frc deve squared
df['Trct_FrcDev_01_squared'] = df['Trct_FrcDev_01']**2

# Define the list of variables to add
variables_to_add = ['gamma01b_units_FMM', 'Trct_FrcDev_01','Trct_FrcDev_01_squared', 'lsf_1_flat_plains','gamma01b_newunits_FMM']


# Define the list of variables to include
variables_to_include = ['ctracts2000', 'wgt'] + variables_to_add


'''
We will use ctracts2000 and gamma01b_units_FMM

ctracts2000 is the census tract identifier for 2000

gamma01b_units_FMM is the estimated housing unit supply elasticity using the finite mixture model (FMM)
approach with the quadratic specification (what 'b' means) and the 2001 (what '01' means) tract developed fraction measure. 
'''

# Get the relevant columns
df = df[variables_to_include]

# Multiply specified variables by weight
for var in variables_to_add:
    df[var] = df[var] * df['wgt']

# Groupby ctracts2000 and sum
agg_columns = {var: 'sum' for var in variables_to_add}
agg_columns.update({'wgt': 'sum'})
df = df.groupby('ctracts2000').agg(agg_columns).reset_index()

#Ensure ctracts2000 is an 11 digit string left padded 0
df['ctracts2000'] = (df['ctracts2000'].astype(str))
df['ctracts2000'] = (df['ctracts2000'].apply(lambda x: x.zfill(11)))

##Setting up bridge

ints = pd.read_csv(os.path.join(config['processed_data'],'Muni-Tract Geobridge Results','muni_tract_geobridge.csv'))

##

#Make new column 'Weight' which is 'Intersection_Area'/'Muni_Area'
ints['Weight'] = ints['Intersection_Area']/ints['Muni_Area']

#Make Tract_GEOID 11 digit string left padded 0
ints['Tract_GEOID'] = ints['Tract_GEOID'].astype(str).apply(lambda x: x.zfill(11))

#Now subset ints for Tract_GEOID in ctracts2000 of df
ints = ints[ints['Tract_GEOID'].isin(df['ctracts2000'])]


##


# Now map df into 2020 tracts using the weights between 2000 and 2020 tracts
merged = pd.merge(df, ints, left_on='ctracts2000', right_on='Tract_GEOID', how='inner')

# Multiply specified variables by weight
for var in variables_to_add:
    merged[f'weighted_{var}'] = merged[var] * merged['Weight']

# Groupby 2020 munis and sum
agg_columns = {f'weighted_{var}': 'sum' for var in variables_to_add}
agg_columns.update({'Weight': 'sum'})
agg_df = merged.groupby('CENSUS_ID_PID6').agg(agg_columns).reset_index()

# Filter out rows with weight <0.75
agg_df = agg_df[agg_df['Weight'] >= 0.75]


# Now divide weighted variables by weight to get weighted average
for var in variables_to_add:
    agg_df[var] = agg_df[f'weighted_{var}'] / agg_df['Weight']

#Set index to CENSUS_ID_PID6
agg_df = agg_df.set_index('CENSUS_ID_PID6')

#Now just keep variabels to add columns
agg_df = agg_df[variables_to_add]


# Export
agg_df.to_csv(os.path.join(config['processed_data'], 'interim_data', 'Housing Elasticity.csv'))