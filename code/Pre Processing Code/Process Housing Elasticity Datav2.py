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

'''
- Need to identify which variables we need
- Then need to use bridge to get to fips place and census county subdivison level
'''

'''
We will use ctracts2000 and gamma01b_units_FMM

ctracts2000 is the census tract identifier for 2000

gamma01b_units_FMM is the estimated housing unit supply elasticity using the finite mixture model (FMM)
approach with the quadratic specification (what 'b' means) and the 2001 (what '01' means) tract developed fraction measure. 
'''

#Get the relevant columns
df = df[['ctracts2000','gamma01b_units_FMM']]


#Ensure ctracts2000 is an 11 digit string left padded 0
df['ctracts2000'] = (df['ctracts2000'].astype(str))
df['ctracts2000'] = (df['ctracts2000'].apply(lambda x: x.zfill(11)))

'''
Bridge census tract to census tract 2020
'''

#Read in 2000 to 2010 bridge
'''
From Brown: https://s4.ad.brown.edu/Projects/Diversity/Researcher/Bridging.htm
'''
bridge_00_10 = pd.read_csv(os.path.join(config['raw_data'],'Geo Crosswalks','crosswalk_2000_2010.csv'))

#Only select relvant columns
bridge_00_10 = bridge_00_10[['trtid00','trtid10','weight']]

#Rename columns
bridge_00_10.columns = ['ctracts2000','ctracts2010','weight_2010']

#Read in 2010 to 2020 bridge
'''
From HUD: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
'''
bridge_10_20 = pd.read_excel(os.path.join(config['raw_data'],'Geo Crosswalks','CENSUS_TRACT_CROSSWALK_2010_to_2020_2010.xlsx'))

#Sort by GEOID_2020
bridge_10_20 = bridge_10_20.sort_values(by='GEOID_2020')

#Only select relvant columns
bridge_10_20 = bridge_10_20[['GEOID_2010','GEOID_2020','RES_RATIO']]

#Rename columns
bridge_10_20.columns = ['ctracts2010','ctracts2020','weight_2020']

#Merge the bridges
bridge = pd.merge(bridge_00_10,bridge_10_20,on='ctracts2010')

#Get aggregate weight column
bridge['weight'] = bridge['weight_2010'] * bridge['weight_2020']

#Groupby both 2000 and 2020 tracts and sum weight
bridge = bridge.groupby(['ctracts2000','ctracts2020']).agg({'weight':'sum'}).reset_index()

#Ensure ctracts2000 is an 11 digit string left padded 0
bridge['ctracts2000'] = (bridge['ctracts2000'].astype(str))
bridge['ctracts2000'] = (bridge['ctracts2000'].apply(lambda x: x.zfill(11)))

#Groupby ctracts2020 and see the sum of weights
test = bridge.groupby('ctracts2020').sum().reset_index()


#Now map df into 2020 tracts using the weights between 2000 and 2020 tracts
df = pd.merge(df,bridge,on='ctracts2000', how = 'left')

#Drop rows with weight = 0
df = df[df['weight'] != 0]

# Step 1: Multiply gamma01b_units_FMM by weight to get weighted values
df['weighted_gamma'] = df['gamma01b_units_FMM'] * df['weight']

# Step 2: Aggregate (sum) the weighted values and weights separately by ctracts2020
agg_df = df.groupby('ctracts2020').agg({'weighted_gamma': 'sum', 'weight': 'sum'}).reset_index()

# Step 3: Calculate the weighted average of gamma01b_units_FMM for each 2020 census tract
agg_df['weighted_avg_gamma'] = agg_df['weighted_gamma'] / agg_df['weight']

# Drop the intermediate sum columns if they are no longer needed
agg_df = agg_df.drop(columns=['weighted_gamma', 'weight'])

def bridge_data(df,bridge, id_vars):

    #In df make tract_2020 a string that is zero padded for 11 chars
    df['tract_2020'] = df['tract_2020'].astype(str).apply(lambda x: x.split('.')[0]).apply(lambda x: x.zfill(11))

    #Drop the first row of the bridge
    bridge = bridge.iloc[1:]

    #Make a proper tract variable in string format
    bridge['tract'] = bridge['county']+ bridge['tract'].apply(lambda x: x.replace('.',''))

    #Drop rows in bridge with last var in id_vars as 99999
    bridge = bridge[bridge[id_vars[-1]] != '99999']

    #Merge in data on tract
    data_merged = bridge.merge(df, left_on = 'tract', right_on = 'tract_2020', how = 'inner')

    index_vars = ['weighted_avg_gamma']

    #Make pop20 an int
    data_merged['pop20'] = data_merged['pop20'].astype(int)

    # Apply population weighting to all index variables
    for var in index_vars:
        weighted_var = f'Weighted_{var}'
        data_merged[weighted_var] = data_merged[var] * data_merged['pop20']

    # Aggregate weighted index variables to the place level
    weighted_vars = [f'Weighted_{var}' for var in index_vars]
    agg_index_vars = {var: 'sum' for var in weighted_vars}
    agg_pop20 = {'pop20': 'sum'}

    aggregated_data = data_merged.groupby(id_vars).agg({**agg_index_vars, **agg_pop20}).reset_index()

    # Calculate final index values
    for var in index_vars:
        weighted_var = f'Weighted_{var}'
        aggregated_data[f'Final_{var}'] = aggregated_data[weighted_var] / aggregated_data['pop20']

    #Just keep variables with 'Final' and id_vars
    aggregated_data = aggregated_data[[col for col in aggregated_data.columns if 'Final' in col] + id_vars]

    #Remove 'Final_' from the columns
    aggregated_data.columns = [col.replace('Final_','') for col in aggregated_data.columns]

    return aggregated_data

#Import the place and county subdivision bridge
place_bridge = pd.read_csv(os.path.join(raw_path, 'Geo Crosswalks', 'geocorr2022_tract_place.csv'), encoding='latin1', low_memory=False)
county_sub_bridge = pd.read_csv(os.path.join(raw_path, 'Geo Crosswalks', 'geocorr2022_tract_cousub.csv'), encoding='latin1', low_memory=False)

place_bridged = bridge_data(agg_df,place_bridge, ['state','place'])
cousub_bridged = bridge_data(agg_df,county_sub_bridge, ['county','cousub20'])

#For cousub_bridged, take the first two digits of county and make the state
cousub_bridged['state'] = cousub_bridged['county'].apply(lambda x: x[:2])
#Now make the last three digits of county the new county var
cousub_bridged['county'] = cousub_bridged['county'].apply(lambda x: x[2:])

#Merge into sample
#Make state and place ints
place_bridged['state'] = place_bridged['state'].astype(int)
place_bridged['place'] = place_bridged['place'].astype(int)

sample_munis = sample[sample['UNIT_TYPE'] == '2 - MUNICIPAL']
munis_merged = pd.merge(sample_munis, place_bridged, left_on = ['FIPS_STATE','FIPS_PLACE'], right_on = ['state','place'], how = 'inner')
#Print merge stats
percent_merged = len(munis_merged)/len(sample_munis)
print(f"Percent of Muni data merged: {percent_merged}")

#Make cousub_bridged merge vars ints
cousub_bridged['state'] = cousub_bridged['state'].astype(int)
cousub_bridged['county'] = cousub_bridged['county'].astype(int)
cousub_bridged['cousub20'] = cousub_bridged['cousub20'].astype(int)

sample_townships = sample[sample['UNIT_TYPE'] == '3 - TOWNSHIP']
townships_merged = pd.merge(sample_townships, cousub_bridged, left_on = ['FIPS_STATE','FIPS_COUNTY','FIPS_PLACE'], right_on = ['state','county','cousub20'], how = 'inner')
#merge stats
percent_merged = len(townships_merged)/len(sample_townships)
print(f"Percent of Township data merged: {percent_merged}")

#Concat the two
sample_merged = pd.concat([munis_merged,townships_merged])

#Just keep census id and the final vars
sample_merged = sample_merged[['CENSUS_ID_PID6','weighted_avg_gamma']]

#Export
sample_merged.to_csv(os.path.join(config['processed_data'], 'interim_data', 'Housing Elasticity.csv'))


