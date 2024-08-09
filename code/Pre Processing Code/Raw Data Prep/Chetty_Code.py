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
Import the data
Map it to 2020 tracts
Map 2020 tracts to 2020 fips place and county subdivision codes
Merge the data with sample
export
'''

# Import the data
data = pd.read_csv(os.path.join(raw_path, 'Chetty Data', 'tract_covariates.csv'))

#Import the tract bridge 2010 to 2020
tract_bridge = pd.read_csv(os.path.join(raw_path, 'Chetty Data', 'us_tract_2010_2020_crosswalk.csv'))

tract_bridge.columns


## Going from 2010 to 2020


import pandas as pd

variables_to_transform = [col for col in data.columns if col not in ['state', 'county', 'tract', 'cz', 'czname']]

# Assuming your data frame is named 'tract_bridge'
tract_bridge_long = pd.melt(tract_bridge,
                            id_vars=['tract_2020'],
                            value_vars=['tract_2010_1', 'tract_2010_2', 'tract_2010_3', 'tract_2010_4', 'tract_2010_5'],
                            var_name='tract_2010_col',
                            value_name='tract_2010')

tract_bridge_long['weight'] = pd.melt(tract_bridge,
                                      id_vars=['tract_2020'],
                                      value_vars=['wgt_1', 'wgt_2', 'wgt_3', 'wgt_4', 'wgt_5'],
                                      var_name='weight_col',
                                      value_name='weight')['weight']

tract_bridge_long = tract_bridge_long[['tract_2020', 'tract_2010', 'weight']]

#Drop na
tract_bridge_long = tract_bridge_long.dropna()

#Ensure tract_2010 is string without any decimals
tract_bridge_long['tract_2010'] = tract_bridge_long['tract_2010'].astype(str).apply(lambda x: x.split('.')[0])

#Zero pad tract_2010 so that its 11 digits
tract_bridge_long['tract_2010'] = tract_bridge_long['tract_2010'].apply(lambda x: x.zfill(11))

#Ensure tract is string
data['tract'] = data['tract'].astype(str)

#Zero pad tract so that its six digits
data['tract'] = data['tract'].apply(lambda x: x.zfill(6))

#Now add a zero padded two digit string state and three digit padded string county to the front
data['state'] = data['state'].apply(lambda x: str(x).zfill(2))
data['county'] = data['county'].apply(lambda x: str(x).zfill(3))
data['tract'] = data['state'] + data['county'] + data['tract']


#Merge in data on tract_2010
data_merged = tract_bridge_long.merge(data, left_on = 'tract_2010', right_on = 'tract', how = 'left')

#Multiply variables in variables to trasnform by weight
for col in variables_to_transform:
    data_merged[col] = data_merged[col] * data_merged['weight']

agg = data_merged.groupby('tract_2020').agg({
    'state': 'first',
    'county': 'first',
    'cz': 'first',
    'czname': 'first',
    **{col: 'sum' for col in data_merged.columns if col not in ['tract_2020', 'state', 'county', 'cz', 'czname']}
})

agg = agg.reset_index()

#vars to keep
vars_to_keep = ['tract_2020','job_density_2013','jobs_highpay_5mi_2015','jobs_total_5mi_2015','gsmn_math_g3_2013']

agg = agg[vars_to_keep]

#Merge in kfr_weighted from tract_bridge
agg = agg.merge(tract_bridge[['tract_2020','kfr_weighted']], on = 'tract_2020', how = 'left')

##Going to place and county subdivisions

#Import the place and county subdivision bridge
place_bridge = pd.read_csv(os.path.join(raw_path, 'Geo Crosswalks', 'geocorr2022_tract_place.csv'), encoding='latin1', low_memory=False)
county_sub_bridge = pd.read_csv(os.path.join(raw_path, 'Geo Crosswalks', 'geocorr2022_tract_cousub.csv'), encoding='latin1', low_memory=False)

def bridge_data(df,bridge, id_vars):

    #In df make tract_2020 a string that is zero padded for 11 chars
    df['tract_2020'] = df['tract_2020'].astype(str).apply(lambda x: x.zfill(11))

    #Drop the first row of the bridge
    bridge = bridge.iloc[1:]

    #Make a proper tract variable in string format
    bridge['tract'] = bridge['county']+ bridge['tract'].apply(lambda x: x.replace('.',''))

    #Drop rows in bridge with last var in id_vars as 99999
    bridge = bridge[bridge[id_vars[-1]] != '99999']

    #Merge in data on tract
    data_merged = bridge.merge(df, left_on = 'tract', right_on = 'tract_2020', how = 'left')

    index_vars = ['job_density_2013',
       'jobs_highpay_5mi_2015', 'jobs_total_5mi_2015', 'gsmn_math_g3_2013',
       'kfr_weighted']

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

place_bridged = bridge_data(agg,place_bridge, ['state','place'])
cousub_bridged = bridge_data(agg,county_sub_bridge, ['county','cousub20'])

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
sample_merged = sample_merged[['CENSUS_ID_PID6','job_density_2013','jobs_highpay_5mi_2015','jobs_total_5mi_2015','gsmn_math_g3_2013','kfr_weighted']]

#Export
sample_merged.to_excel(os.path.join(data_path, 'interim_data', 'Chetty_Data.xlsx'))








