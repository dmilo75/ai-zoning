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

# Load the data
bridge = pd.read_csv(os.path.join(data_path,'School District Geobridge','muni_school_district_bridge.csv'))



# Calculate coverage percentages
bridge['Muni_Coverage'] = bridge['Intersection_Area'] / bridge['Muni_Area']
bridge['School_Coverage'] = bridge['Intersection_Area'] / bridge['School_Area']

# Calculate the product of the two coverage percentages
bridge['Coverage_Product'] = bridge['Muni_Coverage'] * bridge['School_Coverage']

# Normalize the weights within each municipality
bridge['Weight'] = bridge.groupby('CENSUS_ID_PID6')['Coverage_Product'].transform(
    lambda x: x / x.sum()
)

#Filter out weight <= .01
bridge = bridge[bridge['Weight'] > .01]

#Now renormalize
bridge['Weight'] = bridge.groupby('CENSUS_ID_PID6')['Coverage_Product'].transform(
    lambda x: x / x.sum()
)

#Load in sample data
sample_data = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"))

#Subset bridge for CENSUS_ID_PID6 in sample data CENSUS_ID_PID6
bridge = bridge[bridge['CENSUS_ID_PID6'].isin(sample_data['CENSUS_ID_PID6'])]

#Keep CENSUS_ID_PID6, School_GEOID and Weight
bridge = bridge[['CENSUS_ID_PID6','School_GEOID','Weight']]

#Now draw in stanford data on testing
stanford_data = pd.read_csv(os.path.join(raw_path,'Stanford Ed Opportunity Project','Opportunity','seda_geodist_poolsub_cs_5.0_updated_20240319.csv'))

#Filter for subcat and subgroup both equal to 'all'
stanford_data = stanford_data[(stanford_data['subcat'] == 'all') & (stanford_data['subgroup'] == 'all')]

#Get relevant variables
rel_vars = ['cs_mn_avg_mth_ol','cs_mn_avg_rla_ol','cs_mn_grd_mth_ol','cs_mn_grd_rla_ol']

#Keep relevant variables and sedalea
stanford_data = stanford_data[['sedalea'] + rel_vars]

#Now draw in standford data on segregation
segregation_data = pd.read_csv(os.path.join(raw_path, 'Stanford Ed Opportunity Project', 'Segregation', 'School_seg_geolea_1.0.csv'))

#Filter for year in 2008 to 2019 to align with opportunity dataset years
segregation_data = segregation_data[segregation_data['year'].between(2008,2019)]

#Relevant variables
rel_vars_seg = ['perflu','perfrl']

#Keep relevant variables and geoleaid
segregation_data = segregation_data[['geoleaid'] + rel_vars_seg]

#Groupby geoleaid and take the mean
segregation_data = segregation_data.groupby('geoleaid').mean().reset_index()

#Makea merged dataframe, outer merge, of segregation_data and stanford_data on geoleaid and sedalea
merged_data = pd.merge(stanford_data, segregation_data, how = 'outer', left_on = 'sedalea', right_on = 'geoleaid')

#Make geoid which is sedalea fillna with geoleaid
merged_data['School_GEOID'] = merged_data['geoleaid'].fillna(merged_data['sedalea'])

#Drop the other geoids
merged_data = merged_data.drop(columns = ['sedalea','geoleaid'])

# Now merge with bridge
merged_bridge = pd.merge(merged_data, bridge, how='inner', on='School_GEOID')

# Multiply each var in rel_vars and rel_vars_seg by Weight
for var in rel_vars + rel_vars_seg:
    merged_bridge[var] = merged_bridge[var] * merged_bridge['Weight']

# Now group by CENSUS_ID_PID6 and sum, including the count of rows
final_df = merged_bridge.groupby('CENSUS_ID_PID6').agg({
    **{var: 'sum' for var in rel_vars + rel_vars_seg},
    'School_GEOID': 'count'
}).rename(columns={'School_GEOID': 'Num_Sch_Dists'})

# Drop Weight as it's no longer needed
final_df = final_df.drop(columns=['Weight'], errors='ignore')

#Export
final_df.to_excel(os.path.join(data_path, 'interim_data', 'Stanford Ed Data.xlsx'))

