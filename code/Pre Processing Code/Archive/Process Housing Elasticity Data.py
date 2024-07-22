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

#Make first column an int
df['ctracts2000'] = df['ctracts2000'].astype('int64')

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

#Make ctracts2000 an int
bridge['ctracts2000'] = bridge['ctracts2000'].astype('int64')

#Now map df into 2020 tracts using the weights between 2000 and 2020 tracts
df = pd.merge(df,bridge,on='ctracts2000')

test = df.groupby('ctracts2000').sum().reset_index()

#Sort by weight
test = test.sort_values(by='weight',ascending=False)

#Take subset of df with ctracts2020 == 1001020100
tests = df[df['ctracts2020'] == 36005006301]

'''
Need to double check this, should have for each 2020 tract the weights being the share of the 2000 tract that it represents
'''

# Step 1: Multiply gamma01b_units_FMM by weight to get weighted values
df['weighted_gamma'] = df['gamma01b_units_FMM'] * df['weight']

# Step 2: Aggregate (sum) the weighted values and weights separately by ctracts2020
agg_df = df.groupby('ctracts2020').agg({'weighted_gamma': 'sum', 'weight': 'sum'}).reset_index()

# Step 3: Calculate the weighted average of gamma01b_units_FMM for each 2020 census tract
agg_df['weighted_avg_gamma'] = agg_df['weighted_gamma'] / agg_df['weight']

# Drop the intermediate sum columns if they are no longer needed
agg_df = agg_df.drop(columns=['weighted_gamma', 'weight'])

#Now we need to convert 2020 tracts to fips places and county subdivisions
geocorr = pd.read_csv(os.path.join(config['raw_data'],'Geo Crosswalks','geocorr2022_2407404376.csv'), encoding='ISO-8859-1')

#Delete first row
geocorr = geocorr.iloc[1:]

#Make a ctract2020 column by appending 'county' and 'tract' but removing the decimal
geocorr['ctracts2020'] = geocorr.apply(lambda x: str(x['county'])+str(x['tract']).replace('.',''),axis=1)

#Now make int64
geocorr['ctracts2020'] = geocorr['ctracts2020'].astype('int64')

#Now just take the relevant columns
geocorr = geocorr[['ctracts2020','place','county','afact','state']]

#Make 'afact' a float
geocorr['afact'] = geocorr['afact'].astype('float')

#Merge with agg_df
full_df = pd.merge(agg_df,geocorr,on='ctracts2020')

#Replace county column by removing first two characters and then making int
full_df['county'] = full_df['county'].apply(lambda x: int(str(x)[2:]))

#Sort by place and cousub
full_df = full_df.sort_values(by=['state','county','place'])

# Step 1: Calculate the product of weighted_avg_gamma and afact
full_df['gamma_afact_product'] = full_df['weighted_avg_gamma'] * full_df['afact']

#Ensure state, county, place, cousub20 are all type int
full_df['state'] = full_df['state'].astype('int64')
full_df['county'] = full_df['county'].astype('int64')
full_df['place'] = full_df['place'].astype('int64')


# Step 2: Aggregate the sum of gamma_afact_product and afact by the specified levels
agg = full_df.groupby(['state', 'county', 'place']).agg(
    weighted_sum_gamma=('gamma_afact_product', 'sum'),
    sum_afact=('afact', 'sum')
).reset_index()

#Sort agg by place cousub20
agg = agg.sort_values(by=['state','county','place'])

# Step 3: Calculate the weighted average of weighted_avg_gamma
agg['weighted_avg_gamma'] = agg['weighted_sum_gamma'] / agg['sum_afact']

# Step 4: Filter rows where the sum of afact is at least .99
filtered_agg = agg[agg['sum_afact'] >= 0.99]

#Sort filtered_agg by place cousub20
filtered_agg = filtered_agg.sort_values(by=['state','county','place'])

#Merge in to sample on state, county, place
merged_sample = pd.merge(sample,filtered_agg,left_on = ['FIPS_STATE','FIPS_COUNTY','FIPS_PLACE'],right_on=['state','county','place'],how='inner')

#Just keep new columns and CENSUS_ID_PID6
merged_sample = merged_sample[['CENSUS_ID_PID6','weighted_avg_gamma']]

#Export in processed data
merged_sample.to_csv(os.path.join(config['processed_data'],'interim_data','Housing Elasticity.csv'),index=False)



