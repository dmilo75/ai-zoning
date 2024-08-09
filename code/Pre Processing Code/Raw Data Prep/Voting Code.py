import pandas as pd
import os
import yaml
import helper_functions as hf

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

#Load in voting block data
voting_blocks = pd.read_csv(os.path.join(raw_path,'2021blockgroupvoting.csv'))

#Load in geo bridges
cousub_bridge = pd.read_csv(os.path.join(raw_path,'Geo Crosswalks','geocorr2022_censusblock_cousub.csv'), encoding='latin1')
place_bridge = pd.read_csv(os.path.join(raw_path,'Geo Crosswalks','geocorr2022_censusblock_place.csv'), encoding='latin1')

#Drop first row which is just a description
cousub_bridge = cousub_bridge.drop(0)
place_bridge = place_bridge.drop(0)

#In place bridge drop place value of 99999
place_bridge = place_bridge[place_bridge['place'] != '99999']

'''
Roadmap
1. Merge voting blocks with bridges
2. Define function to make new weighted values 
3. Get county subdivision and place level data
4. merge into sample and export 

'''
# Function to pad the tract numbers to six digits
def pad_tract(tract):
    if pd.isna(tract):
        return '000000'
    tract_str = str(tract)
    if '.' in tract_str:
        parts = tract_str.split('.')
        part1 = parts[0].zfill(4)
        part2 = parts[1].zfill(2)
    else:
        part1 = tract_str.zfill(4)
        part2 = '00'
    return part1 + part2

def merge_data(bridge, voting_df,var,aggvars):
    # Ensure the tract column is treated as a string
    bridge['tract'] = bridge['tract'].astype(str)

    # Apply the function to the tract column
    bridge['tract_padded'] = bridge['tract'].apply(pad_tract)

    # Create BLOCKGROUP_GEOID in bridge
    bridge['BLOCKGROUP_GEOID'] = (bridge['county'].astype(str) +
                                  bridge['tract_padded'] +
                                  bridge['blockgroup'].astype(str))

    # Remove preceding 0's
    bridge['BLOCKGROUP_GEOID'] = bridge['BLOCKGROUP_GEOID'].apply(lambda x: x.lstrip('0'))

    # Ensure BLOCKGROUP_GEOID in voting_blocks is a string
    voting_df['BLOCKGROUP_GEOID'] = voting_df['BLOCKGROUP_GEOID'].astype(str)

    # Merge voting_blocks with bridge on BLOCKGROUP_GEOID
    merged_data = pd.merge(voting_df, bridge, on='BLOCKGROUP_GEOID', how='left')

    # Drop rows without an 'afact' as these are census block groups without a place/cousub
    merged_data = merged_data.dropna(subset=['afact'])

    # Convert relevant columns to numeric, forcing errors to NaN
    for col in ['REP', 'DEM', 'LIB', 'OTH','afact']:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

    # Multiply the columns 'REP', 'DEM', 'LIB', and 'OTH' by 'afact'
    for col in ['REP', 'DEM', 'LIB', 'OTH']:
        merged_data[col] = merged_data[col] * merged_data['afact']

    #Make the var column an int
    merged_data[var] = merged_data[var].astype('int64')

    # Group by state, county, and place and sum the columns
    aggregated_data = merged_data.groupby(['STATE']+aggvars).agg({
        'REP': 'sum',
        'DEM': 'sum',
        'LIB': 'sum',
        'OTH': 'sum',
    }).reset_index()

    #Make total votes column
    aggregated_data['total_votes'] = aggregated_data['REP'] + aggregated_data['DEM'] + aggregated_data['LIB'] + aggregated_data['OTH']

    #Now make percent democrat column
    aggregated_data['percent_democrat'] = aggregated_data['DEM'] / aggregated_data['total_votes']

    #Make state lowercase
    aggregated_data['STATE'] = aggregated_data['STATE'].apply(lambda x: x.lower())

    #Drop columns used to make percent democrat
    aggregated_data = aggregated_data.drop(columns = ['REP','DEM','LIB','OTH','total_votes'])

    return aggregated_data


#Process cousub data
cousub_data = merge_data(cousub_bridge, voting_blocks.copy(), 'cousub20',['county','cousub20'])

#Make a new county column with only last three digits of county
cousub_data['county'] = cousub_data['county'].apply(lambda x: int(str(x)[-3:]))

#Process place data
place_data = merge_data(place_bridge, voting_blocks.copy(), 'place',['place'])

#Merge into sample
sample_merged = pd.merge(sample,cousub_data,left_on = ['STATE','FIPS_COUNTY_ADJ','FIPS_PLACE'],right_on=['STATE','county','cousub20'],how='left')

#Merge place data
sample_merged2 = pd.merge(sample_merged,place_data,left_on = ['STATE','FIPS_PLACE'],right_on=['STATE','place'],how='left')

#Find the duplicate state place pair in sample_merged2
duplicates = sample_merged2[sample_merged2.duplicated(subset = ['STATE','FIPS_PLACE'])]

sample_merged2['percent_democrat'] = sample_merged2['percent_democrat_x'].fillna(sample_merged2['percent_democrat_y'])

final_df = sample_merged2[['CENSUS_ID_PID6','percent_democrat']]



#Export in processed data
final_df.to_excel(os.path.join(data_path,'interim_data','Voting Data.xlsx'))

