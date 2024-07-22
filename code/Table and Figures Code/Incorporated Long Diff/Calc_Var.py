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

os.chdir('../../../')

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

year = '2020'

#Draw in all shape files for places and county subdivisions
base_shape_path = config['shape_files']

shape_path = os.path.join(base_shape_path,str(int(year) + 2))


var = 'Housing'

var_name = {
    'Housing': 'Housing Units',
    'Population': 'POPULATION',
}

block_var_name = {
    'Housing': {
        '2020': 'HOUSING20',
    },
    'Population': {
        '2020':'POP20',
    }

}

# Draw in housing data
place_data = pd.read_excel(os.path.join(config['raw_data'], 'Census Data', f'Housing_Place_{year}.xlsx'), index_col=0)
county_subdivision_data = pd.read_excel(os.path.join(config['raw_data'], 'Census Data', f'Housing_Cousub_{year}.xlsx'), index_col=0)

#%% Functions/mappings

#Define the states with overlapping governance of municipalities and townships
overlap_states = {
    'in': {'name': 'Indiana', 'fips': 18},
    'ct': {'name': 'Connecticut', 'fips': 9},
    'il': {'name': 'Illinois', 'fips': 17},
    'ks': {'name': 'Kansas', 'fips': 20},
    'mi': {'name': 'Michigan', 'fips': 26},
    'mn': {'name': 'Minnesota', 'fips': 27},
    'ms': {'name': 'Mississippi', 'fips': 28},
    'ne': {'name': 'Nebraska', 'fips': 31},
    'ny': {'name': 'New York', 'fips': 36},
    'oh': {'name': 'Ohio', 'fips': 39},
    'vt': {'name': 'Vermont', 'fips': 50}
}



#Manually remove select defunt (merged into other) munis/townships
to_remove = {}

to_remove['2022'] = [
    164921, #North Rich Township (Kansas) merged into Rich Township in 2017: https://en.wikipedia.org/wiki/North_Rich_Township,_Anderson_County,_Kansas
    176012, #Millican Texas was unincorporated: https://en.wikipedia.org/wiki/Millican,_Texas
    251374, #Peaster Texas is now unincorporated: https://en.wikipedia.org/wiki/Peaster,_Texas
    ]

to_remove['2012'] = [
15201010100000, #Covered bridge Indiana, couldn't confirm that this is an actual town
26201510100000, #Population of only 4, doesnt have shape file https://en.wikipedia.org/wiki/Friedenswald,_Missouri
26202510100000, #No shape file https://en.wikipedia.org/wiki/Grayson,_Missouri
34203810100000, # Foontana dam, no shape file, low population
]


#Function to get all fips place shape files for a state
def get_place_shapes(state_fips):
    path_municipalities = os.path.join(shape_path,'Places',f'tl_2022_{state_fips:02d}_place',f'tl_2022_{state_fips:02d}_place.shp')
    gdf_municipalities = gpd.read_file(path_municipalities)
    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(int)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities

#Function to get county subdivisions 
def get_cousub_shapes(state_fips):
    path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_2022_{state_fips:02d}_cousub',f'tl_2022_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub 

#Function to get block shape files
def get_block_shapes(state_fips):
    path_block = os.path.join(shape_path,'Blocks',f'tl_2022_{state_fips:02d}_tabblock20',f'tl_2022_{state_fips:02d}_tabblock20.shp')
    gdf_block = gpd.read_file(path_block)
    return gdf_block
            
#Match census of governments to shape data
def match_shapes(places,cousubs,bridge, state_govs):


    #Drop 'State' from bridge
    bridge = bridge.drop(columns = ['State'])

    #Merge into state_govs on CENSUS_ID_PID6
    state_govs = state_govs.merge(bridge, on = 'CENSUS_ID_PID6')

    #Make the type object for GEOID
    state_govs['GEOID'] = state_govs['GEOID'].astype(str)

    #Remove preceding 0's from GEOID in places
    places['GEOID'] = places['GEOID'].str.lstrip('0')

    #Now subset places for GEOID in state_govs
    places_filt = places[places['GEOID'].isin(state_govs['GEOID'])]

    #Remove preceding 0's from GEOID in cousubs
    cousubs['GEOID'] = cousubs['GEOID'].str.lstrip('0')

    #Then subset cousubs for GEOID in state_govs
    cousubs_filt = cousubs[cousubs['GEOID'].isin(state_govs['GEOID'])]

    #Get a list of the 'Geometry' column from both places_filt and cousubs_filt and append, call it matched
    matched = places_filt['geometry'].tolist() + cousubs_filt['geometry'].tolist()

    return matched
    
def get_block_intersections(matched_shapes, block_shapes):
    
    # Initialize a dictionary to store the percentage overlaps for each block
    block_overlaps = {}

    # Loop over all block shapes
    for _, block in block_shapes.iterrows():
        block_id = block['GEOID20']
        block_geom = block.geometry
        block_area = block_geom.area
        block_bounds = box(*block_geom.bounds)
        overlaps = 0

        # Loop over each local government shape
        for local_gov_geom in matched_shapes:
            local_gov_bounds = box(*local_gov_geom.bounds)

            # Check if bounding boxes intersect before calculating actual intersection
            if block_bounds.intersects(local_gov_bounds):
                # Calculate the intersection area
                intersection = block_geom.intersection(local_gov_geom)
                if not intersection.is_empty:
                    # Calculate the percentage overlap and add to the list if it's non-zero
                    overlap_percentage = (intersection.area / block_area) * 100
                    
                    #In practice almost every block is 100% within or 0% within a local gov but we set a cutoff at 50 
                    if overlap_percentage > 50:
                        overlaps = overlaps + 1

        # Store the list of overlaps in the dictionary with block_id as the key
        block_overlaps[block_id] = overlaps

    # Return the dictionary
    return block_overlaps

#Get the population adjustment factor for overlap state
def get_adjustment_factor(matched_shapes,state_fips):
    
    #Get all block shape files
    block_shapes = get_block_shapes(state_fips)
    
    #Get block intersections
    block_counts = get_block_intersections(matched_shapes,block_shapes)

    #Identify blocks with multiple matches and count how many extra matches they have
    processed_block_counts = {k: v - 1 for k, v in block_counts.items() if v > 1}
    
    #Now count the population adjustment factor
    total_sum = 0

    # Iterate over the processed block counts
    for block_id, count in processed_block_counts.items():
        # Find the row in block_shapes that matches the block ID
        block_row = block_shapes[block_shapes['GEOID'+year[-2:]] == block_id]
    
        # Check if there's a matching row
        if not block_row.empty:
            # Multiply the count by the var and add to total
            total_sum += count * block_row.iloc[0][block_var_name[var][year]]
            
    return total_sum

#Get the percent of the population covered by local governments
def get_per_pop(state_fips,state_govs,matched_shapes):

    #1. Sum the population of local govs
    local_gov_pop = state_govs[var_name[var]].sum()
    
    #2. If an overlap state then get adjustment factor
    if any(state_info['fips'] == state_fips for state_info in overlap_states.values()):
        adjustment_factor = get_adjustment_factor(matched_shapes,state_fips)
    else:
        adjustment_factor = 0

    #3. Get the adjusted populatin
    adjusted = local_gov_pop - adjustment_factor
    
    return local_gov_pop, adjustment_factor, adjusted

#%%Main code

def main(sample,state_fips):

    results = pd.Series([])
    
    #Get set of all local governments
    state_govs = sample[sample['FIPS_STATE'] == state_fips]
    
    #Get fips places
    places = get_place_shapes(state_fips)
    
    #Get county subvidisions
    cousubs = get_cousub_shapes(state_fips)

    #Load in bridge
    bridge = pd.read_excel(os.path.join(config['processed_data'],'Shape File Bridges',f'cog_{str(int(year)+2)}_bridge.xlsx'))

    # If var == 'Housing' then merge in housing data
    if var == 'Housing':
        state_govs = merge_housing(state_govs, places,cousubs,bridge)
    
    #Match set of local governments with shape files
    matched_shapes = match_shapes(places,cousubs,bridge, state_govs)
    
    #Get the % population by local govs
    results.loc['Unadjusted'], results.loc['Adjustment Factor'], results.loc['Adjusted'] = get_per_pop(state_fips,state_govs,matched_shapes)
    
    #Get the number of townships
    results.loc['Num Townships'] = len(state_govs[state_govs['UNIT_TYPE'] == '3 - TOWNSHIP'].index)
    
    #Get the number of munis
    results.loc['Num Munis'] = len(state_govs[state_govs['UNIT_TYPE'] == '2 - MUNICIPAL'].index)
    
    return results



def merge_housing(state_govs,places,cousubs,bridge):

    '''
    Need to update fips codes based on shape files
    '''

    #Merge in GEOID into state_govs with bridge
    merged = state_govs.merge(bridge, on = 'CENSUS_ID_PID6')

    #Make GEOID str
    merged['GEOID'] = merged['GEOID'].astype(str)

    #Remove preceding 0's from GEOID in places
    places['GEOID'] = places['GEOID'].str.lstrip('0')

    #Now merge in PLACEFP from places on GEOID
    merged1 = merged.merge(places[['GEOID','PLACEFP']], on = 'GEOID', how = 'left')

    #Remove preceding 0's from GEOID in cousubs
    cousubs['GEOID'] = cousubs['GEOID'].str.lstrip('0')

    #Now merge in COUSUBFP from cousubs on GEOID
    merged2 = merged1.merge(cousubs[['GEOID','COUNTYFP','COUSUBFP']], on = 'GEOID', how = 'left')

    #If state if connecticut then we need to bridge the county fips
    if state_govs['FIPS_STATE'].iloc[0] == 9:
        ct_bridge = pd.read_excel(os.path.join(config['raw_data'],'GEO Crosswalks','CT_Bridge.xlsx'))
        #Make a dictionary to map column 'CENSUS_ID_PID6' to 'FIPS_COUNTY'
        ct_bridge_dict = dict(zip(ct_bridge['CENSUS_ID_PID6'],ct_bridge['FIPS_COUNTY']))

        #Now update COUNTYFP accordingly
        merged2['COUNTYFP'] = merged2['CENSUS_ID_PID6'].map(ct_bridge_dict)

    #Merge in place_data on 'state' and 'place'
    merged3 = pd.merge(merged2, place_data, left_on = ['FIPS_STATE', 'PLACEFP'], right_on = ['state','place'], how = 'left')

    #Merge in county_subdivision_data on 'state', 'county', and 'county subdivision'
    merged4 = pd.merge(merged3, county_subdivision_data, left_on = ['FIPS_STATE', 'COUNTYFP', 'COUSUBFP'], right_on = ['state','county','county subdivision'], how = 'left')

    #Making housing units column merging x and y and drop other merge vars
    merged4['Housing Units'] = merged4['Housing Units_x'].fillna(merged4['Housing Units_y'])
    merged4 = merged4.drop(columns = ['Housing Units_x', 'Housing Units_y','state_x','place','state_y','county','county subdivision','COUNTYFP','COUSUBFP','PLACEFP','GEOID'])

    #Find rows where 'Housing Units' is null and missing rows dataframe
    missing_rows = merged4[merged4['Housing Units'].isnull()]

    #If any missing rows then print the sum of the population of the missing rows
    if not missing_rows.empty:
        print(missing_rows['POPULATION'].sum())

    return merged4[merged4['Housing Units'].notnull()]

# Load the sample data
sample = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))

#Filter out defunct munis/townships
sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove[str(int(year)+2)])]


#Filter out counties
sample = sample[sample['UNIT_TYPE'] != '1 - COUNTY']



all_states = sample['FIPS_STATE'].unique()

# Define the sample to loop over
samples = {
    'cog_sample': sample
}

# Create a list of all task combinations
task_combinations = [(sample_name, state) for sample_name in samples for state in all_states]
# Check existing files and remove completed tasks from the list
if var == 'Population':
    pop_coverage_folder = os.path.join(config['processed_data'], 'pop_coverage'+year)
else:
    pop_coverage_folder = os.path.join(config['processed_data'], 'housing_coverage'+year)
os.makedirs(pop_coverage_folder, exist_ok=True)

completed_tasks = []
for filename in os.listdir(pop_coverage_folder):
    if filename.endswith('.pkl'):
        sample_name, state = filename[:-4].split('#')  
        completed_tasks.append((sample_name, state))

task_combinations = [task for task in task_combinations if task not in completed_tasks]

# SLURM environment setup
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
num_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

# Assign tasks to this node
tasks_per_node = len(task_combinations) // num_nodes
extra_tasks = len(task_combinations) % num_nodes

if task_id < extra_tasks:
    start_idx = task_id * (tasks_per_node + 1)
    end_idx = start_idx + tasks_per_node + 1
else:
    start_idx = task_id * tasks_per_node + extra_tasks
    end_idx = start_idx + tasks_per_node

tasks_for_node = task_combinations[start_idx:end_idx]

# Process each task and save the results
for sample_name, state in tasks_for_node:
    print(state)
    result = main(samples[sample_name], state)

    filename = os.path.join(pop_coverage_folder, f'{sample_name}#{state}.pkl')
    
    # Save result as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

