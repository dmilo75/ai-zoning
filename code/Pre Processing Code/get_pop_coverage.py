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

#Load config file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
#Draw in all shape files for places and county subdivisions
shape_path = config['shape_files']

#Load in state shape file
path_states = os.path.join(shape_path,'States','tl_2022_us_state.shp')
state_shapes = gpd.read_file(path_states)
state_shapes['STATEFP'] = state_shapes['STATEFP'].astype(int)

#Load in state population
state_pop = pd.read_csv(os.path.join(config['raw_data'],'2022 Population Data','state_population.csv'))

#%% Functions/mappings

#Manual mapping of unique ids between census of government and shape files for munis that don't match
#We map PID6 IDs to Geoids 
manual_mapping = {
    101399:'0812910', #Central City Colorado
    207643:'0915061030', #Town of Pomfret, Connecticut
    189810:'1275641',#Weeki Wachee, Florida
    161834:'1303440',#Athens, Georgia
    183701:'1500390810',#Honolulu, Hawaii 
    162945:'1706765240',#Rocky Run-Wilcox, Illinois
    164742:'2028412',#Unified gov of Greeley County, Kansas
    165866:'2148006',#Metro gov of louisville - jefferson county, Kentucky
    224809:'2527060',#Greenfield, Massachussetts 
    128148:'2502156000',#Town of Randolph, Massachussetts 
    166545:'2502308130',#Town of Bridgewater, Massachussetts 
    186122:'2713708',#City of Credit River, Minnesota
    169115:'2904248', #village of Bellerive, Missouri
    108124:'3011397',#CITY AND COUNTY OF BUTTE-SILVER BOW, Montana
    171132:'3727324',#Village of Grandfather, North Carolina
    175728:'4752006',#Metro of nashville-davison, Tennessee
    176097:'4821310',#Corral City, Texas (Renamed to Draper)
    113699:'4873493',#City of Peco, Taxes
    248439:'5589550',#Yorkville, Wisconsin
    }

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

def match(row, which, places,cousubs):
    if which == 'Muni':
        # Unique ID for 'Muni'
        uid = row['FIPS_PLACE']
        # Match into shape files
        selected = places[places['PLACEFP'] == uid]
    else:  # 'Township'
        # Unique ID and county for 'Township'
        uid = row['FIPS_PLACE']
        county = row['FIPS_COUNTY']
        # Match into shape files
        selected = cousubs[(cousubs['COUSUBFP'] == uid) & (cousubs['COUNTYFP'] == county)]

    if len(selected) == 1:
        return selected.iloc[0]['geometry']
    else:
        try:
            geoid = manual_mapping[row['CENSUS_ID_PID6']]
            if (which == 'Township') or (row['UNIT_NAME'] == 'honolulu'): #Honoloulou is exception
                return cousubs[cousubs['GEOID'] == geoid].iloc[0]['geometry']
            if which == 'Muni':    
                return places[places['GEOID'] == geoid].iloc[0]['geometry']
        except:
            print(row['UNIT_NAME'])
            print(row)
            print(which)
            raise ValueError("Not unique match")
            
#Match census of governments to shape data
def match_shapes(places,cousubs,state_govs):
    
    matched = []
    
    for index, row in state_govs.iterrows():
        #Match munis with places
        if row['UNIT_TYPE'] == '2 - MUNICIPAL':
            matched.append(match(row,'Muni',places,cousubs))
            
        #Match townships with county subs
        elif row['UNIT_TYPE'] == '3 - TOWNSHIP':
            matched.append(match(row,'Township',places,cousubs))
            
        #Otherwise, county gov and skip
        else:
            continue
    
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
        block_row = block_shapes[block_shapes['GEOID20'] == block_id]
    
        # Check if there's a matching row
        if not block_row.empty:
            # Multiply the count by the population in 'POP20' and add to total
            total_sum += count * block_row.iloc[0]['POP20']
            
    return total_sum

#Get the percent of the population covered by local governments
def get_per_pop(state_fips,state_pop,state_govs,matched_shapes):
    
    #1. Get the state population
    pop = state_pop[state_pop['STATE'] == state_fips].iloc[0]['POPESTIMATE2021']
    
    #2. Sum the population of local govs
    local_gov_pop = state_govs['POPULATION'].sum()
    
    #3. If an overlap state then get adjustment factor
    if any(state_info['fips'] == state_fips for state_info in overlap_states.values()):
        adjustment_factor = get_adjustment_factor(matched_shapes,state_fips)
    else:
        adjustment_factor = 0
        
    per_pop = 100*(local_gov_pop - adjustment_factor)/pop
    
    return per_pop, local_gov_pop-adjustment_factor, pop

#%%Main code

def main(sample,state_fips):
    
    #Manually remove select defunt (merged into other) munis/townships
    to_remove = [
        164921, #North Rich Township (Kansas) merged into Rich Township in 2017: https://en.wikipedia.org/wiki/North_Rich_Township,_Anderson_County,_Kansas
        176012, #Millican Texas was unincorporated: https://en.wikipedia.org/wiki/Millican,_Texas
        251374, #Peaster Texas is now unincorporated: https://en.wikipedia.org/wiki/Peaster,_Texas
        ] 
    
    sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove)]
    
    #Filter out counties
    sample = sample[sample['UNIT_TYPE'] != '1 - COUNTY']
    
    results = pd.Series([])
    
    #Get set of all local governments
    state_govs = sample[sample['FIPS_STATE'] == state_fips]
    
    #Get fips places
    places = get_place_shapes(state_fips)
    
    #Get county subvidisions
    cousubs = get_cousub_shapes(state_fips)
    
    #Match set of local governments with shape files
    matched_shapes = match_shapes(places,cousubs,state_govs)
    
    #Get the % population by local govs
    results.loc['% Pop'], results.loc['Local Gov Pop'], results.loc['State Pop'] = get_per_pop(state_fips,state_pop,state_govs,matched_shapes)
    
    #Get the number of townships
    results.loc['Num Townships'] = len(state_govs[state_govs['UNIT_TYPE'] == '3 - TOWNSHIP'].index)
    
    #Get the number of munis
    results.loc['Num Munis'] = len(state_govs[state_govs['UNIT_TYPE'] == '2 - MUNICIPAL'].index)
    
    return results

# Load the sample data
sample = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))
all_states = sample['FIPS_STATE'].unique()

# Samples to loop over
'''
Could add sources one by one here too
'''

our_sample =pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))

samples = {
    'cog_sample': pd.read_excel(os.path.join(config['raw_data'],'census_of_gov.xlsx')),
    'our_sample': our_sample,
    'alp':our_sample[our_sample['Source'] == 'American Legal Publishing'],
    'ord':our_sample[our_sample['Source'] == 'Ordinance.com'],
    'mc':our_sample[our_sample['Source'] == 'Municode'],
    
}


# Create a list of all task combinations
task_combinations = [(sample_name, state) for sample_name in samples for state in all_states]

# Check existing files and remove completed tasks from the list
pop_coverage_folder = os.path.join(config['processed_data'], 'pop_coverage')
os.makedirs(pop_coverage_folder, exist_ok=True)

completed_tasks = []
for filename in os.listdir(pop_coverage_folder):
    if filename.endswith('.pkl'):
        sample_name, state = filename[:-4].split('#')  
        completed_tasks.append((sample_name, state))

task_combinations = [task for task in task_combinations if task not in completed_tasks]

# SLURM environment setup
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
num_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', len(task_combinations)))

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
    result = main(samples[sample_name], state)
    filename = os.path.join(pop_coverage_folder, f'{sample_name}#{state}.pkl')
    
    # Save result as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

    
