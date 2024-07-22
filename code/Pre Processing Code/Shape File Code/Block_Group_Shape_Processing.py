"""
This code takes in a list of CENSUS PID6 IDs and returns the population of the group
"""
import geopandas as gpd
import utm
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import os
import pandas as pd
from itertools import combinations
import math
from shapely.ops import unary_union
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2
os.chdir('../../')
import logging
import portalocker  # For file locking
import time
from shapely.ops import nearest_points


# Get the node ID
node_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

# Define the path for the log file
log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bg_logs', f'progress_node_{node_id}.log')

# Create the directory for the log file if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)


#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

year = '2022'

#Draw in all shape files for places and county subdivisions
base_shape_path = config['shape_files']

shape_path = os.path.join(base_shape_path,year)



## Functions/mappings



#Manually remove select defunt (merged into other) munis/townships
to_remove = {}

to_remove['2022'] = [
    164921, #North Rich Township (Kansas) merged into Rich Township in 2017: https://en.wikipedia.org/wiki/North_Rich_Township,_Anderson_County,_Kansas
    176012, #Millican Texas was unincorporated: https://en.wikipedia.org/wiki/Millican,_Texas
    251374, #Peaster Texas is now unincorporated: https://en.wikipedia.org/wiki/Peaster,_Texas
    ]


#Function to get all fips place shape files for a state
def get_place_shapes(state_fips):

    path_municipalities = os.path.join(shape_path,'Places',f'tl_{year}_{state_fips:02d}_place',f'tl_{year}_{state_fips:02d}_place.shp')
    gdf_municipalities = gpd.read_file(path_municipalities)
    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(int)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities


#Function to get county subdivisions 
def get_cousub_shapes(state_fips):
    path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_{year}_{state_fips:02d}_cousub',f'tl_{year}_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub

def load_block_group_shapes(state_fips):
    path_block_groups = os.path.join(shape_path, 'Block Groups', f'tl_{year}_{state_fips:02d}_bg', f'tl_{year}_{state_fips:02d}_bg.shp')
    gdf_block_groups = gpd.read_file(path_block_groups)
    return gdf_block_groups


def calculate_incorporated_area(block_group, matched_shapes):
    # Get potential matches
    potential_matches = matched_shapes[matched_shapes.intersects(block_group.geometry)]

    # Calculate incorporated area
    if not potential_matches.empty:
        incorporated_area = block_group.geometry.intersection(potential_matches.unary_union).area
    else:
        incorporated_area = 0

    return incorporated_area
def calculate_percent_incorporated(block_group, incorporated_area):
    total_area = block_group.geometry.area
    percent_incorporated = (incorporated_area / total_area) * 100 if total_area > 0 else 0
    return percent_incorporated

def get_nearest_metro(block_group, metro_centers):
    block_group_point = block_group.geometry.centroid
    lon1, lat1 = block_group_point.x, block_group_point.y

    closest_center = None
    min_distance = float('inf')

    for _, center_row in metro_centers.iterrows():
        lon2, lat2 = center_row['Longitude'], center_row['Latitude']
        distance = haversine_distance(lon1, lat1, lon2, lat2)

        if distance < min_distance:
            closest_center = center_row['msa']
            min_distance = distance

    return closest_center, min_distance

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

    #Concat places_filt and cousubs_filt
    combined = pd.concat([places_filt,cousubs_filt])

    #Merge in CENSUS_ID_PID6 from state_govs
    merged = combined.merge(state_govs, left_on = 'GEOID', right_on = 'GEOID')

    #Now just keep 'CENSUS_ID_PID6' and 'geometry'
    merged = merged[['CENSUS_ID_PID6','geometry']]

    return merged





#%%Main code

def load_shapes(sample,state_fips):
    
    #Get set of all local governments
    state_govs = sample[sample['FIPS_STATE'] == state_fips]
    
    #Get fips places
    places = get_place_shapes(state_fips)
    
    #Get county subvidisions
    cousubs = get_cousub_shapes(state_fips)

    #Load in bridge
    bridge = pd.read_excel(os.path.join(config['processed_data'],'Shape File Bridges',f'cog_{year}_bridge.xlsx'))

    #Match set of local governments with shape files
    matched_shapes = match_shapes(places,cousubs,bridge, state_govs)

    return matched_shapes


def get_rows(block_group_shapes):

    #Shuffle the row order in the block group shapes with fixed seed
    block_group_shapes = block_group_shapes.sample(frac=1, random_state=42)

    #Reset the index
    block_group_shapes = block_group_shapes.reset_index(drop=True)

    # Get the number of nodes and the task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

    # Calculate the number of block groups per node
    block_groups_per_node = len(block_group_shapes) // num_nodes
    extra_block_groups = len(block_group_shapes) % num_nodes

    # Determine the start and end indices for this node's subsample
    if task_id < extra_block_groups:
        start_idx = task_id * (block_groups_per_node + 1)
        end_idx = start_idx + block_groups_per_node + 1
    else:
        start_idx = task_id * block_groups_per_node + extra_block_groups
        end_idx = start_idx + block_groups_per_node

    # Return the subsample for this node
    return block_group_shapes.iloc[start_idx:end_idx]


#Function to cache all state files needed
def get_all_shapes(sample):
    all_states = sample['FIPS_STATE'].unique()
    matched_shapes = []
    block_group_shapes = []

    for state in all_states:
        matched_shapes.append(load_shapes(sample, state))
        block_group_shapes.append(load_block_group_shapes(state))

    matched_shapes = pd.concat(matched_shapes)
    block_group_shapes = pd.concat(block_group_shapes)

    return matched_shapes, block_group_shapes


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 3958.8  # Earth's radius in miles

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance  # Distance in miles


def distance_to_nearest_incorporated(block_group, matched_shapes):

    # If the block group intersects with any incorporated area, distance is 0
    if matched_shapes.intersects(block_group.geometry).any():
        return 0

    # Find the nearest point on any incorporated area to the block group
    block_group_geom = block_group.geometry
    nearest_geom = matched_shapes.geometry.unary_union

    # Get the nearest points between the block group and the nearest incorporated area
    nearest_point_block, nearest_point_incorporated = nearest_points(block_group_geom, nearest_geom)

    # Calculate the distance with haversine
    distance = haversine_distance(nearest_point_block.x, nearest_point_block.y, nearest_point_incorporated.x, nearest_point_incorporated.y)

    return distance

def get_area(block_group_row, original_crs):

    # Convert the Series back to a GeoDataFrame
    block_group = gpd.GeoDataFrame([block_group_row], columns=block_group_row.index).set_geometry('geometry')
    block_group.crs = original_crs  # Set the original CRS

    # Re-project to a projected CRS
    block_group = block_group.to_crs(epsg=5070)  # You can choose another suitable projected CRS

    # Calculate area in square meters and convert to acres
    area_acres = block_group.geometry.area / 4046.85642

    return area_acres.iloc[0]

def process_block_group(block_group, matched_shapes, metro_centers, original_crs = 4269):

    #Get area in acres
    area_acres = get_area(block_group,original_crs)

    # Calculate incorporated area and percentage
    incorporated_area = calculate_incorporated_area(block_group, matched_shapes)
    percent_incorporated = calculate_percent_incorporated(block_group, incorporated_area)

    # Calculate distance to nearest incorporated area
    distance_to_incorporated = distance_to_nearest_incorporated(block_group, matched_shapes)

    # Get nearest metro
    nearest_metro, distance_to_metro = get_nearest_metro(block_group, metro_centers)

    return {
        "STATEFP": block_group['STATEFP'],
        "COUNTYFP": block_group['COUNTYFP'],
        "TRACTCE": block_group['TRACTCE'],
        "BLKGRPCE": block_group['BLKGRPCE'],
        "Area_Acres": area_acres,
        "Percent_Incorporated": percent_incorporated,
        "Distance_to_Incorporated": distance_to_incorporated,
        "Nearest_Metro": nearest_metro,
        "Distance_to_Metro": distance_to_metro
    }

# Function to append results to a CSV with portalocker
def append_to_csv(data, csv_path):
    backoff = 1
    max_backoff = 32
    while backoff <= max_backoff:
        try:
            with open(csv_path, 'a', newline='') as f:
                portalocker.lock(f, portalocker.LOCK_EX)  # Lock the file
                data.to_csv(f, header=f.tell()==0, index=False)  # Write header only if file is empty
                portalocker.unlock(f)  # Unlock the file
            break
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
            time.sleep(backoff)
            backoff *= 2
    if backoff > max_backoff:
        logging.error("Max backoff reached, could not write to CSV.")


##Main code
logging.info("Starting the main code execution.")

# Create a new directory for storing the results
results_dir = os.path.join(config['processed_data'], 'Block Group Analysis Results')
os.makedirs(results_dir, exist_ok=True)
logging.info(f"Results directory created at {results_dir}.")

# Draw in the census of governments
sample = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))
logging.info("Census of governments data loaded.")

# Filter out defunct munis/townships
sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove[year])]
logging.info("Defunct municipalities/townships filtered out.")

# Filter out counties
sample = sample[sample['UNIT_TYPE'] != '1 - COUNTY']
logging.info("Counties filtered out.")

# Sort by state
sample = sample.sort_values(by='FIPS_STATE')
logging.info("Sample sorted by state.")


# Get all shape files for incorporated areas and block groups
matched_shapes, block_group_shapes = get_all_shapes(sample)
logging.info("All shape files for incorporated areas and block groups loaded.")

# Load metro centers
metro_centers = pd.read_csv(os.path.join(config['raw_data'], 'MSA Center Points', 'city_centers.csv'))
logging.info("Metro centers data loaded.")

# Get the subset of block groups for this node
subset_block_groups = get_rows(block_group_shapes)
logging.info(f"Subset of block groups for this node obtained: {len(subset_block_groups)} block groups.")

# Ensure the CSV file exists before appending
csv_path = os.path.join(results_dir, 'block_group_analysis_results.csv')
if not os.path.exists(csv_path):
    open(csv_path, 'w').close()

# Load existing results if the CSV file exists
if os.path.exists(csv_path):
    existing_results = pd.read_csv(csv_path)
    processed_block_groups = existing_results[['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE']]

    #Two digit zero pad state, three digit zero pad county, six digit zero pad tract and make block a string
    processed_block_groups['STATEFP'] = processed_block_groups['STATEFP'].apply(lambda x: str(x).zfill(2))
    processed_block_groups['COUNTYFP'] = processed_block_groups['COUNTYFP'].apply(lambda x: str(x).zfill(3))
    processed_block_groups['TRACTCE'] = processed_block_groups['TRACTCE'].apply(lambda x: str(x).zfill(6))
    processed_block_groups['BLKGRPCE'] = processed_block_groups['BLKGRPCE'].apply(lambda x: str(x).zfill(1))

else:
    existing_results = pd.DataFrame()
    processed_block_groups = pd.DataFrame(columns=['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE'])



# Function to check if a block group is already processed
def is_processed(block_group, processed_block_groups):

    return not processed_block_groups[
        (processed_block_groups['STATEFP'] == block_group['STATEFP']) &
        (processed_block_groups['COUNTYFP'] == block_group['COUNTYFP']) &
        (processed_block_groups['TRACTCE'] == block_group['TRACTCE']) &
        (processed_block_groups['BLKGRPCE'] == block_group['BLKGRPCE'])
    ].empty

# Filter out already processed block groups
subset_block_groups2 = subset_block_groups[~subset_block_groups.apply(is_processed, axis=1, processed_block_groups=processed_block_groups)]

logging.info(f"Subset of block groups for this node obtained: {len(subset_block_groups)} block groups after filtering out processed ones.")


results_list = []
row_count = 0

# Process each block group
for idx, block_group in subset_block_groups.iterrows():
    logging.info(f"Processing block group {idx + 1}/{len(subset_block_groups)}.")

    result = process_block_group(block_group, matched_shapes, metro_centers)
    results_list.append(result)
    row_count += 1

    if row_count >= 100:
        results_df = pd.DataFrame(results_list)
        append_to_csv(results_df, csv_path)
        results_list = []  # Clear the list
        row_count = 0  # Reset the row count

# Append any remaining results after the loop
if results_list:
    results_df = pd.DataFrame(results_list)
    append_to_csv(results_df, csv_path)

logging.info("All block groups processed.")
