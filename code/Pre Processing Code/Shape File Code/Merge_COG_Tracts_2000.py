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
os.chdir('../../../')
import logging
import portalocker  # For file locking
import time

# Get the node ID
node_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

# Define the path for the log file
log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tract_bridge_logs', f'progress_node_{node_id}.log')

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

def get_block_shapes(state_fips):
    path_block = os.path.join(base_shape_path,year, 'Blocks', f'tl_{year}_{state_fips:02d}_tabblock20',
                              f'tl_{year}_{state_fips:02d}_tabblock20.shp')
    gdf_block = gpd.read_file(path_block)
    return gdf_block

#Function to get county subdivisions 
def get_cousub_shapes(state_fips):
    path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_{year}_{state_fips:02d}_cousub',f'tl_{year}_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub


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

def get_rows(sample):
    # Get the number of nodes and the task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

    # Sort the sample by state
    sample = sample.sort_values('FIPS_STATE')

    # Calculate the number of rows per node
    rows_per_node = len(sample) // num_nodes
    extra_rows = len(sample) % num_nodes

    # Determine the start and end indices for this node's subsample
    if task_id < extra_rows:
        start_idx = task_id * (rows_per_node + 1)
        end_idx = start_idx + rows_per_node + 1
    else:
        start_idx = task_id * rows_per_node + extra_rows
        end_idx = start_idx + rows_per_node

    # Return the subsample for this node
    return sample.iloc[start_idx:end_idx]

#Function to get tract shapes in 2000
def get_tract_shapes(state_fips):
    path_tract = os.path.join(base_shape_path, '2002', 'Census Tracts', f'tl_2010_{state_fips:02d}_tract00',
                              f'tl_2010_{state_fips:02d}_tract00.shp')
    gdf_tract = gpd.read_file(path_tract)
    #Rename CTIDFP00 to GEOID
    gdf_tract['GEOID'] = gdf_tract['CTIDFP00']

    return gdf_tract

#Function to cache all state files needed
def get_all_shapes(sample):
    all_states = sample['FIPS_STATE'].unique()
    matched_shapes = []
    tract_shapes = []

    for state in all_states:
        matched_shapes.append(load_shapes(sample, state))
        tract_shapes.append(get_tract_shapes(state))

    matched_shapes = pd.concat(matched_shapes)
    tract_shapes = pd.concat(tract_shapes)

    # Ensure they are in the same CRS
    tract_shapes = tract_shapes.to_crs(matched_shapes.crs)

    return matched_shapes, tract_shapes





from shapely.geometry import box


def subset_potential_matches(buffered_geometry, gdf):
    bbox = box(*buffered_geometry.bounds)
    possible_matches_index = list(gdf.sindex.intersection(bbox.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    return possible_matches[possible_matches.intersects(buffered_geometry)]




def calculate_intersection(muni_geometry, tract_shapes):
    intersections = []
    potential_matches = subset_potential_matches(muni_geometry, tract_shapes)

    for idx, tract in potential_matches.iterrows():
        intersection = muni_geometry.intersection(tract.geometry)
        if not intersection.is_empty:
            intersections.append({
                "Tract_GEOID": tract.GEOID,
                "Intersection_Area": intersection.area,
                "Tract_Area": tract.geometry.area
            })

    return intersections

def main_calculation(row, matched_shapes, tract_shapes):
    muni_geometry = matched_shapes[matched_shapes['CENSUS_ID_PID6'] == row['CENSUS_ID_PID6']]['geometry'].iloc[0]
    muni_area = muni_geometry.area

    intersections = calculate_intersection(muni_geometry, tract_shapes)

    results = []
    for intersection in intersections:
        results.append({
            "CENSUS_ID_PID6": row['CENSUS_ID_PID6'],
            "Tract_GEOID": intersection['Tract_GEOID'],
            "Intersection_Area": intersection['Intersection_Area'],
            "Muni_Area": muni_area,
            "Tract_Area": intersection['Tract_Area']
        })

    return results

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

# Main code
logging.info("Starting the main code execution.")

results_dir = os.path.join(config['processed_data'], 'Muni-Tract Geobridge Results')
os.makedirs(results_dir, exist_ok=True)
logging.info(f"Results directory created at {results_dir}.")

csv_path = os.path.join(results_dir, 'muni_tract_geobridge.csv')

sample = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'), index_col = 0)

logging.info("Census of governments data loaded.")

sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove[year])]
logging.info("Defunct municipalities/townships filtered out.")

sample = sample[sample['UNIT_TYPE'] != '1 - COUNTY']
logging.info("Counties filtered out.")

sample = sample.sort_values(by='FIPS_STATE')
logging.info("Sample sorted by state.")

subset = get_rows(sample)
logging.info(f"Subset of rows for this node obtained: {len(subset)} rows.")


matched_shapes, tract_shapes = get_all_shapes(subset)
logging.info("All shape files for incorporated areas and tracts loaded.")


if not os.path.exists(csv_path):
    open(csv_path, 'w').close()

all_results = []
for idx, row in subset.iterrows():
    logging.info(f"Processing row {idx + 1}/{len(subset)}")
    results = main_calculation(row, matched_shapes, tract_shapes)
    all_results.extend(results)
    if len(all_results) >= 50:
        results_df = pd.DataFrame(all_results)
        append_to_csv(results_df, csv_path)
        all_results = []

if all_results:
    results_df = pd.DataFrame(all_results)
    append_to_csv(results_df, csv_path)

logging.info("Processing completed.")