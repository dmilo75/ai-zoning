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

# Get the node ID
node_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

# Define the path for the log file
log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'school_bridge_logs', f'progress_node_{node_id}.log')

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

results_dir = os.path.join(config['processed_data'], 'School District Geobridge')
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, 'muni_school_district_bridge.csv')


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


#Get school shapes
def get_school_shapes():

    #School shapes is under raw data, Stanford Ed Opportunity Project, Geo Crosswalk, SEDA-2019-District-Shapefile
    school_shapes = gpd.read_file(os.path.join(config['raw_data'],'Stanford Ed Opportunity Project','Geo Crosswalk','SEDA-2019-District-Shapefile','SEDA-2019-District-Shapefile.shp'))

    return school_shapes

#Function to cache all state files needed
def get_all_shapes(sample):

    # Draw in school shapes
    school_shapes = get_school_shapes()

    #Get unique states
    all_states = sample['FIPS_STATE'].unique()

    #matched shapes dictionary
    matched_shapes = []

    #Get all state files
    for state in all_states:
        matched_shapes.append(load_shapes(sample,state))

    #Turn into dataframe
    matched_shapes = pd.concat(matched_shapes)

    #Make sure they are in the same crs
    school_shapes = school_shapes.to_crs(matched_shapes.crs)


    return matched_shapes, school_shapes





from shapely.geometry import box


def subset_potential_matches(buffered_geometry, gdf):
    """
    Subset a GeoDataFrame to only include potential matches within the buffered geometry
    using spatial indexing for improved efficiency.

    :param buffered_geometry: The buffered geometry to check against
    :param gdf: The GeoDataFrame to subset
    :return: A subset of the input GeoDataFrame
    """
    # Create a bounding box of the buffered geometry
    bbox = box(*buffered_geometry.bounds)

    # Use spatial index to get candidates
    possible_matches_index = list(gdf.sindex.intersection(bbox.bounds))
    possible_matches = gdf.iloc[possible_matches_index]

    # Perform the final intersection test
    return possible_matches[possible_matches.intersects(buffered_geometry)]


def calculate_intersection(muni_geometry, school_shapes):
    """
    Calculate intersection between a municipality and school districts.

    :param muni_geometry: Geometry of the municipality
    :param school_shapes: GeoDataFrame of school district shapes
    :return: List of dictionaries containing intersection information
    """
    intersections = []
    potential_matches = subset_potential_matches(muni_geometry, school_shapes)

    for idx, school_district in potential_matches.iterrows():
        intersection = muni_geometry.intersection(school_district.geometry)
        if not intersection.is_empty:
            intersections.append({
                "School_GEOID": school_district.GEOID,
                "Intersection_Area": intersection.area,
                "School_Area": school_district.geometry.area
            })

    return intersections

##

def main_calculation(row, matched_shapes, school_shapes):
    muni_geometry = matched_shapes[matched_shapes['CENSUS_ID_PID6'] == row['CENSUS_ID_PID6']]['geometry'].iloc[0]
    muni_area = muni_geometry.area

    intersections = calculate_intersection(muni_geometry, school_shapes)

    results = []
    for intersection in intersections:
        results.append({
            "CENSUS_ID_PID6": row['CENSUS_ID_PID6'],
            "School_GEOID": intersection['School_GEOID'],
            "Intersection_Area": intersection['Intersection_Area'],
            "Muni_Area": muni_area,
            "School_Area": intersection['School_Area']
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

logging.info("Starting the main code execution.")

# Create a new directory for storing the results
results_dir = os.path.join(config['processed_data'], 'Unincorporated Area Results')
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

# Get all shape files for incorporated areas
matched_shapes,school_shapes = get_all_shapes(sample)
logging.info("All shape files for incorporated areas loaded.")

# Get the subset of rows for this node
subset = get_rows(sample)
logging.info(f"Subset of rows for this node obtained: {len(subset)} rows.")

# Ensure the CSV file exists before appending
if not os.path.exists(csv_path):
    open(csv_path, 'w').close()

# In the main loop:
all_results = []
for idx, row in subset.iterrows():
    logging.info(f"Processing row {idx + 1}/{len(subset)}")
    results = main_calculation(row, matched_shapes, school_shapes)
    all_results.extend(results)

    if len(all_results) >= 50:
        results_df = pd.DataFrame(all_results)
        append_to_csv(results_df, csv_path)
        all_results = []

# Don't forget to write any remaining results
if all_results:
    results_df = pd.DataFrame(all_results)
    append_to_csv(results_df, csv_path)
