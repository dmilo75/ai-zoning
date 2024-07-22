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
log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uninc_logs', f'progress_node_{node_id}.log')

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

results_dir = os.path.join(config['processed_data'], 'Unincorporated Area Results')
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, 'unincorporated_area_results.csv')


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

def get_water_shape(state_fips):

    # Ensure state_fips is a string and zero-padded if necessary
    state_fips = str(state_fips).zfill(2)

    water_path = os.path.join(config['shape_files'], '2022', 'Water', state_fips)

    # List to hold all water shapes
    all_water_shapes = []

    # Loop over each folder in the water_path
    for folder_name in os.listdir(water_path):
        folder_path = os.path.join(water_path, folder_name)
        if os.path.isdir(folder_path):
            # Construct the file path
            file_path = os.path.join(folder_path, f'tl_2022_{state_fips}{folder_name}_areawater.shp')
            if os.path.exists(file_path):
                # Read the shape file and append to the list
                water_shape = gpd.read_file(file_path)
                all_water_shapes.append(water_shape)
            else:
                print(f"File not found: {file_path}")

    # If no files are found, raise an error
    if not all_water_shapes:
        raise FileNotFoundError(f"No water shape files found for state FIPS {state_fips}")

    # Concatenate all water shapes into a single GeoDataFrame
    water_shapes = gpd.GeoDataFrame(pd.concat(all_water_shapes, ignore_index=True))

    # Return the GeoDataFrame of water shapes
    return water_shapes

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


#Function to cache all state files needed
def get_all_shapes(sample):

    #Get unique states
    all_states = sample['FIPS_STATE'].unique()



    #matched shapes dictionary
    matched_shapes = []
    block_shapes = []
    water_shapes = []

    #Get all state files
    for state in all_states:
        matched_shapes.append(load_shapes(sample,state))
        block_shapes.append(get_block_shapes(state))
        water_shapes.append(get_water_shape(state))

    #Turn into dataframe
    matched_shapes = pd.concat(matched_shapes)
    block_shapes = pd.concat(block_shapes)
    water_shapes = pd.concat(water_shapes)

    return matched_shapes, block_shapes, water_shapes


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


import utm
from shapely.geometry import Point
from pyproj import CRS, Transformer
from shapely.ops import transform


def get_buffer_radius(geometry, buffer_miles):
    # Get lat long centroid
    lon, lat = geometry.centroid.x, geometry.centroid.y

    # Get UTM zone information
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)

    # Create UTM CRS for the specific zone
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone_number, 'ellps': 'WGS84'})

    # Create transformer
    transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)

    # Transform geometry to UTM
    utm_geometry = transform(transformer.transform, geometry)

    # Get buffer in meters
    buffer_meters = buffer_miles * 1609.34

    # Apply buffer
    buffered_geometry = utm_geometry.buffer(buffer_meters)

    # Transform back to WGS84
    wgs84_transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    buffered_wgs84 = transform(wgs84_transformer.transform, buffered_geometry)

    return buffered_wgs84


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


def calculate_overlap(geometry, buffered_geometry, matched_shapes, water_shapes):
    """
    Calculate the overlap between the buffered geometry (excluding the original geometry)
    and matched shapes, excluding water areas.

    :param geometry: The original geometry
    :param buffered_geometry: The buffered geometry of the area of interest
    :param matched_shapes: GeoDataFrame of matched shapes
    :param water_shapes: GeoDataFrame of water shapes
    :return: A dictionary containing various overlap statistics
    """
    # Calculate the ring area (buffered area excluding the original geometry)
    ring_area = buffered_geometry.difference(geometry)

    # Subset potential matches
    potential_matched_shapes = subset_potential_matches(ring_area, matched_shapes)
    potential_water_shapes = subset_potential_matches(ring_area, water_shapes)

    # Calculate water area within the ring
    water_area = unary_union(potential_water_shapes.geometry).intersection(ring_area)

    # Calculate ring area without water
    ring_area_ex_water = ring_area.difference(water_area)

    # Calculate intersection area with matched shapes, excluding water
    intersection_area = potential_matched_shapes.geometry.intersection(ring_area_ex_water).area.sum()

    # Calculate total area (ring area without water)
    total_area = ring_area_ex_water.area

    # Calculate percent overlap
    percent_overlap = (intersection_area / total_area) * 100 if total_area > 0 else 0

    # Calculate unincorporated area
    unincorporated_area = total_area - intersection_area

    return {
        "percent_overlap": percent_overlap,
        "unincorporated_area": unincorporated_area,
        "total_area": total_area
    }


def get_area_shapes(buffered_geometry, matched_shapes):
    """
    Get shape files for unincorporated area and total area.

    :param buffered_geometry: The buffered geometry
    :param matched_shapes: GeoDataFrame of matched shapes
    :return: Dictionary with unincorporated and total area shapes
    """
    # Subset potential matches
    potential_matched_shapes = subset_potential_matches(buffered_geometry, matched_shapes)

    # Get unincorporated area
    incorporated_area = unary_union(potential_matched_shapes.geometry)
    unincorporated_area = buffered_geometry.difference(incorporated_area)

    return {
        "unincorporated_area": unincorporated_area,
        "total_area": buffered_geometry
    }


def calculate_adjusted_variables(area_shape, block_shapes, variables):
    """
    Calculate adjusted variables based on partial overlaps.

    :param area_shape: The shape to intersect with blocks
    :param block_shapes: GeoDataFrame of block shapes
    :param variables: List of variables to adjust ('HOUSING20', 'POP20', etc.)
    :return: Dictionary of adjusted variable sums
    """
    # Calculate percent overlap for each block
    block_shapes['percent_overlap'] = block_shapes.geometry.apply(
        lambda x: x.intersection(area_shape).area / x.area if x.intersects(area_shape) else 0
    )

    # Calculate adjusted variables
    adjusted_vars = {}
    for var in variables:
        block_shapes[f'adjusted_{var}'] = block_shapes[var] * block_shapes['percent_overlap']
        adjusted_vars[var] = block_shapes[f'adjusted_{var}'].sum()

    return adjusted_vars


##

def main_calculation(row,matched_shapes, block_shapes, water_shapes, buffer = 10):

    #Get the geometry of the row
    geometry = matched_shapes[matched_shapes['CENSUS_ID_PID6'] == row['CENSUS_ID_PID6']]['geometry'].iloc[0]

    #Add a buffer
    buffered_geometry = get_buffer_radius(geometry, buffer)

    # Calculate overlap statistics
    overlap_stats = calculate_overlap(geometry, buffered_geometry, matched_shapes, water_shapes)

    # Get area shapes
    area_shapes = get_area_shapes(buffered_geometry, matched_shapes)

    # Variables to adjust
    variables = ['HOUSING20', 'POP20']

    # Calculate adjusted variables for unincorporated and total areas
    # Use total_area_with_water for block intersections
    unincorporated_vars = calculate_adjusted_variables(area_shapes["unincorporated_area"], block_shapes, variables)
    total_vars = calculate_adjusted_variables(area_shapes["total_area"], block_shapes, variables)

    return {
        "ID": row['CENSUS_ID_PID6'],
        "percent_land_unincorporated": overlap_stats['percent_overlap'],
        "unincorporated_housing_units": unincorporated_vars['HOUSING20'],
        "total_housing_units": total_vars['HOUSING20'],
        "unincorporated_population": unincorporated_vars['POP20'],
        "total_population": total_vars['POP20'],
        "unincorporated_area": area_shapes["unincorporated_area"].area,
        "total_area": area_shapes["total_area"].area
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
matched_shapes, block_shapes, water_shapes = get_all_shapes(sample)
logging.info("All shape files for incorporated areas loaded.")

# Get the subset of rows for this node
subset = get_rows(sample)
logging.info(f"Subset of rows for this node obtained: {len(subset)} rows.")

# Ensure the CSV file exists before appending
if not os.path.exists(csv_path):
    open(csv_path, 'w').close()

# Loop over rows and call function
for idx, row in subset.iterrows():
    results = []
    for buffer in [10, 25, 50]:
        logging.info(f"Processing row {idx+1}/{len(subset)} with buffer {buffer}.")
        result = main_calculation(row, matched_shapes, block_shapes, water_shapes, buffer)
        results.append(result)
    results_df = pd.DataFrame(results)
    append_to_csv(results_df, csv_path)
logging.info("All rows processed.")

