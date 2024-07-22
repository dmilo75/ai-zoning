"""
This code takes in a list of CENSUS PID6 IDs and returns the population of the group
"""
import geopandas as gpd
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import os
import pandas as pd
from itertools import combinations
import math
from shapely.ops import transform
import pyproj
from math import radians, sin, cos, sqrt, atan2
os.chdir('../../')

#


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

##

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

def get_node_group_combinations(sample):


    # Get the number of nodes and the task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

    # Sort the sample by state
    sample = sample.sort_values('FIPS_STATE')

    # Determine the number of groups
    n = int(math.sqrt(2 * num_nodes)) + 1
    while (n * (n-1) // 2) < num_nodes:
        n += 1

    # Split the sample into n groups
    group_size = len(sample) // n
    groups = [sample.iloc[i:i+group_size] for i in range(0, len(sample), group_size)]

    # Generate all pairwise combinations of groups
    group_combinations = list(combinations(range(len(groups)), 2))

    # Add self-comparisons
    group_combinations += [(i, i) for i in range(len(groups))]

    # Distribute combinations across nodes
    combinations_per_node = len(group_combinations) // num_nodes
    extra_combinations = len(group_combinations) % num_nodes

    if task_id < extra_combinations:
        start_idx = task_id * (combinations_per_node + 1)
        end_idx = start_idx + combinations_per_node + 1
    else:
        start_idx = task_id * combinations_per_node + extra_combinations
        end_idx = start_idx + combinations_per_node

    # Get the combinations for this node
    node_combinations = group_combinations[start_idx:end_idx]

    return [(groups[i], groups[j]) for i, j in node_combinations]

def get_unique_states(group_pairs):
    unique_states = set()
    for group1, group2 in group_pairs:
        unique_states.update(group1['FIPS_STATE'].unique())
        unique_states.update(group2['FIPS_STATE'].unique())
    return list(unique_states)

#Function to cache all state files needed
def get_states_files(group_pairs):

    #Get unique states
    unique_states = get_unique_states(group_pairs)

    #Take union of all dataframes in group_pairs
    sample = pd.concat([group1 for group1, group2 in group_pairs] + [group2 for group1, group2 in group_pairs])

    #Drop duplicate CENSUS_ID_PID6
    sample = sample.drop_duplicates(subset = 'CENSUS_ID_PID6')

    #matched shapes dictionary
    matched_shapes = []

    #Get all state files
    for state in unique_states:
        matched_shapes.append(load_shapes(sample,state))

    #Turn into dataframe
    matched_shapes = pd.concat(matched_shapes)

    return matched_shapes



def main_calculation(group_from, group_to, shape_files):
    # Prepare GeoSeries for geometries from group_from and group_to using the CENSUS_ID_PID6 column
    geometries_from = shape_files.set_index('CENSUS_ID_PID6').loc[group_from['CENSUS_ID_PID6'], 'geometry']
    geometries_to = shape_files.set_index('CENSUS_ID_PID6').loc[group_to['CENSUS_ID_PID6'], 'geometry']

    # Create a projection to calculate distances in meters
    proj_meters = pyproj.Proj(proj='aea', lat_1=geometries_from.total_bounds[1], lat_2=geometries_from.total_bounds[3],
                              lat_0=(geometries_from.total_bounds[1] + geometries_from.total_bounds[3])/2,
                              lon_0=(geometries_from.total_bounds[0] + geometries_from.total_bounds[2])/2)

    project = pyproj.Transformer.from_proj(pyproj.Proj(proj='latlong'), proj_meters, always_xy=True).transform

    # Initialize an empty list to store the distances
    distances = []

    # Calculate pairwise distances and store them in the list
    for id1, geometry_from in geometries_from.items():
        geom_from_proj = transform(project, geometry_from)
        for id2, geometry_to in geometries_to.items():
            geom_to_proj = transform(project, geometry_to)

            distance = geom_from_proj.distance(geom_to_proj) / 1609.34  # Convert meters to miles
            distances.append({'ID1': id1, 'ID2': id2, 'distance': distance})

    # Convert the list to a DataFrame
    distance_df = pd.DataFrame(distances)

    # Make filename
    filename = f'{group_from["CENSUS_ID_PID6"].iloc[0]}_{group_to["CENSUS_ID_PID6"].iloc[0]}.csv'

    # Save the DataFrame to a CSV file
    distance_df.to_csv(os.path.join(config['processed_data'], 'Distance Matrices', filename), index=False)

    return


##Main code

#Ensure folder in processed data called distance_matrisx exists
os.makedirs(os.path.join(config['processed_data'],'Distance Matrices'), exist_ok = True)

#Draw in the census of governments
sample = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))

#Filter out defunct munis/townships
sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove[year])]

#Filter out counties
sample = sample[sample['UNIT_TYPE'] != '1 - COUNTY']

#Sort by state
sample = sample.sort_values(by = 'FIPS_STATE')

group_pairs = get_node_group_combinations(sample)

shape_files = get_states_files(group_pairs)

print(f"Node will process {len(group_pairs)} group pairs")

for i, (group_from, group_to) in enumerate(group_pairs):
    print()

    main_calculation(group_from,group_to,shape_files)





##


