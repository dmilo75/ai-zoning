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
import logging
import sys
os.chdir('../../../')

#

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Ignore keyboard interrupts to allow graceful exit
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Install the exception handler
sys.excepthook = log_exception

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

var = 'Housing'
year = '2020'

fix_2000 = True

if fix_2000:
    cog_year = '2002'
else:
    cog_year = str(int(year)+2)

#Draw in all shape files for places and county subdivisions
base_shape_path = config['shape_files']

shape_path = os.path.join(base_shape_path,cog_year)

#Set up logging
log_folder = os.path.join('code','Table and Figures Code', 'Incorporated Long Diff', 'error_logs')
os.makedirs(log_folder, exist_ok=True)
log_filename = f'{year}_error_log_{os.environ.get("SLURM_ARRAY_TASK_ID", "0")}.log'
log_filepath = os.path.join(log_folder, log_filename)

logging.basicConfig(
    filename=log_filepath,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

block_var_name = {
    'Housing': {
        '2020': 'HOUSING20',
        '2010': 'HOUSING10',
        '2000':'Housing Units',
    },
    'Population': {
        '2020':'POP20',
        '2010':'POP10',
        '2000':'Population',
    }

}

## Functions/mappings



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

to_remove['2002'] = []


#Function to get all fips place shape files for a state
def get_place_shapes(state_fips):
    if cog_year == '2002':
        path_municipalities = os.path.join(shape_path,'Places',f'tl_2002_{state_fips:02d}_place',f'tl_2010_{state_fips:02d}_place00.shp')
    else:
        path_municipalities = os.path.join(shape_path,'Places',f'tl_{year}_{state_fips:02d}_place',f'tl_{year}_{state_fips:02d}_place.shp')
    gdf_municipalities = gpd.read_file(path_municipalities)
    if cog_year == '2002':
        #Remove 00 suffix from columns
        gdf_municipalities.columns = gdf_municipalities.columns.str.replace('00','')

        #Rename PLCIDFP to GEOID
        gdf_municipalities = gdf_municipalities.rename(columns = {'PLCIDFP':'GEOID'})

    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(int)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities


#Function to get county subdivisions 
def get_cousub_shapes(state_fips):
    if cog_year == '2002':
        path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_2002_{state_fips:02d}_cousub',f'tl_2010_{state_fips:02d}_cousub00.shp')
    else:
        path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_{year}_{state_fips:02d}_cousub',f'tl_{year}_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    if cog_year == '2002':
        #Remove 00 suffix from columns
        gdf_cousub.columns = gdf_cousub.columns.str.replace('00','')

        #Rename COSBIDFP to GEOID
        gdf_cousub = gdf_cousub.rename(columns = {'COSBIDFP':'GEOID'})

    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub

#Function to get block shape files
def get_block_shapes(state_fips):
    if year == '2020':
        path_block = os.path.join(base_shape_path,str(int(year) + 2), 'Blocks', f'tl_{str(int(year) + 2)}_{state_fips:02d}_tabblock20',
                                  f'tl_{str(int(year) + 2)}_{state_fips:02d}_tabblock20.shp')
    elif year == '2010':
        path_block = os.path.join(base_shape_path,str(int(year) + 2), 'Blocks', f'tabblock2010_{state_fips:02d}_pophu',
                                  f'tabblock2010_{state_fips:02d}_pophu.shp')
    else:
        path_block = os.path.join(base_shape_path,str(int(year) + 2), 'Blocks', f'tl_2010_{state_fips:02d}_tabblock00',
                                  f'tl_2010_{state_fips:02d}_tabblock00.shp')

    gdf_block = gpd.read_file(path_block)

    # If 2000 then need to merge with housing and population data too
    if year == '2000':
        census_data = pd.read_csv(os.path.join(config['raw_data'], '2000_census_block_data',f"census_block_data_{state_fips:02d}.csv"))
        #Make state two digit zero padded, county 3 digit, tract 6 and make block four digit
        census_data['state'] = census_data['state'].astype(str).str.zfill(2)
        census_data['county'] = census_data['county'].astype(str).str.zfill(3)
        census_data['tract'] = census_data['tract'].astype(str).str.zfill(6)
        census_data['block'] = census_data['block'].astype(str).str.zfill(4)

        gdf_block = pd.merge(census_data,gdf_block,left_on = ['state','county','tract','block'], right_on = ['STATEFP00','COUNTYFP00','TRACTCE00','BLOCKCE00'], how = 'inner')

        #If not a perfect 1-1 matching then raise error
        if len(gdf_block) != len(census_data):
            raise ValueError('Block shape file and census data do not match')

        #Double check for duplicates
        if gdf_block.duplicated(subset = ['state','county','tract','block']).sum() > 0:
            raise ValueError('Duplicates in block shape file')

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


# Function to calculate percent overlap
def calculate_percent_overlap(block_shapes, merged_shapes):
    percent_overlap_list = []

    for block in block_shapes.itertuples():
        block_shape = block.geometry
        intersection = block_shape.intersection(merged_shapes)

        if not intersection.is_empty:
            percent_overlap = (intersection.area / block_shape.area)
        else:
            percent_overlap = 0.0

        percent_overlap_list.append(percent_overlap)

    return percent_overlap_list

#Get the percent of the population covered by local governments
def get_per_pop(state_fips,matched_shapes):

    '''
    Make some merged shape file
    Then loop over each block and see if it intersects with a local government
    Will calculate the percentage overlap
    '''

    #First, get a merged shape object
    merged_shapes = gpd.GeoSeries(matched_shapes).unary_union

    #Now pull in block data
    block_shapes = get_block_shapes(state_fips)

    # Calculate the percent overlap for each block shape
    percent_overlap_list = calculate_percent_overlap(block_shapes, merged_shapes)

    # Add the percent overlap to the block_shapes GeoDataFrame
    block_shapes['percent_overlap'] = percent_overlap_list

    #Multiply variable by percent_overlap
    block_shapes['adjusted_var'] = block_shapes[block_var_name[var][year]] * block_shapes['percent_overlap']

    return block_shapes['adjusted_var'].sum()



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
    bridge = pd.read_excel(os.path.join(config['processed_data'],'Shape File Bridges',f'cog_{cog_year}_bridge.xlsx'))

    #Match set of local governments with shape files
    matched_shapes = match_shapes(places,cousubs,bridge, state_govs)
    
    #Get the % population by local govs
    results.loc[var] = get_per_pop(state_fips,matched_shapes)

    return results



if fix_2000:
    sample = pd.read_excel(os.path.join(config['raw_data'],'Census of Governments', 'census_of_gov_02_preprocessed.xlsx'))

else:

    # Load the sample data
    if year == '2020':
        sample = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))
    elif year == '2010':
        sample = pd.read_excel(os.path.join(config['raw_data'],'Census of Governments', 'census_of_gov_12_preprocessed.xlsx'))
    elif year == '2000':
        sample = pd.read_excel(os.path.join(config['raw_data'],'Census of Governments', 'census_of_gov_02_preprocessed.xlsx'))
    else:
        raise ValueError(f"Unsupported year: {year}")

#Filter out defunct munis/townships
sample = sample[~sample['CENSUS_ID_PID6'].isin(to_remove[cog_year])]


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
    pop_coverage_folder = os.path.join(config['processed_data'], 'pop_coverage'+year+'Block')
else:
    pop_coverage_folder = os.path.join(config['processed_data'], 'housing_coverage'+year+'Block')


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

print("Made it to start of for loop")

# Process each task and save the results
for sample_name, state in tasks_for_node:
    print(state)

    filename = os.path.join(pop_coverage_folder, f'{sample_name}#{state}.pkl')

    #If file already exists then continue
    if os.path.exists(filename):
        print('file already exists')
        continue

    try:
        result = main(samples[sample_name], state)

        # Save result as a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(result, f)

    except Exception as e:
        logging.error(f"Error processing task ({sample_name}, {state}): {str(e)}")




##


