import pandas as pd
import os
import yaml
import helper_functions as hf
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from math import radians, sin, cos, sqrt, asin

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

# Read in sample data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"), index_col=0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])

base_shape_path = config['shape_files']

shape_path = os.path.join(base_shape_path,'2022')
cog_year = 2022


#Function to get all fips place shape files for a state
def get_place_shapes(state_fips):
    path_municipalities = os.path.join(shape_path,'Places',f'tl_{cog_year}_{state_fips:02d}_place',f'tl_{cog_year}_{state_fips:02d}_place.shp')
    gdf_municipalities = gpd.read_file(path_municipalities)
    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(int)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities

#Function to get county subdivisions
def get_cousub_shapes(state_fips):
    path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_{cog_year}_{state_fips:02d}_cousub',f'tl_{cog_year}_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub

def match_shapes(places,cousubs,bridge, state_govs):


    #Drop 'State' from bridge
    bridge = bridge.drop(columns = ['State'])

    #Merge into state_govs on CENSUS_ID_PID6
    state_govs = state_govs.merge(bridge, on = 'CENSUS_ID_PID6')

    #Make the type object for GEOID
    state_govs['GEOID'] = state_govs['GEOID'].astype(str)

    #Make muni ids which is GEOID and CENSUS_ID_PID6
    muni_ids = state_govs[['GEOID','CENSUS_ID_PID6']]

    #Remove preceding 0's from GEOID in places
    places['GEOID'] = places['GEOID'].str.lstrip('0')

    #Now subset places for GEOID in state_govs
    places_filt = places.merge(muni_ids, on = 'GEOID', how = 'inner')

    #Remove preceding 0's from GEOID in cousubs
    cousubs['GEOID'] = cousubs['GEOID'].str.lstrip('0')

    #Then subset cousubs for GEOID in state_govs
    cousubs_filt = cousubs.merge(muni_ids, on = 'GEOID', how = 'inner')

    #Concatanate the two
    matched = pd.concat([places_filt,cousubs_filt])

    return matched


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3958.8  # Radius of earth in miles
    return c * r

def get_nearest_metro(matched_shapes):
    # Read in CSV of metro centers
    metro_centers = pd.read_csv(os.path.join(config['raw_data'], 'MSA Center Points', 'city_centers.csv'))

    # Convert the CRS from EPSG:4269 to EPSG:4326
    matched_shapes = matched_shapes.to_crs(epsg=4326)

    # Adding new columns to matched_shapes
    matched_shapes['Nearest Metro Name'] = None
    matched_shapes['Miles to Metro Center'] = None

    # Iterating through each row in matched_shapes
    for idx, row in matched_shapes.iterrows():
        shape_point = row.geometry.centroid  # We use the centroid of the geometry
        lon1, lat1 = shape_point.x, shape_point.y

        closest_center = None
        min_distance = float('inf')

        for _, center_row in metro_centers.iterrows():
            lon2, lat2 = center_row['Longitude'], center_row['Latitude']
            distance = haversine(lon1, lat1, lon2, lat2)

            if distance <= 100 and distance < min_distance:
                closest_center = center_row['msa']
                min_distance = distance

        if closest_center:
            matched_shapes.at[idx, 'Nearest Metro Name'] = closest_center
            matched_shapes.at[idx, 'Miles to Metro Center'] = min_distance

    return matched_shapes


def get_vars(matched_shapes):

    matched_shapes = matched_shapes.set_index('CENSUS_ID_PID6')

    #Get the area in acres
    matched_shapes['Land Area Acres'] = matched_shapes['ALAND'] / 4046.85642

    #Nearest metro calculation
    matched_shapes = get_nearest_metro(matched_shapes)

    #Only keep new vars and census id
    new_vars = ['Land Area Acres','Nearest Metro Name','Miles to Metro Center']
    matched_shapes = matched_shapes.reset_index()[new_vars+['CENSUS_ID_PID6']]

    return matched_shapes

def main(sample, state_fips):

    # Get set of all local governments
    state_govs = sample[sample['FIPS_STATE'] == state_fips]

    # Get fips places
    places = get_place_shapes(state_fips)

    # Get county subvidisions
    cousubs = get_cousub_shapes(state_fips)

    # Load in bridge
    bridge = pd.read_excel(os.path.join(config['processed_data'], 'Shape File Bridges', f'cog_{cog_year}_bridge.xlsx'))

    # Match set of local governments with shape files
    matched_shapes = match_shapes(places, cousubs, bridge, state_govs)

    #Function to extract variables
    new_vars = get_vars(matched_shapes)

    return new_vars

new_vars = []

#Run for all states
for state in sample['FIPS_STATE'].unique():
    print(f"Running for state {state}")
    new_vars.append(main(sample, state))

new_vars_df = pd.concat(new_vars)

#Export to interim data
new_vars_df.to_excel(os.path.join(data_path, 'interim_data', 'Shape_data.xlsx'))




