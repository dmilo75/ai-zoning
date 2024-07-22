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
from fuzzywuzzy import process, fuzz

os.chdir('../../')

#Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
#Draw in all shape files for places and county subdivisions
base_shape_path = config['shape_files']

year = '2002'

shape_path = os.path.join(base_shape_path,year)

#%% Functions/mappings

#Manual mapping of unique ids between census of government and shape files for munis that don't match
#We map PID6 IDs to Geoids 
manual_mapping = {}

manual_mapping['2022'] = {
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

manual_mapping['2012'] = {}

manual_mapping['2002'] = {
    15204900800000: '1836003',
    27204700100000:'3011397',
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

to_remove['2002'] = []

#Function to get all fips place shape files for a state
def get_place_shapes(state_fips):
    if year == '2002':
        path_municipalities = os.path.join(shape_path,'Places',f'tl_2002_{state_fips:02d}_place',f'tl_2010_{state_fips:02d}_place00.shp')
    else:
        path_municipalities = os.path.join(shape_path,'Places',f'tl_{year}_{state_fips:02d}_place',f'tl_{year}_{state_fips:02d}_place.shp')
    gdf_municipalities = gpd.read_file(path_municipalities)
    if year == '2002':
        #Remove 00 suffix from columns
        gdf_municipalities.columns = gdf_municipalities.columns.str.replace('00','')

        #Rename PLCIDFP to GEOID
        gdf_municipalities = gdf_municipalities.rename(columns = {'PLCIDFP':'GEOID'})

    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(int)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities

#Function to get county subdivisions 
def get_cousub_shapes(state_fips):
    if year == '2002':
        path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_2002_{state_fips:02d}_cousub',f'tl_2010_{state_fips:02d}_cousub00.shp')
    else:
        path_cousub = os.path.join(shape_path,'County Subdivisions',f'tl_{year}_{state_fips:02d}_cousub',f'tl_{year}_{state_fips:02d}_cousub.shp')
    gdf_cousub = gpd.read_file(path_cousub)
    if year == '2002':
        #Remove 00 suffix from columns
        gdf_cousub.columns = gdf_cousub.columns.str.replace('00','')

        #Rename COSBIDFP to GEOID
        gdf_cousub = gdf_cousub.rename(columns = {'COSBIDFP':'GEOID'})

    gdf_cousub['COUSUBFP'] = gdf_cousub['COUSUBFP'].astype(int)
    gdf_cousub['COUNTYFP'] = gdf_cousub['COUNTYFP'].astype(int)
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub


def clean_name(name):
    # Define the list of unit types
    unit_types = [
        'CITY ',
        'CONSOLIDATED GOVERNMENT ',
        'CORPORATION ',
        'BOROUGH ',
        'MUNICIPALITY ',
        'VILLAGE ',
        'TOWN ',
        'PLANTATION ',
        'CIVIL TOWNSHIP ',
        'CHARTER TOWNSHIP ',
        'TOWNSHIP ',
    ]

    for unit_type in unit_types:
        if name.startswith(unit_type):
            name = name.replace(unit_type, '').strip().lower()
            break

    # Remove the word "of"
    name = name.replace(' of ', '').replace('of ','').strip()

    #Remove the term 'city' at the end
    name = name.replace(' city','').strip()

    #Remove the term ' peninsula' at the end
    name = name.replace(' peninsula','').strip()

    #Remove ' gateway' at the end
    name = name.replace(' gateway','').strip()

    return name

def clean_shape_name(name):

    #Make name lowercase
    name = name.lower().strip()

    #Remove the term 'City'
    name = name.replace(' city','').strip()

    #Remove ' town' from the end too
    name = name.replace(' town','').strip()

    #REmove ' peninsula' from the end
    name = name.replace(' peninsula','').strip()

    return name

def fuzzy_holder(row, which, places, cousubs):
    match =  fuzzy_matching(row, which, places, cousubs)

    #If no match and township then try muni and vice versa
    if match == None:
        if which == 'Township':
            match = fuzzy_matching(row, 'Muni', places, cousubs)
        else:
            match = fuzzy_matching(row, 'Township', places, cousubs)


    return match

def fuzzy_matching(row, which, places, cousubs, threshold=85):
    # Determine which GeoDataFrame to use based on 'which'
    if which == 'Muni':
        gdf = places.copy()
    else:
        gdf = cousubs.copy()
        gdf = cousubs[gdf['COUNTYFP'] == row['FIPS_COUNTY']]

    # Clean the name from the row
    name = clean_name(row['UNIT_NAME'])

    #Make name column lowercase in gdf
    gdf['NAME'] = gdf['NAME'].apply(lambda x: clean_shape_name(x))

    # Perform fuzzy matching
    choices = gdf['NAME'].tolist()
    best_match, score = process.extractOne(name, choices, scorer=fuzz.ratio)

    # Get the index of the best match
    best_match_index = choices.index(best_match)

    # Debugging prints
    print("Comparing:", name, "to", best_match, "with score:", score)

    # Check if the score meets the threshold
    if score >= threshold:
        matched_row = gdf.iloc[best_match_index]
        matched_geoid = matched_row['GEOID']
        # Return the GeoID and optionally other data
        return matched_geoid
    else:
        return None


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
        return selected.iloc[0]['GEOID']
    else:
        if year in ['2002','2012']:

            #Try manual matching
            try:
                geoid = manual_mapping[year][row['CENSUS_ID_PID6']]
                if (which == 'Township'):
                    return cousubs[cousubs['GEOID'] == geoid].iloc[0]['GEOID']
                if which == 'Muni':
                    return places[places['GEOID'] == geoid].iloc[0]['GEOID']
            except:
                if len(selected) > 1:
                    raise ValueError("Not unique match")
                    print(selected)
                else:
                    geoid =  fuzzy_holder(row,which,places,cousubs)

                    #If none then raise error
                    if geoid == None:
                        print(row['UNIT_NAME'])
                        print(row)
                        print(which)
                        if year == '2002':
                            return None
                        raise ValueError("Couldn't fuzzy match")
                    else:
                        return geoid

            #Here we will go into some kind of fuzzy matching function
        else:

            try:
                geoid = manual_mapping[year][row['CENSUS_ID_PID6']]
                if (which == 'Township') or (row['UNIT_NAME'] == 'CITY AND COUNTY OF HONOLULU'): #Honoloulou is exception
                    return cousubs[cousubs['GEOID'] == geoid].iloc[0]['GEOID']
                if which == 'Muni':
                    return places[places['GEOID'] == geoid].iloc[0]['GEOID']
            except:
                print(row['UNIT_NAME'])
                print(row)
                print(which)
                if len(selected) > 1:
                    raise ValueError("Not unique match")
                    print(selected)
                else:

                    raise ValueError("No match found")


'''
name = 'CENTENNIAL'

test = cousubs[cousubs['NAMELSAD'].str.contains(name.title())]

test = places[places['NAMELSAD'].str.contains(name.title())]


print(test['GEOID'])

'''


#Match census of governments to shape data
def match_shapes(places,cousubs,state_govs):
    
    matched = pd.Series([])


    for index, row in state_govs.iterrows():

        #Match munis with places
        if row['UNIT_TYPE'] == '2 - MUNICIPAL':
            matched.loc[row['CENSUS_ID_PID6']] = match(row,'Muni',places,cousubs)
            
        #Match townships with county subs
        elif row['UNIT_TYPE'] == '3 - TOWNSHIP':
            matched.loc[row['CENSUS_ID_PID6']] = match(row,'Township',places,cousubs)
            
        #Otherwise, county gov and skip
        else:
            continue
    
    return matched

if year == '2012':
    cog = pd.read_excel(os.path.join(config['raw_data'], 'Census of Governments','census_of_gov_12_preprocessed.xlsx'))
elif year == '2022':
    cog = pd.read_excel(os.path.join(config['raw_data'], 'Census of Governments','census_of_gov_22.xlsx'))
elif year == '2002':
    cog = pd.read_excel(os.path.join(config['raw_data'], 'Census of Governments','census_of_gov_02_preprocessed.xlsx'))
else:
    raise ValueError("Invalid year specified")


all_states = cog['FIPS_STATE'].unique()

cog = cog[~cog['CENSUS_ID_PID6'].isin(to_remove[year])]

#Filter out counties
cog = cog[cog['UNIT_TYPE'] != '1 - COUNTY']

#Define bridge
bridge = pd.DataFrame(columns = ['CENSUS_ID_PID6','GEOID','State'])

condition_go = False


for state_fips in all_states:

    print(state_fips)

    #Get set of all local governments
    state_govs = cog[cog['FIPS_STATE'] == state_fips]

    # Get fips places
    places = get_place_shapes(state_fips)
    
    # Get county subvidisions
    cousubs = get_cousub_shapes(state_fips)
    
    # Match set of local governments with shape files
    matched_shapes = match_shapes(places, cousubs, state_govs)

    #Make dataframe
    matched_shapes = matched_shapes.reset_index()
    matched_shapes.columns = ['CENSUS_ID_PID6','GEOID']
    matched_shapes['State'] = state_fips

    #Concat to bridge
    bridge = pd.concat([bridge,matched_shapes])

#Find rows in cog that have a 'None' for GEOID in bridge
missing = bridge[bridge['GEOID'].isnull()]

missing_rows = cog[cog['CENSUS_ID_PID6'].isin(missing['CENSUS_ID_PID6'])]

#Drop rows that have no match
bridge = bridge.dropna()

#Save the bridge to excel
bridge.to_excel(os.path.join(config['processed_data'],'Shape File Bridges',f'cog_{year}_bridge.xlsx'),index = False)


