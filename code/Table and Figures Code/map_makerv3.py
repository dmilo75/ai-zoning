import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import yaml
import math
import matplotlib.colors as mcolors

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

shape_path = config['shape_files']
figures_path = config['figures_path']

#Make yes_no_qs all strings
yes_no_qs = [4, 5, 6, 8, 9, 11, 13, 14, 17, 20, 21] + ['17w']
yes_no_qs = [str(q) for q in yes_no_qs]
numerical =['2','22','27mean', '27min','28Min']

def ensure_directories(base_path, subfolders):
    for folder in [base_path] + [os.path.join(base_path, sub) for sub in subfolders]:
        os.makedirs(folder, exist_ok=True)

ensure_directories(os.path.join(figures_path, 'All Maps New'), ['Individual Maps', 'Combined Maps', 'Legends', 'Standalone Maps'])

# Define the states where townships are used
township_states = ['ct', 'mn', 'oh', 'il', 'mo', 'pa', 'in', 'ne', 'ri', 'ks', 'nh', 'sd', 'me', 'nj', 'vt', 'ma', 'ny', 'wi', 'mi', 'nd']

#Pull in census of governments file
cog = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))

#Draw in bridge between shapes and cog
bridge = pd.read_excel(os.path.join(config['processed_data'], 'Shape File Bridges','cog_2022_bridge.xlsx'))

#Make 'GEOID' column type string
bridge['GEOID'] = bridge['GEOID'].astype(str)

##

def calculate_boundaries(lat, lon, length):
    delta_lat = length / 111.32
    delta_long = length / (111.32 * math.cos(math.radians(lat)))
    return [lon - delta_long / 2, lon + delta_long / 2, lat - delta_lat / 2, lat + delta_lat / 2]

def get_old_name(geoid, sample):
    row = sample[sample['CENSUS_ID_PID6'] == int(geoid)].iloc[0]
    return row['UNIT_NAME'].lower() + '#' + str(row['FIPS_PLACE']) + '#' + row['State']

def clean_answer(answer):
    try:
        return int(answer)
    except:
        if answer == 'Yes':
            return 1
        elif answer == 'No':
            return 0
        else:
            return np.nan

def update_fips(row):
    fips_mapping = {'27060': '27100', '56000': '55955', '08130': '08085'}
    return fips_mapping.get(row['PLACEFP'], row['PLACEFP'])

def load_municipalities( state_fips, shape_path):
    path_municipalities = os.path.join(shape_path,"2022", "Places", f"tl_2022_{state_fips}_place", f"tl_2022_{state_fips}_place.shp")
    gdf_municipalities = gpd.read_file(path_municipalities)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities

def get_block_shapes(state_fips,shape_path):
    state_fips = int(state_fips)
    path_block = os.path.join(shape_path,"2022",'Blocks',f'tl_2022_{state_fips:02d}_tabblock20',f'tl_2022_{state_fips:02d}_tabblock20.shp')
    gdf_block = gpd.read_file(path_block)
    return gdf_block

def load_county_subdivisions(state_fips, shape_path):
    path_cousub = os.path.join(shape_path,"2022", "County Subdivisions", f"tl_2022_{state_fips}_cousub", f"tl_2022_{state_fips}_cousub.shp")
    gdf_cousub = gpd.read_file(path_cousub)
    gdf_cousub = gdf_cousub.rename(columns={'COUSUBFP': 'PLACEFP'})
    gdf_cousub['TYPE'] = 'cousub'
    return gdf_cousub

def process_municipalities(state_abbrev, gdf_municipalities, gdf_cousub):
    if state_abbrev in township_states:
        gdf_municipalities = pd.concat([gdf_municipalities, gdf_cousub], ignore_index=True)
        gdf_municipalities['PLACEFP'] = gdf_municipalities.apply(update_fips, axis=1)
        gdf_municipalities.sort_values(by="TYPE", ascending=False, inplace=True)
        gdf_municipalities.drop_duplicates(subset="PLACEFP", keep="first", inplace=True)
    return gdf_municipalities

def load_water_data(state_fips, shape_path):
    base_path_water = os.path.join(shape_path,"2022", "Water", str(state_fips))
    county_fips = [name for name in os.listdir(base_path_water) if os.path.isdir(os.path.join(base_path_water, name))]
    gdfs_water = []
    for fips in county_fips:
        path_water = os.path.join(base_path_water, fips, f"tl_2022_{state_fips}{fips}_areawater.shp")
        gdfs_water.append(gpd.read_file(path_water))
    return gdfs_water

def validate_fips_from_files(gdf_municipalities, all_ids):

    #Get list of ids from gdf_municipalities
    shape_ids = gdf_municipalities['CENSUS_ID_PID6'].astype(int).tolist()

    #Check if all ids is a subset of shape ids
    if not set(all_ids).issubset(set(shape_ids)):
        #Print which ids are missing
        missing_ids = set(all_ids) - set(shape_ids)
        print(f"Missing ids: {missing_ids}")

        #Print names of towns from cog via missing ids
        missing_munis = cog[cog['CENSUS_ID_PID6'].isin(missing_ids)]
        missing_names = missing_munis['UNIT_NAME'].to_list()
        print(f"Missing names: {missing_names}")



def filter_municipalities(gdf_municipalities, all_ids):

    #Ensure all ids is a list of ints
    all_ids = [int(x) for x in all_ids]

    municipalities_with_text = gdf_municipalities[gdf_municipalities['CENSUS_ID_PID6'].astype(int).isin(all_ids)]
    municipalities_without_text = gdf_municipalities[~gdf_municipalities['CENSUS_ID_PID6'].astype(int).isin(all_ids)]
    return municipalities_with_text, municipalities_without_text


#Function to bridge fips place codes that are inconsistent between shape file and census of gov dataset
def clean_fips(gdf_municipalities):

    #For now we just hard code in the changes, if we find many then we will port to Excel

    if 'COUNTYFP' in gdf_municipalities.columns:

        # Oyster Bay township fips place 56000 missing, because should be fps of 55955, state code 36, county fips 59
        gdf_municipalities.loc[(gdf_municipalities['STATEFP'] == '36') & (gdf_municipalities['COUNTYFP'] == '059') & (gdf_municipalities['PLACEFP'] == '55955'), 'PLACEFP'] = '56000'

    return gdf_municipalities




def add_census_id_pid6(gdf_municipalities):

    #Remove preceding 0 in GEOID from gdf_municipalities
    gdf_municipalities['GEOID'] = gdf_municipalities['GEOID'].str.lstrip('0')

    #Merge in the bridge on GEOID
    gdf_municipalities2 = pd.merge(gdf_municipalities, bridge[['GEOID','CENSUS_ID_PID6']], on = 'GEOID', how = 'inner')

    return gdf_municipalities2


def load_geodata(state_abbrevs, state_fipses, shape_path):
    all_gdf_municipalities = []
    all_gdfs_water = []
    all_block_data = []

    for state_abbrev, state_fips in zip(state_abbrevs, state_fipses):

        #Load munis
        gdf_municipalities = load_municipalities(state_fips, shape_path)

        #Load county subdivisions
        gdf_cousub = load_county_subdivisions( state_fips, shape_path) if state_abbrev in township_states else None

        #Combine
        gdf_municipalities = process_municipalities(state_abbrev, gdf_municipalities, gdf_cousub)

        #Append to list of all
        all_gdf_municipalities.append(gdf_municipalities)

        #Load water data
        gdfs_water = load_water_data(state_fips, shape_path)
        all_gdfs_water.extend(gdfs_water)

        #Load block data
        gdf_block = get_block_shapes(state_fips, shape_path)
        all_block_data.append(gdf_block)

    gdf_municipalities = pd.concat(all_gdf_municipalities, ignore_index=True)
    gdf_municipalities = clean_fips(gdf_municipalities)

    #Add census id pid6 to gdf_municipalities
    gdf_municipalities = add_census_id_pid6(gdf_municipalities)

    gdf_water_area = pd.concat(all_gdfs_water, ignore_index=True)

    # Load the counties geodata
    path_counties = os.path.join(shape_path,"2022", "Counties", "tl_2022_us_county.shp")
    gdf_counties = gpd.read_file(path_counties)
    gdf_counties = gdf_counties[gdf_counties['STATEFP'].isin(state_fipses)]

    block_data = pd.concat(all_block_data, ignore_index=True)


    return gdf_municipalities, gdf_water_area, gdf_counties, block_data

def filter_holder(gdf_municipalities, all_ids):

    #Ensure that all munis from sample have a shape file
    validate_fips_from_files(gdf_municipalities, all_ids)

    municipalities_with_text, municipalities_without_text = filter_municipalities(gdf_municipalities, all_ids)


    return municipalities_with_text, municipalities_without_text


def create_data_dict(municipalities_with_text, municipalities_without_text, muni_answer_map, question):
    data_dict = {}

    condition_no_answer = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: np.isnan(muni_answer_map.get(x, np.nan)))

    if question in yes_no_qs:
        condition_yes = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: muni_answer_map.get(x, '') == 1.0)
        condition_no = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: muni_answer_map.get(x, '') == 0.0)
        data_dict['Yes'] = {'data': municipalities_with_text[condition_yes], 'color': '#1E90FF'}
        data_dict['No'] = {'data': municipalities_with_text[condition_no], 'color': '#FF4500'}
    else:
        municipalities_with_text['Answer'] = municipalities_with_text['CENSUS_ID_PID6'].apply(
            lambda x: muni_answer_map.get(x, np.nan))
        min_answer = municipalities_with_text['Answer'].min()
        max_answer = municipalities_with_text['Answer'].max()
        data_dict['Numerical'] = {'data': municipalities_with_text, 'cmap': 'Reds', 'vmin': min_answer,
                                  'vmax': max_answer}

    if not municipalities_without_text.empty:
        data_dict["Out of Sample"] = {'data': municipalities_without_text, 'color': '#D3D3D3'}
    if not municipalities_with_text[condition_no_answer].empty:
        data_dict["I Don't Know"] = {'data': municipalities_with_text[condition_no_answer], 'color': '#666666'}

    return data_dict

'''
Should make the below recursive 
'''
def assign_z_order(gdf):
    # Initialize the z-order column with default value 1 (on top)
    gdf['z_order'] = 2
    gdf['container_name'] = None  # New column for the name of the containing town

    #Reset index
    gdf = gdf.reset_index(drop=True)

    # Create a spatial index for the geometries in the GeoDataFrame
    spatial_index = gdf.sindex

    # Check for containment using spatial index to reduce comparisons
    for idx, shape in gdf.iterrows():

        shape = gdf.loc[idx]
        # Find possible candidates for interaction using the spatial index
        possible_matches_index = list(spatial_index.intersection(shape.geometry.bounds))
        # Ensure not comparing the shape with itself
        possible_matches_index.remove(idx)
        possible_matches = gdf.loc[possible_matches_index]

        # Further refine to those that actually contain the shape and are in the same state
        for idx2, other_shape in possible_matches.iterrows():
            # Ensure that we're not comparing the same item and they are in the same state
            if idx != idx2 and shape.geometry.contains(other_shape.geometry) and shape['STATEFP'] == other_shape[
                'STATEFP']:
                gdf.at[idx, 'z_order'] = 1  # This shape contains another, so it goes on the bottom

    return gdf


def plot_density_map(gdf_municipalities, gdf_counties, gdf_water_area, block_data, boundaries, city, figures_path):


    # Calculate housing unit density and then plot
    block_data['Density'] = block_data['HOUSING20'] / (block_data['ALAND20'] / 4046.86)

    # Filter out any blocks with 0 density or 0 ALAND20
    block_data = block_data[(block_data['Density'] > 0) & (block_data['ALAND20'] > 0)]

    # Calculate percentiles
    percentiles = np.percentile(block_data['Density'], np.arange(0, 101, 5))

    # Create a colormap and norm based on percentiles
    cmap_red = plt.get_cmap('Reds')
    cmap_green = plt.get_cmap('Greens')
    norm = mcolors.BoundaryNorm(percentiles, cmap_red.N, clip=True)

    # Perform a spatial join to identify intersecting blocks
    block_data = block_data.set_geometry('geometry')
    gdf_municipalities = gdf_municipalities.set_geometry('geometry')

    intersecting_blocks = gpd.sjoin(block_data, gdf_municipalities, how='inner', op='intersects')

    # Mark intersecting and non-intersecting blocks
    block_data['Intersect'] = block_data.index.isin(intersecting_blocks.index)

    # Standalone Map
    color_county = '#A9A9A9'  # Darker gray for counties

    # Create a new figure for the standalone map with legend
    fig_standalone, ax_standalone = plt.subplots(figsize=(8, 10),
                                                 dpi=100)  # Adjust the figsize for better slide dimensions
    ax_standalone.fill_between(x=[boundaries[0], boundaries[1]], y1=boundaries[2], y2=boundaries[3], color='#ADD8E6',
                               zorder=0)
    gdf_counties.plot(ax=ax_standalone, color=color_county, edgecolor='black', linewidth=0.5, zorder=0)



    # Plot intersecting blocks in green
    block_data[block_data['Intersect']].plot(ax=ax_standalone, column='Density', cmap=cmap_green, norm=norm,
                                             edgecolor='none')

    # Plot non-intersecting blocks in red
    block_data[~block_data['Intersect']].plot(ax=ax_standalone, column='Density', cmap=cmap_red, norm=norm,
                                              edgecolor='none')

    # Add color bars
    cbar_ax1 = fig_standalone.add_axes([0.9, 0.2, 0.02, 0.6])  # Adjust the position and size of the color bar
    sm1 = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm1.set_array([])
    fig_standalone.colorbar(sm1, cax=cbar_ax1, ticks=[])  # Remove ticks/labels from the first color bar

    cbar_ax2 = fig_standalone.add_axes([0.93, 0.2, 0.02, 0.6])  # Adjust the position and size of the color bar
    sm2 = plt.cm.ScalarMappable(cmap=cmap_green, norm=norm)
    sm2.set_array([])
    cbar2 = fig_standalone.colorbar(sm2, cax=cbar_ax2, format='%.2f')  # Round labels to the hundredths place
    cbar2.ax.tick_params(labelsize=12)  # Increase the font size of the colorbar labels
    cbar2.set_label('Housing Units per Acre (2020)', fontsize = 12)  # Add a label to the second color bar

    # Adjust the main plot to make space for the color bars
    fig_standalone.subplots_adjust(right=0.88)

    gdf_water_area.plot(ax=ax_standalone, color='#ADD8E6', edgecolor='black', linewidth=0, zorder=5)
    ax_standalone.add_patch(
        mpatches.Rectangle((boundaries[0], boundaries[2]), boundaries[1] - boundaries[0], boundaries[3] - boundaries[2],
                           fill=False, edgecolor='black', linewidth=1))
    ax_standalone.set_title(city, fontsize=15)
    ax_standalone.set_xlim(boundaries[0:2])
    ax_standalone.set_ylim(boundaries[2:])
    ax_standalone.set_xticks([])
    ax_standalone.set_yticks([])
    ax_standalone.axis('off')

    # Save the figure with the map and integrated legend in the 'Housing Density' directory
    standalone_save_path = os.path.join(figures_path, "All Maps New", "Housing Density", f"{city}_standalone.png")
    fig_standalone.savefig(standalone_save_path, dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig_standalone)



    ##

    return

def plot_map(data_dict, gdf_counties, gdf_water_area, boundaries, city, question_text, figures_path, question):

    #Standalone Map
    color_county = '#A9A9A9'  # Darker gray for counties

    # Create a new figure for the standalone map with legend
    fig_standalone, ax_standalone = plt.subplots(figsize=(8, 10),
                                                 dpi=100)  # Adjust the figsize for better slide dimensions
    ax_standalone.fill_between(x=[boundaries[0], boundaries[1]], y1=boundaries[2], y2=boundaries[3], color='#ADD8E6', zorder = 0)
    gdf_counties.plot(ax=ax_standalone, color=color_county, edgecolor='black', linewidth=0.5, zorder = 0)

    legend_handles = []
    for label, info in data_dict.items():
        need_plot_label = True

        info['data'] = assign_z_order(info['data'])

        grouped = info['data'].groupby('z_order')
        for z_order, group in grouped:
            if label == 'Numerical':
                group.plot(ax=ax_standalone, column='Answer', cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'],
                           edgecolor='black', zorder=z_order)
            elif label == 'Out of Sample':
                group.plot(ax=ax_standalone, color=info['color'], edgecolor='black', zorder=z_order-1)
            else:
                group.plot(ax=ax_standalone, color=info['color'], edgecolor='black', zorder=z_order)

            if need_plot_label:
                if label == 'Numerical':
                    sm = plt.cm.ScalarMappable(cmap=info['cmap'],
                                               norm=plt.Normalize(vmin=info['vmin'], vmax=info['vmax']))
                    sm._A = []
                    cbar = fig_standalone.colorbar(sm, ax=ax_standalone, fraction=0.046, pad=0.04)
                    legend_handles.append(cbar)
                else:
                    patch = mpatches.Patch(color=info['color'], label=label)
                    legend_handles.append(patch)
                need_plot_label = False

    gdf_water_area.plot(ax=ax_standalone, color='#ADD8E6', edgecolor='black', linewidth=0, zorder = 5)
    ax_standalone.add_patch(
        mpatches.Rectangle((boundaries[0], boundaries[2]), boundaries[1] - boundaries[0], boundaries[3] - boundaries[2],
                           fill=False, edgecolor='black', linewidth=1))
    ax_standalone.set_title('Question: '+question_text+'\n'+city, fontsize=15)
    ax_standalone.set_xlim(boundaries[0:2])
    ax_standalone.set_ylim(boundaries[2:])
    ax_standalone.set_xticks([])
    ax_standalone.set_yticks([])
    ax_standalone.axis('off')

    offset = -0.125 + (int(len(legend_handles)/2)-1)*-.05

    import matplotlib.colorbar

    # Set legend with patch handles and colorbar
    legend_handles_no_cbar = [h for h in legend_handles if not isinstance(h, matplotlib.colorbar.ColorbarBase)]
    legend = ax_standalone.legend(handles=legend_handles_no_cbar, fontsize=15, loc='lower center', ncol=2,
                                  bbox_to_anchor=(0.5, offset), frameon=False)

    # Save the figure with the map and integrated legend in the 'Standalone Maps' directory
    standalone_save_path = os.path.join(figures_path, "All Maps New", "Standalone Maps",
                                        f"{str(question)}_{city}_standalone.png")

    fig_standalone.savefig(standalone_save_path, dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig_standalone)


def load_data_for_muni_and_question( params, question, results, shape_path, gdf_municipalities):
    state_fipses = params['fips']

    #Filter for state
    res = results[results['FIPS_STATE'].astype(str).isin(state_fipses)]


    #Get all ids
    all_ids = res['CENSUS_ID_PID6'].tolist()

    # Load the geodata and prepare the municipalities data
    municipalities_with_text, municipalities_without_text = filter_holder(gdf_municipalities, all_ids)

    #Map id to answer
    muni_answer_map = results[['CENSUS_ID_PID6', question]].set_index('CENSUS_ID_PID6')[question].to_dict()
    
    # Create the data dictionary for plotting
    data_dict = create_data_dict(municipalities_with_text, municipalities_without_text, muni_answer_map, question)

    return data_dict

def calculate_city_boundaries(full_params):
    for city, params in full_params.items():
        boundary_length = 0.75 * 100 if city == 'Boston' else 100
        params['boundaries'] = calculate_boundaries(params['lat'], params['long'], boundary_length)
    return full_params

def load_summary_data(config):
    return pd.read_excel(os.path.join(config['raw_data'], "Sample Data.xlsx"))

def load_and_preprocess_results(model):

    #Pull in light data
    df = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Comprehensive Data.xlsx'))

    return df

##Main code

# Define the parameters for each city
full_params = {
'Atlanta': {'abbrev': ['ga'], 'fips': ['13'], 'lat': 33.7488, 'long': -84.3877, 'loc': 'upper left'},
'San Francisco': {'abbrev': ['ca'], 'fips': ['06'], 'lat': 37.7749, 'long': -122.4194, 'loc': 'lower left'},
'Chicago': {'abbrev': ['il', 'in'], 'fips': ['17', '18'], 'lat': 41.8781, 'long': -87.6298, 'loc': 'upper right'},
#'Philadelphia': {'abbrev': ['pa','nj','de'], 'fips': ['42','34','10'], 'lat': 39.9526, 'long': -75.1652, 'loc': 'upper left'},
#'Boston': {'abbrev': ['ma'], 'fips': ['25'], 'lat': 42.3601, 'long': -71.0589, 'loc': 'upper left'},
'Houston': {'abbrev': ['tx'], 'fips': ['48'], 'lat': 29.7604, 'long': -95.3698, 'loc': 'upper left'},
}

#full_params['New York City'] = {'abbrev': ['ny', 'nj', 'ct'], 'fips': ['36', '34', '09'], 'lat': 40.7128, 'long': -74.0060, 'loc': 'upper right'}

# Calculate the boundaries for each city
full_params = calculate_city_boundaries(full_params)

# Define the question map
question_map = {
'Question 17':'Affordable Housing',
'Question 28Min': 'Minimum minimum lot size',
#'First_PC': 'First Principal Component',
#'Second_PC': 'Second Principal Component',
}

questions = pd.read_excel(os.path.join(config['raw_data'], 'Questions.xlsx'))

#for i, row in questions.iterrows():
    #question_id = str(row['ID'])
   # if question_id not in ['27','28']:
        #question_map[question_id] = row['Pioneer Question']

# Load the summary data
summary_data = load_summary_data(config)

# Define the model name
model = "latest_combined"

# Load and preprocess the results data (pulls in light data)
results = load_and_preprocess_results(model)

# Loop over each city and question
for city, params in full_params.items():
    print(city)

    #Load the geodata for the city
    gdf_municipalities, gdf_water_area, gdf_counties, block_data = load_geodata(params['abbrev'], params['fips'], shape_path)

    #Make housing unit density map
    plot_density_map(gdf_municipalities, gdf_counties, gdf_water_area,block_data, params['boundaries'], city, figures_path)

    for question, question_text in question_map.items():
        print(question)


        # Load the data for the current municipality and question
        data_dict = load_data_for_muni_and_question( params, question, results, shape_path, gdf_municipalities)

        # Plot the map
        plot_map(data_dict, gdf_counties, gdf_water_area, params['boundaries'], city, question_text, figures_path, question)



##

