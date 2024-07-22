import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import yaml
import math

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

shape_path = config['shape_files']
figures_path = config['figures_path']
yes_no_qs = config['yes_no_qs']
#Make yes_no_qs all strings
yes_no_qs = [str(q) for q in yes_no_qs] + ['17w']
numerical = config['numerical'] + ['27mean', '27min','28Min']

def ensure_directories(base_path, subfolders):
    for folder in [base_path] + [os.path.join(base_path, sub) for sub in subfolders]:
        os.makedirs(folder, exist_ok=True)

ensure_directories(os.path.join(figures_path, 'All Maps New'), ['Individual Maps', 'Combined Maps', 'Legends', 'Standalone Maps'])

# Define the states where townships are used
township_states = ['ct', 'mn', 'oh', 'il', 'mo', 'pa', 'in', 'ne', 'ri', 'ks', 'nh', 'sd', 'me', 'nj', 'vt', 'ma', 'ny', 'wi', 'mi', 'nd']

#Pull in census of governments file
cog = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))

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
    path_municipalities = os.path.join(shape_path, "Places", f"tl_2022_{state_fips}_place", f"tl_2022_{state_fips}_place.shp")
    gdf_municipalities = gpd.read_file(path_municipalities)
    gdf_municipalities['TYPE'] = 'place'
    return gdf_municipalities

def load_county_subdivisions(state_fips, shape_path):
    path_cousub = os.path.join(shape_path, "County Subdivisions", f"tl_2022_{state_fips}_cousub", f"tl_2022_{state_fips}_cousub.shp")
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
    base_path_water = os.path.join(shape_path, "Water", str(state_fips))
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

    # Convert directly to string and strip leading zeros
    gdf_municipalities['STATEFP'] = gdf_municipalities['STATEFP'].astype(str).str.lstrip('0')
    gdf_municipalities['PLACEFP'] = gdf_municipalities['PLACEFP'].astype(str).str.lstrip('0')

    try:
        gdf_municipalities['COUNTYFP'] = gdf_municipalities['COUNTYFP'].astype(str).str.lstrip('0')
        has_townships = True
    except:
        has_townships = False


    cog['FIPS_STATE'] = cog['FIPS_STATE'].astype(str)
    cog['FIPS_COUNTY'] = cog['FIPS_COUNTY'].astype(str)
    cog['FIPS_PLACE'] = cog['FIPS_PLACE'].astype(str)

    # Splitting the data into 'place' and 'cousub'
    places = gdf_municipalities[gdf_municipalities['TYPE'] == 'place']
    cousubs = gdf_municipalities[gdf_municipalities['TYPE'] == 'cousub']

    # Merging for 'place'
    merged_places = pd.merge(
        places,
        cog[['FIPS_STATE', 'FIPS_PLACE', 'CENSUS_ID_PID6']],
        how='left',
        left_on=['STATEFP', 'PLACEFP'],
        right_on=['FIPS_STATE', 'FIPS_PLACE']
    )
    if has_townships:
        # Merging for 'cousub'
        merged_cousubs = pd.merge(
            cousubs,
            cog[['FIPS_STATE', 'FIPS_COUNTY', 'FIPS_PLACE', 'CENSUS_ID_PID6']],
            how='left',
            left_on=['STATEFP', 'COUNTYFP', 'PLACEFP'],
            right_on=['FIPS_STATE', 'FIPS_COUNTY', 'FIPS_PLACE']
        )

        # Concatenate the results back into a single dataframe
        final_df = pd.concat([merged_places, merged_cousubs])

    else:
        final_df = merged_places

    #Drop rows with mising census id
    final_df = final_df[final_df['CENSUS_ID_PID6'].notnull()]


    return final_df


def load_geodata(state_abbrevs, state_fipses, all_ids, shape_path):
    all_gdf_municipalities = []
    all_gdfs_water = []

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

    gdf_municipalities = pd.concat(all_gdf_municipalities, ignore_index=True)
    gdf_municipalities = clean_fips(gdf_municipalities)

    #Add census id pid6 to gdf_municipalities
    gdf_municipalities = add_census_id_pid6(gdf_municipalities)

    #Ensure that all munis from sample have a shape file
    validate_fips_from_files(gdf_municipalities, all_ids)

    municipalities_with_text, municipalities_without_text = filter_municipalities(gdf_municipalities, all_ids)
    gdf_water_area = pd.concat(all_gdfs_water, ignore_index=True)

    return municipalities_with_text, municipalities_without_text, gdf_water_area


def create_data_dict(municipalities_with_text, municipalities_without_text, muni_answer_map, question):
    data_dict = {}

    condition_no_answer = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: np.isnan(muni_answer_map.get(x, np.nan)))

    if question in yes_no_qs:
        condition_yes = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: muni_answer_map.get(x, '') == 1.0)
        condition_no = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: muni_answer_map.get(x, '') == 0.0)
        data_dict['Yes'] = {'data': municipalities_with_text[condition_yes], 'color': '#1E90FF'}
        data_dict['No'] = {'data': municipalities_with_text[condition_no], 'color': '#FF4500'}

    elif question in numerical:
        answers = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: muni_answer_map.get(x, np.nan)).dropna().astype(float)
        min_answer = answers.min()
        q1, q2, q3 = answers.quantile([0.25, 0.5, 0.75])
        condition_q1 = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: min_answer <= muni_answer_map.get(x, -1) < q1)
        condition_q2 = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: q1 <= muni_answer_map.get(x, -1) < q2)
        condition_q3 = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: q2 <= muni_answer_map.get(x, -1) < q3)
        condition_q4 = municipalities_with_text['CENSUS_ID_PID6'].apply(lambda x: q3 <= muni_answer_map.get(x, -1))
        if not municipalities_with_text[condition_q1].empty:
            data_dict['First Quartile'] = {'data': municipalities_with_text[condition_q1], 'color': '#FFEBEB'}
        if not municipalities_with_text[condition_q2].empty:
            data_dict['Second Quartile'] = {'data': municipalities_with_text[condition_q2], 'color': '#FFA5A5'}
        if not municipalities_with_text[condition_q3].empty:
            data_dict['Third Quartile'] = {'data': municipalities_with_text[condition_q3], 'color': '#FF4C4C'}
        if not municipalities_with_text[condition_q4].empty:
            data_dict['Fourth Quartile'] = {'data': municipalities_with_text[condition_q4], 'color': '#8B0000'}
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

def plot_map(data_dict, gdf_counties, gdf_water_area, boundaries, city, question_text, figures_path, question):

    #Standalone Map


    # Create a new figure for the standalone map with legend
    fig_standalone, ax_standalone = plt.subplots(figsize=(8, 10),
                                                 dpi=100)  # Adjust the figsize for better slide dimensions
    ax_standalone.fill_between(x=[boundaries[0], boundaries[1]], y1=boundaries[2], y2=boundaries[3], color='#ADD8E6', zorder = 0)
    gdf_counties.plot(ax=ax_standalone, color=color_county, edgecolor='black', linewidth=0.5, zorder = 0)

    legend_handles = []
    for label, info in data_dict.items():
        need_plot_label = True

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
    ax_standalone.set_title('Question: '+question_text+'\n'+editable_title, fontsize=15)
    ax_standalone.set_xlim(boundaries[0:2])
    ax_standalone.set_ylim(boundaries[2:])
    ax_standalone.set_xticks([])
    ax_standalone.set_yticks([])
    ax_standalone.axis('off')

    offset = -0.165 + (1-int(len(legend_handles)/2))*-.05

    import matplotlib.colorbar

    # Set legend with patch handles and colorbar
    legend_handles_no_cbar = [h for h in legend_handles if not isinstance(h, matplotlib.colorbar.ColorbarBase)]
    legend = ax_standalone.legend(handles=legend_handles_no_cbar, fontsize=20, loc='lower center', ncol=2,
                                  bbox_to_anchor=(0.5, offset), frameon=False)


def create_combined_image(question, figures_path, full_params):
    imgs = [Image.open(os.path.join(figures_path, f"All Maps New\Individual Maps\{question_text}_{city}.png")) for city in full_params.keys()]
    img_width, img_height = imgs[0].size
    legend_img = Image.open(os.path.join(figures_path, f"All Maps New\Legends\{question}_legend.png"))
    new_legend_width = img_width * 2
    aspect_ratio = legend_img.width / legend_img.height
    new_legend_height = int(new_legend_width / aspect_ratio)
    legend_img = legend_img.resize((new_legend_width, new_legend_height))
    new_img = Image.new("RGBA", (img_width * 2, img_height * 2 + new_legend_height), "white")

    for i, img in enumerate(imgs):
        row = i // 2
        col = i % 2
        x_offset = col * img_width
        y_offset = row * img_height
        new_img.paste(img, (x_offset, y_offset))

    new_img.paste(legend_img, (0, img_height * 2))

    if new_img.mode != 'RGB':
        new_img = new_img.convert('RGB')

    new_img.save(os.path.join(figures_path, f"All Maps New\\Combined Maps\\{question}_combined.jpg"), 'JPEG', quality=20)
    print(f"Combined image for question {question} saved.")

def load_data_for_muni_and_question( params, question, results, shape_path):
    state_abbrevs = params['abbrev']
    state_fipses = params['fips']

    res = results[results['Question'] == question].copy()

    #Filter for state abbrevs
    res['State'] = res['Muni'].apply(lambda x: x.split('#')[2])
    res = res[res['State'].isin(state_abbrevs)]

    #Turn to list of dictionaries
    filtered_data = res.to_dict('records')

    #Get all ids
    all_ids = [item['CENSUS_ID_PID6'] for item in filtered_data]

    # Load the geodata and prepare the municipalities data
    municipalities_with_text, municipalities_without_text, gdf_water_area = load_geodata(state_abbrevs, state_fipses,all_ids, shape_path)

    #Map id to answer
    muni_answer_map = {item['CENSUS_ID_PID6']: item['Answer'] for item in filtered_data}
    
    # Create the data dictionary for plotting
    data_dict = create_data_dict(municipalities_with_text, municipalities_without_text, muni_answer_map, question)

    # Load the counties geodata
    path_counties = os.path.join(shape_path, "Counties", "tl_2022_us_county.shp")
    gdf_counties = gpd.read_file(path_counties)
    gdf_counties = gdf_counties[gdf_counties['STATEFP'].isin(state_fipses)]

    return data_dict, gdf_counties, gdf_water_area

def calculate_city_boundaries(full_params):
    for city, params in full_params.items():
        boundary_length = 0.75 * 100 if city == 'Boston' else 100
        params['boundaries'] = calculate_boundaries(params['lat'], params['long'], boundary_length)
    return full_params

def load_summary_data(config):
    return pd.read_excel(os.path.join(config['raw_data'], "Sample Data.xlsx"))

def load_and_preprocess_results(model):

    #Pull in light data
    df = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Light Data.xlsx'))
    df['Answer'] = df['Answer'].apply(clean_answer).astype(float)

    #Now try to pull in PCA indices
    pca = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Overall_Index.xlsx'))

    #Take the first three principal component columns
    pca = pca[['Muni','First_Principal_Component','Second_Principal_Component','Third_Principal_Component']]

    # Rename to PCA1, PCA2, and PCA3
    pca.columns = ['Muni','PCA1','PCA2','PCA3']

    #Extract census id
    pca['CENSUS_ID_PID6'] = pca['Muni'].apply(lambda x: x.split('#')[1])

    #Reshape long
    pca = pca.melt(id_vars = ['Muni','CENSUS_ID_PID6'], value_vars = ['PCA1','PCA2','PCA3'], var_name = 'Component', value_name = 'Answer')

    #Rename 'Component' as 'Question'
    pca['Question'] = pca['Component']

    #Drop 'Component'
    pca = pca.drop('Component',axis = 1)

    #Append to df
    df = pd.concat([df,pca],ignore_index = True)

    #Ensure census id is type int
    df['CENSUS_ID_PID6'] = df['CENSUS_ID_PID6'].astype(int)

    #Ensure answer is type float
    df['Answer'] = df['Answer'].astype(float)


    return df

##Main code

# Define the parameters for each city
full_params = { }
'''
'San Francisco': {'abbrev': ['ca'], 'fips': ['06'], 'lat': 37.7749, 'long': -122.4194, 'loc': 'lower left'},
'Chicago': {'abbrev': ['il', 'in'], 'fips': ['17', '18'], 'lat': 41.8781, 'long': -87.6298, 'loc': 'upper right'},
'Raleigh': {'abbrev': ['nc'], 'fips': ['37'], 'lat': 35.787743, 'long': -78.644257, 'loc': 'upper right'},
'Philadelphia': {'abbrev': ['pa','nj','de'], 'fips': ['42','34','10'], 'lat': 39.9526, 'long': -75.1652, 'loc': 'upper left'},
'Atlanta': {'abbrev': ['ga'], 'fips': ['13'], 'lat': 33.7488, 'long': -84.3877, 'loc': 'upper left'},
'Boston': {'abbrev': ['ma'], 'fips': ['25'], 'lat': 42.3601, 'long': -71.0589, 'loc': 'upper left'},
'Houston': {'abbrev': ['tx'], 'fips': ['48'], 'lat': 29.7604, 'long': -95.3698, 'loc': 'upper left'},
'Dallas': {'abbrev': ['tx'], 'fips': ['48'], 'lat': 32.7767, 'long': -96.7970, 'loc': 'upper left'},
}
'''
full_params['New York City'] = {'abbrev': ['ny', 'nj', 'ct'], 'fips': ['36', '34', '09'], 'lat': 40.7128, 'long': -74.0060, 'loc': 'upper right'}

# Calculate the boundaries for each city
full_params = calculate_city_boundaries(full_params)

# Define the question map
question_map = {
    '17': 'Does the zoning bylaw/ordinance include any mandates\nor incentives for development of affordable units?',
    '28Min': 'Minimum minimum lot size',
'PCA1': 'First Principal Component',
'PCA2': 'Second Principal Component',
}

# Load the summary data
summary_data = load_summary_data(config)

# Define the model name
model = "national_run_may"

# Load and preprocess the results data (pulls in light data)
results = load_and_preprocess_results(model)

# Loop over each city and question
for city, params in full_params.items():
    print(city)

    for question, question_text in question_map.items():
        print(question)

        # Load the data for the current municipality and question
        data_dict, gdf_counties, gdf_water_area = load_data_for_muni_and_question( params, question, results, shape_path)

        # Plot the map
        plot_map(data_dict, gdf_counties, gdf_water_area, params['boundaries'], city, question_text, figures_path, question)

    break

# Create the combined images for each question
#for question in question_map.keys():
    #create_combined_image(question, figures_path, full_params)




