import geopandas as gpd
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import yaml

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

shape_path = config['shape_files']
text_path = config['muni_text']
figures_path = config['figures_path']
yes_no_qs = config['yes_no_qs']
numerical = config['numerical'] + ['27mean', '27min']  # Add in lot size questions

model = "llama-13b"  # Which model results you want to run maps for

# Check to see if we have the necessary export folders and if not then make them

# Folders to create within 'All Maps'
subfolders = ['Individual Maps', 'Combined Maps', 'Legends', 'Standalone Maps']

# Initialize with the 'All Maps' folder
current_folder = os.path.join(figures_path, 'All Maps New')

# Loop to create folders
for folder in [current_folder] + [os.path.join(current_folder, sub) for sub in subfolders]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Parameters for each city, the lat and long are for city centers
full_params = {
    'San Francisco': {'abbrev': ['ca'], 'fips': ['06'], 'lat': 37.7749, 'long': -122.4194, 'loc': 'lower left'},
    'Chicago': {'abbrev': ['il', 'in'], 'fips': ['17', '18'], 'lat': 41.8781, 'long': -87.6298, 'loc': 'upper right'},
    # Since Chicago is so close to Indiana we map Indiana as well
    'Atlanta': {'abbrev': ['ga'], 'fips': ['13'], 'lat': 33.7488, 'long': -84.3877, 'loc': 'upper left'},
    'Boston': {'abbrev': ['ma'], 'fips': ['25'], 'lat': 42.3601, 'long': -71.0589, 'loc': 'upper left'}
}

length = 100  # Length of each side of the square in km

# Make boundaries
for city, params in full_params.items():
    lat = params['lat']
    long = params['long']

    if city == 'Boston':  # For boston we use a roughly 75kmx75km boundary to focus on just Massachussetts
        L = .75 * length
    else:
        L = length

    # Calculate the new boundaries: [long_min, long_max, lat_min, lat_max]
    delta_lat = L / 111.32  # Change in latitude
    delta_long = L / (111.32 * math.cos(math.radians(lat)))  # Change in longitude

    boundaries = [long - delta_long / 2, long + delta_long / 2, lat - delta_lat / 2, lat + delta_lat / 2]

    # Add the new boundaries to the dictionary
    params['boundaries'] = boundaries

# List of questions to make maps for
question_map = {
    17: 'Does the zoning bylaw/ordinance include any mandates\nor incentives for development of affordable units?',
    '27min': 'Minimum minimum lot size',
}

# %%Loop over each city and question and make the map
for city, params in full_params.items():
    for question, question_text in question_map.items():

        params = full_params[city]

        state_abbrevs = params['abbrev']
        state_fipses = params['fips']

        # %%Load in model output

        # Let's use Sample Data Excel to map to state and filter
        sample = pd.read_excel(os.path.join(config['raw_data'], "Sample Data.xlsx"))

        # res = pd.read_excel(os.path.join(config['processed_data'],model+'.xlsx'),index_col = 0)
        res = pd.read_excel(os.path.join(config['processed_data'], "Full Run.xlsx"), index_col=0)
        res.rename(columns={'fips': 'CENSUS_ID_PID6'}, inplace=True)
        res = pd.merge(res, sample[['CENSUS_ID_PID6','FIPS_PLACE']], on='CENSUS_ID_PID6', how='left')
        res = res.rename(columns={'FIPS_PLACE': 'fips'})
        res['Muni'] = res['Muni'] + '#' + res['muni_fips'].astype(str)

        # Update 'Question' based on 'Type'
        mask = res['Question'] == 27
        res.loc[mask & (res['Type'] == 'Mean'), 'Question'] = '27mean'
        res.loc[mask & (res['Type'] == 'Min'), 'Question'] = '27min'

        all_data = res.to_dict('records')

        # Filter the sample dataframe based on 'State' column for multiple states if given
        sample = sample[sample['State'].isin(state_abbrevs)]

        # Convert FIPS_PLACE to string and then concatenate with UNIT_NAME
        sample['FIPS_PLACE_str'] = sample['FIPS_PLACE'].astype(str)

        # Extract the UNIT_NAME and FIPS_PLACE pairs from sample to a set for faster lookup
        valid_pairs = set(sample[['UNIT_NAME', 'FIPS_PLACE_str']].agg('#'.join, axis=1))

        # Filter all_data based on the criteria
        filtered_data = [
            item for item in all_data
            if item['Muni'].split('#')[0] + "#" + item['Muni'].split('#')[1] in valid_pairs and item.get(
                'Question') == question
        ]

        # %%Load in data

        all_gdf_municipalities = []
        all_fips_from_files = []
        all_gdfs_water = []

        # Update a few FIPS codes for Massachusetts before the matching process
        fips_mapping = {
            '27060': '27100',
            '56000': '55955',
            '08130': '08085',
        }


        def update_fips(row):
            """Update FIPS codes based on the mapping."""
            if row['PLACEFP'] in fips_mapping:
                return fips_mapping[row['PLACEFP']]
            return row['PLACEFP']


        for state_abbrev, state_fips in zip(state_abbrevs, state_fipses):

            # Load the municipalities shapefile
            # path_municipalities = shape_path+rf"\\Places\tl_2022_{state_fips}_place\tl_2022_{state_fips}_place.shp"
            path_municipalities = os.path.join(shape_path, "Places", f"tl_2022_{state_fips}_place",
                                               f"tl_2022_{state_fips}_place.shp")
            gdf_municipalities = gpd.read_file(path_municipalities)
            gdf_municipalities['TYPE'] = 'place'

            # Townships are a common form of local-level government in Massachussetts which correspond to county subdivisions
            if state_abbrev == 'ma':
                # Load county subdivisions for Massachusetts
                path_cousub = os.path.join(shape_path, "County Subdivisions", f"tl_2022_{state_fips}_cousub",
                                           f"tl_2022_{state_fips}_cousub.shp")
                gdf_cousub = gpd.read_file(path_cousub)
                gdf_cousub = gdf_cousub.rename(columns={'COUSUBFP': 'PLACEFP'})
                gdf_cousub['TYPE'] = 'cousub'

                # Combine municipalities and county subdivisions
                gdf_municipalities = pd.concat([gdf_municipalities, gdf_cousub], ignore_index=True)

                # Apply the update_fips function to the 'PLACEFP' column
                gdf_municipalities['PLACEFP'] = gdf_municipalities.apply(update_fips, axis=1)

                # Handle prioritizing 'place' over 'cousub' in case of duplicate matches
                gdf_municipalities.sort_values(by="TYPE", ascending=False, inplace=True)
                gdf_municipalities.drop_duplicates(subset="PLACEFP", keep="first", inplace=True)

            all_gdf_municipalities.append(gdf_municipalities)

            # Get list of all FIPS codes from text file names
            fips_from_files = sample[sample['State'] == state_abbrev]['FIPS_PLACE'].to_list()
            all_fips_from_files.extend(fips_from_files)

            # Base path where state-wise water area files are located
            base_path_water = os.path.join(shape_path, "Water", str(state_fips))

            # Automatically generate the list of county FIPS codes from subdirectories
            county_fips = [name for name in os.listdir(base_path_water) if
                           os.path.isdir(os.path.join(base_path_water, name))]

            # Loop through each FIPS code, load the shapefile, and merge
            for fips in county_fips:
                path_water = os.path.join(base_path_water, fips, f"tl_2022_{state_fips}{fips}_areawater.shp")
                all_gdfs_water.append(gpd.read_file(path_water))

        # Concatenate all loaded municipalities (if there's more than one state)
        gdf_municipalities = pd.concat(all_gdf_municipalities, ignore_index=True)

        # Ensure that every text file FIPS for Massachusetts matched.
        if 'ma' in state_abbrevs:
            unmatched_fips_from_files = set([str(fips) for fips in all_fips_from_files]) - set(
                gdf_municipalities['PLACEFP'].astype(int).astype(str).tolist())

            if unmatched_fips_from_files:
                raise ValueError(
                    f"Some FIPS from text files did not match the shapefile data: {unmatched_fips_from_files}")

            # Filter out gdf_municipalities entries that don't have a matching FIPS in the text files
            gdf_municipalities = gdf_municipalities[gdf_municipalities['PLACEFP'].astype(int).isin(all_fips_from_files)]

        # Combine all loaded water geodataframes
        gdf_water_area = pd.concat(all_gdfs_water, ignore_index=True)

        # Split gdf_municipalities
        gdf_municipalities_with_text = gdf_municipalities[
            gdf_municipalities['PLACEFP'].astype(int).isin(all_fips_from_files)]
        gdf_municipalities_without_text = gdf_municipalities[
            ~gdf_municipalities['PLACEFP'].astype(int).isin(all_fips_from_files)]

        # Create a dictionary from filtered_data for easier lookup
        muni_answer_map = {item['Muni'].split('#')[1]: item['Answer'] for item in filtered_data}

        # Get list of FIPS codes from gdf_municipalities_with_text but not in filtered_data
        missing_fips = gdf_municipalities_with_text[
            ~gdf_municipalities_with_text['PLACEFP'].astype(int).astype(str).isin(muni_answer_map.keys())]

        # Append these to gdf_municipalities_without_text
        gdf_municipalities_without_text = pd.concat([gdf_municipalities_without_text, missing_fips], axis=0)

        # Now, remove the unmatched FIPS from gdf_municipalities_with_text
        gdf_municipalities_with_text = gdf_municipalities_with_text[
            gdf_municipalities_with_text['PLACEFP'].astype(int).astype(str).isin(muni_answer_map.keys())]

        # Create an empty data_dict to store the geodataframe slices and their associated colors
        data_dict = {}

        condition_no_answer = gdf_municipalities_with_text['PLACEFP'].apply(
            lambda x: np.isnan(muni_answer_map.get(str(int(x)), np.nan))
        )

        # Conditions and splits for Yes/No questions
        if question in yes_no_qs:

            # Existing conditions for yes_no questions
            condition_yes = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: muni_answer_map.get(str(int(x)), '') == 1
            )
            condition_no = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: muni_answer_map.get(str(int(x)), '') == 0
            )

            data_dict['Yes'] = {'data': gdf_municipalities_with_text[condition_yes], 'color': '#1E90FF'}  # DodgerBlue
            data_dict['No'] = {'data': gdf_municipalities_with_text[condition_no], 'color': '#FF4500'}  # OrangeRed


        # Conditions and splits for numerical questions
        elif question in numerical:
            # Determine quartiles for numerical questions
            answers = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: muni_answer_map.get(str(int(x)), np.nan)
            ).dropna().astype(float)
            q1 = answers.quantile(0.25)
            q2 = answers.quantile(0.5)
            q3 = answers.quantile(0.75)

            # Define conditions based on quartiles

            condition_q1 = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: 0 <= muni_answer_map.get(str(int(x)), -1) < q1
            )
            condition_q2 = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: q1 <= muni_answer_map.get(str(int(x)), -1) < q2
            )
            condition_q3 = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: q2 <= muni_answer_map.get(str(int(x)), -1) < q3
            )
            condition_q4 = gdf_municipalities_with_text['PLACEFP'].apply(
                lambda x: q3 <= muni_answer_map.get(str(int(x)), -1)
            )

            # Split based on conditions and add to the data dictionary with gradient of red colors
            if not gdf_municipalities_with_text[condition_q1].empty:
                data_dict['First Quartile'] = {'data': gdf_municipalities_with_text[condition_q1],
                                               'color': '#FFEBEB'}  # Very light red
            if not gdf_municipalities_with_text[condition_q2].empty:
                data_dict['Second Quartile'] = {'data': gdf_municipalities_with_text[condition_q2],
                                                'color': '#FFA5A5'}  # Light red
            if not gdf_municipalities_with_text[condition_q3].empty:
                data_dict['Third Quartile'] = {'data': gdf_municipalities_with_text[condition_q3],
                                               'color': '#FF4C4C'}  # Bright red
            if not gdf_municipalities_with_text[condition_q4].empty:
                data_dict['Fourth Quartile'] = {'data': gdf_municipalities_with_text[condition_q4],
                                                'color': '#8B0000'}  # Dark red

        # Add "Out of Sample" to data_dict if it's non-empty
        if not gdf_municipalities_without_text.empty:
            data_dict["Out of Sample"] = {
                'data': gdf_municipalities_without_text,
                'color': '#D3D3D3'  # Light gray
            }

        # Add "I don't know"
        if not gdf_municipalities_with_text[condition_no_answer].empty:
            data_dict["I Don't Know"] = {'data': gdf_municipalities_with_text[condition_no_answer],
                                         'color': '#666666'}  # Dark Grey

        path_counties = os.path.join(shape_path, "Counties", "tl_2022_us_county.shp")
        gdf_counties = gpd.read_file(path_counties)
        gdf_counties = gdf_counties[gdf_counties['STATEFP'].isin(state_fipses)]  # Filter for state

        # %%Plotting code

        # Simplified plotting using the data_dict
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        # Draw a rectangle for the background inside the boundaries
        boundaries = params['boundaries']  # [xmin, xmax, ymin, ymax]
        ax.fill_between(x=[boundaries[0], boundaries[1]], y1=boundaries[2], y2=boundaries[3], color='#ADD8E6')

        # Plot county boundaries
        color_county = '#A9A9A9'  # Darker gray for counties
        gdf_counties.plot(ax=ax, color=color_county, edgecolor='black', linewidth=0.5)

        # Loop through data_dict to plot each category
        print(data_dict.items())
        legend_handles = []
        for label, info in data_dict.items():
            info['data'].plot(ax=ax, color=info['color'], edgecolor='black')
            patch = mpatches.Patch(color=info['color'], label=label)
            legend_handles.append(patch)

        # Plot water bodies
        gdf_water_area.plot(ax=ax, color='#ADD8E6', edgecolor='black', linewidth=0)

        # Add a thin black border around the entire map
        ax.add_patch(patches.Rectangle((boundaries[0], boundaries[2]), boundaries[1] - boundaries[0],
                                       boundaries[3] - boundaries[2],
                                       fill=False, edgecolor='black', linewidth=1))

        ax.set_title(city, fontsize=20)

        # Zoom into the specified area
        ax.set_xlim(boundaries[0:2])
        ax.set_ylim(boundaries[2:])

        # Remove longitude and latitude data (ticks and labels)
        ax.set_xticks([])
        ax.set_yticks([])

        # Turn off the axis border
        ax.axis('off')

        # Construct the path for the saved image
        save_path = os.path.join(figures_path, "All Maps New", "Individual Maps", f"{str(question)}_{city}.png")

        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        offset = -0.165

        if len(legend_handles) > 4:
            offset = -0.24

        # Add the legend to the map
        ax.legend(handles=legend_handles, fontsize=20, loc='lower center', ncol=len(legend_handles),
                  bbox_to_anchor=(0.5, offset), ncols=2, frameon=False)

        # Set or edit the title
        editable_title = f"{question_text}\n{city}"  # Replace this with your desired title
        ax.set_title(editable_title, fontsize=20)

        # Save the figure with the map and integrated legend in the 'Standalone Maps' directory
        standalone_save_path = os.path.join(figures_path, "All Maps New", "Standalone Maps",
                                            f"{str(question)}_{city}_standalone.png")
        fig.savefig(standalone_save_path, dpi=300, bbox_inches='tight')

        # Show the figure with the map and legend (optional)
        plt.show()

        # %%

        # Create a new figure and axis for the legend
        fig_legend, ax_legend = plt.subplots(figsize=(20, 2.5), dpi=300)  # Adjust the size as needed
        ax_legend.axis('off')

        # Add the legend using the handles
        ax_legend.legend(handles=legend_handles, fontsize=30, facecolor='#FFFFFF', loc='center', ncol=2,
                         edgecolor='none')

        # Save the legend figure
        fig_legend.savefig(os.path.join(figures_path, "All Maps new", "Legends", f"{str(question)}_legend.png"),
                           dpi=300)

        # Show the legend (optional)
        # plt.show()

# %%Loop over each question and make 2x2 plot with legend


# Loop through each question
for question in question_map.keys():

    # %%

    # Initialize an empty list to hold the images for each city
    imgs = []

    # Loop through each city and read in the images
    for city in full_params.keys():
        imgs.append(Image.open(os.path.join(figures_path, f"All Maps New\Individual Maps\{str(question)}_{city}.png")))

    img_width, img_height = imgs[0].size

    # Load the legend image
    legend_img = Image.open(os.path.join(figures_path, f"All Maps New\Legends\{str(question)}_legend.png"))

    # Calculate new dimensions for the legend
    new_legend_width = img_width * 2
    aspect_ratio = legend_img.width / legend_img.height
    new_legend_height = int(new_legend_width / aspect_ratio)

    # Resize the legend while maintaining its original aspect ratio
    legend_img = legend_img.resize((new_legend_width, new_legend_height))

    # Create a new image with enough height to accommodate the 2x2 grid and the resized legend
    new_img = Image.new("RGBA", (img_width * 2, img_height * 2 + new_legend_height), "white")

    # Paste each city image into the new image to form a 2x2 grid
    for i, img in enumerate(imgs):
        row = i // 2  # Row index (0 or 1)
        col = i % 2  # Column index (0 or 1)

        x_offset = col * img_width
        y_offset = row * img_height

        new_img.paste(img, (x_offset, y_offset))

    # Paste the resized legend image below the 2x2 grid
    new_img.paste(legend_img, (0, img_height * 2))

    # Save the new image
    new_img.save(os.path.join(figures_path, f"All Maps New\Combined Maps\{str(question)}_combined.png"))

    print(f"Combined image for question {question} saved.")


