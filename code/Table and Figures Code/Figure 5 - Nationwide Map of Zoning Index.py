import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd

import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = 'latest_combined'


#Draw in excel file of index from model
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Overall_Index.xlsx'))

#Get state from 'Muni'
df['State'] = df['Muni'].apply(lambda x: x.split('#')[2])

#Sort by state
df = df.drop(columns = 'Muni').sort_values(by = 'State')

#Count number of obs per 'State'
state_counts = df['State'].value_counts()

#Groupby 'State' and average 'First_Principal_Component', drop other columns
df = df.groupby('State').mean()

# Step 1: Load the shapefile
gdf_states = gpd.read_file(os.path.join(config['shape_files'],'State_Maps','cb_2018_us_state_500k.shp'))

# Add a column to the GeoDataFrame to indicate if the state has a small number of observations
gdf_states['small_obs'] = gdf_states['STUSPS'].apply(lambda x: state_counts.get(x.lower(), 0) < 2)


# Convert df to a DataFrame and reset index
df = df.reset_index()

#Rename 'State' to 'StateAbbrev'
df = df.rename(columns = {'State':'StateAbbrev'})

#Make 'StateAbbrev' uppercase
df['StateAbbrev'] = df['StateAbbrev'].apply(lambda x: x.upper())

# Step 2: Merge the geodataframe with your data
gdf_states = gdf_states.merge(df, left_on='STUSPS', right_on='StateAbbrev')

# Step 3: Plot the map

##
# Sizes for different versions
sizes = {'large': (18, 14), 'small': (.4*18, .4*14)}

# Define the font size variable
fontsize = 18

# Now, let's define the polygons to clip Alaska and Hawaii
polygon_ak = Polygon([(-170, 50), (-170, 72), (-130, 72), (-130, 50)])  # Alaska
polygon_hi = Polygon([(-161, 18), (-161, 23), (-154, 23), (-154, 18)])  # Hawaii

# Clip Alaska and Hawaii
ak_gdf = gdf_states[gdf_states['STUSPS'] == 'AK'].copy()
hi_gdf = gdf_states[gdf_states['STUSPS'] == 'HI'].copy()

ak_gdf['geometry'] = ak_gdf['geometry'].intersection(polygon_ak)
hi_gdf['geometry'] = hi_gdf['geometry'].intersection(polygon_hi)

# Loop over df columns except 'StateAbbrev'
for col in df.columns[1:]:
    if col not in ['First_Principal_Component', 'Second_Principal_Component']:
        continue

    for size_name, figsize in sizes.items():
        # Create figure and axes with different sizes
        fig, ax_main = plt.subplots(1, figsize=figsize)
        ax_main.set_aspect('equal')

        # Plot the mainland states with more than 2 observations
        mainland_more_than_2 = gdf_states[~gdf_states['STUSPS'].isin(['AK', 'HI']) & ~gdf_states['small_obs']]
        if not mainland_more_than_2.empty:
            mainland_more_than_2.plot(column=col, ax=ax_main, legend=False)

        # Set the colorbar label font size
        cb = fig.colorbar(ax_main.collections[0], ax=ax_main)
        cb.set_label(label='State Average Index Value', size=fontsize)

        # Adjust the colorbar tick label font size
        cb.ax.tick_params(labelsize=fontsize)

        # Plot the mainland states with less than 2 observations
        mainland_less_than_2 = gdf_states[~gdf_states['STUSPS'].isin(['AK', 'HI']) & gdf_states['small_obs']]
        if not mainland_less_than_2.empty:
            mainland_less_than_2.plot(color='grey', ax=ax_main, legend=False)

        # Create inset axes for Alaska and Hawaii
        ax_ak = fig.add_axes([0.1, 0.1, 0.2, 0.2], anchor='SW', axes_class=plt.Axes)
        ax_hi = fig.add_axes([0.3, 0.1, 0.15, 0.15], anchor='SW', axes_class=plt.Axes)

        # Plot Alaska and Hawaii
        if not ak_gdf[~ak_gdf['small_obs']].empty:
            ak_gdf[~ak_gdf['small_obs']].plot(column=col, ax=ax_ak)
        else:
            ak_gdf[ak_gdf['small_obs']].plot(color='grey', ax=ax_ak, legend=False)
        if not hi_gdf[~hi_gdf['small_obs']].empty:
            hi_gdf[~hi_gdf['small_obs']].plot(column=col, ax=ax_hi)
        else:
            hi_gdf[hi_gdf['small_obs']].plot(color='grey', ax=ax_hi, legend=False)

        # Set aspect ratio for the insets to match the main axis
        ax_ak.set_aspect('equal')
        ax_hi.set_aspect('equal')

        # Remove axis for all plots
        ax_main.axis('off')
        ax_ak.axis('off')
        ax_hi.axis('off')

        # Adjust Alaska and Hawaii positions
        ax_ak.set_xlim([-170, -130])
        ax_ak.set_ylim([50, 72])
        ax_hi.set_xlim([-161, -154])
        ax_hi.set_ylim([18, 23])

        ax_main.axis('off')

        # Save the figure
        plt.savefig(os.path.join(config['figures_path'], f"Figure 5 - Map of {col.replace('_', ' ')} - {size_name}.png"), dpi = 300)

        plt.show()


