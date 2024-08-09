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
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Comprehensive Data.xlsx'))

def make_gdf_state(df):

    #Count number of obs per 'State'
    state_counts = df['FIPS_STATE'].value_counts()

    #Keep columns POPULATION, First_PC, Second_PC, FIPS_STATE
    df = df[['FIPS_STATE','POPULATION','First_PC','Second_PC','Question 28Min','Question 17']]

    # Define a function to calculate weighted averages
    def weighted_avg(df, values, weights):
        return (df[values] * df[weights]).sum() / df[weights].sum()

    # Group by 'FIPS_STATE' and calculate both normal and weighted averages
    state_averages = df.groupby('FIPS_STATE').agg(
        First_PC_mean=('First_PC', 'mean'),
        Second_PC_mean=('Second_PC', 'mean'),
        min_lot_mean = ('Question 28Min', 'mean'),
        affordable_mean = ('Question 17', 'mean')
    ).reset_index()

    # Calculate population-weighted averages using apply within groupby
    state_weighted_averages = df.groupby('FIPS_STATE').apply(
        lambda x: pd.Series({
            'First_PC_weighted': weighted_avg(x, 'First_PC', 'POPULATION'),
            'Second_PC_weighted': weighted_avg(x, 'Second_PC', 'POPULATION'),
            'min_lot_weighted': weighted_avg(x, 'Question 28Min', 'POPULATION'),
            'affordable_weighted': weighted_avg(x, 'Question 17', 'POPULATION')
        })
    ).reset_index()

    # Merge normal and weighted averages
    combined_averages = pd.merge(state_averages, state_weighted_averages, on='FIPS_STATE')

    # Step 1: Load the shapefile
    gdf_states = gpd.read_file(os.path.join(config['shape_files'],'State_Maps','cb_2018_us_state_500k.shp'))

    # Add a column to the GeoDataFrame to indicate if the state has a small number of observations
    gdf_states['small_obs'] = gdf_states['STATEFP'].apply(lambda x: state_counts.get(int(x), 0) < 2)


    #Mak STATEFP an int in gdf
    gdf_states['STATEFP'] = gdf_states['STATEFP'].astype(int)


    # Step 2: Merge the geodataframe with your data
    gdf_states = gdf_states.merge(combined_averages, left_on='STATEFP', right_on='FIPS_STATE')

    return gdf_states

gdf_states = make_gdf_state(df)

##Make GDF Counties

def make_gdf_counties(df):
    # Count number of obs per 'State'
    state_counts = df[['FIPS_STATE','FIPS_COUNTY']].value_counts()

    # Keep columns POPULATION, First_PC, Second_PC, FIPS_STATE
    df = df[['FIPS_STATE','FIPS_COUNTY', 'POPULATION', 'First_PC', 'Second_PC', 'Question 28Min', 'Question 17']]

    # Define a function to calculate weighted averages
    def weighted_avg(df, values, weights):
        return (df[values] * df[weights]).sum() / df[weights].sum()

    # Group by 'FIPS_STATE' and calculate both normal and weighted averages
    state_averages = df.groupby(['FIPS_STATE','FIPS_COUNTY']).agg(
        First_PC_mean=('First_PC', 'mean'),
        Second_PC_mean=('Second_PC', 'mean'),
        min_lot_mean=('Question 28Min', 'mean'),
        affordable_mean=('Question 17', 'mean')
    ).reset_index()

    # Calculate population-weighted averages using apply within groupby
    state_weighted_averages = df.groupby(['FIPS_STATE','FIPS_COUNTY']).apply(
        lambda x: pd.Series({
            'First_PC_weighted': weighted_avg(x, 'First_PC', 'POPULATION'),
            'Second_PC_weighted': weighted_avg(x, 'Second_PC', 'POPULATION'),
            'min_lot_weighted': weighted_avg(x, 'Question 28Min', 'POPULATION'),
            'affordable_weighted': weighted_avg(x, 'Question 17', 'POPULATION')
        })
    ).reset_index()

    # Merge normal and weighted averages
    combined_averages = pd.merge(state_averages, state_weighted_averages, on=['FIPS_STATE','FIPS_COUNTY'])

    # Step 1: Load the shapefile
    gdf_states = gpd.read_file(os.path.join(config['shape_files'], 'County_Maps', 'cb_2018_us_county_500k.shp'))

    #Make STATEFP and COUNTYFP ints
    gdf_states['STATEFP'] = gdf_states['STATEFP'].astype(int)
    gdf_states['COUNTYFP'] = gdf_states['COUNTYFP'].astype(int)

    # Step 2: Merge the geodataframe with your data
    gdf_states = gdf_states.merge(combined_averages, left_on=['STATEFP','COUNTYFP'], right_on=['FIPS_STATE','FIPS_COUNTY'], how = 'left')

    #Drop fips of territories
    gdf_states = gdf_states[gdf_states['STATEFP'] < 60]

    return gdf_states

gdf_counties = make_gdf_counties(df)


##

'''
Make colorbar horizontal and then try to loop through images and show data
'''

get_label = {
    'First_PC_mean': {'label': 'Average First PC', 'unit': '', 'quantile': False},
    'Second_PC_mean': {'label': 'Average Second PC', 'unit': '', 'quantile': False},
    'First_PC_weighted': {'label': 'Population Weighted First PC', 'unit': '', 'quantile': False},
    'Second_PC_weighted': {'label': 'Population Weighted Second PC', 'unit': '', 'quantile': False},
    'min_lot_mean': {'label': 'Average Minimum Lot Size', 'unit': 'Square Feet', 'quantile': True},
    'min_lot_weighted': {'label': 'Population Weighted Minimum Lot Size', 'unit': 'Square Feet', 'quantile': True},
    'affordable_mean': {'label': 'Average Affordable Housing', 'unit': 'Percent', 'quantile': False},
    'affordable_weighted': {'label': 'Population Weighted Affordable Housing', 'unit': 'Percent', 'quantile': False}
}

from matplotlib import colors
import numpy as np


def plot_map_for_size(figsize, gdf_states, fontsize, col):
    # Set up color mapping
    if get_label[col]['quantile']:
        cmap = plt.cm.viridis
        quantiles = np.linspace(0, 1, 7)  # 20 quantiles
        bins = gdf_states[col].quantile(quantiles).values
        # Ensure bins are unique and remove any NaN values
        bins = np.unique(bins)
        bins = bins[~np.isnan(bins)]
        norm = colors.BoundaryNorm(bins, cmap.N)
        col_to_plot = col
    else:
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=gdf_states[col].min(), vmax=gdf_states[col].max())
        col_to_plot = col

    # Now, let's define the polygons to clip Alaska and Hawaii
    polygon_ak = Polygon([(-170, 50), (-170, 72), (-130, 72), (-130, 50)])  # Alaska
    polygon_hi = Polygon([(-161, 18), (-161, 23), (-154, 23), (-154, 18)])  # Hawaii

    # Clip Alaska and Hawaii
    ak_gdf = gdf_states[gdf_states['STATEFP'] == 2].copy()
    hi_gdf = gdf_states[gdf_states['STATEFP'] == 15].copy()

    ak_gdf['geometry'] = ak_gdf['geometry'].intersection(polygon_ak)
    hi_gdf['geometry'] = hi_gdf['geometry'].intersection(polygon_hi)

    # Create figure and axes with different sizes
    fig, ax_main = plt.subplots(1, figsize=figsize)
    ax_main.set_aspect('equal')

    # Plot the mainland states with more than 2 observations
    mainland_more_than_2 = gdf_states[~gdf_states['STATEFP'].isin([2, 15])]
    if not mainland_more_than_2.empty:
        mainland_more_than_2.plot(column=col_to_plot, ax=ax_main, cmap=cmap, norm=norm, legend=False)

    # Plot any with missing data as grey
    mainland_missing_data = mainland_more_than_2[mainland_more_than_2[col_to_plot].isnull()]
    if not mainland_missing_data.empty:
        mainland_missing_data.plot(ax=ax_main, color='grey', legend=False)

    # Set the colorbar label font size
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_main, orientation='horizontal', pad=0.05, aspect=50, location='bottom')
    cb.set_label(label=get_label[col]['unit'], size=fontsize)

    # Adjust the colorbar tick label font size
    cb.ax.tick_params(labelsize=fontsize)

    # Create inset axes for Alaska and Hawaii
    ax_ak = fig.add_axes([0.05, 0.22, 0.25, 0.25], anchor='SW', axes_class=plt.Axes)
    ax_hi = fig.add_axes([0.3, 0.22, 0.15, 0.15], anchor='SW', axes_class=plt.Axes)

    # Plot Alaska and Hawaii
    ak_gdf.plot(column=col_to_plot, ax=ax_ak, cmap=cmap, norm=norm)
    hi_gdf.plot(color='grey', ax=ax_hi, legend=False)

    # Plot any missing data as grey for Alaska
    ak_missing_data = ak_gdf[ak_gdf[col_to_plot].isnull()]
    if not ak_missing_data.empty:
        ak_missing_data.plot(ax=ax_ak, color='grey', legend=False)

    # Plot any missing data as grey for Hawaii
    hi_missing_data = hi_gdf[hi_gdf[col_to_plot].isnull()]
    if not hi_missing_data.empty:
        hi_missing_data.plot(ax=ax_hi, color='grey', legend=False)

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

    county = 'COUNTYFP' in gdf_states.columns

    if county:
        extra_label = 'County'
    else:
        extra_label = 'State'


    #Save to figures folder with 300 dpi
    plt.savefig(os.path.join(config['figures_path'],f'Figure - {extra_label} Map {col}.png'), dpi = 300)

    ax_main.set_title(get_label[col]['label'], fontsize=fontsize, fontweight='bold')

    #Save fig again with 100 dpi in county and state subfolders
    plt.savefig(os.path.join(config['figures_path'],'All Maps New',extra_label,f'{col}.png'), dpi = 100)

    plt.show()




# Define the font size variable
fontsize = 12


# Loop over df columns except 'StateAbbrev'
for col in get_label.keys():

    plot_map_for_size((5,2.5), gdf_states, fontsize, col)

    plot_map_for_size((5,2.5), gdf_counties, fontsize, col)





