import os
import pandas as pd
import pickle
import geopandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_bridge(bridge_path):
    return pd.read_excel(bridge_path)


def find_matching_file(filename, bridge, directory):
    row = bridge[bridge['filename'] == filename.split('\\')[-1].replace('.pkl', '.xlsx')]
    if row.empty:
        print(f"No match found for {filename}")
        return None

    match_result = row.iloc[0]['match_result']
    for file in os.listdir(directory):
        if str(match_result) in file:
            return os.path.join(directory, file)

    print(f"No matching file found for {filename}")
    return None


def load_and_merge_data(filename, filepath):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    gdf = geopandas.GeoDataFrame(data, geometry=data['geometry'])
    df = pd.read_excel(filepath)
    zone_col = df.columns[0]
    merged = gdf.merge(df, how='left', left_on='district', right_on=zone_col)

    #If More than 25% of rows didnt merge
    if merged.isnull().sum().sum()/len(merged.index) > 0.25:
        raise ValueError(f"Data not entirely merged for {filename}")

    return merged


def determine_color(row):
    if pd.isnull(row['Single_Family_Or_Duplex_Only']) and pd.isnull(row['Allows_Multi_Family']):
        return 'blue'
    elif (row['Single_Family_Or_Duplex_Only'] == 1) or (row['Single_Family_Only'] == 1):
        return 'red'
    elif row['Allows_Multi_Family'] == 1:
        return 'green'
    else:
        return 'grey'


def create_map(gdf):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, color=gdf['color'])

    red_patch = mpatches.Patch(color='red', label='Single Family or Duplex Only')
    green_patch = mpatches.Patch(color='green', label='Allows Multi Family')
    grey_patch = mpatches.Patch(color='grey', label='Neither')
    blue_patch = mpatches.Patch(color='blue', label='Both Null')
    plt.legend(handles=[red_patch, green_patch, grey_patch, blue_patch])

    plt.title(gdf.iloc[0]['city'])
    return fig


def save_map(fig, output_directory, city):
    output_filename = os.path.join(output_directory, f"{city}_zoning_map.png")
    fig.savefig(output_filename)
    plt.close(fig)
    print(f"Map saved as {output_filename}")

def create_zoning_map(filename, bridge_path, directory, output_directory):
    bridge = load_bridge(bridge_path)
    filepath = find_matching_file(filename, bridge, directory)
    if filepath is None:
        return

    try:
        gdf = load_and_merge_data(filename, filepath)
        gdf['color'] = gdf.apply(determine_color, axis=1)
        fig = create_map(gdf)
        save_map(fig, output_directory, gdf.iloc[0]['city'])
        return gdf
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

if __name__ == "__main__":
    input_directory = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\Zoning Map Data\with_county_state_checked"
    bridge_path = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\Zoning Map Data\cog_bridge.xlsx"
    directory = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Zoning Map Analysis"
    output_directory = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Zoning Map Analysis\Maps"

    os.makedirs(output_directory, exist_ok=True)

    gdfs = []

    for filename in os.listdir(input_directory):
        if filename.endswith('.pkl'):
            full_path = os.path.join(input_directory, filename)
            gdf = create_zoning_map(full_path, bridge_path, directory, output_directory)
            if gdf is not None:
                gdfs.append(gdf)


##Code for distance to city center

from shapely.geometry import Point
from shapely.validation import make_valid
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer

def get_district_data(gdf,latitude,longitude):


    gdf = gdf.copy()

    city = gdf['city'][0]


    gdf = gdf.to_crs(epsg=32616)  # Example: UTM zone 16N

    # Create a transformer to convert from lat/long to the GeoDataFrame's CRS
    transformer = Transformer.from_crs("epsg:4326", gdf.crs, always_xy=True)


    # Convert City Hall coordinates to the GeoDataFrame's CRS
    city_hall_x, city_hall_y = transformer.transform(longitude, latitude)
    city_hall_point = Point(city_hall_x, city_hall_y)
    city_hall = gpd.GeoSeries([city_hall_point], crs=gdf.crs)


    # Ensure all geometries in the GeoDataFrame are valid
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom))

    # Convert distances from miles to meters
    distances_miles = [0, 1, 2, 5, 10,20]
    distances_meters = [d * 1609.34 for d in distances_miles]

    # Plot the map with the city hall and rings
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, color=gdf['color'])

    # Plot the City Hall
    city_hall.plot(ax=ax, marker='*', color='red', markersize=100)

    # Plot rings
    for distance in distances_meters[1:]:  # Skip the first distance (0 miles)
        ring = city_hall.buffer(distance)
        ring.boundary.plot(ax=ax, color='blue')

    #Set title as city
    plt.title(city)

    plt.show()



    data  = []

    # Calculate percentage of single-family and multi-family housing within each ring
    for i in range(1, len(distances_meters)):
        inner_distance = distances_meters[i - 1]
        outer_distance = distances_meters[i]

        inner_ring = city_hall.buffer(inner_distance)
        outer_ring = city_hall.buffer(outer_distance)
        ring_area = outer_ring.difference(inner_ring)


        # Subset geometries that intersect with the ring area
        intersecting_geometries = gdf[gdf.geometry.intersects(ring_area.unary_union)]

        # Clip geometries to the ring area and calculate areas
        clipped_areas = intersecting_geometries.geometry.intersection(ring_area.unary_union)
        clipped_gdf = gpd.GeoDataFrame(intersecting_geometries.drop(columns='geometry'), geometry=clipped_areas)

        clipped_gdf['area'] = clipped_gdf.geometry.area

        total_area = clipped_gdf['area'].sum()
        single_family_area = clipped_gdf[clipped_gdf['color'] == 'red']['area'].sum()
        multi_family_area = clipped_gdf[clipped_gdf['color'] == 'green']['area'].sum()

        percent_single_family = (single_family_area / total_area) * 100 if total_area > 0 else 0
        percent_multi_family = (multi_family_area / total_area) * 100 if total_area > 0 else 0

        data.append({
            'bucket': str(distances_miles[i-1])+' - '+str(distances_miles[i])+' miles',
            'percent_single_family': percent_single_family,
            'percent_multi_family': percent_multi_family
        })

    return pd.DataFrame(data)

##

df = get_district_data(gdfs[1],41.8838,-87.6321)



centers = pd.read_csv(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\MSA Center Points\city_centers.csv")



##

locations = {
    "Cary": {"lat": 35.789211, "long": -78.779897},
    "Chicago": {"lat": 41.8838, "long": -87.6321},
    "Detroit": {"lat": 42.331399, "long": -83.045756},
    "Frederick": {"lat": 39.415722, "long": -77.412566},
    "Henderson": {"lat": 36.029696, "long": -114.981679},
    "Kansas City MO": {"lat": 39.100942, "long": -94.578105},
    "Kent": {"lat": 47.380602, "long": -122.237418},
    "Los Angeles City": {"lat": 34.053470, "long": -118.242342},
    "Mesa": {"lat": 33.417629, "long": -111.822962},
    "Parma": {"lat": 41.384301, "long": -81.733997},
    "Riverside": {"lat": 33.980242, "long": -117.375881},
    "Roswell": {"lat": 34.022440, "long": -84.359685},
    "Round Rock": {"lat": 30.508192, "long": -97.676563},
    "San Antonio": {"lat": 29.424523, "long": -98.495409},
    "San Diego": {"lat": 32.717234, "long": -117.162963},
    "San Francisco": {"lat": 37.779894, "long": -122.419337},
    "San Mateo": {"lat": 37.547598, "long": -122.314803},
    "Tampa": {"lat": 27.947836, "long": -82.457316}
}


dfs = []

indices = [1, 4, 5, 8, 13]
selected_gdfs = [gdfs[i] for i in indices]


for gdf in selected_gdfs:
    city = gdf['city'][0]
    city_hall = locations[city]
    print(city)
    df = get_district_data(gdf, city_hall['lat'], city_hall['long'])
    dfs.append(df)


'''
Need to get lat long pairs of city hall for all 18 cities then can loop through
'''

##

full_df = pd.concat(dfs)

#Gropuby bucket and average columns
full_df = full_df.groupby('bucket').mean().reset_index()


import matplotlib.pyplot as plt

# Assuming full_df is your dataframe
full_df.set_index('bucket', inplace=True)

#Drop index '10 - 20 miles'
full_df = full_df.drop('10 - 20 miles')


# Sort the index by the first digit before the '-'
full_df = full_df.reindex(sorted(full_df.index, key=lambda x: int(x.split(' - ')[0])))

# Define custom colors for the bars
colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange

# Plotting the bar chart with custom colors
ax = full_df[['percent_single_family', 'percent_multi_family']].plot(kind='bar', figsize=(5.8, 3.3), color=colors)

plt.title('Five City Average: Land Use By Distance to City Hall')
plt.xlabel('Distance From City Hall')
plt.ylabel('Share of Land')
plt.xticks(rotation=0)
plt.legend(['Exclusively Single Family/Duplexes', 'Allows Multi Family'])
plt.tight_layout()

#Save figure
plt.savefig(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures\Slides - Districts Distance From Center", dpi=300)

plt.show()


