import geopandas as gpd
import pandas as pd

# File paths
paths = {
    'shapefile': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\Stanford Ed Opportunity Project\Geo Crosswalk\SEDA-2019-District-Shapefile\SEDA-2019-District-Shapefile.shp",
    'opportunity_csv': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\Stanford Ed Opportunity Project\Opportunity\seda_geodist_poolsub_cs_5.0_updated_20240319.csv",
    'geocorr_csv': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\GEO Crosswalks\geocorr2022_school_place.csv"
}

# Load data
data = {
    'district_shapefile': gpd.read_file(paths['shapefile']),
    'opportunity_data': pd.read_csv(paths['opportunity_csv']),
    'geocorr_bridge': pd.read_csv(paths['geocorr_csv'], encoding='latin1').drop(0)
}

# Preprocess data
data['district_shapefile']['GEOID'] = data['district_shapefile']['GEOID'].astype(int)
data['geocorr_bridge']['ID'] = (data['geocorr_bridge']['state'] + data['geocorr_bridge']['sduni20']).astype(int)

# Extract unique IDs
unique_ids = {
    'sedalea': set(data['opportunity_data']['sedalea'].astype(int)),
    'geoid': set(data['district_shapefile']['GEOID']),
    'geocorr': set(data['geocorr_bridge']['ID'])
}

# Find intersections and unique IDs
intersections = {
    'sedalea_geoid': unique_ids['sedalea'].intersection(unique_ids['geoid']),
    'sedalea_geocorr': unique_ids['sedalea'].intersection(unique_ids['geocorr']),
    'geoid_geocorr': unique_ids['geoid'].intersection(unique_ids['geocorr']),
    'all': set.intersection(*unique_ids.values())
}

unique_only = {
    'sedalea': unique_ids['sedalea'] - set.union(unique_ids['geoid'], unique_ids['geocorr']),
    'geoid': unique_ids['geoid'] - set.union(unique_ids['sedalea'], unique_ids['geocorr']),
    'geocorr': unique_ids['geocorr'] - set.union(unique_ids['sedalea'], unique_ids['geoid'])
}

# Filter data for unique IDs
filtered_data = {
    'opportunity': data['opportunity_data'][data['opportunity_data']['sedalea'].astype(int).isin(unique_only['sedalea'])],
    'shapefile': data['district_shapefile'][~data['district_shapefile']['GEOID'].isin(intersections['all'])],
    'geocorr': data['geocorr_bridge'][data['geocorr_bridge']['ID'].isin(unique_only['geocorr'])]
}

#Get filtered data of opportunity data not in geocorr
opportunity_geocorr = data['opportunity_data'][~data['opportunity_data']['sedalea'].astype(int).isin(intersections['sedalea_geocorr'])]

#Keep one row per sedelea
opportunity_geocorr = opportunity_geocorr.drop_duplicates(subset='sedalea')

#Now of the sedaleas in opportunity_geocorr, how many are in geoid?
opportunity_geoid = opportunity_geocorr[opportunity_geocorr['sedalea'].astype(int).isin(unique_ids['geoid'])]

# Print summary
print("Unique IDs in each dataset:")
for key, value in unique_ids.items():
    print(f"{key}: {len(value)}")

print("\nIntersections:")
for key, value in intersections.items():
    print(f"{key}: {len(value)}")

print("\nUnique IDs only in each dataset:")
for key, value in unique_only.items():
    print(f"{key}: {len(value)}")

print("\nFiltered data sizes:")
for key, value in filtered_data.items():
    print(f"{key}: {len(value)}")


