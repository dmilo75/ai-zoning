import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml

#Load config file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Read in sample data
sample = pd.read_excel(config['raw_data']+"Sample Data.xlsx",index_col  = 0)

#%%Urban rural classification 

# File paths
base_path = config['shape_files']
urban_shapefile_path = os.path.join(base_path, "Urban\\tl_2020_us_uac20.shp")
urban_shape = gpd.read_file(urban_shapefile_path)

sample_urban = sample.copy()

for var in ['FIPS_STATE','FIPS_PLACE','FIPS_COUNTY']:
    sample_urban[var] = sample_urban[var].astype(int)
    
sample_urban['% Urban'] = None

# Loop over each fips state in the sample_urban dataframe
for fips_state in sample_urban['FIPS_STATE'].unique():

    
    # Ensure fips_state is a string with two characters
    fips_state_str = str(fips_state).zfill(2)
    
    places_path = os.path.join(base_path,f"Places\\tl_2022_{fips_state_str}_place\\tl_2022_{fips_state_str}_place.shp")
    cousub_path = os.path.join(base_path,f"County Subdivisions\\tl_2022_{fips_state_str}_cousub\\tl_2022_{fips_state_str}_cousub.shp")

    
    # Load fips place shapefile for the state
    fips_places = gpd.read_file(places_path)
    
    #Convert matching columns to ints
    for var in ['PLACEFP']:
        fips_places[var] = fips_places[var].astype(int)

    
    # Filter rows for current state and UNIT_TYPE == '2 - Municipality'
    municipal_rows = sample_urban[(sample_urban['FIPS_STATE'] == fips_state) & (sample_urban['UNIT_TYPE'] == '2 - MUNICIPAL')]
    
    # Try to match rows and integrate geodata
    for index, row in municipal_rows.iterrows():
        matched = fips_places[fips_places['PLACEFP'] == row['FIPS_PLACE']]
        if matched.empty:
            print(f"No match found for {row['FIPS_PLACE']} in state {fips_state}")
        else:
            sample_urban.at[index, 'geometry'] = matched.iloc[0]['geometry']
    
    # If there are any rows with UNIT_TYPE == '3 - Township'
    township_rows = sample_urban[(sample_urban['FIPS_STATE'] == fips_state) & (sample_urban['UNIT_TYPE'] == '3 - TOWNSHIP')]
    if not township_rows.empty:
        # Load county subdivision file for the state
        county_subdivisions = gpd.read_file(cousub_path)
        
        #Convert matching vars to ints
        for var in [ 'COUSUBFP', 'COUNTYFP']:
            county_subdivisions[var] = county_subdivisions[var].astype(int)
        
        # Try to match rows and integrate geodata
        for index, row in township_rows.iterrows():
            matched = county_subdivisions[(county_subdivisions['COUSUBFP'] == row['FIPS_PLACE']) & 
                                          (county_subdivisions['COUNTYFP'] == row['FIPS_COUNTY'])]
            if matched.empty:
                print(f"No match found for {row['FIPS_PLACE']} in county subdivision of state {fips_state}")
            else:
                sample_urban.at[index, 'geometry'] = matched.iloc[0]['geometry']
    
    # Population '% Urban'
    buffer_distance = 0.1  
    
    for index, row in sample_urban[sample_urban['FIPS_STATE'] == fips_state].iterrows():
        if pd.notna(row.geometry):
            # Filter urban_shape for geometries nearby
            nearby_urban_areas = urban_shape[urban_shape.intersects(row.geometry.buffer(buffer_distance))]
            
            if len(nearby_urban_areas) == 0:
                sample_urban.at[index, '% Urban'] = 0
            
            else:
            
                # Calculate the total overlap with all nearby urban areas
                total_overlap = sum(nearby_urban_areas.geometry.iloc[i].intersection(row.geometry).area for i in range(nearby_urban_areas.shape[0]))
                
                percent_urban = (total_overlap / row.geometry.area) * 100
                sample_urban.at[index, '% Urban'] = percent_urban


#%% Plot distribution of '% Urban'
sample_urban['% Urban'].dropna().hist(bins=100)
plt.xlabel('% Urban')
plt.ylabel('Frequency')
plt.title('Distribution of % Urban')
plt.show()


#%%Export urban dataframe

sample_urban[['FIPS_PLACE', 'FIPS_STATE','% Urban']].to_excel(os.path.join(config['raw_data'],"urban_raw.xlsx"))
