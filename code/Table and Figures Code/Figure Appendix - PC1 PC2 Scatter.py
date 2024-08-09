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

model = 'augest_latest'


#Draw in excel file of index from model
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Comprehensive Data.xlsx'))

#Drop row with 166775 as CENSUS_ID_PID6
df = df[df['CENSUS_ID_PID6'] != 166775]

##Find rows




##Graph

import matplotlib.pyplot as plt

# Create the scatter plot
plt.figure(figsize=(5.8*1.7, 3.3*1.7))

# Plot all points
plt.scatter(df['First_PC'], df['Second_PC'], alpha=0.6, color='blue')

# List of specific CENSUS_ID_PID6 values
specific_ids = [161455, 161165, 162271,193551,133602,172358,106387,128142,161174]

# Filter the dataframe for these specific points
specific_points = df[df['CENSUS_ID_PID6'].isin(specific_ids)]

# Plot and label specific points
for _, point in specific_points.iterrows():
    plt.scatter(point['First_PC'], point['Second_PC'], color='red', s=100, zorder=5)
    plt.annotate(f"{point['UNIT_NAME'].title()}, {point['STATE'].upper()}",
                 (point['First_PC'], point['Second_PC']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, color='red',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

# Set labels and title
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Add vertical and horizontal lines through the origin
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Ensure the origin (0,0) is included in the plot
plt.xlim(plt.xlim()[0], max(plt.xlim()[1], 0))
plt.ylim(plt.ylim()[0], max(plt.ylim()[1], 0))

# Show the plot
plt.tight_layout()
plt.show()