import pandas as pd
import numpy as np
import os
import yaml

# Load configuration
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data
data = pd.read_excel(os.path.join(config['raw_data'], 'Sample Data.xlsx'))
distance_matrix = pd.read_csv(os.path.join(config['processed_data'], 'Compact Distance Matrices', 'combined_lower_triangular_distances.csv'))

# Subset distance_matrix for only CENSUS_ID_PID6 in data
distance_matrix = distance_matrix[distance_matrix['ID1'].isin(data['CENSUS_ID_PID6']) & distance_matrix['ID2'].isin(data['CENSUS_ID_PID6'])]

#Now save the distance matrix
distance_matrix.to_csv(os.path.join(config['processed_data'], 'Compact Distance Matrices', 'combined_lower_triangular_distances_sample.csv'), index=False)


