import pandas as pd
import numpy as np
import os
import yaml
from itertools import combinations

# Set working directory and load configuration
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

'''
This has to be run on the supercomputer, need all munis to calculate this 
'''

#load data
data = pd.read_excel(os.path.join(config['raw_data'], 'Sample Data.xlsx'))

all_ids = set(data['CENSUS_ID_PID6'])

# Load distance matrix
distance_matrix = pd.read_csv(os.path.join(config['processed_data'], 'Compact Distance Matrices', 'combined_lower_triangular_distances.csv'))

# Make ID1 and ID2 ints and ensure ID1 < ID2 in distance_matrix
distance_matrix['ID1'] = distance_matrix['ID1'].astype(int)
distance_matrix['ID2'] = distance_matrix['ID2'].astype(int)
distance_matrix['ID1'], distance_matrix['ID2'] = np.where(
    distance_matrix['ID1'] < distance_matrix['ID2'],
    (distance_matrix['ID1'], distance_matrix['ID2']),
    (distance_matrix['ID2'], distance_matrix['ID1'])
)

#Require either ID1 or ID2 to be in all_ids
distance_matrix = distance_matrix[(distance_matrix['ID1'].isin(all_ids)) | (distance_matrix['ID2'].isin(all_ids))]

#Drop duplicates
distance_matrix = distance_matrix.drop_duplicates(subset = ['ID1','ID2'])

#Drop where ID1 == ID2
distance_matrix = distance_matrix[distance_matrix['ID1'] != distance_matrix['ID2']]

# Function to count occurrences within a certain distance
def count_within_distance(distance_matrix, max_distance):
    filtered = distance_matrix[distance_matrix['distance'] <= max_distance]
    counts_ID1 = filtered['ID1'].value_counts()
    counts_ID2 = filtered['ID2'].value_counts()
    all_counts = counts_ID1.add(counts_ID2, fill_value=0).astype(int)
    return all_counts

# Calculate counts for distances 10, 25, 50
counts_10 = count_within_distance(distance_matrix, 10)
counts_25 = count_within_distance(distance_matrix, 25)
counts_50 = count_within_distance(distance_matrix, 50)

# Create a DataFrame to hold the results
result_df = pd.DataFrame(index = list(all_ids))

result_df['Neighbors_10'] = result_df.index.map(counts_10).fillna(0).astype(int)
result_df['Neighbors_25'] = result_df.index.map(counts_25).fillna(0).astype(int)
result_df['Neighbors_50'] = result_df.index.map(counts_50).fillna(0).astype(int)


#Export to interim data
result_df.to_csv(os.path.join(config['processed_data'],'interim_data', 'Neighbor_Counts.csv'))
