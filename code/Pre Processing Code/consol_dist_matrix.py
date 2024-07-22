import os
import pandas as pd
import numpy as np
import yaml
import gc

os.chdir('../../')

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

input_dir = os.path.join(config['processed_data'], 'Distance Matrices')
output_dir = os.path.join(config['processed_data'], 'Compact Distance Matrices')
os.makedirs(output_dir, exist_ok=True)

# Load Sample Data
sample_data = pd.read_excel(os.path.join(config['raw_data'], "Sample Data.xlsx"))
sample_ids = set(sample_data['CENSUS_ID_PID6'].astype(str))


def process_csv_chunk(file_path, chunk_size=100000):
    chunk_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['ID1', 'ID2', 'distance']):
        chunk_list.append(chunk)
        gc.collect()  # Free memory after each chunk
    return pd.concat(chunk_list, ignore_index=True)


def combine_and_process_csvs():
    combined_df_list = []

    # Read and combine all CSV files in chunks
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            chunk_df = process_csv_chunk(file_path)
            combined_df_list.append(chunk_df)
            print(f"Processed {filename}")

    # Concatenate all dataframes
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    print("All files combined")

    # Convert distances to integers (round to nearest mile)
    combined_df['distance'] = np.round(combined_df['distance']).astype(np.uint16)

    # Drop duplicates
    combined_df.drop_duplicates(subset=['ID1', 'ID2'], keep='first', inplace=True)

    # Sort by ID1, then ID2 for better organization
    combined_df.sort_values(['ID1', 'ID2'], inplace=True)

    # Subset for Sample Data
    sample_df = combined_df[combined_df['ID1'].isin(sample_ids) & combined_df['ID2'].isin(sample_ids)]

    # Save Sample dataset
    output_filename_sample = os.path.join(output_dir, 'combined_lower_triangular_distances_sample.csv')
    sample_df.to_csv(output_filename_sample, index=False)
    print(f"Sample combined lower triangular distances saved to {output_filename_sample}")

    # Save full dataset
    output_filename_full = os.path.join(output_dir, 'combined_lower_triangular_distances_full.csv')
    combined_df.to_csv(output_filename_full, index=False)
    print(f"Full combined lower triangular distances saved to {output_filename_full}")

    # Free memory
    del combined_df_list, combined_df, sample_df
    gc.collect()


# Run the processing
combine_and_process_csvs()
