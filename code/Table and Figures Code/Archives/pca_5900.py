# import pickle
# import os
# import yaml
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import re
# import seaborn as sns
# from sklearn.manifold import TSNE
# from matplotlib import cm
# from sklearn.decomposition import PCA
# import plotly.express as px
# import pandas as pd
# import random
#
# random.seed(10)
#
#
# def transform_to_standard_tree(original_tree):
#     def transform_node(key, value):
#         node = {'name': key}
#         # Check if the current node is a leaf node by looking for 'Leaf_Text'
#         if isinstance(value, dict) and 'Leaf_Text' in value:
#
#             for key in value.keys():
#                 node[key] = value[key]
#
#         else:
#             # If the node is not a leaf, recursively transform its children
#             children = []
#             for child_key, child_value in value.items():
#                 if isinstance(child_value, dict):  # Ensure it's a nested dictionary
#                     transformed_child = transform_node(child_key, child_value)
#                     children.append(transformed_child)
#             if children:
#                 node['children'] = children
#         return node
#
#     # Start the transformation with a pseudo-root to handle top-level nodes uniformly
#     pseudo_root = {'Root': original_tree}
#     # Since 'Root' is a pseudo-node, directly return its children to avoid nesting under 'Root'
#     return transform_node('Root', pseudo_root)['children']
#
# def flatten_tree(tree):
#     '''
#     Flattens a hierarchical dictionary into a list of leaf nodes, based on the new tree format,
#     where the tree is given as a list with one entry that is the root node.
#
#     Parameters:
#     - tree: A list containing the root node of the hierarchical dictionary in the new tree format.
#
#     Returns:
#     - A list of leaf node dictionaries.
#     '''
#     leaf_nodes = []
#
#     def _flatten(node):
#         # Check if node is a dictionary and has 'Leaf_Text', indicating it's a leaf node
#         if isinstance(node, dict) and 'Leaf_Text' in node:
#             leaf_nodes.append(node)
#         elif isinstance(node, dict) and 'children' in node:  # If the node has children, recurse on them
#             for child in node['children']:
#                 _flatten(child)
#         # If the node is not a dictionary or does not have 'children', there's nothing to do
#
#     # Start processing from the root node, which is the first item in the list
#     if tree and isinstance(tree, list):
#         _flatten(tree[0])
#     return leaf_nodes
#
#
# # Load the sample data
# sample_df = pd.read_excel('/Users/david_dzz/Dropbox/Inclusionary Zoning/Github/ai-zoning/Sample Data.xlsx')
#
# # Get the list of CENSUS_ID_PID6 values from the sample data
# census_id_list = sample_df['CENSUS_ID_PID6'].tolist()
#
# # Path to the directory containing pickle files
# pickle_files_dir = '/Users/david_dzz/Dropbox/Inclusionary Zoning/Github/ai-zoning/processed data/All Old Embeddings'
#
# # List to hold the names of pickle files that match the CENSUS_ID_PID6 in the sample data
# matching_pickle_files = []
#
# # Regex pattern to extract the CENSUS_ID_PID6 value from pickle file names
# pattern = r'#(\d+)#'
#
# # Iterate over the pickle files in the directory
# for filename in os.listdir(pickle_files_dir):
#     if filename.endswith('.pkl'):
#         # Extract CENSUS_ID_PID6 from the filename
#         match = re.search(pattern, filename)
#         if match:
#             census_id = int(match.group(1))
#             # Check if this ID is in the list from the sample data
#             if census_id in census_id_list:
#                 matching_pickle_files.append(filename)
#
# # Now `matching_pickle_files` contains the names of pickle files to include in further analysis
# print(f'Found {len(matching_pickle_files)} matching pickle files.')
#
# loaded_pickles = {}
# numbers_in_filenames = []
# state_code = []
# all_data = []
#
# import numpy as np
#
# # Earlier parts of the code remain the same...
#
# # Adjusted part starts here
# all_data = []
# state_codes = []  # Renamed for clarity
#
# for pickle_filename in matching_pickle_files:
#     full_path = os.path.join(pickle_files_dir, pickle_filename)
#     try:
#         with open(full_path, 'rb') as file:
#             pickle_content = pickle.load(file)
#             transformed = transform_to_standard_tree(pickle_content[0])  # Adjust based on actual data structure
#             flattened = flatten_tree(transformed)
#             embeddings = [node['Embeddings'][0] for node in flattened if "Embeddings" in node]
#
#             # Calculate the average embedding if embeddings list is not empty
#             if embeddings:
#                 avg_embedding = np.mean(embeddings, axis=0)
#                 all_data.append(avg_embedding)
#
#                 # Extract state code from filename and add to state_codes list
#                 state_code = pickle_filename.split('#')[-2].lower()  # Adjusted extraction based on filename pattern
#                 state_codes.append(state_code)
#             # Process the pickle content as before
#     except EOFError:
#         print(f"Error loading {pickle_filename}. The file may be empty or corrupted.")
#         continue
#
#
# # load pickle files
# print(state_code)
#
#
#
# # apply flatten_data function to all embedding pickle files
# flatten_data = []
# #
# # all_transformed_data = []
# # for file in all_data:
# #     transformed_file = transform_to_standard_tree(file)
# #     all_transformed_data.append(transformed_file)
# #
# # print("yes")
# # for data in all_transformed_data:
# #     flatten = flatten_tree(data)
# #     flatten_data.append(flatten)
# #
# # average_embeddings = []
# # for flatten in flatten_data:
# #     embeddings = []
# #     for d in flatten:
# #         if "Embeddings" in d:
# #             embeddings.append(d['Embeddings'][0])
# #     array = np.array(embeddings)
# #     average_embeddings.append(array.sum(axis=0)/len(array))
#
#
# # average_embeddings = [x for x in average_embeddings if str(x) != 'nan']
# averaged_embeddings = np.array(all_data)
# print(averaged_embeddings)
# # PCA
#
# pca = PCA(n_components=2)
# pca.fit(averaged_embeddings)
# pca_proj = pca.transform(averaged_embeddings)
#
# colors = np.random.rand(len(state_code))
# pca_df = pd.DataFrame(pca_proj)
# pca_df['state code'] = state_code
# pca_df['state code'] = pca_df['state code'].astype('category')
# print(pca_df)
# # # Get unique state codes
# # unique_states = pca_df['state code'].unique()
# # num_states = len(unique_states)
# # color_map = plt.cm.get_cmap('tab10', num_states)
# # sns.scatterplot(data=pca_df, x=0, y=1, hue='state code', palette='Set1')
# # plt.title('PCA of all towns according to states')
# # plt.savefig(os.path.join(config['figures_path'],'PCA and TSNE results','pca averaged embeddings.png'))
# # plt.legend(loc='right')
# # plt.show()
# # print()
# # Set up the color palette
# num_unique_states = pca_df['state code'].nunique()
# palette = sns.color_palette("hsv", num_unique_states)  # 'hsv' is a good choice for many distinct categories
#
# # Plotting
# plt.figure(figsize=(14, 10))  # Adjust the size of the figure here
#
# # Use a categorical palette and increase point size
# sns.scatterplot(data=pca_df, x=0, y=1, hue='state code', palette=palette, s=100, alpha=0.7)  # s is the size of points, alpha is the transparency
#
# plt.title('PCA of All Towns Categorized by State Codes', fontsize=18)
# plt.xlabel('PCA Component 1', fontsize=14)
# plt.ylabel('PCA Component 2', fontsize=14)
#
# # Move the legend outside the plot to avoid overlap with points
# plt.legend(title='State Code', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', title_fontsize=14)
#
# # Tight layout often solves the issue of cutting off parts of the plot
# plt.tight_layout()
#
# # # Save the figure
# # plt.savefig('/mnt/data/pca_averaged_embeddings_improved.png', bbox_inches='tight')  # Modify the path as needed
#
# # Show the plot
# # plt.savefig(os.path.join(config['figures_path'],'PCA and TSNE results','improved pca averaged embeddings across nation.png'))
# plt.show()
