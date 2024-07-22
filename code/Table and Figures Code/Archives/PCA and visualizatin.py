import pickle
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import random
#Generate 5 random numbers between 10 and 30
random.seed(10)
randomlist = random.sample(range(0, 107), 5)

# Load file paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# load embeddings data
dir_path = os.path.join(config['processed_data'],"Embeddings")
all_files = [f for f in os.listdir(dir_path) if any(char.isdigit() for char in f)]
numbers_in_filenames = []

# Define a regex pattern to match numbers
pattern = re.compile(r'\d+')

# Loop through each file in the directory
for filename in all_files:
    # Search for numbers in the filename
    numbers_found = pattern.findall(filename)
    # Add the numbers to the list if any were found
    if numbers_found:
        numbers_in_filenames.extend(numbers_found)

# load pickle files
all_data = []
for file in all_files:
    with open(os.path.join(dir_path, file), 'rb') as f:
        all_data.extend(pickle.load(f))

# define a function to flatten trees
def flatten_tree(tree):
    '''
    Flattens a hierarchical dictionary into a list of leaf nodes, based on the new tree format,
    where the tree is given as a list with one entry that is the root node.

    Parameters:
    - tree: A list containing the root node of the hierarchical dictionary in the new tree format.

    Returns:
    - A list of leaf node dictionaries.
    '''
    leaf_nodes = []

    def _flatten(node):
        # Check if node is a dictionary and has 'Leaf_Text', indicating it's a leaf node
        if isinstance(node, dict) and 'Leaf_Text' in node:
            leaf_nodes.append(node)
        elif isinstance(node, dict) and 'children' in node:  # If the node has children, recurse on them
            for child in node['children']:
                _flatten(child)
        # If the node is not a dictionary or does not have 'children', there's nothing to do

    # Start processing from the root node, which is the first item in the list
    if tree and isinstance(tree, list):
        _flatten(tree[0])
    return leaf_nodes

# apply flatten_data function to all embedding pickle files
flatten_data = []
for data in all_data:
    flatten = flatten_tree([data])
    flatten_data.append(flatten)

# Get average embedding vectors for each pickle file (which is for each town)
# and form a numpy array to store all these data

average_embeddings = []
for flatten in flatten_data:
    embeddings = []
    for d in flatten:
        if "Embeddings" in d:
            embeddings.append(d['Embeddings'][0])
    array = np.array(embeddings)
    average_embeddings.append(array.sum(axis=0)/len(array))

# average_embeddings = [x for x in average_embeddings if str(x) != 'nan']
averaged_embeddings = np.array(average_embeddings)
# PCA
pca = PCA(n_components=2)
pca.fit(averaged_embeddings)
pca_proj = pca.transform(averaged_embeddings)
# TSNE
tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(averaged_embeddings)

# create a dataframe that links pca data with geoid
pca_df = pd.DataFrame(pca_proj, index=numbers_in_filenames)
pca_df['geoid'] = numbers_in_filenames
tsne_df = pd.DataFrame(tsne_proj, index=numbers_in_filenames)
tsne_df['geoid'] = numbers_in_filenames

#
#
# # plot data
# plt.scatter(pca_df[0],pca_df[1])
# plt.title('PCA of all towns')
# plt.savefig(os.path.join(config['figures_path'],'PCA and TSNE results','pca averaged embeddings.png'))
# plt.show()
# plt.scatter(tsne_df[0],tsne_df[1])
# plt.title('TSNE of all towns')
# plt.savefig(os.path.join(config['figures_path'],'PCA and TSNE results','tsne averaged embeddings.png'))
# plt.show()
#
#
# # Here we randomly pick 5 towns,
# # Do pca and TSNE
# flatten_data_ = []
# for i in randomlist:
#     flatten_data_.append(flatten_data[i])
#
# pca_data = []
# tsne_data = []
# for flatten in flatten_data_:
#     embeddings = []
#     for d in flatten:
#         if "Embeddings" in d:
#             embeddings.append(d['Embeddings'][0])
#     array = np.array(embeddings)
#     # Create a two dimensional PCA projection of the embeddings
#     pca = PCA(n_components=2)
#     pca.fit(array)
#     pca_proj = pca.transform(array)
#     pca_data.append(pca_proj)
#
#     # Create a two dimensional t-SNE projection of the embeddings
#     tsne = TSNE(2, verbose=1)
#     tsne_proj = tsne.fit_transform(array)
#     tsne_data.append(tsne_proj)
#
#
# def plot_extraction(data,filename,method):
#     plt.figure(figsize=(8, 6))
#     for i, sublist in enumerate(data):
#         x = sublist[:, 0]
#         y = sublist[:, 1]
#         plt.scatter(x, y, label=f"{i+1}")
#
#     # Add labels and legend
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title(method + ' Scatter Plot of 5 Towns')
#     plt.legend()
#     plt.show()
#
#     # Save plot
#     plt.savefig(os.path.join(config['figures_path'],'PCA and TSNE results', filename))
#
# plot_extraction(tsne_data, "TSNE of 5 Towns", "TSNE")
# plot_extraction(pca_data,"PCA of 5 Towns", "PCA")

# load Full Run data
df = pd.read_excel(os.path.join(config['processed_data'],"Full Run.xlsx"),index_col = 0)
question_list = [4, 5, 6, 8, 9, 11, 13, 14, 17, 20, 21]

# link pca data to answers of yes/no questions by loading Full Run data
pca_df['geoid'] = pca_df['geoid'].astype(int)
tsne_df['geoid'] = pca_df['geoid'].astype(int)
merged_tsne_df = tsne_df
merged_pca_df = pca_df
for i in question_list:
    df2 = df[df['Question']==i][['geoid', 'Answer']]
    merged_tsne_df = pd.merge(merged_tsne_df, df2 , on='geoid', how='inner')
    merged_pca_df = pd.merge(merged_pca_df, df2 , on='geoid', how='inner')
column_names = ['first component','second component','geoid', 'Q4', 'Q5', 'Q6','Q8','Q9','Q11','Q13','Q14','Q17','Q20','Q21']
merged_pca_df.columns = column_names
merged_tsne_df.columns = column_names


def plot_by_answer(question,df,method):
# Do scatter plot, coloring by answers
    colors = ['green' if val == 'Yes' else 'blue' for val in df[question]]

    # Scatter plot
    plt.scatter(df['first component'], df['second component'], c=colors)

    # Add labels and title
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.title(method + ' Scatter Plot with Color based on ' + question)
    plt.savefig(os.path.join(config['figures_path'], 'PCA and TSNE results', method + ' clustering based on ' + question))

    # Show plot
    plt.show()

# plotting
question_list = ['Q4', 'Q5', 'Q6','Q8','Q9','Q11','Q13','Q14','Q17','Q20','Q21']
for i in question_list:
    plot_by_answer(i,merged_tsne_df,'T-SNE')
    plot_by_answer(i,merged_pca_df,'PCA')






