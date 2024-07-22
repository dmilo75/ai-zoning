
import pandas as pd
import numpy as np
from readability import Readability
import yaml
import os
import pickle
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def reading_level(text):
    '''
    We will use flesch kincaid
    '''
    try:
        score = Readability(text)
        r = score.flesch_kincaid()
        return r.score
    # If less than 100 words, return nan
    except:
        return np.nan

def calculate_reading_stats(tree):
    total_tokens = 0
    reading_scores = []

    def is_leaf(node):
        return "Leaf_Text" in node

    def process_node(node, path):
        nonlocal total_tokens
        if is_leaf(node):  # Check if the node is a leaf
            leaf_text = node['Leaf_Text']
            tokens = node.get('Tokens', len(leaf_text.split()))  # Use 'Tokens' if available, else count words
            total_tokens += tokens  # Sum up tokens
            score = reading_level(leaf_text)  # Calculate reading level
            reading_scores.append(score)  # Accumulate reading scores
        else:
            # Recursively process each child if the node has 'children'
            if 'children' in node:
                for child in node['children']:
                    process_node(child, path + (node['name'],))

    # Initialize traversal
    if isinstance(tree, list):
        for root_node in tree:
            process_node(root_node, ())
    else:
        process_node(tree, ())

    # Calculate mean and standard deviation of reading scores and drop nan
    reading_scores = [score for score in reading_scores if not np.isnan(score)]
    mean_reading_level = np.mean(reading_scores)
    std_dev_reading_level = np.std(reading_scores)

    return total_tokens, mean_reading_level, std_dev_reading_level

def process_muni(muni_name):

    id = int(muni_name.split('#')[1])

    # Read in pickle file
    with open(os.path.join(config['embeddings'],muni_name+'.pkl'), "rb") as fIn:
        raw_dic = pickle.load(fIn)

    tokens, mean_reading_level, std_dev_reading_level = calculate_reading_stats(raw_dic)

    return tokens, mean_reading_level, std_dev_reading_level, id


#Import sample data excel
sample_data = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))

#Filter for massachussets
sample_data = sample_data[sample_data['State'] == 'ma']

#Set up dataframe to hold pid6, length, and reading level stats
df = pd.DataFrame(columns = ['CENSUS_ID_PID6','length','mean_reading_level','std_dev_reading_level'])

#Iterate over the rows of sample_data
for i, row in sample_data.iterrows():

    try:

        print('starting')

        #Get muni_name
        muni_name = row['Muni'].lower()+'#'+str(row['CENSUS_ID_PID6'])+'#'+row['State']

        #process the muni
        tokens, mean_reading_level, std_dev_reading_level, id = process_muni(muni_name)
        #Now append on row to dataframe using concat
        df = df.append({'CENSUS_ID_PID6':id,'length':tokens,'mean_reading_level':mean_reading_level,'std_dev_reading_level':std_dev_reading_level},ignore_index=True)

        print('made it')

    except Exception as e:
        print(e)
        continue


#Now, export to raw_data
df.to_excel(os.path.join(config['raw_data'],'Muni Reading Stats.xlsx'),index=False)
