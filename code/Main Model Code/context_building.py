import pickle
import random
random.seed(2)
import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import string
import gpt_functions as cg
import yaml
import pandas as pd
import cohere
from dotenv import load_dotenv
load_dotenv()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Configure the cohere client
co = cohere.Client(os.getenv('cohere_key'))

# %% Pull in data on question embeddings
with open(os.path.join(config['embeddings'], "Questions.pkl"), "rb") as fIn:
    da = pickle.load(fIn)

# Extract the embeddings array and the 'Questions' DataFrame
embeddings = da['embeddings']
questions_df = da['Questions']

# Create a map from question text ('Question Detail' column) to its embedding
question_embeddings = {questions_df.iloc[i]['Question Detail']: embeddings[i] for i in range(len(questions_df))}

#Get keywords
keywords = pd.read_excel(os.path.join(config['raw_data'],'Keywords.xlsx'))

#Max tokens
max_tokens = config['max_tokens']*1000


##Context building functions

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


def sort_chunks(question_embedding, node_list):
    '''
    Sorts the relevant chunks in a hierarchical dictionary tree in terms of the maximum cosine similarity
    of the text embeddings with the question embedding and stores the maximum cosine similarity score in each leaf node.

    Parameters:
    - question_embedding: A list of floats representing the question embedding.
    - tree: A hierarchical dictionary where leaf nodes contain text information and embeddings.

    Returns:
    - A list of leaf node dictionaries sorted by the maximum cosine similarity of their embeddings with the question embedding.
      Each dictionary includes a new key 'max_cosine_similarity' with the corresponding similarity score.
    '''

    # Iterate through each item and calculate the maximum cosine similarity
    for item in node_list:
        embeddings = item.get("Embeddings", [])

        # Ensure question_embedding is a 1-D array
        if isinstance(question_embedding, list):
            question_embedding = np.array(question_embedding)  # Convert to numpy array if it's a list

        scores = [1 - cosine(question_embedding, emb) for emb in embeddings]

        item["max_cosine_similarity"] = max(scores) if scores else -1

    # Sort the list based on the maximum cosine similarity scores
    sorted_node_list = sorted(node_list, key=lambda x: x.get("max_cosine_similarity", -1), reverse=True)

    return sorted_node_list

#Keyword reranker
def rerank_keyword(sorted_da, question):

    # Get the keywords for the question
    keywords = get_keywords(question)

    # Initialize the list for the reordered items
    reordered = []

    # Temporary list to hold items not yet reordered
    remaining_items = sorted_da[:]

    for keyword in keywords:
        # Temporary list for this iteration to collect items with the current keyword
        this_keyword_items = []

        # Use a while loop to iterate through remaining_items so we can safely remove items
        i = 0
        while i < len(remaining_items):
            item = remaining_items[i]
            # Check if the current keyword is in this item
            if keyword.lower() in item['Leaf_Text'].lower():
                # Add to this keyword's list and remove from remaining items
                this_keyword_items.append(item)
                remaining_items.pop(i)
            else:
                # Move to next item if current doesn't contain the keyword
                i += 1

        # Extend the reordered list with items containing the current keyword
        reordered.extend(this_keyword_items)

    # Finally, add any remaining items that did not match any keywords
    reordered.extend(remaining_items)

    return reordered


#Reranking parent function
def reranker(sorted_da, question, type):
    '''
    Reranks the sorted_da based on the question
    '''

    if type == 'semantic':
        return semantic_reranker(sorted_da, question)
    elif type == 'keyword':
        return rerank_keyword(sorted_da,question)
    else:
        return sorted_da

def semantic_reranker(node_list, question):
    question_string = question['Question Rephrase']

    # Get the top documents according to the initial criteria
    docs = [x['Leaf_Text'] for x in node_list[0:50]]

    # Assuming `co.rerank` returns a new ranking based on the reranker algorithm
    response = co.rerank(
        model='rerank-english-v3.0',
        query=question_string,
        documents=docs,
    )

    # Extract relevance scores and indices from the reranker response
    relevance_scores = [x.relevance_score for x in response]
    relevance_indices = [x.index for x in response]

    # Extract max_cosine_similarity scores from node_list
    max_cosine_scores = [x['max_cosine_similarity'] for x in node_list]

    # Normalize relevance scores and max_cosine_scores using z-scores
    relevance_scores_z = (relevance_scores - np.mean(relevance_scores)) / np.std(relevance_scores)
    max_cosine_scores_z = (max_cosine_scores - np.mean(max_cosine_scores)) / np.std(max_cosine_scores)

    # Combine the normalized scores using the relevance indices
    combined_scores = [0] * len(node_list)
    for i, idx in enumerate(relevance_indices):
        combined_scores[idx] = relevance_scores_z[i] + max_cosine_scores_z[idx]

    # Assign the combined score to each node in node_list
    for i, node in enumerate(node_list):
        node['combined_score'] = combined_scores[i]

    # Sort node_list based on the combined scores in descending order
    sorted_node_list = sorted(node_list[0:50], key=lambda x: x.get('combined_score', float('-inf')), reverse=True)

    return sorted_node_list


def select_chunks(sorted_text, max_tokens = max_tokens):
    '''
    Selects relevant text chunks from sorted_text based on their relevance to a question_string.
    Can filter out irrelevant chunks based on a relevance determination, and will stop
    after finding a certain number of consecutive irrelevant chunks if filtering is enabled.

    Parameters:
    - sorted_text: List of dictionaries, each containing at least 'Tokens' keys.
    - question_string: The question or context string to determine relevance against.
    - filter: Boolean indicating whether to filter out irrelevant chunks.
    - threshold: The number of irrelevant sections allowed before stopping.

    Returns:
    - A list of dictionaries representing the selected text chunks.
    '''

    selected_entries = []
    total_tokens = 0

    for entry in sorted_text:
        tokens_in_entry = entry['Tokens']  # Assuming 'Tokens' is an integer
        if total_tokens + tokens_in_entry > max_tokens:
            break

        selected_entries.append(entry)
        total_tokens += tokens_in_entry


    return selected_entries


#Function to get embeddings
def find_embedding(text):
    '''
    Get the embedding for a text string
    '''

    #First, check if embedding is already in the dictionary
    if text in question_embeddings:
        return question_embeddings[text]

    #Otherwise, we need to calculate the embedding
    else:
        embedding = cg.get_embedding(text)

        #Store the embedding in the dictionary
        question_embeddings[text] = embedding

    return embedding

def process_text(text,keywords):

    # Create a translation table to replace each punctuation character with a space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Replace punctuation in the text with spaces
    text = text.translate(translator)

    #Whenever we find a keyword we want to replace spaces within the keyword with a '_'
    for keyword in keywords:
        text = text.lower().replace(keyword, keyword.replace(" ","_"))

    return text

def get_keywords(question):

    keys = keywords[question['ID']].dropna()

    return keys.to_list()

def load_tree(muni):
    embeddings_file_path = os.path.join(config['embeddings'], muni['Muni'] + '.pkl')

    with open(embeddings_file_path, "rb") as fIn:
        tree = pickle.load(fIn)

    return tree

# Creates the context section
def context_builder(question,state,muni):
    '''
    Builds the context using an embeddings approach
    '''

    #Get the tree
    tree = load_tree(muni)

    #Define type based on state
    if state == 'double_checking':
        type = question['Double Check Context']
    else:
        type = 'semantic'

    # Get question_string embedding
    question_embedding = find_embedding(question['Question Detail'])

    # Flatten the tree to get a list of leaf nodes
    node_list = flatten_tree(tree)

    # Now, sort the embeddings by cosine similarity
    sorted_da = sort_chunks(question_embedding, node_list)

    #Now rerank based on the question
    reranked = reranker(sorted_da, question, type)

    # Now select enough entries to fill the context window
    selected_text = select_chunks(reranked)

    #Now join together the strings and add two hashtags to sifnify new sections
    combined_text = '\n\n'.join(['##'+x['Leaf_Text'] for x in selected_text])

    return combined_text
