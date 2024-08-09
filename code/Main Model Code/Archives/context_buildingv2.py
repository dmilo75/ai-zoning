import pickle
import random
random.seed(2)
import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from rank_bm25 import BM25Okapi
import string
import gpt_functions as cg
import yaml
import pandas as pd
import cohere
co = cohere.Client('5xbKAvQhLaciaxYqOT2k9mh81wFttU8aL3cWNu9u')
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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

#If determine_relevant returns a value that is not 'DO NOT EXPLORE' or 'EXPLORE', we need to parse the response
def parse_explore(prompt,response_text):

    #Make a new prompt with the old prompt and the response text
    newprompt = prompt + '\nUser: ' + response_text

    tools = getattr(cg, 'parse_explore_section')
    tool_choice = {"type": "function", "function": {"name": "parse_explore_section"}}

    response = cg.gpt_api_wrapper(cg.gpt_api_call, (newprompt, 'gpt-3.5-turbo-0125', tools,tool_choice, '', [],1))

    # Get json string
    json_string = response.choices[0].message.tool_calls[0].function.arguments

    #Parse json
    response_dic = json.loads(json_string)

    return response_dic['explore']

# Function to use gpt to determine whether a chunk of text is relevant to a question or not
def determine_relevant(node, question):

    Task = (
    "You are a legal research assistant tasked with finding relevant sections from a zoning ordinance for a research question."
    "a section from a municipal zoning ordinance. You get to see the text path"
    "(where in the table of contents the section is located) and the section text. "
    "Your task is to determine whether the section contains relevant evidence for the question. You reply with a detailed "
    "argument thinking step by step about whether any of part of the section has relevant evidence for the question. Following your argument you "
    "conclude with 'RELEVANT' if the section has relevant evidence and 'NOT RELEVANT' if the section does not."
    )

    prompt = []



    prompt.extend([
        f"Zoning Question: {question['Question Detail']}",
        f"Section Text: {node['Leaf_Text']}",
    ])

    prompt = '\n'.join(prompt)
    response = cg.gpt_api_wrapper(cg.gpt_api_call, (prompt, 'gpt-3.5-turbo-0125', None, None, Task, [],1))

    response_text = response.choices[0].message.content

    print(prompt)
    print(response_text)

    if 'NOT RELEVANT' in response_text:
        explore = False
    elif 'RELEVANT' in response_text:
        explore = True
    else:
        '''
        Send to parser in this case, with tool function to determine whether to explore or not
        '''

        return parse_explore(prompt,response_text)


    return explore


def toc_navigator(question, tree):
    '''
    Traverse a tree structure and return the relevant sections based on a question.

    :param question: The question for which relevant sections are to be found.
    :param tree: The hierarchical dictionary representing the tree with summaries.
    :return: A list of relevant sections (leaf nodes) from the tree.

    - The first call (root node) always traverses all children.
    - Subsequent calls check for relevance before traversing further.

    To add
    - As a first step we could break down the tree until there is at least x amount of nodes
    '''


    relevant_sections = []
    not_relevant_sections = []

    def traverse_tree(question, nodes, section_name= ""):
        nonlocal relevant_sections
        nonlocal not_relevant_sections

        for node in nodes:
            current_section_name = node.get('name', section_name)  # Use the node's name or inherit from parent

            # Check if the current node is relevant
            whether_explore = determine_relevant(node, question)

            if 'Leaf_Text' in node:
                if not whether_explore:
                    not_relevant_sections.append(node)
                else:
                    relevant_sections.append(node)

            # If the node has children, recursively process each child
            elif 'children' in node and whether_explore:
                traverse_tree(question, node['children'], section_name=current_section_name)

    # Start the traversal from the children of the root
    traverse_tree(question, tree[0]['children'])

    return relevant_sections, not_relevant_sections

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


#Reranking parent function
def reranker(sorted_da, question):
    '''
    Reranks the sorted_da based on the question
    '''

    #Get the reranking style
    reranking_style = question['Reranker']

    if reranking_style == 'semantic':
        return semantic_reranker(sorted_da, question)
    elif reranking_style == 'keyword':
        return rerank_bm25(sorted_da,question)
    else:
        return sorted_da

# Reciprocal Rank Fusion (RRF) algorithm, a way to aggregate multiple rankings
def rrf(all_rankings: list[list[int]]):

    k = 1
    scores = {}
    for algorithm_ranks in all_rankings:
        for rank, idx in enumerate(algorithm_ranks):
            if idx in scores:
                scores[idx] += 1 / (k + rank)
            else:
                scores[idx] = 1 / (k + rank)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores

def semantic_reranker(node_list, question):
    question_string = question['Question Rephrase']

    # Get the top documents according to the initial criteria
    docs = [x['Leaf_Text'] for x in node_list[0:50]]

    # Assuming `co.rerank` returns a new ranking based on the reranker algorithm
    response = co.rerank(
        model='rerank-english-v2.0',
        query=question_string,
        documents=docs,
    )

    # Extract the new order from the reranker response
    new_order = [x.index for x in response]

    # Prepare rankings for RRF: original ranking and reranker's new order
    all_rankings = [list(range(len(node_list))), new_order]

    # Apply RRF to combine the rankings
    sorted_indices = rrf(all_rankings)

    # Reorder node_list based on the sorted indices from RRF
    sorted_node_list = [node_list[idx] for idx, _ in sorted_indices]

    return sorted_node_list

def select_chunks(sorted_text, question, filter=True, threshold=10, max_tokens = max_tokens):
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
    irrelevant_count = 0  # Counter for consecutive irrelevant entries

    for entry in sorted_text:
        tokens_in_entry = entry['Tokens']  # Assuming 'Tokens' is an integer
        if total_tokens + tokens_in_entry > max_tokens:
            break

        # Determine if relevant with gpt, if choosing to filter
        if filter:
            relevance = determine_relevant(entry, question)
            if not relevance:
                irrelevant_count += 1
                if irrelevant_count > threshold:
                    break  # Terminate the loop if threshold is exceeded
                continue  # Skip adding this entry to selected_entries
        else:
            relevance = True

        # Reset irrelevant counter if a relevant entry is found
        irrelevant_count = 0

        if relevance:
            selected_entries.append(entry)
            total_tokens += tokens_in_entry

    return selected_entries


#Function to get embeddings
def get_embedding(text):
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

def rerank_bm25(sorted_da,question):

    #Get the keywords
    keywords = get_keywords(question)

    # Create the entire corpus of text
    corpus = [process_text(x['Leaf_Text'], keywords) for x in sorted_da]

    # Make keywords lowercase now and replace spaces with underscores
    keywords = [keyword.lower().replace(" ", "_") for keyword in keywords]

    # Tokenize the corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    # Create the bm25 object
    bm25 = BM25Okapi(tokenized_corpus)

    # Get the scores
    doc_scores = bm25.get_scores(keywords)

    # Add the scores to the sorted_da
    for i in range(len(sorted_da)):
        sorted_da[i]['bm25'] = doc_scores[i]

    # Now make a new sorted list based on bm25
    sorted_da_bm25 = sorted(sorted_da, key=lambda x: x['bm25'], reverse=True)

    # Apply RRF
    # Original ranking
    original_ranking = [i for i in range(len(sorted_da))]
    # BM25 ranking
    bm25_ranking = [sorted_da.index(item) for item in sorted_da_bm25]

    # Combine rankings for RRF
    final_rankings = rrf([bm25_ranking,original_ranking])

    # Create a dictionary of scores
    scores_dict = {idx: score for idx, score in final_rankings}

    # Sort sorted_da by RRF scores, using the original indices of items in sorted_da
    sorted_da_final = sorted(sorted_da, key=lambda x: scores_dict[sorted_da.index(x)], reverse=True)

    return sorted_da_final


# Creates the context section
def embeddings_context_builder(question, tree,filter = True):
    '''
    Builds the context using an embeddings approach
    '''

    # Get question_string embedding
    question_embedding = get_embedding(question['Question Detail'])

    # Flatten the tree to get a list of leaf nodes
    node_list = flatten_tree(tree)

    # Now, sort the embeddings by cosine similarity
    sorted_da = sort_chunks(question_embedding, node_list)

    #Now rerank based on the question
    reranked = reranker(sorted_da, question)

    # Now select enough entries to fill the context window
    selected_text = select_chunks(reranked, question, filter = False)

    return selected_text

def toc_nav_context_builder(question, tree):

    #Get the relevant sections by navigating the tree
    relevant_sections, not_relevant_sections = toc_navigator(question, tree)

    #Now, check how many tokens we have
    tokens = sum([section['Tokens'] for section in relevant_sections])

    #If we have too many tokens, we need to filter
    if tokens > max_tokens:

        question_string = question['Question Detail']

        #First, sort the leaf nodes
        sorted_text = sort_chunks(question_embeddings[question_string], relevant_sections)

        #Second, select the chunks (text is already filtered)
        return select_chunks(sorted_text, question, filter = False)

    #If we have too few tokens, we need to add more
    else:
        #First get all nodes
        all_nodes = flatten_tree(tree)

        #Second, remove the relevant and not relevant sections
        for section in relevant_sections:
            all_nodes.remove(section)
        for section in not_relevant_sections:
            all_nodes.remove(section)

        #Third, sort the leaf nodes
        sorted_text = sort_chunks(question_embeddings[question['Question Detail']], all_nodes)

        #Fourth, append on sorted_text to relevant text
        relevant_sections.extend(select_chunks(sorted_text, question, filter = False, max_tokens = max_tokens-tokens))

        return relevant_sections

