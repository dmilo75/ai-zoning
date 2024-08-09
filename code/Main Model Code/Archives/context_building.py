import pickle
import random
random.seed(2)
import os
import json

from scipy.spatial.distance import cosine
import gpt_functions as cg
import yaml
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

#Max tokens
max_tokens = config['max_tokens']*1000

##Context building functions

#If determine_relevant returns a value that is not 'DO NOT EXPLORE' or 'EXPLORE', we need to parse the response
def parse_explore(prompt,response_text):

    #Make a new prompt with the old prompt and the response text
    newprompt = prompt + '\nUser: ' + response_text

    tools = getattr(cg, 'parse_explore_section')
    tool_choice = {"type": "function", "function": {"name": "parse_explore_section"}}

    response = cg.gpt_api_wrapper(cg.gpt_api_call, (newprompt, 'gpt-3.5-turbo-0125', tools,tool_choice, '', []))

    # Get json string
    json_string = response.choices[0].message.tool_calls[0].function.arguments

    #Parse json
    response_dic = json.loads(json_string)

    return response_dic['explore']

# Function to use gpt to determine whether a chunk of text is relevant to a question or not
def determine_relevant(node, question):

    Task = (
    "You are a municipal zoning ordinance expert. You are given a zoning question and details "
    "about a section from a municipal zoning ordinance. These details include the section path "
    "(where in the table of contents the section is located), a summary of the text in the section, "
    "and when applicable a list of subsections you can read within the section if you explore further. "
    "Your task is to determine whether exploring the section is relevant to answering the question. You reply with a brief "
    "argument explaining whether exploring the section is relevant to the question. Following your argument you "
    "conclude with 'EXPLORE' if the section is relevant and 'DO NOT EXPLORE' if the section is not relevant."
    )

    prompt = []

    prompt.extend([
        f"Zoning Question: {question['Question Detail']}",
        f"Section Path: {'->'.join(node['Tree_Path'][1:]+(node['name'],))}",
        f"Section Summary: {node['Summary']}",
    ])


    if 'children' in node:
        sub_sections = [child['name'] for child in node['children']]
        prompt.append(f"Subsections in this Section: {', '.join(sub_sections)}")

    prompt = '\n'.join(prompt)
    response = cg.gpt_api_wrapper(cg.gpt_api_call, (prompt, 'gpt-3.5-turbo-0125', None, None, Task, []))

    response_text = response.choices[0].message.content

    if 'DO NOT EXPLORE' in response_text:
        explore = False
    elif 'EXPLORE' in response_text:
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
        scores = [1 - cosine(question_embedding, emb) for emb in embeddings]
        item["max_cosine_similarity"] = max(scores) if scores else -1

    # Sort the list based on the maximum cosine similarity scores
    sorted_node_list = sorted(node_list, key=lambda x: x.get("max_cosine_similarity", -1), reverse=True)

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


# Creates the context section
def embeddings_context_builder(question, tree,filter = True):
    '''
    Builds the context using an embeddings approach
    '''
    question_string = question['Question Detail']

    # Get question_string embedding
    question_embedding = question_embeddings[question_string]

    # Flatten the tree to get a list of leaf nodes
    node_list = flatten_tree(tree)

    # Now, sort the embeddings by cosine similarity
    sorted_da = sort_chunks(question_embedding, node_list)

    # Now select enough entries to fill the context window
    selected_text = select_chunks(sorted_da, question, filter = filter)

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

