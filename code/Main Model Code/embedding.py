import os
import pickle
import pandas as pd
import tiktoken
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import gpt_functions as cg
from dotenv import load_dotenv
load_dotenv()

#Whether to skip certain operations (it will still do these operations for new munis)
skip_text_process = False
#If set to false runs embedding code even if embedding file exists already
skip_embed = True

# Load filepaths
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

text_path = config['muni_text']
embed_path = config['embeddings']
pre_embed_path = config['pre_embed']

# Check if embed_path exists
if not os.path.exists(embed_path):
    # If not, create the directory
    os.makedirs(embed_path)

# Whether or not we are using SLURM job arrays
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0)) # Id for first job array id
    task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0)) # Id for last job array id
    num_nodes = task_max - task_min + 1 # Total number of job array ids
else:
    task_id = 0
    num_nodes = 1

# Openai
client = OpenAI(api_key=os.getenv('openai_key'))

#Load in tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Setup text splitter
threshold = 1000 #Rough number of tokens to limit text chunks at

'''
Could consider using tiktoken as length here instead
'''

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name = 'gpt-3.5-turbo',
    chunk_size=threshold,
    chunk_overlap=2*threshold/10,
)

 #%%Functions
    
def token_len(string):
    return len(encoding.encode(string))

def is_leaf(node):
    return 'Leaf_Text' in node
    
def split_tree(tree, threshold, path=()):
    '''
    Recursively splits up text chunks in a tree structure that are too long to embed, and adds a 'Tree_Path' attribute to all nodes.
    :param tree: The hierarchical dictionary or a list of dictionaries to process.
    :param threshold: The length at which to split text chunks.
    :param path: The current path in the tree (initially empty).
    :return: A hierarchical dictionary with text chunks split as needed and 'Tree_Path' for all nodes.
    '''

    def process_node(node, path):
        # Assign 'Tree_Path' to every node
        node["Tree_Path"] = path
        if is_leaf(node):  # Check if the node is a leaf
            leaf_text = node['Leaf_Text']
            if token_len(leaf_text) > threshold:  # Split text if it exceeds the threshold
                split_texts = text_splitter.split_text(leaf_text)
                # Check if there are multiple parts
                if len(split_texts) > 1:
                    # Create a new subtree for each split text
                    return {'children': [{'name':f"Part_{index}",'Leaf_Text': text, 'Part': index,
                                          'Tree_Path': path + (node['name'],), 'Has_Table': node['Has_Table']}
                                         for index, text in enumerate(split_texts, start=1)],
                            'Tree_Path':node['Tree_Path'], 'name': node['name']}
                else:
                    # Only one part, so keep the original structure
                    node['Leaf_Text'] = split_texts[0]
                    node['Part'] = 0
                    return node
            else:
                node['Part'] = 0
                return node
        else:
            # Recursively process each child if the node has 'children'
            if 'children' in node:
                node['children'] = [process_node(child, path + (node['name'],)) for child in node['children']]
            return node

    # Check if the input is a list of nodes and process each node accordingly
    if isinstance(tree, list):
        return [process_node(root_node, path) for root_node in tree]
    else:
        return process_node(tree, path)


def clean_tree(tree):

    def merge_leaves(siblings, target_index):
        target_leaf = siblings[target_index]

        # Determine the indices of the previous and next siblings
        prev_index = target_index - 1 if target_index - 1 >= 0 else None
        next_index = target_index + 1 if target_index + 1 < len(siblings) else None

        # Determine the best candidate for merging based on 'Leaf_Text' length, ensuring they are leaf nodes
        candidates = []
        if prev_index is not None and is_leaf(siblings[prev_index]):
            candidates.append((prev_index, len(siblings[prev_index]['Leaf_Text']),'Previous'))
        if next_index is not None and is_leaf(siblings[next_index]):
            candidates.append((next_index, len(siblings[next_index]['Leaf_Text']),'Next'))

        # Sort candidates by their 'Leaf_Text' length to prioritize the shorter one
        candidates.sort(key=lambda x: x[1])

        # Merge with the best candidate if available
        if candidates:
            best_match_index = candidates[0][0]

            #Determine if previous of next sibling
            if candidates[0][2] == 'Previous':
                siblings[best_match_index]['Leaf_Text'] = siblings[best_match_index]['Leaf_Text'] + f"\n\n{target_leaf['name']}\n" + target_leaf['Leaf_Text']
                siblings[best_match_index]['name'] = siblings[best_match_index]['name'] + " & " + target_leaf['name']
            else:
                siblings[best_match_index]['Leaf_Text'] = target_leaf['Leaf_Text'] + f"\n\n{siblings[best_match_index]['name']}\n" + siblings[best_match_index]['Leaf_Text']
                siblings[best_match_index]['name'] = target_leaf['name'] + " & " + siblings[best_match_index]['name']


            siblings[best_match_index]['Has_Table'] = siblings[best_match_index].get('Has_Table', False) or target_leaf.get('Has_Table', False)

            del siblings[target_index]

    def process_node(node):
        if is_leaf(node):

            #Whether we need to drop a node
            if len(node['Leaf_Text'].strip()) == 0:
                return None  # Signal to remove this node

            if len(node['Leaf_Text']) < 200:
                # Mark the node for merging but don't merge here, as it requires sibling context
                return {'needs_merging': True, **node}
            else:
                return node
        else:
            # Process child nodes
            to_delete = []
            if 'children' in node:  # Ensure we have 'children' to process
                for i, child in enumerate(node['children']):
                    result = process_node(child)
                    if result is None:
                        to_delete.append(i)
                    elif result.get('needs_merging', False):
                        merge_leaves(node['children'], i)
                # Remove nodes in reverse order to not mess up the indices
                for index in reversed(to_delete):
                    del node['children'][index]
            return node

    # If the input is a list of nodes (as the root level could be), process each root-level node
    if isinstance(tree, list):
        for i, root_node in enumerate(tree):
            tree[i] = process_node(root_node)
    else:  # Single node case
        tree = process_node(tree)

    return tree

def transform_to_standard_tree(original_tree):
    def transform_node(key, value):
        node = {'name': key}
        # Check if the current node is a leaf node by looking for 'Leaf_Text'
        if isinstance(value, dict) and 'Leaf_Text' in value:
            node['Leaf_Text'] = value['Leaf_Text']
            # Carry over the 'Has_Table' attribute for leaf nodes
            if 'Has_Table' in value:
                node['Has_Table'] = value['Has_Table']
        else:
            # If the node is not a leaf, recursively transform its children
            children = []
            for child_key, child_value in value.items():
                if isinstance(child_value, dict):  # Ensure it's a nested dictionary
                    transformed_child = transform_node(child_key, child_value)
                    children.append(transformed_child)
            if children:
                node['children'] = children
        return node

    # Start the transformation with a pseudo-root to handle top-level nodes uniformly
    pseudo_root = {'Root': original_tree}
    # Since 'Root' is a pseudo-node, directly return its children to avoid nesting under 'Root'
    return transform_node('Root', pseudo_root)['children']

def prepare_text(tree, path=()):
    '''
    Recursively prepares text in a hierarchical tree structure for embedding.
    :param tree: The hierarchical dictionary or a list of dictionaries to process.
    :param path: The current path in the tree (initially empty).
    :return: The modified tree with prepared text for embedding.
    '''

    def process_node(node, path):
        if is_leaf(node):  # Check if the node is a leaf

            #Add path to leaf text
            node['Leaf_Text'] = 'Section Path: '+'->'.join(node['Tree_Path'][1:])+'->'+node['name'] + '\nSection Text: ' + node['Leaf_Text'].strip()

            # Make embed text same as leaf text (adds flexibility to embed a summary or other text)
            node['Embed_Text'] = '->'.join(node['Tree_Path'])+'\n'+  node['Leaf_Text']

            # Store the number of tokens in the leaf text
            node['Tokens'] = token_len(node['Leaf_Text'])

            return node
        else:
            # Recursively process each child if the node has 'children'
            if 'children' in node:
                for i, child in enumerate(node['children']):

                    node['children'][i] = process_node(child, path + (node['name'],))


            return node

    # Handle the case where the input might be a list of nodes
    if isinstance(tree, list):
        return [process_node(node, path) for node in tree]
    else:
        return process_node(tree, path)

def embed_text(tree):
    '''
    Recursively embeds text in a hierarchical tree structure.
    :param tree: The hierarchical dictionary or a list of dictionaries to process.
    :return: The modified tree with text embeddings.
    '''

    def process_node(node):
        if is_leaf(node):  # Check if the node is a leaf
            text_to_embed = node['Embed_Text']
            text_list = [text_to_embed]  # Split the text if needed
            embeddings = embed(text_list)  # Create embeddings
            node['Embeddings'] = embeddings
        else:
            # Recursively process each child if the node has 'children'
            if 'children' in node:
                for child in node['children']:
                    process_node(child)  # Recursive call
        return node

    # Handle the case where the input might be a list of nodes
    if isinstance(tree, list):
        return [process_node(node) for node in tree]
    else:
        return process_node(tree)
def embed(text_list):
    '''
    Embed the list of text using either OpenAI's embeddings API, VoyageAI's embeddings API, or an open source method.

    :param text_list: List of texts to be embedded.
    :param method: The embedding method to use ('openai', 'voyageai', or 'opensource').
    :return: List of embeddings.
    '''

    embeddings = []

    # Iterate over each text in the list and get its embedding
    for text in text_list:

        # Extract the embedding and add to the list
        embeddings.append(cg.get_embedding(text))

    return embeddings



def embed_questions(questions):
    
    #Regular method
    embeddings = embed(questions)
    
    return embeddings

#Function to get filepath for row in sample data excel
def get_path(row):

    source_mapping = {
        'Municode': 'municode',
        'American Legal Publishing': 'alp',
        'Ordinance.com': 'ordDotCom'
    }

    #1. Find source folder
    source_name = row['Source']
    source_folder = source_mapping[source_name]

    #2. Find state
    state_folder = row['State']

    #3. Find filename
    muni_name = row['UNIT_NAME']
    id_num = row['CENSUS_ID_PID6']
    filename = muni_name+"#"+str(id_num)+"#"+state_folder+".pkl"

    #4. Return filepath
    return os.path.join(text_path,source_folder,state_folder,filename)



def get_files(type = 'all', sample = None):
    # Read in sample data
    sample_data = pd.read_excel(os.path.join(config['raw_data'], 'Sample Data.xlsx'))

    # Generate file paths
    file_paths = [get_path(row) for _, row in sample_data.iterrows()]

    # Split files across nodes
    num_files = len(file_paths)
    files_per_task = num_files // num_nodes
    extra_files = num_files % num_nodes

    # Determine which text trees we will embed on this node
    if task_id < extra_files:
        start_idx = task_id * (files_per_task + 1)
        end_idx = start_idx + files_per_task + 1
    else:
        start_idx = task_id * files_per_task + extra_files
        end_idx = start_idx + files_per_task

    files_for_task = file_paths[start_idx:end_idx]

    return files_for_task

##Main code
files_for_task = get_files(type = 'all')

# If task_id is 0, then embed questions
if task_id == 0:
    questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))
    
    embeddings = embed_questions(questions['Question Detail'].to_list())
    
    with open(embed_path+"/Questions.pkl", "wb") as fOut:
        pickle.dump({'Questions': questions, 'embeddings': embeddings}, fOut)
        
# Processing the selected files for each node
for file_path in files_for_task:
    print(file_path)
    
    #Determine the filepath where to save the embedding
    file_extension = os.path.basename(file_path)
    
    # Check if the pickle file already exists
    if os.path.exists(os.path.join(embed_path,file_extension)) and skip_embed:
        continue  # Skip this loop iteration
    
    if os.path.exists(os.path.join(pre_embed_path,file_extension)) and skip_text_process:
        with open(os.path.join(pre_embed_path,file_extension), 'rb') as file:
            prepared_tree = pickle.load(file)
    else:

        #Read in the tree
        with open(file_path, 'rb') as file:
            tree = pickle.load(file)

        #First, transform heirarchical dictionary structure to the standard python tree format
        standard_tree = transform_to_standard_tree(tree)

        #Second, merge small sections together and drop empty sections
        cleaned_tree = clean_tree(standard_tree)

        #Second, split up text chunks that are too long to embed
        tree_split = split_tree(cleaned_tree, threshold)

        print("Prepping")
        #Fourth, we prepare the text of the tree before embedding by assigning the text to embed and counting token lengths
        prepared_tree = prepare_text(tree_split)
        
        #Cache processed tree to avoid recalling LLM 
        with open(os.path.join(pre_embed_path,file_extension), "wb") as fOut:
            pickle.dump(prepared_tree, fOut)

    print("Embedding")
    #Fourth, we embed the text
    embed_tree = embed_text(prepared_tree)
    print("Embedding finished")

    #Save the embedding
    with open(os.path.join(embed_path,file_extension), "wb") as fOut:
        pickle.dump(embed_tree, fOut)




    