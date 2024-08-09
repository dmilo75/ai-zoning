from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import pandas as pd
import os
import yaml
import openai
import threading
import queue
import random
import time
import tiktoken

#Whether to skip certain operations (it will still do these operations for new munis)
skip_text_process = True
skip_embed = True

#Whether to call an LLM to summarize text with tables
summ_text = False

# Load filepaths
with open('../../config.yaml', 'r') as file:
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


#Load in tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Load the embedding model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Setup text splitter
treshold = 3000 #Rough number of tokens to limit text chunks at

'''
Could consider using tiktoken as length here instead
'''

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name = 'gpt-3.5-turbo',
    chunk_size=treshold,
    chunk_overlap=treshold/10,
)

encoding_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name = 'gpt-3.5-turbo',
    chunk_size=128,
    chunk_overlap=30,
)

voyage_api_key = "pa-PyJnp62FZd_G4DUT3oOnifdKRq9PSKum4-90TYUJCIk"

#%%OpenAI Setup

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key = "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")
        
def call_openai_api_ask(prompt, text, model, output_queue):
    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        output_queue.put(res)
    except openai.OpenAIError as e:
        output_queue.put(e)

def chat_gpt_api_call(thread_func, args, timeout):
    '''
    Wrapper for calling OpenAI (Chat GPT). We allow for periods where the server times out or is slow
    '''
    for delay_secs in (2 ** x for x in range(6)):
        randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
        sleep_dur = delay_secs + randomness_collision_avoidance
        time.sleep(sleep_dur)
        
        output_queue = queue.Queue()
        thread = threading.Thread(target=thread_func, args=(*args, output_queue))
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            print("The API call did not return a response within the timeout period. Retrying...")
            continue
        
        try:
            res = output_queue.get_nowait()
            if isinstance(res, openai.OpenAIError):
                print(f"Error: {res}. Retrying...")
                continue
            else:
                return res['choices'][0]['message']['content']
        except queue.Empty:
            print("The API call did not return a response within the timeout period. Retrying...")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying...")
            continue
    
    raise TimeoutError("The API call did not return a response after multiple attempts.")

def chat_gpt(prompt,text,model):
    
    return chat_gpt_api_call(call_openai_api_ask,(prompt,text,model),120)

 #%%Functions
 
def flatten_tree(tree, path=()):
    """
    Recursively flatten a hierarchical dictionary.

    :param tree: The hierarchical dictionary to flatten.
    :param path: The current path in the tree (initially empty).
    :return: A list of leaf dictionaries with an added 'Tree_Path' key.
    """
    # Check if the current dictionary is a leaf node
    if "Leaf_Text" in tree and "Has_Table" in tree:
        # Add the 'Tree_Path' key to the leaf node
        tree["Tree_Path"] = path
        return [tree]
    else:
        # Recursively process each child
        leaves = []
        for key, subtree in tree.items():
            leaves.extend(flatten_tree(subtree, path + (key,)))
        return leaves
   
    
def token_len(string):
    return len(encoding.encode(string))

    
def split_flattened_tree(flattened_tree, threshold = treshold):
    '''
    Splits up text chunks that are too long to embed
    '''
    new_tree = []
    for item in flattened_tree:
        leaf_text = item['Leaf_Text']
        if token_len(leaf_text) > threshold: #Number of characters/4 is a rough proxy for tokens
            split_texts = text_splitter.split_text(leaf_text)  
            for index, text in enumerate(split_texts, start=1):
                new_item = item.copy()
                new_item['Leaf_Text'] = text
                new_item['Part'] = index
                new_item['Tree_Path'] = new_item['Tree_Path'][:-1] + (f"{new_item['Tree_Path'][-1]}",)
                new_tree.append(new_item)
        else:
            item['Part'] = 0
            new_tree.append(item)
    return new_tree


def summarize(text):
    
    '''
    Create a text_to_embed parameter for each leaf node. For table text we send the raw text to an LLM
    For non table text we just set this as the regular text
    '''
    
    prompt = "Briefly summarize the below below text that contains a table: "
    
    summary = chat_gpt(prompt,text,"gpt-3.5-turbo")
    
    return summary

def prepare_text(tree_list):
    
    for item in tree_list:

        #First, add the path as a string to the embed_text 
        item['Leaf_Text'] = "Text Path: "+"->".join(item['Tree_Path'])+"\n"+item['Leaf_Text']
        
        
        if summ_text:
        
            #If we do not have a table then this is the text to embed
            if not item['Has_Table']:
                item['Embed_Text'] = item['Leaf_Text']
            
            #Otherwise, if there is a table then we want to embed a summary of the text
            else:
                item['Embed_Text'] = summarize(item['Leaf_Text'])
                
        else:
            item['Embed_Text'] = item['Leaf_Text']
        
        #Store the number of tokens in the leaf text
        item['Tokens'] = token_len(item['Leaf_Text'])
        
    return tree_list
   
#Split the text we are about to embed into smaller chunks
def split_embed_text(text):
    
    #split_texts = encoding_text_splitter.split_text(text)  
    
    #return split_texts

    return [text]

def embed_text(tree_list):
     
    for item in tree_list:
        
        text_to_embed = item['Embed_Text']
        text_list = split_embed_text(text_to_embed)
        embeddings = embed(text_list)
        item['Embeddings'] = embeddings
     
    return tree_list

def embed(text_list, method='openai'):
    '''
    Embed the list of text using either OpenAI's embeddings API, VoyageAI's embeddings API, or an open source method.

    :param text_list: List of texts to be embedded.
    :param method: The embedding method to use ('openai', 'voyageai', or 'opensource').
    :return: List of embeddings.
    '''

    # If using open source embeddings
    if method == 'opensource':
        # Assuming 'model' is a pre-trained model from an open source library like sentence-transformers
        embeddings = model.encode(text_list)

    # If using VoyageAI
    elif method == 'voyageai':
        import voyageai 
        from voyageai import get_embeddings

        # Set your Voyage API key
        voyageai.api_key = "pa-kNNSjjFPPsOgVQm8NjE_hRJ80D2ZGPU0glgMk9fy6LA"

        # Embed the documents using VoyageAI
        embeddings = get_embeddings(text_list, model="voyage-01")

    # If using OpenAI's embeddings
    else:

        embeddings = []
        # Iterate over each text in the list and get its embedding
        for text in text_list:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"  # Specify the model you want to use
            )
            # Extract the embedding and add to the list
            embeddings.append(response.data[0].embedding)

    return embeddings

#Used to rephrase questions using Chat GPT
def rephrase(questions):
    # Prepare the prompt
    prompt = """
        You are an AI language model assistant. Your task is to generate 3 different versions
        of the given user question to retrieve relevant documents from a vector database.
        By generating multiple perspectives on the user question, your goal is to help the
        user overcome some of the limitations of distance-based similarity search. Provide
        these alternative questions separated by newlines. Original question:
    """
    
    # Generate rephrased versions
    rephrased = [chat_gpt(prompt,x, "gpt-4-1106-preview") for x in questions]

    # Split the rephrased questions into separate lists
    rephrased_split = [[version.strip() for version in rephrase.split('\n')] for rephrase in rephrased]

    # Initialize lists for embeddings
    avg_embeddings = []

    # Embed each rephrased question and calculate average embedding
    for trio in rephrased_split:
        embeddings = [model.encode(question) for question in trio]
        avg_embedding = sum(embeddings) / len(embeddings)
        avg_embeddings.append(avg_embedding)

    return avg_embeddings

#Hypothetical answers
def HyDE(questions):
    
    '''
    Use GPT 4 for this task since not much data
    '''
    
    prompt = "Please write the subsection of a municipal zoning ordinance that contains the answer to the following question. Do not provide any explanation.\n"
    
    hypo_answers = [chat_gpt(prompt,x,"gpt-4-1106-preview") for x in questions]
    
    embeddings = model.encode(hypo_answers)
    
    return embeddings

def embed_questions(questions):
    
    #Rephrase questions a few ways 
    #embeddings = rephrase(questions)
    
    #Hypothetical answers
    #embeddings = HyDE(questions)
    
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

def get_files():


    """
    Ideally, we take the list of files to embed from the list of selected munis not from all text

    1. Import sample data

    2. Loop through each row and get file path

    3.
    """

    sample_data = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))

    #loop over each row in sample_data and get file path
    file_paths = []
    for index, row in sample_data.iterrows():
        #Send row to function that gets file path
        file_paths.append(os.path.join(text_path,row['CENSUS_ID_PID6']+".pkl"))

    num_files = len(file_paths)
    files_per_task = num_files // num_nodes
    extra_files = num_files % num_nodes

    #Determine which text trees we will embed on this node
    if task_id < extra_files:
        start_idx = task_id * (files_per_task + 1)
        end_idx = start_idx + files_per_task + 1
    else:
        start_idx = task_id * files_per_task + extra_files
        end_idx = start_idx + files_per_task

    files_for_task = file_paths[start_idx:end_idx]
    
    return files_for_task

#%%Main code

files_for_task = get_files()

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
        '''
        Needs to read in pickle file now
        '''
        with open(file_path, 'rb') as file:
            tree = pickle.load(file)
    
        #First flatten the tree
        print("Flattening")
        flattened_tree = flatten_tree(tree)
        
        print("Splitting")
        #Second, split up text chunks that are too long to embed
        split_tree = split_flattened_tree(flattened_tree,4000)
        
        print("Prepping")
        #Third, we prepare the text of the tree before embedding
        prepared_tree = prepare_text(split_tree)
        
        #Cache processed tree to avoid recalling LLM 
        with open(os.path.join(pre_embed_path,file_extension), "wb") as fOut:
            pickle.dump(prepared_tree, fOut)

    print("Embedding")
    #Fourth, we embed the text
    embed_tree = embed_text(prepared_tree)
    
    print("Embedding finished")
    
    #%%


    #Save the embedding
    with open(os.path.join(embed_path,file_extension), "wb") as fOut:
        pickle.dump(embed_tree, fOut)



    
    
    