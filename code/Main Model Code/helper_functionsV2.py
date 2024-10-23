import time
import pickle
import random
import openai
import json
import pandas as pd
import os
from sentence_transformers import util 
import sys
import glob
import gpt_functions as cg
import korhelper as kh
import yaml
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
#%%Config parameters

use_chatgpt = os.environ.get('USE_CHATGPT') 
model_type = os.environ.get('MODEL_TYPE')

#How many nearest neighbors to use
num_neighbors = config['num_neighbors']

questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))

yes_no_qs = questions[questions['Question Type'] == 'Binary']['ID'].to_list()
numerical = questions[questions['Question Type'] == 'Numerical']['ID'].to_list()
categorical = questions[questions['Question Type'] == 'Categorical']['ID'].to_list()
lot_size = questions[questions['Question Type'] == 'lot_size']['ID'].to_list()

relevant_questions = questions['ID'].to_list()

#%%OpenAI setup

#Openai
if use_chatgpt == 'True':
    openai.api_key = config['openai_key'] #OpenAI Key
    openai.Engine.list()  # check we have authenticated

#%%Llama setup
if use_chatgpt == 'False':
    
    sys.path.append(config['exllama_path'])
    from model import ExLlama, ExLlamaCache, ExLlamaConfig
    from tokenizer import ExLlamaTokenizer
    from generator import ExLlamaGenerator

    # Directory containing model, tokenizer, generator
    model_directory =  config[model_type+"_path"]
    
    # Locate files we need within that directory
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]
    
    # Create config, model, tokenizer and generator
    llamaConfig = ExLlamaConfig(model_config_path)               # create config from config.json
    llamaConfig.model_path = model_path                          # supply path to model weights file
    llamaConfig.max_seq_len = 4096
    
    model = ExLlama(llamaConfig)                                 # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
    
    cache = ExLlamaCache(model)                             # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
    
    # Configure generator 
    generator.disallow_tokens([tokenizer.eos_token_id])
    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = 0.1
    generator.settings.top_p = 0.90
    generator.settings.top_k = 100
    generator.settings.typical = 0.5

#%% Pull in data on question embeddings
with open(config['embeddings']+"/Questions.pkl", "rb") as fIn:
    data = pickle.load(fIn)

# Extract the list of question details and the embeddings
question_details = data['Questions']  # This is a list of question details (texts)
embeddings = data['embeddings']

# Create a map from question text to its embedding
question_to_embedding = dict(zip(question_details, embeddings))

#%% Get nearest neighbors
def find_nearest_neighbors(question, contexts, context_embeddings):

    # Get the query embedding from question_embeddings using the 'question' index
    query_emb = question_to_embedding[question['Question Detail']]

    # Compute dot scores between query_emb and context_embeddings
    scores = util.dot_score(query_emb, context_embeddings)[0].cpu().tolist()

    # Get the indices of the scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    # Extract the indices of the top 'n' contexts
    top_n_indices = sorted_indices[:num_neighbors]

    # Return the corresponding contexts using these indices
    nearest_neighbors = [contexts[i] for i in top_n_indices]

    return nearest_neighbors


#%% Create context text block

def create_context(question,contexts,context_embeddings):
    
    contexts = find_nearest_neighbors(question,contexts,context_embeddings)
    
    context = "Context: \n \n"
    for text in contexts:
        context += text + "\n \n \n"
    return context


#%% Wrapper around the chat completion

def llama_response(context, question,muni):

    #Get the question
    query = question['Question Detail']

    #Get instructions for the type of question
    if question['ID'] in categorical:
        instructions1 = "Answer the question using the context with a list of entities and explain your answer. Say 'I don't know' if you do not know in the beginning. Only say 'I don't know' if the context is completely irrelevant. The words 'I don't know' should be used only when you have very low confidence."
        instructions2 = "Reply with the list of entities that the answer provides to the question delimited by commas. Do not provide any explanation"
        response_format = 'Entities in list format: '
    elif question['ID'] in numerical:
        instructions1 = "Answer the question using the context with an integer and explain your answer. Say 'I don't know' if you do not know. Let me know the integer, or 'I don't know' if you don't know in the beginning. Only say 'I don't know' if the context is completely irrelevant. The words 'I don't know' should be used only in extreme cases. The words 'I don't know' should be used only when you have very low confidence."
        instructions2 = "Reply with the integer that the answer provides to the question. Do not provide any explanation"
        response_format = "Integer: "
    elif question['ID'] in lot_size:
        instructions1 = "Reply with a list of zoning districts with their respective lot size minimums. Do this for as many districts as possible. Say 'I don't know' if you do not know if you don't know in the beginning. Only say 'I don't know' if the context is completely irrelevant. The words 'I don't know' should be used only in extreme cases. The words 'I don't know' should be used only when you have very low confidence."
        instructions2 = "Format the answer to the question as 'District Name' : integer. Only provide one integer for each district and list districts in a numbered list. For example, we would reformat '1. District 1: Lot area minimum 12,000 square feet \n 2. District 2: Lot area minimum 5,000 square feet (3,000 square feet for legacy units)' as '1. District 1: 12000 \n 2. District 2: 5000'"
        response_format = 'Formatted answer: '
    else:
        instructions1 = "Answer the question using the context with 'Yes', 'No', or 'I don't know' and explain your answer. Tell 'Yes', 'No', or 'I don't know' in the beginning. Only say 'I don't know' if the context is completely irrelevant. The words 'I don't know' should be used only in extreme cases. The words 'I don't know' should be used only when you have very low confidence."
        instructions2 = "Reply with whether the answer to the question is 'Yes' or 'No'. Do not provide any explanation."
        response_format = "Answer: "         

    start_time = time.time()
    prompt = "[INST] " + context + '\n\n' + query + "\n Instructions: " + instructions2 + ". [/INST]"
    while True:
        try:
            if question['ID'] == 2 or question['ID'] == 27:
                output = generator.generate_simple(prompt, max_new_tokens=250)
            else:
                output = generator.generate_simple(prompt, max_new_tokens=200)
            break  # Exit the loop if it succeeds.
        except:
            
            # If an exception occurs, shorten the context.
            context = context.rsplit("\n \n \n", 1)[0]
            
            # If no context left, raise value error
            if not context:
                raise ValueError("Context has been shortened to the point of being empty.")
            
            # Update the prompt with the shortened context and retry.
            prompt = "[INST] " + context + '\n\n' + query + "\n Instructions: " + instructions1 + ". [/INST]"
    
    answer = output[len(prompt):].strip()

    end_time = time.time()
    
    #Extract the specific answer from the unstructured response
    prompt = "[INST] Question: "+query+"\n\n Answer: "+answer+"\n\n Instructions: "+instructions2+". [/INST] "+response_format
    finalOutput = kh.korHelper(question, answer, prompt)
    print('origAns ', answer)
    print('koranswer ', finalOutput)
    end_time = time.time()
    elapsed_time = round(end_time-start_time)
    print("Time taken :"+str(elapsed_time))

    #Format the response object
    response = {}
    response['Answer'] = finalOutput
    response['Explanation'] = answer
    response['Compute Time'] = elapsed_time
    response['Muni'] = muni
    response['Question'] = question['ID']
    response['Context'] = context
    
    return response


#%%Helper functions for Chat GPT

#Which type of function to use
def determine_function_type(question):
    if question['Question Type'] == 'Binary': 
        function = "answer_question_yes_no"
    elif question['Question Type'] == 'Numerical': 
        function = "answer_question_numerical"
    elif question['Question Type'] == 'Categorical': 
        function = "answer_question_categorical"
    elif question['Question Type'] == 'Lot Size': 
        function = "answer_question_lot_size"
    return function


def parse_response(res, question,muni):
    
    
    json_string = res['choices'][0]['message']['function_call']['arguments']

    try:
        response = json.loads(json_string)
    except:

        cleaned_json = json_string
        
        clean_vals = {}
        
        for key in ['Explanation']:
            try:
                # Split by the given key
                pre_value = cleaned_json.split(f'\"{key}\": \"')[0] + f'\"{key}\": \"'
                value = cleaned_json.split(f'\"{key}\": \"')[1].split('\",\n')[0]
                
                # Replace the original value with 'dummy text'
                cleaned_json = pre_value + "dummy text" + cleaned_json.split(value)[1]
        
                # Store the original value in the clean dictionary
                clean_vals[key] = value
            except:
                continue
        
        response = json.loads(cleaned_json)
        for val in clean_vals.keys():
            response[val] = clean_vals[val]

    response['Question'] = question['ID']
    response['Muni'] = muni
    return response


def call_chat_completion(context,question,muni,model):
    
    #Get query
    query = question['Question Detail']
    
    
    res = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        functions=getattr(cg,determine_function_type(question)),
        function_call={"name": determine_function_type(question)},
        messages=[
            {"role": "system", "content": "You are a legal analyst. Use the following context from a municipal ordinance about\
                zoning laws to answer the question."},
            {"role": "user", "content": context},
            {"role": "user", "content": "Question: " + query}
        ]
    )
    
    #Parse response
    response = parse_response(res, question,muni)
    
    return response

#%%Chat GPT model

def chat_gpt(context,question,muni,model):
    max_retries = 5  # Max number of retries
    base_delay = 1.0  # Initial delay in seconds
    max_delay = 16.0  # Max delay in seconds
 
    for retry in range(max_retries):
        try:
            return call_chat_completion(context, question, muni, model)
        
        except Exception as e:  # Catch the specific exception you expect here
            if retry == max_retries - 1:  # If we've tried max_retries times
                print(e)
                raise  # Re-raise the last exception
            # Calculate next sleep time
            sleep_time = (2 ** retry) * base_delay
            noise = random.uniform(0.5, 1.5)  # Add noise
            sleep_time = sleep_time * noise
            sleep_time = min(sleep_time, max_delay)  # Ensure it's below max delay
            time.sleep(sleep_time)
            










