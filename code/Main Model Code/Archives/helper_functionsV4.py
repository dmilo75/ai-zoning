# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 08:51:57 2023

@author: Dan's Laptop
"""
import time
import pickle
import random
random.seed(2)
import json
import pandas as pd
import os

import gpt_functions as cg
import context_building as cb
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
#%%Config parameters

model_type = os.environ.get('MODEL_TYPE')



#Whether to just use trianing data
just_training = True

#Random seed for chat gpt
seed = 42

questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))

yes_no_qs = questions[questions['Question Type'] == 'Binary']['ID'].to_list()
numerical = questions[questions['Question Type'] == 'Numerical']['ID'].to_list()
categorical = questions[questions['Question Type'] == 'Categorical']['ID'].to_list()
lot_size = questions[questions['Question Type'] == 'lot_size']['ID'].to_list()

relevant_questions = questions['ID'].to_list()

#%Context creation
def create_context(question,tree,context_type):

    if context_type == 'embeddings':
        texts =  cb.embeddings_context_builder(question, tree,filter = False)
    elif context_type == 'embeddings_and_filter':
        texts =  cb.embeddings_context_builder(question, tree,filter = True)
    elif context_type == 'toc_navigator':
        texts =  cb.toc_nav_context_builder(question, tree)
    else:
        raise ValueError('Invalid context type')

    context = '\n\n'.join([x['Leaf_Text'] for x in texts])

    return context

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




def make_multi_shot_messages(multi_shot,question):
    
    messages = []
    
    for entry in multi_shot:
        
        prompt = f"Question:\n{question}\n\nContext:{entry['Input']}"
        
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": entry['Output']})
    
    return messages

#Get the open-ended answer from Chat GPT
def get_open_ended(entries,model,sys_info,multi_shot = []):

    prompt_list = []
    for entry in entries:
        prompt_list.append(entry+':\n'+entries[entry])
    
    prompt = '\n\n'.join(prompt_list)
    
    #Call the llm
    response = cg.gpt_api_wrapper(cg.gpt_api_call, (prompt, model, None, None, sys_info,multi_shot))
    
    #Extract text from response
    response_text = response.choices[0].message.content
    
    return response_text

#Parse the answer now
def get_parsed_answer(prompt,question):
    
    sys_info = "You are a data entry analyst. You are presented with a question and an expert answer. You parse the expert answer to the question."

    #Get the tools
    tools = getattr(cg,determine_function_type(question))
    tool_choice = {"type": "function", "function": {"name": determine_function_type(question)}}
    
    #Call the llm
    response = cg.gpt_api_wrapper(cg.gpt_api_call,(prompt,'gpt-3.5-turbo-1106',tools,tool_choice,sys_info,[]))
    
    #Get json string
    json_string = response.choices[0].message.tool_calls[0].function.arguments
    
    try:
        response = json.loads(json_string)
        
        # Check if expected keys are in the response and handle accordingly
        if 'Answer' not in response:
            response['Answer'] = 'JSON read error on the below:\n' + json_string
        if 'Dont_Know' not in response:
            response['Dont_Know'] = True
        
    except:
        response = {
            'Answer':'JSON read error on the below:\n'+json_string,
            'Dont_Know':True,
            }
    
    return response

def load_or_fetch_question_details(question):
    """
    Check if the question details are already in a text file.
    If not, fetch the details using a hypothetical function and store them in a text file.
    """
    # Define the directory and file path
    directory = os.path.join(config['raw_data'], 'question_details')
    file_path = os.path.join(directory, f"{question['ID']}.txt")

    # Load existing data
    with open(file_path, 'r', encoding='utf-8') as file:
        question_details = file.read()
    return question_details



#Wrapper for binary questions
def binary_question(context,question,model):
    

    #Have research assistant expand on the question
    #question_background = load_or_fetch_question_details(question)

    #sys_info = "You are a municpal zoning ordinance expert. You use the following context which follows 'Context: ' from a municipal ordinance\
            #to answer the question which follows 'Question: '. Background information on the question is provided too. You reply with step by step thoughts that answer the question and you justify each step with explanations and evidence. At the end of your argument you explictly state your answer or say that you do not know the answer. The question requires a binary response of 'Yes' or 'No'."

    sys_info = "You are a municpal zoning ordinance expert. You use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
                to answer the question which follows 'Question: '. You think step by step and justify each step with explanations and evidence. At the end of your response you explictly state your answer or say that you do not know the answer."

    prompt = {
        #'Instructions':instructions+'\n'+question_conditions,
        'Question':question['Question Detail'],
        #'Background Information on Question':question_background,
        'Context':context,
        }

    expert_answer = get_open_ended(prompt,model,sys_info, multi_shot = [])
    
    #Get question string
    question_string = question.loc['Question Detail']
        
    #Build prompt
    prompt = f"Question: {question_string}\n\nAnswer: {expert_answer}"

    #Extract final answer
    response = get_parsed_answer(prompt,question)
    
    #Add in open ended answer
    response['Explanation'] = expert_answer

    return response

#Wrapper for numerical quesitons
def numerical_question(context,question,model):

    #System information
    sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
        to answer the question which follows 'Question: '. You think step by step and justify each step with explanations and evidence. When you do not know the answer to a question you say that you do not know and explain why."
    
    #First, build the prompt
    prompt = {
        'Question': question.loc['Question Detail'],
        'Context': context
        }
    
    #First, ask Chat GPT the question in an open-ended format
    open_ended_answer = get_open_ended(prompt,model,sys_info,multi_shot = [])
    
    #Get question string
    question_string = question.loc['Question Detail']
        
    #Build prompt
    prompt = f"Question: {question_string}\n\nAnswer: {open_ended_answer}"

    #Extract final answer
    response = get_parsed_answer(prompt,question)
    
    #Add in open ended answer
    response['Explanation'] = open_ended_answer
    
    return response 

#Wrapper for lot size question
def lot_size_question(context,question,model):

    #System information
    sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
            to answer the question which follows 'Question: '. You think step by step and justify each step with explanations and evidence. When you do not know the answer to a question you say that you do not know and explain why."

    #First, build the prompt
    prompt = {
        'Question': question.loc['Question Detail'],
        'Context': context
        }
    
    #First, ask Chat GPT the question in an open-ended format
    open_ended_answer = get_open_ended(prompt,model,sys_info,multi_shot = [])
    
    #Get question string
    question_string = question.loc['Question Detail']
        
    #Build prompt
    prompt = f"Question: {question_string}\n\nAnswer: {open_ended_answer}"

    #Extract final answer
    response = get_parsed_answer(prompt,question)
    
    #Add in open ended answer
    response['Explanation'] = open_ended_answer
    
    return response



#Run Chat GPT on the context/question
def chat_gpt(context,question,muni,model):
    
    #Get question type and send to relevant wrapper
    question_type = question['Question Type']
    print(question_type)
    if question_type == 'Numerical':
        response = numerical_question(context,question,model)
    elif question_type == 'Binary':
        response = binary_question(context,question,model)
    elif question_type == 'Lot Size':
        response = lot_size_question(context,question,model)
    else:
        raise ValueError('Invalid question type')
    
    #Now add other relevant info to return
    response['Muni'] = muni
    response['Question'] = question['ID']
    response['Context'] = context

    return response

#%%Setup functions

#If running the code on many nodes then split the list of munis across the nodes
def split_munis_across_nodes(muni_list):
    
    #If running on multiple nodes
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        #The number of nodes we are parallelizing off of
        task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0)) #Id for first job array id
        task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0)) #Id for last job array id
        num_nodes = task_max - task_min + 1 #Total number of job array ids
        
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_munis = len(muni_list)
        
        # Calculate municipalities per node
        munis_per_task = num_munis // num_nodes
        extra_munis = num_munis % num_nodes
        
        # If task_id is less than extra_munis, it takes one more municipality
        if task_id < extra_munis:
            start_idx = task_id * (munis_per_task + 1)
            end_idx = start_idx + munis_per_task + 1
        else:
            start_idx = task_id * munis_per_task + extra_munis
            end_idx = start_idx + munis_per_task
        munis_for_task = muni_list[start_idx:end_idx]
        
        return munis_for_task
     
    return muni_list

#Filter muni_list for training munis only
def filter_for_training(muni_list):
    
    training_file = os.path.join(config['raw_data'],"training.pickle")
    with open(training_file,"rb") as fIn:
        training = pickle.load(fIn)
        
    training = training.to_list()
    print("Pre filter")
    print(len(training))
    
    total_training = [entry for entry in muni_list if entry['Muni'] in training]
    
    print("Length of training")
    print(len(total_training))

    return total_training


def filter_muni_list_by_files(muni_list, filepath):
    # List all .pkl files in the specified directory
    files = [f for f in os.listdir(filepath) if f.endswith('.pkl')]

    # Extract the base names without the .pkl extension
    file_basenames = set(os.path.splitext(f)[0] for f in files)

    # Filter muni_list where 'Muni' matches any file basename
    filtered_list = [entry for entry in muni_list if entry['Muni'] in file_basenames]

    return filtered_list


def get_munis():
    sample_data = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))
    
    # Create list of dictionaries
    muni_list = []
    for index, row in sample_data.iterrows():
        entry = {'State': row['State'],
                 'Muni': f"{row['UNIT_NAME']}#{row['CENSUS_ID_PID6']}#{row['State']}"}
        muni_list.append(entry)
    
    #Whether to just use training sample
    if just_training:
        muni_list = filter_for_training(muni_list)
        
    #Whether to use a small sample 
    if os.environ['SAMPLE_SIZE'] != 'all':
        #Take a smaller random sample for testing purposes
        muni_list = random.sample(muni_list,int(os.environ['SAMPLE_SIZE']))

    #Filter here for small sample for testing purposes
    #filepath = "/vast/dm4766/context_building_refine/embed/Full Summaries/"
    #muni_list = filter_muni_list_by_files(muni_list, filepath)
    
    #Split across parallelization 
    munis_for_task = split_munis_across_nodes(muni_list)
    
    return munis_for_task

