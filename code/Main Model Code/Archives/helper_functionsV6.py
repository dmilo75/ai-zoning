###Old Code

import pickle
import random
random.seed(2)
import json
import statistics
import pandas as pd
import os
import tiktoken
import copy
import io
import claude_functions as cf
import llama_functions as lf
import gemini_functions as gf
import gpt_functions as cg
import context_buildingv4 as cb
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MAX_BATCH_TOKENS = 250000000
    
#%%Config parameters

model_type = os.environ.get('MODEL_TYPE')

#Read in questions
questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))

#Get the list of questions
yes_no_qs = questions[questions['Question Type'] == 'Binary']['ID'].to_list()
numerical = questions[questions['Question Type'] == 'Numerical']['ID'].to_list()
categorical = questions[questions['Question Type'] == 'Categorical']['ID'].to_list()
lot_size = questions[questions['Question Type'] == 'lot_size']['ID'].to_list()
relevant_questions = questions['ID'].to_list()

#Load in subtasks excel
subtasks_df = pd.read_excel(os.path.join(config['raw_data'],'Subtasks.xlsx'))

#%Context creation

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
        if question['ID'] == 28:
            function = "answer_question_lot_size_residential"
        else:
            function = "answer_question_lot_size"
    return function


def make_multi_shot_messages(multi_shot,question):
    
    messages = []
    
    for entry in multi_shot:
        
        prompt = f"Question:\n{question}\n\nContext:{entry['Input']}"
        
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": entry['Output']})
    
    return messages




#Function to get token count
def get_token_count(openai_object):
    encoding = tiktoken.encoding_for_model(openai_object[0]["body"]["model"])

    token_count = 0

    #If the object is a list then we need to combine all messages first
    messages = [x['body']['messages'] for x in openai_object]
    messages = [item for sublist in messages for item in sublist]

    for message in messages:
        token_count += len(encoding.encode(message["content"]))

    return token_count

#Function to calculate costs
def calc_cost(response,model):

    if 'llama' in model:
        return 0

    if 'gemini' in model:
        return 0

    if 'claude' in model:

        #First, get the input tokens
        input_tokens = response.usage.input_tokens

        #Second, get the output tokens
        output_tokens = response.usage.output_tokens

        #Third, get the cost per input token from the model
        input_cost = cf.cost_dict[model]['Input']

        #Fourth, get the cost per output token from the model
        output_cost = cf.cost_dict[model]['Output']

    #Otherwise, use the gpt model
    else:

        #First, get the input tokens
        input_tokens = response.usage.prompt_tokens

        #Second, get the output tokens
        output_tokens = response.usage.completion_tokens

        #Third, get the cost per input token from the model
        input_cost = cg.cost_dict[model]['Input']

        #Fourth, get the cost per output token from the model
        output_cost = cg.cost_dict[model]['Output']

    #Fifth, calculate the cost
    cost = input_cost*input_tokens/1000000 + output_cost*output_tokens/1000000


    return cost



#Get the open-ended answer from Chat GPT
def get_open_ended(entries,model,sys_info,multi_shot = [],n = 1):

    prompt_list = []
    for entry in entries:
        prompt_list.append(entry+':\n'+entries[entry])
    
    prompt = '\n\n'.join(prompt_list)

    #Call the llm
    if 'gemini' in model:
        response = gf.gemini_api_wrapper(gf.gemini_api_call, (prompt, model, sys_info))
    elif 'llama' in model:
        response = lf.llama2_api_wrapper(lf.llama2_api_call, (prompt, model, sys_info))
    elif 'claude' in model:
        response = cf.claude_api_wrapper(cf.claude_api_call, (prompt, model, sys_info, []))
    else:
        response = cg.gpt_api_wrapper(cg.gpt_api_call, (prompt, model, None, None, sys_info,multi_shot,n))

    #Calculate costs
    cost = calc_cost(response,model)

    if 'gemini' in model:
        try:
            response_texts = [response.text]
        #If blocked for content reasons (strangely) then report that it doesn't know
        except Exception as e:
            print(e)
            print(response)
            response_texts = ['ANSWER: I DON\'T KNOW']

    elif 'llama' in model:
        response_texts = [response['generation']]

    elif 'claude' in model:
        #Extract text from response
        choices = response.content

        #Extract the text from each choice
        response_texts = [choice.text for choice in choices]
    else:
        #Extract text from response
        choices = response.choices

        #Extract the text from each choice
        response_texts = [choice.message.content for choice in choices]
    
    return response_texts, cost


#Function to use LLM to parse response
def llm_parse_response(question,expert_answer):


    # Get question string
    question_string = question.loc['Question Detail']

    #Try to split expert_answer for the part that comes after 'ANSWER:'
    try:
        expert_answer = expert_answer.split('ANSWER:')[1]
    except:
        pass

    # Build prompt
    prompt = f"Question: {question_string}\n\nAnswer: {expert_answer}"

    sys_info = "You are a data entry analyst. You are presented with a question and an expert answer. You parse the expert answer to the question."

    # Get the tools
    tools = getattr(cg, determine_function_type(question))
    tool_choice = {"type": "function", "function": {"name": determine_function_type(question)}}

    #Determine model
    if question['Question Type'] != 'Lot Size':
        model = 'gpt-3.5-turbo-0125'

    #Use gpt 4 to parse the lot size question, since its a more complicated question to parse
    else:
        model = 'gpt-4-0125-preview'

    # Call the llm
    response = cg.gpt_api_wrapper(cg.gpt_api_call, (prompt, model, tools, tool_choice, sys_info, [], 1))

    # Get json string
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
            'Answer': 'JSON read error on the below:\n' + json_string,
            'Dont_Know': True,
        }
    return response

#Function to extract textual answer from LLM
def extract_textual_answer(response,question):

    #First, get part of response after 'Answer:'
    try:
        answer = response.lower().split('answer:')[1]
    except:
        return False

    #Second, clean up any asterisk characters
    answer = answer.replace('*','').strip()

    #Check if answer is I Don't Know
    if answer == 'i don\'t know':
        results = {}
        results['Answer'] = None
        results['Dont_Know'] = True

    #Third, check if valid answer for type of question
    if question['Question Type'] == 'Binary':
        if answer not in ['yes','no']:
            return False
        else:
            clean_answer = answer.title()
    elif question['Question Type'] == 'Numerical':
        try:
            int(answer)
            clean_answer = int(answer)
        except:
            return False
    #For the lot size question we need the LLM to parse the response
    elif question['Question Type'] == 'Lot Size':
        return False

    results = {}
    results['Answer'] = clean_answer
    results['Dont_Know'] = False

    return results

#Parse the answer now
def get_parsed_answer(question,expert_answer):

    #First, try to directly extract keyword response
    response = extract_textual_answer(expert_answer,question)

    #Check if response is a boolean of false
    if response == False:

        print('manually extracting')

        #If that fails, then use the LLM to parse the response
        llm_response = llm_parse_response(question,expert_answer)

        return llm_response

    return response

def load_or_fetch_question_details(question):
    """
    Check if the question details are already in a text file.
    If not, fetch the details using a hypothetical function and store them in a text file.
    """
    # Define the directory and file path
    directory = os.path.join(config['raw_data'], 'question_backgrounds')
    file_path = os.path.join(directory, f"{question['ID']}.txt")

    # Load existing data
    with open(file_path, 'r', encoding='utf-8') as file:
        question_details = file.read()
    return question_details


#binary aggregation
def aggregate_binary(parsed_responses):

    # First, drop any answers that are I Don't Know
    valid_responses = [x for x in parsed_responses if x['Dont_Know'] == False]

    # Get the answers
    answers = [x['Answer'] for x in valid_responses]

    #If the majority of responses are I don't know then return I Don't Know
    if sum([x['Dont_Know'] for x in parsed_responses]) > len(parsed_responses)/2:
        response = {'Answer':None,'Dont_Know':True, 'Answers':answers}
        return response

    #Check if theres a tie between yes and no
    if answers.count('Yes') == answers.count('No'):
        response = {'Answer':None,'Dont_Know':True, 'Answers':answers}
        return response
    elif answers.count('Yes') > answers.count('No'):
        response = {'Answer':'Yes','Dont_Know':False, 'Answers':answers}
        return response
    else:
        response = {'Answer':'No','Dont_Know':False, 'Answers':answers}
        return response



#Numerical aggregation
def aggregate_numerical(parsed_responses):

    # Filter out responses where "I Don't Know" is True
    valid_responses = [response for response in parsed_responses if not response['Dont_Know']]

    # If the majority of responses are "I Don't Know", return that indication
    if len(valid_responses) <= len(parsed_responses) / 2:
        return {'Answer': None, 'Dont_Know': True, 'Answers': []}

    # Extract numerical answers
    answers = [int(response['Answer']) for response in valid_responses]

    # Calculate the median directly as the preferred method for handling unique values
    median_answer = statistics.median(answers)
    return {'Answer': median_answer, 'Dont_Know': False, 'Answers': answers}

#Function to aggregate answers
def aggregate_answers(question,expert_answers):

    #If only one answer then we can directly parse it
    if len(expert_answers) == 1:
        response = get_parsed_answer(question,expert_answers[0])
        return response

    #Otherwise, we first need to parse all responses and then aggregate
    parsed_responses = []
    for answer in expert_answers:
        parsed_responses.append(get_parsed_answer(question,answer))

    #Now we need to aggregate the responses depending on the question type
    if question['Question Type'] == 'Binary':
        response = aggregate_binary(parsed_responses)
    elif question['Question Type'] == 'Numerical':
        response = aggregate_numerical(parsed_responses)
    else:
        raise ValueError('Unsupported question type')

    return response

#Wrapper for binary questions
def binary_question(subtasks,context,question,model, double_check):

    #Pull in question background information
    question_background = load_or_fetch_question_details(question)

    #Define question string
    if double_check:
        if not pd.isnull(question['Double Check Question']):
            question_string = question['Double Check Question']
        else:
            question_string = question['Question Rephrase']
    else:
        question_string = question['Question Rephrase']

    sys_info = "You are a municpal zoning ordinance expert. You use the following context which follows 'Context: ' from a municipal ordinance to answer the question which follows 'Question: '. You first review the background information on the question following 'Background Information on Question:' and treat it as additional instructions. You assume that the context includes all of the relevant legal information for the question. You review the context thoroughly for evidence to answer the question. When you do cannot find any relevant information in the context, you realize that the town does not have relevant laws for the question and you reference the question background for how to handle this situation. You think step by step and justify each step with explanations and evidence from the context. At the end of your argument you review what the answer should be and then explictly state your answer in the format of 'ANSWER: ' and then one of 'YES', 'NO', or 'I DON'T KNOW'."

    prompt = {
        #'Instructions':instructions+'\n'+question_conditions,
        'Question':question_string,
        'Background Information on Question':question_background,
        'Context': context
        }

    # Add in subtasks
    for subtask in subtasks:
        prompt[subtask['Answer Description']] = subtask['Answer']




    expert_answers, cost = get_open_ended(prompt,model,sys_info, multi_shot = [], n = 1)

    response = aggregate_answers(question,expert_answers)

    #Add in open ended answer
    response['Explanation'] = '\n\n\n'.join(expert_answers)

    return response, cost

#Wrapper for numerical quesitons
def numerical_question(subtasks,context,question,model,double_check):

    #System information
    sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws to answer the question which follows 'Question: '. You think step by step and justify each step with explanations and evidence from the context. At the end of your argument you explictly state your answer in the format of 'ANSWER: ' followed by an integer or 'I DON'T KNOW'.."
    
    #First, build the prompt
    prompt = {
        'Question': question.loc['Question Detail'],
        'Context': context
        }

    #Add in subtasks
    for subtask in subtasks:
        prompt[subtask['Answer Description']] = subtask['Answer']

    expert_answers, cost = get_open_ended(prompt, model, sys_info, multi_shot=[], n = 1)

    response = aggregate_answers(question, expert_answers)

    # Add in open ended answer
    response['Explanation'] = '\n\n\n'.join(expert_answers)

    return response, cost

#Wrapper for lot size question
def lot_size_question(subtasks,context,question,model, double_check):


    # Pull in question background information
    question_background = load_or_fetch_question_details(question)

    #System information
    sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
            to answer the question which follows 'Question: '. Refer to the question background section for detailed instructions on how to answer the question. You think step by step and justify each step with explanations and evidence from the context. At the end of your answer you say 'ANSWER:' and then reply with a CSV format with a column for 'District Name', 'Minimum Lot Size', 'Unit', and perhaps more depending on the question background. Ensure that you only include one row per district."

    #First, build the prompt
    prompt = {
        'Question': question.loc['Question Detail'],
        'Background Information on Question': question_background,
        'Context': context
        }

    # Add in subtasks
    for subtask in subtasks:
        prompt[subtask['Answer Description']] = subtask['Answer']

    expert_answers, cost = get_open_ended(prompt, model, sys_info, multi_shot=[])

    response = aggregate_answers(question, expert_answers)

    # Add in open ended answer
    response['Explanation'] = '\n\n\n'.join(expert_answers)

    return response, cost

#Wrapper for subtask questions

def subtask_question(subtask,tree):

    #Model information
    model = os.environ['MODEL_TYPE']

    #Build the context
    context = cb.context_builder(subtask,tree,'semantic')

    #System information
    sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
            to answer the question which follows 'Question: '. You justify your answer with explanations and evidence from the context. When you do not know the answer to a question you say that you do not know and explain why."

    #First, build the prompt
    prompt = {
        'Question': subtask.loc['Subtask Instructions'],
        'Context': context
        }

    #First, ask Chat GPT the question in an open-ended format
    open_ended_answer, cost = get_open_ended(prompt,model,sys_info,multi_shot = [], n = 1)

    results = {}
    results['Answer'] = open_ended_answer[0]
    results['Answer Description'] = subtask['Subtask Results Description']
    results['Context'] = context

    print(open_ended_answer[0])

    return results, cost


#Function to get subtasks
def get_subtasks(question, tree):
    # Initialize list of completed subtasks
    completed_subtasks = []

    # Initialize cost of subtasks
    subtask_cost = 0

    # Convert question ID to string for comparison
    question_id_str = str(question['ID'])

    # Loop over each row in the DataFrame
    for i, subtask in subtasks_df.iterrows():

        # Check if 'Subtask Questions' is not empty or null
        if pd.notnull(subtask['Subtask Questions']):
            # Normalize the subtask questions string: trim spaces and ensure it is a string
            subtask_questions_str = str(subtask['Subtask Questions']).strip()

            # Split 'Subtask Questions' by comma and strip spaces to handle any formatting,
            # also handle case where there might not be a comma
            subtask_questions = [x.strip() for x in
                                 subtask_questions_str.split(',')] if ',' in subtask_questions_str else [
                subtask_questions_str]

            # Check if question ID is in the list of subtask questions
            if question_id_str in subtask_questions:
                # Get the subtask answered (assuming subtask_question is a function you've defined)
                response, cost = subtask_question(subtask, tree)

                #Add on cost
                subtask_cost += cost

                # Add to completed subtasks
                completed_subtasks.append(response)

    return completed_subtasks, subtask_cost

#Function to double check answer
def double_check_question(subtasks,response,tree,question,muni,model):

    print('Double Checking Answer')

    #get type of context for double checking from question
    context_type = question['Double Check Context']

    #Create new context
    new_context = cb.context_builder(question,tree,context_type)

    #Run the model
    response, cost = run_model(subtasks,new_context,question,model, double_check = True)

    return response, cost

#Function to determine whether to double check question
def double_check(subtasks,response,tree,question,muni,model):

    #Get the prior
    prior = question['Prior']

    #If no prior for the question then just return response
    if (prior == "") or pd.isnull(prior):
        return response, 0

    #Check if the answer is I Don't Know or not the pior
    if (response['Dont_Know'] == True) or (response['Answer'] != prior):
        response, cost = double_check_question(subtasks,response,tree,question,muni,model)
        return response, cost

    else:
        return response, 0

#Function to pick which question type to run
def run_model(subtasks,context,question,model, double_check = False):

    # Get question type and send to relevant wrapper
    question_type = question['Question Type']
    print(question_type)

    # Now run the relevant question type
    if question_type == 'Numerical':
        response, cost = numerical_question(subtasks, context, question, model, double_check)
    elif question_type == 'Binary':
        response, cost = binary_question(subtasks, context, question, model, double_check)
    elif question_type == 'Lot Size':
        response, cost = lot_size_question(subtasks, context, question, model, double_check)
    else:
        raise ValueError('Invalid question type')

    response['Context'] = context

    return response, cost


#Run Chat GPT on the context/question
def chat_gpt(tree, question, muni, model):
    # First, build the context
    context = cb.context_builder(question, tree, 'semantic')

    # Check if any subtasks need to be answered first
    subtasks, subtask_cost = get_subtasks(question, tree)

    # Run the relevant model
    response, cost = run_model(subtasks, context, question, model)

    # Deep copy the response dictionary from run_model()
    first_attempt = copy.deepcopy(response)

    # Now, send to the double-checker
    response, double_check_cost = double_check(subtasks, response, tree, question, muni, model)

    # Create the final response dictionary
    final_response = {
        'Answer': response['Answer'],
        'Dont_Know': response['Dont_Know'],
        'Explanation': response['Explanation'],
        'Context': response['Context'],
        'Muni': muni,
        'Question': question['ID'],
        'Cost': cost + subtask_cost + double_check_cost,
        'Subtasks': subtasks
    }

    # Add the 'First_Attempt' key if double checking was performed
    if double_check_cost > 0:
        final_response['First_Attempt'] = first_attempt

    return final_response

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
def filter_pioneer(muni_list,type):

    #Read in training data
    if type == 'training':
        training_file = os.path.join(config['raw_data'],"training.pickle")
        with open(training_file,"rb") as fIn:
            training = pickle.load(fIn)

    #Read in testing data
    elif type == 'testing':
        testing_file = os.path.join(config['raw_data'],"testing.pickle")
        with open(testing_file,"rb") as fIn:
            training = pickle.load(fIn)

    elif type == 'wharton':
        wharton_file = os.path.join(config['raw_data'],"wharton_training.pkl")
        training = pd.read_pickle(wharton_file)

    #Otherwise, just return the muni_list
    else:
        return muni_list

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


def get_munis(type):
    sample_data = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))
    
    # Create list of dictionaries
    muni_list = []
    for index, row in sample_data.iterrows():
        entry = {'State': row['State'],
                 'Muni': f"{row['UNIT_NAME']}#{row['CENSUS_ID_PID6']}#{row['State']}"}
        muni_list.append(entry)
    
    #Whether to just use training or testing sample
    muni_list = filter_pioneer(muni_list,type)
        
    #Whether to use a small sample 
    if os.environ['SAMPLE_SIZE'] != 'all':
        #print what seed random has
        #random.seed(5)
        #Take a smaller random sample for testing purposes
        muni_list = random.sample(muni_list,int(os.environ['SAMPLE_SIZE']))

    #Filter here for small sample for testing purposes
    #filepath = "/vast/dm4766/context_building_refine/embed/Full Summaries/"
    #muni_list = filter_muni_list_by_files(muni_list, filepath)
    
    #Split across parallelization 
    munis_for_task = split_munis_across_nodes(muni_list)
    
    return munis_for_task

