
import pickle
import pandas as pd
import os
import tiktoken
import random
import gpt_functions as gpt
from question_muni_pair import QuestionMuniPair
import yaml
from dotenv import load_dotenv
load_dotenv()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


MAX_BATCH_TOKENS = config['max_batch_tokens']
MAX_REQUESTS = config['max_batch_requests']

##Helper functions for QA_Code

def load_pair_queue(folder_path):
    pair_queue_file = os.path.join(folder_path, f'pair_queue{slurm_name()}.pkl')
    if os.path.exists(pair_queue_file):
        with open(pair_queue_file, 'rb') as file:
            return pickle.load(file)
    else:

        # The list of munis to process
        munis_for_task = get_munis()

        questions = pd.read_excel(os.path.join(config['raw_data'], 'Questions.xlsx'))

        #Check whether testing mode
        if config['testing_mode'] == 'True':
            print("Testing Mode so only running one question and one muni")
            munis_for_task = munis_for_task[:1]
            questions = questions[:1]

        return [QuestionMuniPair(row, muni, config['model_type'])
                      for muni in munis_for_task
                      for index, row in questions.iterrows()]

def get_num_nodes():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        # The number of nodes we are parallelizing off of
        task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))  # Id for first job array id
        task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0))  # Id for last job array id
        num_nodes = task_max - task_min + 1  # Total number of job array ids
        return num_nodes
    else:
        return 1

def load_results(folder_path):
    results_file = os.path.join(folder_path, f"output{slurm_name()}.pkl")

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def load_errors(folder_path):
    errors_file = os.path.join(folder_path, f"errors{slurm_name()}.pkl")

    if os.path.exists(errors_file):
        with open(errors_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []


def save_results(results,folder_path):

    # Save the updated results after each batch
    results_file = os.path.join(folder_path, f"output{slurm_name()}.pkl")

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

def save_errors(errors,folder_path):

    # Save the updated results after each batch
    results_file = os.path.join(folder_path, f"errors{slurm_name()}.pkl")

    with open(results_file, 'wb') as f:
        pickle.dump(errors, f)


def save_pair_queue(pair_queue,folder_path):

    status_check_file = os.path.join(folder_path, f'status_check_needed{slurm_name()}.txt')
    pair_queue_file = os.path.join(folder_path, f'pair_queue{slurm_name()}.pkl')

    if len(pair_queue) > 0:
        print("Still remaining pairs")
        with open(pair_queue_file, 'wb') as file:
            pickle.dump(pair_queue, file)
        with open(status_check_file, 'w') as file:
            file.write("True")
    else:
        print("No remaining pairs")
        # If the pair_queue is empty, delete the pair_queue pickle file
        if os.path.exists(pair_queue_file):
            os.remove(pair_queue_file)
        with open(status_check_file, 'w') as file:
            file.write("False")


def check_status(folder_path):
    status_check_file = os.path.join(folder_path, f'status_check_needed{slurm_name()}.txt')

    # Check if the status file exists
    if not os.path.exists(status_check_file):
        return True

    # If the file exists, read the status
    with open(status_check_file, 'r') as file:
        status = file.read().strip()

    # Return True if the status is "True", otherwise False
    return status == "True"

def flatten_batches(batch):
    #Flatten the list of lists of batches needed
    return [item for sublist in batch for item in sublist]


def build_model(pair_queue, results, errors, folder_path, fake_batch = False):

    batch = []
    batch_tokens = 0

    for pair in reversed(pair_queue): #Iterate in reverse to not corrupt indices when removing

        #Try to process the pair
        try:
            print(pair.state)
            print(pair.action)
            pair.process()
            print(pair.state)
            print(pair.action)

            if pair.state == 'completed':
                results.append(pair.get_final_response())
                pair_queue.remove(pair)
            else:
                if batch_tokens + get_token_count(pair.openai_object) > MAX_BATCH_TOKENS/get_num_nodes():
                    print('Hit Token Limit')
                    break

                # Check for request limit
                if len(flatten_batches(batch)) + len(flatten_batches([pair.openai_object])) > MAX_REQUESTS/get_num_nodes():
                    print('Hit Request Limit')
                    break

                batch.append(pair.openai_object)
                batch_tokens += get_token_count(pair.openai_object)

        #If any error, like failed to locate the muni's embeddings, then add to errors with the pair and the specific error to rerun later
        except Exception as e:
            print(e)
            dic = {
                'pair': pair,
                'error': e
            }
            errors.append(dic)
            pair_queue.remove(pair)

    if len(batch) > 0:

        flat_batches = flatten_batches(batch)

        if fake_batch:
            batch_results = generate_real_response(flat_batches)

            # Save batch results to file
            batch_results_file = os.path.join(folder_path, f"batch_results{slurm_name()}.pkl")
            with open(batch_results_file, 'wb') as f:
                pickle.dump(batch_results, f)

            batch_id = 1

        else:

            batch_data = {
                "input_file_id": gpt.upload_jsonl(flat_batches),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }

            batch_id = gpt.create_batch(batch_data)

    else:
        batch_id = None

    return batch_id, results, errors, pair_queue

def retrieve_batch(batch_id, folder_path, fake_batch = False):

    if fake_batch:
        batch_results_file = os.path.join(folder_path, f"batch_results{slurm_name()}.pkl")
        if os.path.exists(batch_results_file):
            with open(batch_results_file, 'rb') as f:
                batch_results = pickle.load(f)
        else:
            batch_results = []

    else:
        batch_results = gpt.retrieve_batch_results(batch_id)

    return batch_results

def generate_real_response(batch):
    responses = []

    for request in batch:
        # Extract the necessary parameters from the request
        model = request["body"]["model"]
        messages = request["body"]["messages"]
        tools = request["body"].get("tools")

        # Make the API call to create a chat completion
        completion = gpt.generate_completion(model,messages,tools,request)

        # Prepare a list to store all choices from the completion
        all_choices = []
        for choice in completion.choices:
            choice_obj = {
                "index": choice.index,
                "message": choice.message,
                "logprobs": choice.logprobs,
                "finish_reason": choice.finish_reason
            }
            all_choices.append(choice_obj)

        # Create the response object including all choices
        response_obj = {
            "id": "batch_req_" + completion.id,
            "custom_id": request["custom_id"],
            "error": None,
            "status_code": 200,
            "response": {
                "body": {
                    "id": completion.id,
                    "object": "chat.completion",
                    "created": completion.created,
                    "model": model,
                    "system_fingerprint": completion.system_fingerprint,
                    "choices": all_choices,  # Include all choices here
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens
                    }
                }
            }
        }
        responses.append(response_obj)

    return responses


def update_pairs(pair_queue, batch_results):
    if len(batch_results) == 0:
        return pair_queue

    # Update the pair_queue with the results from the batch
    for pair in pair_queue:
        for openai_obj in pair.openai_object:  # Now iterating over a list of openai_objects
            result = next((r for r in batch_results if r['custom_id'] == openai_obj['custom_id']), None)
            if result:
                pair.response.append(result)

    return pair_queue

def slurm_name():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        return f"_{int(os.environ['SLURM_ARRAY_TASK_ID'])}"
    else:
        return "_1"

def load_batch_id(folder_path):

    batch_id_file = os.path.join(folder_path, f"batch_id{slurm_name()}.txt")

    if os.path.exists(batch_id_file):
        with open(batch_id_file, 'r') as file:
            return file.read().strip()
    else:
        return None

def save_batch_id(batch_id,folder_path):

    batch_id_file = os.path.join(folder_path, f"batch_id{slurm_name()}.txt")

    with open(batch_id_file, 'w') as file:
        file.write(str(batch_id))

##Helper functions for question_muni_pair



def get_token_count(openai_object):
    encoding = tiktoken.encoding_for_model(openai_object[0]["body"]["model"])

    token_count = 0

    #If the object is a list then we need to combine all messages first
    messages = [x['body']['messages'] for x in openai_object]
    messages = [item for sublist in messages for item in sublist]

    for message in messages:
        token_count += len(encoding.encode(message["content"]))

    return token_count

##Getting munis
def get_munis():
    sample_data = pd.read_excel(os.path.join(config['raw_data'], 'Sample Data.xlsx'))

    # Create list of dictionaries
    muni_list = []
    for index, row in sample_data.iterrows():
        entry = {'State': row['State'],
                 'Muni': f"{row['UNIT_NAME']}#{row['CENSUS_ID_PID6']}#{row['State']}"}
        muni_list.append(entry)

    #Print number of munis
    print(f"Number of munis: {len(muni_list)}")

    #Find all munis we have embeddings for
    embedded_munis = get_embeddings()

    #Now filter down for munis we have embeddings for
    muni_list = [x for x in muni_list if x['Muni'] in embedded_munis]

    #Print number of munis
    print(f"Number of munis with embeddings: {len(muni_list)}")

    # Whether to use a small sample
    if config['sample_size'] != 'all':
        # print what seed random has
        # random.seed(5)
        # Take a smaller random sample for testing purposes
        muni_list = random.sample(muni_list, int(config['sample_size'] ))

    # Split across parallelization
    munis_for_task = split_munis_across_nodes(muni_list)

    return munis_for_task

#Function get all available embeddings
def get_embeddings():
    # Now get the list of all embeddings
    embed_path = config['embeddings']

    # Get list of all filenames
    filenames = os.listdir(embed_path)

    # Remove the .pkl suffix
    filenames = [x[:-4] for x in filenames]

    return filenames

# If running the code on many nodes then split the list of munis across the nodes
def split_munis_across_nodes(muni_list):
    # If running on multiple nodes
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        # The number of nodes we are parallelizing off of
        task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))  # Id for first job array id
        task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0))  # Id for last job array id
        num_nodes = task_max - task_min + 1  # Total number of job array ids

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

