###gpt_functions.py
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:48:13 2023

@author: Dan's Laptop
"""

from openai import OpenAI
import openai
import queue
import threading
import yaml
import os
import random
import time
import json
import tempfile
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Python functions used to call chat gpt

# Openai
client = OpenAI(api_key=config['openai_key'])

#Random seed for chat gpt
seed = 42

#Cost dictionary
cost_dict = {
'gpt-4-0125-preview': {'Input':10,'Output':30},
'gpt-3.5-turbo-0125': {'Input':0.5,'Output':1.5},
'gpt-3.5-turbo': {'Input':0.5,'Output':1.5},
'gpt-4-turbo-2024-04-09': {'Input':10,'Output':30},
'gpt-4o': {'Input':5,'Output':15},
}

##

def delete_old_files():

    # Get the current time
    now = datetime.now()
    # Define the threshold date (1 week ago)
    threshold = now - timedelta(days=7)

    # Retrieve the list of files
    files = client.files.list()

    # Iterate through the files and delete those older than the threshold date
    for file in files:
        try:
            # Convert the file creation time from Unix timestamp to datetime
            file_creation_time = datetime.fromtimestamp(file.created_at)
            # Check if the file is older than the threshold
            if file_creation_time < threshold:
                # Delete the file
                client.files.delete(file.id)
                print(f"Deleted file: {file.filename} created at {file_creation_time}")
        except:
            pass





##

def create_batch(batch_data):
    # Creating a batch using the client instance
    response = client.batches.create(
        input_file_id=batch_data["input_file_id"],
        endpoint=batch_data["endpoint"],
        completion_window=batch_data["completion_window"]
    )
    # Return the batch ID using attribute access
    return response.id

def generate_completion(model, messages, tools, request, response_format = {"type": "text"}):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=request['body']['temperature'],
        seed=request['body']['seed'],
        n=request['body'].get('n', 1),
        response_format = response_format,
    )
    return completion

def retrieve_batch_results(batch_id):

    # Check if the batch ID is None
    if batch_id is None:
        #Squeeze in the file management here
        delete_old_files()
        return []

    # Retrieve the batch using the client instance
    response = client.batches.retrieve(batch_id)

    # Print out the response status
    print(f"Batch status: {response.status}")

    if response.status == "completed":
        output_file_id = response.output_file_id
        # Download the output file content as binary data
        output_file_content = client.files.content(output_file_id)
        # Read the binary content and decode it as UTF-8
        output_data = output_file_content.read().decode('utf-8').splitlines()
        return [json.loads(line) for line in output_data]
    elif response.status == "failed":
        # Print the error messages and raise an exception
        print("Error:", response.errors)
        raise Exception(f"Batch processing failed: {response.errors}")
    else:
        return []


def upload_jsonl(data):
    # Create a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode='w+', encoding='utf-8')
    try:
        # Write the JSONL data to the temporary file
        for item in data:
            json.dump(item, temp)
            temp.write('\n')
        # Close the file to ensure all data is written and file is not locked
        temp.close()

        # Upload the file to OpenAI
        with open(temp.name, 'rb') as file_obj:
            response = client.files.create(
                file=file_obj,
                purpose="batch"  # Ensure the purpose is set correctly for your use case
            )
    finally:
        # Ensure the temporary file is removed after uploading
        os.unlink(temp.name)  # Remove the file after it's closed

    # Return the file ID
    return response.id


#Function to get embeddings
def get_embedding(text):

    #Get the embedding
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # Specify the model you want to use
    )
    # Extract the embedding and return
    return response.data[0].embedding


# Ask Chat GPT for an open-ended answer
def gpt_api_call(output_queue, prompt, model, tools, tool_choice, sys_info, messages, n):
    # Create a fresh copy of messages for this call to ensure it starts with the intended state
    messages_copy = messages.copy()

    # The messages we will send to GPT
    messages_copy.append({"role": "user", "content": prompt})

    # If we want to provide system instructions, then send that first
    if sys_info != "":
        messages_copy.insert(0, {"role": "system", "content": sys_info})

    #Set temperature based on n, since with n > 1 we want more randomness
    if n > 1:
        temperature = 1
    else:
        temperature = 0.0

    try:
        res = client.chat.completions.create(
            model=model,  # Which model from OpenAI to use
            temperature=temperature,  # Randomness in the response
            tools=tools,  # Tools that we give OpenAI
            tool_choice=tool_choice,  # Require the LLM to use a specific tool
            seed=seed,  # The random seed, not entirely deterministic but close
            messages=messages_copy,
            n = n,
        )

        #Print all inputs to the chat completion
        print(model)
        print(temperature)
        print(tools)
        print(tool_choice)
        print(seed)
        for message in messages_copy:
            print(message['role'])
            print(message['content'])
        print(n)

        output_queue.put(res)
    except openai.OpenAIError as e:
        output_queue.put(e)



def gpt_api_wrapper(thread_func, args, timeout=360):
    '''
    Wrapper for calling OpenAI (Chat GPT). We allow for periods where the server times out or is slow
    '''
    for delay_secs in (2 ** x for x in range(6)):
        randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
        sleep_dur = delay_secs + randomness_collision_avoidance
        time.sleep(sleep_dur)

        output_queue = queue.Queue()
        thread = threading.Thread(target=thread_func, args=(output_queue, *args))
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
                return res
        except queue.Empty:
            print("The API call did not return a response within the timeout period. Retrying...")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying...")
            continue

    raise TimeoutError("The API call did not return a response after multiple attempts.")


#%%Functions
#https://json-schema.org/understanding-json-schema/

#Functions for OpenAI calls which we can use to specify response format
answer_question_yes_no = [
            {
                "type": "function",
                "function": {
                    "name": "answer_question_yes_no",
                    "description": "Send answer to question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Dont_Know": {
                                "type": "boolean",
                                "description": "Whether you don't know the answer.",
                                    },
                            "Answer": {
                                "type": "string",
                                "enum": ["Yes","No"],
                                "description": "The answer to the question.",
                            },
                        },
                        "required": ["Dont_Know","Answer"],
                    },
                    
                }
            }
            ]

keyword_search = [
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Receive an array of keywords for searching",
            "parameters": {
                "type": "object",
                "properties": {
                    "Keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of keywords for the search."
                    }
                },
                "required": ["Keywords"]
            }
        }
    }
]



answer_question_numerical = [
            {
                "type": "function",
                "function": {
                    "name": "answer_question_numerical",
                    "description": "Send answer to question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Dont_Know": {
                                "type": "boolean",
                                "description": "Whether you don't know the answer.",
                                    },
                            "Answer": {
                                "type": "integer",
                                "description": "The answer to the question.",
                            },
                        },
                        "required": ["Dont_Know","Answer"],
                    },
                    
                }
            }
       ]

answer_question_lot_size = [
    {
     "type": "function",
     "function": {
        "name": "answer_question_lot_size",
        "description": "Send answer to question",
        "parameters": {
            "type": "object",
            "properties": {
                
                "Dont_Know": {
                    "type": "boolean",
                    "description": "Whether you don't know the answer.",
                        },
                "Answer": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "district_name": {
                                "type": "string",
                                "description": "Name of the district."
                            },
                            "min_lot_size": {
                                "type": "integer",
                                "description": "Minimum lot size for the district"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["Square Feet","Acres"],
                                "description": "The unit of measurement for the lot size."
                            },
                        },
                        "required": ["district_name", "min_lot_size", "unit"]
                    },
                    "description": "A list of districts with their corresponding lot size. Each district must be a separate object with the keys 'district_name', 'min_lot_size', and 'unit'.",
                },
                
            },
            "required": ["Dont_Know","Answer"],
        },
    }
}
]

answer_question_lot_size_residential = [
    {
        "type": "function",
        "function": {
            "name": "answer_question_lot_size_residential",
            "description": "Send answer to question",
            "parameters": {
                "type": "object",
                "properties": {

                    "Dont_Know": {
                        "type": "boolean",
                        "description": "Whether you don't know the answer.",
                    },
                    "Answer": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "district_name": {
                                    "type": "string",
                                    "description": "Name of the district."
                                },
                                "min_lot_size": {
                                    "type": "integer",
                                    "description": "Minimum lot size for the district"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["Square Feet", "Acres"],
                                    "description": "The unit of measurement for the lot size."
                                },
                                "estate": {
                                    "type": "boolean",
                                    "description": "Whether or not the district is an estate district."
                                },
                            },
                            "required": ["district_name", "min_lot_size", "unit", "estate"]
                        },
                        "description": "A list of districts with their corresponding lot size. Each district must be a separate object with the keys 'district_name', 'min_lot_size', 'unit', and 'estate'.",
                    },

                },
                "required": ["Dont_Know", "Answer"],
            },
        }
    }
]

#Function to determine whether text is relevant to a question
determine_relevant = [
    {
        "type": "function",
        "function": {
            "name": "whether_relevant",
            "description": "Determine whether the presented section is relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Argument": {
                        "type": "string",
                        "description": "Detailed and lengthy argument explaining whether the presented section is relevant to the question.",
                    },
                    "Relevant": {
                        "type": "boolean",
                        "description": "Whether the presented section of text is relevant to the question",
                    },
                },
                "required": ["Argument","Relevant"],
            },

        }
    }
]


parse_explore_section = [
    {
        "type": "function",
        "function": {
            "name": "parse_explore_section",
            "description": "Determine whether the user said to explore the section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explore": {
                        "type": "boolean",
                        "description": "Whether the user said to explore the section."
                    },
                },
                "required": ["explore"]
            }
        }
    }
]

