import yaml
import os
import json
import logging
import boto3
import queue
import threading
import time
import random
from botocore.exceptions import ClientError

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cost_dict = {
    "meta.llama2-70b-chat-v1": {'Input': 0.0015, 'Output': 0.0030},
}

def generate_text(model_id, body):
    """
    Generate text using Meta Llama 2 Chat on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (JSON): The text that the model generated, token information, and the
        reason the model stopped generating text.
    """

    logger.info("Generating text with Meta Llama 2 Chat model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime')

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    return response_body


def llama2_api_call(output_queue, prompt, model_id, system_prompt, max_gen_len=2048, temperature=0, top_p=0.9):


    body = json.dumps({
        "prompt": f"[INST]{system_prompt}[\INST]\n\n###User: {prompt}\n\n###Detailed answer to question (Remember to follow the instructions and think step by step): ",
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p
    })



    try:
        res = generate_text(model_id, body)
        output_queue.put(res)
    except ClientError as e:
        output_queue.put(e)


def llama2_api_wrapper(thread_func, args, timeout=360):
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
            if isinstance(res, ClientError):
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