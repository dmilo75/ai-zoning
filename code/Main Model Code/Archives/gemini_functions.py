import yaml
import os
import time
import queue
import threading
import time
import random
import google.generativeai as genai

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure the API key
genai.configure(api_key=config['gemini_key'])

# Set up the generation configuration
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Set up the safety settings

safety_settings = [
{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
},
{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
},
{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
},
{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
},
]


def create_gemini_model(model_name):
    return genai.GenerativeModel(model_name=model_name,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)


def gemini_api_call(output_queue, prompt, model_name, system_prompt):
    try:
        model = create_gemini_model(model_name)
        convo = model.start_chat(history=[])

        # Prepend the system prompt to the user's prompt
        combined_prompt = f"###Instructions:\n{system_prompt}\n\n###User:\n{prompt}\n\n###Detailed answer to question using the context and the question background as well as following the instructions:"

        res = convo.send_message(combined_prompt)
        output_queue.put(res)
    except Exception as e:
        output_queue.put(e)


def gemini_api_wrapper(thread_func, args, timeout=360):
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
            if isinstance(res, Exception):
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