import yaml
import os
import time
import anthropic
import queue
import threading
import time
import random

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Python functions used to call claude

cost_dict = {
"claude-3-opus-20240229": {'Input':15,'Output':75},
"claude-3-sonnet-20240229": {'Input':3,'Output':15},
}

client = anthropic.Anthropic(
    api_key=config['claude_key'],
)

def claude_api_call(output_queue, prompt, model, system_prompt, messages):
    messages_copy = messages.copy()
    messages_copy.append({"role": "user", "content": prompt})

    try:
        res = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            stream=False,
            system=system_prompt,
            messages=messages_copy,
        )
        output_queue.put(res)
    except anthropic.APIError as e:
        output_queue.put(e)

def claude_api_wrapper(thread_func, args, timeout=600):
    for delay_secs in (4 * 2 ** x for x in range(8)):
        randomness_collision_avoidance = random.randint(0, 5000) / 1000.0
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
            if isinstance(res, anthropic.APIError):
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