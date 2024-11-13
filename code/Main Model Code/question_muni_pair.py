###New Code
import yaml
import context_building as cb
import gpt_functions as cg
import pandas as pd
import json
import statistics
import os
from dotenv import load_dotenv
load_dotenv()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = 42

#Load in subtasks excel
subtasks_df = pd.read_excel(os.path.join(config['raw_data'],'Subtasks.xlsx'))


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

def get_subtasks(question):
    question_id_str = str(question['ID'])
    for i, subtask in subtasks_df.iterrows():
        if pd.notnull(subtask['Subtask Questions']):
            subtask_questions_str = str(subtask['Subtask Questions']).strip()
            subtask_questions = [x.strip() for x in subtask_questions_str.split(',')] if ',' in subtask_questions_str else [subtask_questions_str]
            if question_id_str in subtask_questions:
                # Instead of calling subtask_question, return the subtask details needed for further processing
                return subtask
    return None

def determine_function_type(question):
    if question['Question Type'] == 'Binary':
        function = "answer_question_yes_no"
    elif question['Question Type'] == 'Numerical':
        function = "answer_question_numerical"
    elif question['Question Type'] == 'Lot Size':
        function = "answer_question_lot_size_residential"
    return function

def get_temperature():

    return 0

def get_seed():

    return 42

def get_top_p():

    return None

def create_openai_object(model, prompt, sys_info, custom_id, tools=None, tool_choice=None, n=1, subtask = False):
    # Start with the basic structure of the response dictionary
    response = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_info},
                {"role": "user", "content": prompt}
            ],
            "n": n,
        }
    }

    if subtask:
        temp = 0

    else:
        temp = get_temperature()

    seed = get_seed()

    top_p = get_top_p()

    if temp is not None:
        response["body"]["temperature"] = temp

    if seed is not None:
        response["body"]["seed"] = seed

    if top_p is not None:
        response["body"]["top_p"] = top_p


    # Add 'tools' to the body only if it is not None
    if tools is not None:
        response["body"]["tools"] = tools

    # Add 'tool_choice' to the body only if it is not None
    if tool_choice is not None:
        response["body"]["tool_choice"] = tool_choice

    return response


class QuestionMuniPair:
    def __init__(self, question, muni, model):
        self.question = question
        self.muni = muni
        self.model = model
        self.state = 'subtasks'
        self.response = []
        self.cost = 0
        self.action = 'query'
        self.openai_object = []
        self.subtasks = []
        self.data = {}
        self.context = None
        self.has_double_checked = False

    #Where to start processing the pair following either intialization or completion of a previous API call
    def process(self):


        #If no response and an openai object then we need to re-reun so just return
        if len(self.response) == 0 and len(self.openai_object) > 0:
            return

        #Reset openai_object
        self.openai_object = []

        #First, process the response if any and transition to the next state
        if len(self.response) > 0:
            self.process_response()

        #If state is complete then return
        if self.state == 'completed':
            return

        #If openai object is empty then we proceed:
        if len(self.openai_object) == 0:
            if self.state == 'subtasks':
                self.process_subtasks()
            else:
                self.process_main()


    def process_response(self):

        #If in subtasks state then we need to process the subtask response
        if self.state == 'subtasks':
            self.response_subtasks()
        else:
            self.response_main()

        #Reset response to empty
        self.response = []

    def response_subtasks(self):
        subtask = get_subtasks(self.question)

        # Try to extract the answer using nested keys
        try:
            answer = self.response[0]['response']['body']['choices'][0]['message']['content']
        except:
            # Fallback if the key structure differs
            answer = self.response[0]['response']['body']['choices'][0]['message'].content

        #Remove three hashtags in a row from answer
        answer = answer.replace('###','')

        #Remove two hashtags in a row
        answer = answer.replace('##','')

        # Construct the subtask dictionary
        subtask = {
            'Answer': answer,
            'Answer Description': subtask['Subtask Results Description'],
            'Context': self.context,
        }

        self.subtasks.append(subtask)

        # Calculate the cost and add it to the total cost
        self.cost += self.calc_cost(self.response[0])

        # Then transition to the main state
        self.state = 'main'

    def calc_cost(self, response):
        # Get the input tokens
        input_tokens = response['response']['body']['usage']['prompt_tokens']

        # Get the output tokens
        output_tokens = response['response']['body']['usage']['completion_tokens']

        #Get the model from the response
        model = response['response']['body']['model']

        # Get the cost per input token from the model
        input_cost = cg.cost_dict[model]['Input']/2 #50% off for batching

        # Get the cost per output token from the model
        output_cost = cg.cost_dict[model]['Output']/2 #50% off for batching

        # Calculate the cost
        cost = input_cost * input_tokens / 1000000 + output_cost * output_tokens / 1000000

        return cost


    def parse_function_response(self, json_string):

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

    def parse_function_call(self):

        #Get parsed responses so far
        parsed_responses = self.data['Parsed Responses']

        #Loop over the responses, parse them, and place them in the list according to their location
        for response in self.response:
            location = response['custom_id'].split('_')[-1]
            try:
                json_string = response['response']['body']['choices'][0]['message']['tool_calls'][0]['function']['arguments']
            except:
                json_string = response['response']['body']['choices'][0]['message'].tool_calls[0].function.arguments
            parsed_response = self.parse_function_response(json_string)
            parsed_responses[int(location)] = parsed_response

        #Aggregate the responses
        return self.aggregate_responses(parsed_responses)


    def response_main(self):

        # Calculate the cost and add it to the total cost
        self.cost += self.calc_cost(self.response[0])

        if self.action != 'parsing':

            # Extract the answer from the open-ended response (method depends on whether using real or fake batching)
            try:
                open_ended = [x['message']['content'] for x in self.response[0]['response']['body']['choices']]
            except:
                open_ended = [x['message'].content for x in self.response[0]['response']['body']['choices']]

            self.data['Explanation'] = open_ended
            self.data['Context'] = self.context
            self.data['Muni'] = self.muni['Muni']
            self.data['Question'] = self.question['ID']
            self.data['Subtasks'] = self.subtasks

            # Parse the answer from the open-ended response
            parsed_response = self.parse_open_ended(open_ended)

            # If parsing needs an LLM call then set action to parsing and return
            if parsed_response is None:
                self.action = 'parsing'
                return

        else:
            parsed_response = self.parse_function_call()
            self.action = 'query'

        self.data['Answer'] = parsed_response['Answer']
        self.data['Dont_Know'] = parsed_response['Dont_Know']
        self.data['All Answers'] = parsed_response['All Answers']

        # Transition to the double-checking state if needed, otherwise finish processing
        if self.should_double_check(parsed_response):
            self.state = 'double_checking'
            self.process_double_check()
        else:
            self.state = 'completed'

    def get_final_response(self):
        if self.state != 'completed':
            raise ValueError("Processing is not complete.")

        final_response = {
            'Answer': self.data['Answer'],
            'Dont_Know': self.data['Dont_Know'],
            'All Answers': self.data['All Answers'],
            'Explanation': self.data['Explanation'],
            'Context': self.data['Context'],
            'Muni': self.data['Muni'],
            'Question': self.data['Question'],
            'Cost': self.cost,
            'Subtasks': self.data['Subtasks'],
            'Prompt': self.data.get('Prompt', 'Prompt not available')
        }

        if 'First Attempt' in self.data:
            final_response['First_Attempt'] = self.data['First Attempt']

        return final_response

    def should_double_check(self, parsed_response):

        #Check if already double checked
        if self.has_double_checked:
            return False

        # Get the prior
        prior = self.question['Prior']

        # If no prior for the question or if the answer is "I Don't Know", don't double-check
        if pd.isnull(prior) or prior == "":
            return False

        # Check if the answer is not the same as the prior
        return parsed_response['Answer'] != prior

    def process_subtasks(self):
        # First, get the subtask
        subtask = get_subtasks(self.question)

        # Check if subtask is None, if so return a list and move to main model
        if subtask is None:
            self.state = 'main'
            self.process_main()
            return

        # Second, build the prompt for the subtask
        openai_object, context = self.subtask_question(subtask)

        # Third, set the openai_object to the prompt
        self.openai_object = [openai_object]

        # Store the context in the state
        self.context = context

    def process_main(self):
        # Build the context using a predefined context builder function
        context = cb.context_builder(self.question, self.state, self.muni)

        # Get question type and prepare the relevant API request object
        question_type = self.question['Question Type']

        if question_type == 'Numerical':
            sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws to answer the question which follows 'Question: '. You think step by step and justify each step with explanations and evidence from the context. At the end of your argument you explictly state your answer in the format of 'ANSWER: ' followed by a number or 'I DON'T KNOW'."
        elif question_type == 'Binary':
            sys_info = "You are a municipal zoning ordinance expert. You use the following context which follows 'Context: ' from a municipal ordinance to answer the question which follows 'Question: '. You first review the background information on the question following 'Background Information on Question:' and treat it as additional instructions. You assume that the context includes all of the relevant legal information for the question. You review the context thoroughly for evidence to answer the question. When you do cannot find any relevant information in the context, you realize that the town does not have relevant laws for the question and you reference the question background for how to handle this situation. You think step by step and justify each step with explanations and evidence from the context. At the end of your argument you review what the answer should be and then explicitly state your answer in the format of 'ANSWER: ' and then one of 'YES', 'NO', or 'I DON'T KNOW'."
        elif question_type == 'Lot Size':
            sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
                        to answer the question which follows 'Question: '. Refer to the question background section for detailed instructions on how to answer the question. You think step by step and justify each step with explanations and evidence from the context. At the end of your answer you say 'ANSWER:' and then reply with a CSV format with a column for 'District Name', 'Minimum Lot Size', 'Unit', and perhaps more depending on the question background. Ensure that you only include one row per district."
        else:
            raise ValueError('Invalid question type')

        openai_object = self.create_question_openai_object(context, sys_info)

        # Set the openai_object to the prepared API request object
        self.openai_object = [openai_object]

        # Store the context in the state
        self.context = context

    def process_double_check(self):

        #Set has double checked to true
        self.has_double_checked = True

        #Restructure data so that all current entries except subtasks are stored in a dictionary inside of data called 'First Attempt'
        first_attempt = self.data.copy()
        first_attempt.pop('Subtasks')
        self.data = {'First Attempt': first_attempt, 'Subtasks': self.subtasks}


    def subtask_question(self, subtask):
        # Build the context using a predefined context builder function
        context = cb.context_builder(self.question, self.state, self.muni)

        # System information setup, providing guidelines for the assistant
        sys_info = "You are a municpal zoning ordinance expert. Use the following context which follows 'Context: ' from a municipal ordinance about zoning laws\
            to answer the question which follows 'Question: '. You justify your answer with explanations and evidence from the context. When you do not know the answer to a question you say that you do not know and explain why."

        # Constructing the prompt with clear separation of elements
        prompt = {
            "System Info": sys_info,
            "Question": subtask['Subtask Instructions'],
            "Context": context
        }

        # Prepare the OpenAI API request object, integrating the prompt in a structured way
        openai_object = create_openai_object(
            model=self.model,
            prompt=self.format_prompt(prompt),
            sys_info=sys_info,
            custom_id=f"{self.muni['Muni']}_subtask_{self.question['ID']}",
            subtask = True
        )

        # Return the prepared API object
        return openai_object, context

    def create_llm_parsing_object(self, answer,location=0):
        #location is the index of the response in the list of responses

        # Get question string
        question_string = self.question['Question Rephrase']

        # Try to split answer for the part that comes after 'ANSWER:'
        try:
            answer = answer.split('ANSWER:')[1]
        except:
            pass

        # Build prompt
        prompt = f"Question: {question_string}\n\nAnswer: {answer}"

        sys_info = "You are a data entry analyst. You are presented with a question and an expert answer. You parse the expert answer to the question."

        # Get the tools
        tools = getattr(cg, determine_function_type(self.question))
        tool_choice = {"type": "function", "function": {"name": determine_function_type(self.question)}}

        openai_object = create_openai_object(
            model=self.model,
            prompt=prompt,
            sys_info=sys_info,
            custom_id=f"{self.muni['Muni']}_parse_{self.question['ID']}_{location}",
            tools=tools,
            tool_choice=tool_choice,
        )

        return openai_object

    def aggregate_binary(self, parsed_responses):
        valid_responses = [resp for resp in parsed_responses if not resp['Dont_Know']]
        if not valid_responses:  # Handle case where all are 'I Don't Know'
            return {'Answer': None, 'Dont_Know': True}

        yes_count = sum(1 for resp in valid_responses if resp['Answer'].lower() == 'yes')
        no_count = sum(1 for resp in valid_responses if resp['Answer'].lower() == 'no')

        if yes_count == no_count:
            return {'Answer': None, 'Dont_Know': True}  # Indeterminate result
        most_common = 'Yes' if yes_count > no_count else 'No'
        return {'Answer': most_common, 'Dont_Know': False}

    def aggregate_numerical(self, parsed_responses):
        valid_responses = [resp for resp in parsed_responses if not resp['Dont_Know']]
        if not valid_responses:
            return {'Answer': None, 'Dont_Know': True}

        numbers = [float(resp['Answer']) for resp in valid_responses]
        median_answer = statistics.median(numbers)
        return {'Answer': median_answer, 'Dont_Know': False}

    def parse_open_ended(self, responses):
        parsed_responses = [self.parse_answer(resp,i) for i, resp in enumerate(responses)]

        self.data['Parsed Responses'] = parsed_responses

        # Handle cases where answer parsing requires LLM parsing
        if any(resp is None for resp in parsed_responses):
            return None  # Triggers LLM parsing in process flow

        return self.aggregate_responses(parsed_responses)

    def aggregate_responses(self, parsed_responses):

        if self.question['Question Type'] == 'Binary':
            response =  self.aggregate_binary(parsed_responses)
        elif self.question['Question Type'] == 'Numerical':
            response =  self.aggregate_numerical(parsed_responses)
        elif self.question['Question Type'] == 'Lot Size':
            response = parsed_responses[0]

        else:
            raise ValueError('Unsupported question type for aggregation')

        response['All Answers'] = parsed_responses

        return response


    def parse_answer(self, answer, location):
        try:
            # Extract the answer and confidence level
            answer_str= answer.split('ANSWER:')[1].strip()
            parsed_answer = self.process_answer(answer_str)

            #Dont know variable is whether parsed answer is None
            dont_know = parsed_answer is None

            return {'Answer': parsed_answer, 'Dont_Know': dont_know}

        except Exception as e:
            # Handle errors and exceptions, sending to LLM if needed
            self.openai_object.append(self.create_llm_parsing_object(answer, location))
            return None  # Indicate that parsing is incomplete due to an error


    def process_answer(self, answer_str):
        # Check if the answer is "I DON'T KNOW"
        if answer_str.upper() == "I DON'T KNOW":
            return None

        # Parse the answer based on the question type
        question_type = self.question['Question Type']
        if question_type == 'Binary':
            if answer_str.upper() in ['YES', 'NO']:
                return answer_str.capitalize()
        elif question_type == 'Numerical':
            return float(answer_str)

        # Raise an error if answer format is unexpected
        raise ValueError(f"Unexpected answer format for {question_type} question")

    def process_confidence(self, confidence_str):
        # Validate confidence string
        if confidence_str not in ['Low', 'High']:
            raise ValueError("Confidence level must be 'Low' or 'High'")
        return confidence_str

    def create_question_openai_object(self, context, sys_info):

        if self.state == 'main':
            question = self.question['Question Rephrase']
        else:
            question = self.question['Double Check Question']

        #If question is null then print self.question then raise error
        if str(question) == 'nan':
            print(self.question)
            print(self.state)
            raise ValueError('Question is null')

        prompt = {
            'Question': question,
        }

        #Add in background information for binary or lot size questions
        if self.question['Question Type'] in ['Binary', 'Lot Size']:
            question_background = load_or_fetch_question_details(self.question)
            prompt['Background Information on Question'] = question_background

        elif self.question['Question Type'] == 'Numerical':
            #For numerical questions, not all questions have question background yet so we use a try catch block to accomodate
            try:
                prompt['Background Information on Question'] = load_or_fetch_question_details(self.question)
                sys_info = sys_info + ' Refer to the question background section for detailed instructions on how to answer the question.'
            except:
                pass


        #Add in context
        prompt['Context'] = context

        for subtask in self.subtasks:
            prompt[subtask['Answer Description']] = subtask['Answer']

        formatted_prompt = self.format_prompt(prompt)

        openai_object = create_openai_object(
            model=self.model,
            prompt=formatted_prompt,
            sys_info=sys_info,
            custom_id=f"{self.muni['Muni']}_main_{self.question['ID']}",
            n=1
        )

        self.data['Prompt'] = formatted_prompt

        return openai_object

    def format_prompt(self, prompt):
        prompt_list = [f"#{entry}:\n{prompt[entry]}" for entry in prompt]
        return '\n\n'.join(prompt_list)

