# Import necessary libraries and modules
import os
import pandas as pd
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from langchain.llms import HuggingFacePipeline
from auto_gptq import exllama_set_max_input_length
import transformers
import yaml

# Read configuration from a yaml file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
# Load the Excel file containing example data into a pandas DataFrame
df = pd.read_excel(config['raw_data']+'/Kor Training.xlsx', engine='openpyxl')

# Get the model type from the environment variable
model_type = os.environ.get('MODEL_TYPE')

# Retrieve model directory from the configuration
model_directory =  config[model_type+"_path"]

# Load the model configuration
model_config = transformers.AutoConfig.from_pretrained(
    model_directory,
)

# Load the language model
modelKor = transformers.AutoModelForCausalLM.from_pretrained(
    model_directory,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
)

# Set the maximum input length for the model
modelKor = exllama_set_max_input_length(modelKor, 4096)

# Initialize the tokenizer for the model
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_directory,
)

# Create a text generation pipeline
generate_text = transformers.pipeline(
    model=modelKor,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0,
    max_new_tokens=512,
)

# Instantiate the HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=generate_text)

# Define question lists
questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))
yesnoList = questions[questions['Question Type'] == 'Binary']['ID'].to_list()
numericalList = questions[questions['Question Type'] == 'Numerical']['ID'].to_list()
categoricalList = questions[questions['Question Type'] == 'Categorical']['ID'].to_list()
lotSizeList = questions[questions['Question Type'] == 'lot_size']['ID'].to_list()

# Filter the DataFrame based on the question types
numericDf = df[df['Question'].isin(numericalList)]
yesnoDf = df[df['Question'].isin(yesnoList)]
categoricalDf = df[df['Question'].isin(categoricalList)]
lotSizeDf = df[df['Question'].isin(lotSizeList)]

# Extract explanation and answer pairs from filtered DataFrames
numeric = [(str(k[1]['Explanation']).replace('_x000D_', ''),str(k[1]['Answer'])) for k in numericDf.iterrows()]
yesno = [(str(k[1]['Explanation']).replace('_x000D_', ''),str(k[1]['Answer'])) for k in yesnoDf.iterrows()]
categorical = [(str(k[1]['Explanation']).replace('_x000D_', ''),str(k[1]['Answer'])) for k in categoricalDf.iterrows()]
lotSize = [(str(k[1]['Explanation']).replace('_x000D_', ''),str(k[1]['Answer'])) for k in lotSizeDf.iterrows()]

def korHelper(question, output, prompt):
    """
    Helper function to extract specific information based on the question type using KOR.
    """
    finalOutput = ''
    
    # Handle numerical type questions
    if question['ID'] in numericalList:
        schema = Object(
            id="zoning districts numeric info",
            description="zoning district information about a particular zone.",
            attributes=[
                Text(
                    id="number",
                    description="The number of zoning districts",
                    examples=numeric,
                ),
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema,encoder_or_encoder_class='json')
        korOutput = chain.run(text=(output))["data"]
        finalOutput = korOutput.get('zoning districts numeric info', [{}])[0].get('number', "I don't know")
        
    # Handle yes or no type questions
    elif question['ID'] in yesnoList:
        schema = Object(
            id="zoning districts yes or no info",
            description="zoning district information about a particular zone.",
            attributes=[
                Text(
                    id="yes or no",
                    description="Decision for housing permissions",
                    examples=yesno,
                ),
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema,encoder_or_encoder_class='json')
        korOutput = chain.run(text=(output))["data"]
        finalOutput = korOutput.get('zoning districts yes or no info', [{}])[0].get('yes or no', "I don't know")
        
    # Handle categorical type questions
    elif question['ID'] in categoricalList:
        schema = Object(
            id="zoning entities info",
            description="single or multiple zoning entities information about a particular zone.",
            attributes=[
                Text(
                    id="entity",
                    description="Permission granting entity or entities for zoning and housing",
                    examples=categorical,
                ),
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema,encoder_or_encoder_class='json')
        korOutput = chain.run(text=(output))["data"]
        finalOutput = korOutput.get('zoning entities info', [{}])[0].get('entity', "I don't know")
        
    # Handle lot size type questions
    elif question['ID'] in lotSizeList:
        schema = Object(
            id="zoning and housing lot size",
            description="different zoning lot sizes.",
            attributes=[
                Text(
                    id="size",
                    description="Information on housing lot sizes",
                    examples=lotSize,
                ),
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema,encoder_or_encoder_class='json')
        korOutput = chain.run(text=(output))["data"]
        finalOutput = korOutput.get('zoning and housing lot size', [{}])[0].get('size', "I don't know")
        
    return str(finalOutput)