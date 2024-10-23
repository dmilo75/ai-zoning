### `gpt_functions.py` README

#### Purpose
The `gpt_functions.py` file serves two primary purposes:
1. **API Wrapper**: Provides a more intuitive interface for interacting with OpenAI's API, minimizing redundancy across the codebase.
2. **Function Call Templates**: Contains function templates designed to structure responses from OpenAI's models in a standardized format, making it easier to parse and interpret model outputs in other parts of the project.

#### Configuration
This file requires the OpenAI API key to be set in the `config.yaml` file. The API key is the only field from the configuration required for the operation of this code.

Example:
```yaml
openai_key: "your-openai-api-key"
```

#### Key Components

1. **Cost Dictionary (`cost_dict`)**:
   The cost dictionary holds the pricing information for various OpenAI models, with pricing per million tokens. If users use a different model not included in the dictionary, they can add it manually. For reference on pricing, a link to OpenAI's pricing page is included in the code.

   Example:
   ```python
   cost_dict = {
       'gpt-4-0125-preview': {'Input':10,'Output':30},
       'gpt-3.5-turbo-0125': {'Input':0.5,'Output':1.5},
       # Add your model here
   }
   ```

2. **Batch Processing Functions**:
   While users do not typically interact directly with these functions, they are responsible for managing batch processing operations that are critical in handling large-scale API requests. These include:
   - `create_batch(batch_data)`: Initiates a batch processing job.
   - `retrieve_batch_results(batch_id)`: Retrieves the results of a batch processing job based on its batch ID.
   - `upload_jsonl(data)`: Uploads data in JSONL format for batch processing.

3. **API Interaction**:
   - **`generate_completion`**: This function calls the OpenAI API to generate a completion based on the provided model, messages, and any tools or parameters specified in the request. It is mainly used for fake batching in the project.
   - **`get_embedding`**: Retrieves embeddings for a given text using OpenAI's embedding models (default: `text-embedding-3-large`). This function is utilized in both the embedding code and the context-building process when the required embedding is not available.

4. **File Management**:
   - **`delete_old_files`**: Periodically deletes files older than one week that are hosted on OpenAI's servers. This helps to avoid exceeding OpenAI's storage limits.

5. **Function Call Templates**:
   These templates are designed to structure how questions and responses are processed by OpenAI. The following function templates are provided:
   
   - **`answer_question_yes_no`**: Used to format yes/no answers to questions.
   - **`answer_question_numerical`**: Formats a numerical answer response.
   - **`answer_question_lot_size_residential`**: Used to send the answer to lot size-related questions, including information about residential districts and estate classification.
   
   Each of these templates specifies how responses should be structured in a JSON-like format to ensure consistency when interacting with the models.

6. **Example Function Call**:
   Here's an example of how a function call template is structured:
   ```python
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
                           "enum": ["Yes", "No"],
                           "description": "The answer to the question.",
                       },
                   },
                   "required": ["Dont_Know", "Answer"],
               },
           }
       }
   ]
   ```

#### Usage Notes
- Users generally don't need to interact directly with this file. It serves as a utility file to streamline interactions with OpenAI models and manage batch processing. Calls to these functions will typically originate from other parts of the project, such as the `question_muni_pair.py` object.
- For users needing to add new models to the project, updates to the `cost_dict` will be necessary to ensure cost calculations remain accurate.

