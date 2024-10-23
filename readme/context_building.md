# Context Builder Module

## Description
This module constructs a custom context for a given question-municipality (muni) pair. It processes a tree of text chunks (nodes) associated with the municipality, ranks these chunks based on their cosine similarity to the question's embedding, and then reranks them using a more advanced semantic similarity method. The final output is a structured context string, which can be used to assist in answering zoning-related questions.

## Helper Functions

### 1. `flatten_tree(tree)`
- **Purpose**: Flattens a hierarchical dictionary (tree) into a list of leaf nodes (text chunks).
- **Usage**: Extracts text chunks from a tree for subsequent ranking.

### 2. `sort_chunks(question_embedding, node_list)`
- **Purpose**: Ranks text chunks by their cosine similarity to the given question's embedding.
- **Usage**: Provides an initial ranking based on similarity scores.

### 3. `reranker(sorted_da, question, type)`
- **Purpose**: Reranks the sorted text chunks based on the method specified:
  - **'semantic'**: Uses Cohere’s semantic reranker.
  - **'keyword'**: Utilizes specific keywords in the text chunks when double-checking.
- **Usage**: Refines the initial ranking using a more advanced similarity approach. We use the Cohere API, for more information please see [here](https://docs.cohere.com/docs/overview).

### 4. `select_chunks(sorted_text, max_tokens)`
- **Purpose**: Selects text chunks based on their relevance, limited by a maximum token count.
- **Usage**: Ensures the context doesn't exceed the allowed number of tokens.

### 5. `find_embedding(text)`

- **Purpose**: Retrieves or computes the embedding for a question.
- **Usage**: Used to get the question’s embedding. If the embedding has already been computed, it pulls it from the stored file. If experimenting with question phrasing, it computes the embedding on the spot.



## Primary Function: `context_builder(question, state, muni)`

- **Purpose**: Constructs a custom context for a given question-municipality (muni) pair.

- **Steps**:
  1. **Load Tree**: Retrieves the hierarchical tree structure of text chunks related to the municipality.
  2. **Flatten Tree**: Converts the tree into a list of leaf nodes containing text chunks.
  3. **Initial Ranking**: Sorts the list of text chunks based on the cosine similarity between each chunk’s embedding and the question’s embedding.
  4. **Determine Context-Building Strategy**: If the question-muni pair is in the double-checking state, the function checks whether a different context-building strategy should be used (either keyword filtering or semantic reranking) based on the question. If not in double-checking, the default is semantic context building.
  5. **Reranking**: After the initial cosine similarity ranking, the function refines the list using either a reranking model (semantic) or keyword filtering, depending on the context-building strategy.
  6. **Chunk Selection**: Adds text chunks in the order determined by the reranking step until the maximum token threshold is reached.
  7. **Context Construction**: Joins the selected chunks into a single structured context string, with each chunk marked by section headers.

- **Returns**: A formatted string containing the most relevant text chunks for answering the given question.



## Configuration
The following parameters in the `config.yaml` file are relevant for this module:

```yaml
embeddings: "processed data/Embeddings" # Directory where the embeddings for all municipalities are stored
raw_data: "raw data" # Folder to store raw data, such as Keywords.xlsx
max_tokens: 4 # Maximum context length in thousands of tokens (e.g., 4000 tokens)
```
