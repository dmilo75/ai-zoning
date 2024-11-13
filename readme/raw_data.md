
### README for Raw Data Directory

#### Introduction
This directory contains several key files essential for the study, combining raw data with processed information used for embedding and question generation. Most of the data columns in these files are derived from the **Census of Governments** dataset. For detailed information on those columns, please refer to the Census of Governments documentation available at this [link](https://www.census.gov/data/datasets/2022/econ/gus/public-use-files.html).

---

### File Descriptions

#### 1. **Sample Data Excel**
This file is a subset of the Census of Governments data, focusing on the municipalities included in this study. The only additional column requiring definition is:

- **Source**: The source of the ordinance information. This can be:
  - **Ordinance.com**
  - **American Legal Publishing**
  - **MuniCode**

For other column definitions, please refer to the [Census of Governments documentation](https://www.census.gov/data/datasets/2022/econ/gus/public-use-files.html).

---

#### 2. **Keywords Excel**
This file is designed to assist in building context around questions by providing an ordered list of keywords. The higher a keyword appears in the list (i.e., the lower the row number), the more important it is for the context-building process.

**Example for Question ID 8**:
- **Question**: "Are attached single family houses (townhouses, 3+ units) listed as an allowed use (by right or special permit)?"
- **Keywords**:
  1. town house
  2. town houses
  3. townhouse
  4. townhouses
  5. attached dwelling
  6. attached dwellings
  7. row house
  8. row houses
  9. rowhouse
  10. rowhouses
  11. attached single family
  12. attached unit
  13. attached units
  14. attached

---

#### 3. **Subtasks Excel**
This file defines various subtasks that are linked to main questions. Each subtask is embedded into the question vector and assists the language model in providing answers.

- **Question Detail**: The exact phrasing of the question used for embedding.
- **Subtask Questions**: The ID of the main question to which this subtask relates.
- **Question Rephrase**: The phrasing of the question as seen by the LLM.
- **Subtask Instructions**: Extra information given to the LLM to assist in answering the subtask.
- **Subtask Results Description**: How the subtask's answer is portrayed to the LLM.

---

#### 4. **Questions Excel**
This file contains the core question data used in the study. It consists of three sheets: **Input Info**, **Archive Questions** (not publicly shared), and **Processed Info**.

**Input Info**:
- **Pioneer Question**: This column is blank if the question does not come from the Pioneer Institute. If the question is from the Pioneer Institute, it contains the exact phrasing used by them.
- **Question Detail**: The phrasing of the question used for embedding.
- **ID**: The unique ID assigned to the question.
- **Question Type**: The category of the question (e.g., binary, lot size, numerical).
- **Question Rephrase**: The phrasing of the question as it appears to the LLM.
- **Prior**: This column is blank if no double checking is performed. If not blank, it contains the prior expected answer, and if that answer is unmet, double checking is triggered.
- **Double Check Context**: Indicates the method used for building context during double checking. The value is either 'keyword' or 'semantic'. All cases in the current study use 'keyword'.
- **Double Check Question**: This column is blank if the question is not rephrased for double checking. If not blank, it contains the rephrased question used for double checking.

**Processed Info**:
- **Positive Means Stricter**: Used for charting purposes, indicating whether a more positive response corresponds to stricter zoning regulations.
- **ID**: The ID of the processed question.
- **Imputed Short**: A short description of the question used when data is imputed, ensuring that all variables are framed in the context of "positive means stricter".
- **Short Question**: A concise phrasing of the question.
- **Full Question**: The complete phrasing of the question.
- **Include**: Whether this question is included in the current study.

---

#### 5. **question_backgrounds Folder**
This folder contains `.txt` files, each named by the question ID. Each file includes the background and assumptions used for the corresponding question. These files provide additional context to help ensure accurate interpretation and processing of each question within the study.

---

---

#### 6. **text Folder**
This folder contains a nested structure that ultimately yields `.pkl` files that contain the zoning ordinances in text form. The text from these `.pkl` files are turned into embeddings, which are fed into the model. Currently, for demonstration purposes, the text folder contains one `.pkl` file that holds the ordinances scraped directly from the website of Wheaton, Illinois

---
