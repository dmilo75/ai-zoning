import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import os
import numpy as np
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def load_adjustments():
    # Load in the manual corrections
    adjustments = pd.read_excel(os.path.join(config['raw_data'], 'binary_adjustments.xlsx'))

    # Filter to 'Correct' == 0
    adjustments = adjustments[adjustments['Whether Correct'] == 0]

    # Options for 'Why Wrong'
    options = ['Wrong - Context', 'Wrong - Interpretation', 'Pioneer Wrong', 'Pioneer Outside', 'Ambiguous']

    # Fix the np.random seed
    np.random.seed(0)

    # Now randomly assign the 'Why Wrong' column with these options with a fixed seed
    adjustments['Why Wrong'] = np.random.choice(options, adjustments.shape[0])

    #Ensure the 'Question' column is a string
    adjustments['Question'] = adjustments['Question'].astype(str)

    return adjustments

adjustments = load_adjustments()

adjustments_counts = adjustments.groupby(['Question', 'Why Wrong']).size().unstack(fill_value=0)

# Plot the updated counts as a stacked bar chart
adjustments_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Figure 2: Breaking down errors by Question')
plt.xlabel('Question')
plt.ylabel('Count of Errors')
plt.xticks(rotation=0)
plt.legend(title='Why Wrong')
plt.tight_layout()

plt.show()