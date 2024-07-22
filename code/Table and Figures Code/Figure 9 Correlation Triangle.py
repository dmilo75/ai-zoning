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

model = 'latest_combined'





short_questions = {
        '1': 'Bylaw Online Availability',
        '2': 'Zoning District Count',
        '3': 'Multifamily By Right',
        '4': 'Multifamily Allowed',
        '5': 'Mixed-Use Buildings',
        '6': 'Conversion To Multifamily',
        '7': 'Multifamily Permit Authority',
        '8': 'Townhouses Allowed',
        '9': 'Age-Restricted Provisions',
        '10': 'Age-Restricted Developments Built',
        '11': 'Accessory Apartments Allowed',
        '12': 'Accessory Apartment Authority',
        '13': 'Flexible Zoning By Right',
        '14': 'Flexible Zoning By Permit',
        '15': 'Flexible Zoning Authority',
        '16': 'Flexible Zoning Built',
        '17w': 'Affordable Mandate',
        '17': 'Affordable Incentive',
        '18': 'Inclusionary Zoning Adopted',
        '19': 'Affordable Units Built',
        '20': 'Permit Cap Or Phasing',
        '21': 'Wetlands Restricted in Lot Size Calc',
        '22': 'Longest Frontage Requirement',
        '23': 'Frontage Measurement',
        '24': 'Lot Shape Requirements',
        '25': 'Height Measurement',
        '26': 'Additional Zoning Notes',
        '28Mean': 'Mean Res Min Lot Size',
        '28Min': 'Minimum Res Min Lot Size',
        '28Max': 'Maximum Res Min Lot Size',
        '30': 'Mandatory Approval Steps',
        '31': 'Distinct Approval Bodies',
        '32': 'Public Hearing Requirements',
        '34': 'Max Review Waiting Time'
    }


#Import data
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Light Data.xlsx'))

#Pivot the dataframe to have questions as columns
pivot_df = df.pivot(index = 'Muni', columns = 'Question', values = 'Answer')

#Map 'Yes' to 1 and 'No' to 0
pivot_df = pivot_df.replace({'Yes':1,'No':0})

#Drop 28Max
pivot_df = pivot_df.drop(columns = '28Max')

#Get correlation between each column and question '27Min'
correlation = pivot_df.corrwith(pivot_df['28Min'])

#Now order columns in pivot_df by correlation with '27Min'
pivot_df = pivot_df[correlation.sort_values(ascending=False).index]

# Calculate the correlation matrix
correlation_matrix = pivot_df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

##

# Increase the figure size for more vertical space
plt.figure(figsize=(8, 7), dpi = 150)

# Plot the heatmap
ax1 = plt.subplot()
sns.heatmap(correlation_matrix, mask=mask, cmap="coolwarm", center=0, annot=False, square=True,
            linewidths=.5, cbar_kws={"shrink": .5, 'label': 'Correlation Coefficient'}, ax=ax1)

# Transform x-axis labels
xticks_labels = [f"{short_questions[str(i)]}" for i in correlation_matrix.columns]
ax1.set_xticks(np.arange(len(xticks_labels)) + 0.5)
ax1.set_xticklabels(xticks_labels, rotation=45, ha='right')

# Transform y-axis labels
yticks_labels = [f"{short_questions[str(i)]}" for i in correlation_matrix.index]
ax1.set_yticks(np.arange(len(yticks_labels)) + 0.5)
ax1.set_yticklabels(yticks_labels, rotation=0)

ax1.tick_params(axis='both', which='major', labelsize=8)

# Adjust x-axis and y-axis labels
ax1.set_xlabel('Zoning Question')
ax1.set_ylabel('Zoning Question')

# Adjust the left and bottom margins for longer labels
plt.subplots_adjust(left=0.3, bottom=0.3)

# Save the figure
plt.savefig(os.path.join(config['figures_path'],'Figure 9 - Correlation Triangle.png'), bbox_inches='tight')

plt.show()


