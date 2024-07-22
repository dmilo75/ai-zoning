from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define the model to use
model = 'latest_combined'

# define function to transform the imputed data
def transform(imputed_df,type):

    if type == 'standard_pca':

        # Define columns to make negative
        cols_neg = ['4', '5', '6', '8', '11', '13', '14']

        # Make negative
        imputed_df[cols_neg] = -imputed_df[cols_neg] + 1

        #Return
        return imputed_df

yes_no_qs = config['yes_no_qs']

#make yes_no_qs strings
yes_no_qs = [str(q) for q in yes_no_qs]

#Load in data
df2 = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Light Data.xlsx'))

##

df = df2.copy()

#Map answer of 'Yes' to 1 and 'No' to 0
df['Answer'] = df['Answer'].replace({'Yes':1,'No':0})

#Pivot the dataframe to have questions as columns
pivot_df = df.pivot(index = 'Muni', columns = 'Question', values = 'Answer')

#Count the percent of rows with '28Min' null or 0
def winsorize_to_null(df, column, percentile=0.99):
    # Calculate the 95th percentile of non-null values
    threshold = np.percentile(df[column].dropna(), percentile * 100)
    # Set values above the threshold to null
    df[column] = np.where(df[column] > threshold, np.nan, df[column])

# Apply the function to the columns '28Min' and '22'
winsorize_to_null(pivot_df, '28Max')
winsorize_to_null(pivot_df, '28Min')
winsorize_to_null(pivot_df, '22')
winsorize_to_null(pivot_df, '34')

#Drop 28Max and 28Mean
pivot_df = pivot_df.drop(columns = ['28Mean'])

#For binary columns impute by making it 0.5, i.e. in between 0 and 1
for col in pivot_df.columns:
    if col not in ['28Min','22','2','28Max','30','31','34']:
        pivot_df[col] = pivot_df[col].fillna(0.5)

# Initialize the KNN Imputer
knn_imputer = KNNImputer(n_neighbors=50, weights='uniform')

# Fit the imputer and transform the DataFrame to fill in the missing values
imputed_data = knn_imputer.fit_transform(pivot_df)

# Convert the imputed numpy array back into a pandas DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=pivot_df.columns, index=pivot_df.index)

#Transform the imputed_df into sub_indices
imputed_df = transform(imputed_df,'standard_pca')

#Make 28Max binary for if its greater than 43560
imputed_df['28Max'] = imputed_df['28Max'].apply(lambda x: 1 if x > 43560 else 0)

# Normalize the imputed data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(imputed_df)

# Apply PCA to the normalized data to obtain the first principal component
pca = PCA(n_components=3)
principal_components = pca.fit_transform(normalized_data)

# Extract the loadings for the first two principal components
loadings = pca.components_[:3, :]  # Gets the first three components

# Convert the loadings into a DataFrame for easier handling
loadings_df = pd.DataFrame(loadings, index=['PC1', 'PC2','PC3'], columns=[f'Feature{i}' for i in range(1, normalized_data.shape[1]+1)])

#Set columns
loadings_df.columns = list(imputed_df.columns)

#Export to processed_data
loadings_df.to_excel(os.path.join(config['processed_data'],'Model Output',model,'Loadings.xlsx'))

#Print the loadings on pca in percent weights
loadings = pca.components_
num_features = loadings.shape[1]
#Loop over each components loadings
for j in range(loadings.shape[0]):
    print("Component",j+1)
    for i in range(num_features):
        print(f"{list(imputed_df.columns)[i]}: {loadings[j, i]:.2f}")
    print()


##

    #print(f"{list(imputed_df.columns)[i]}: {loadings[2, i]:.2f}")

# Add the first principal component as a new column to the DataFrame
imputed_df['First_Principal_Component'] = principal_components[:, 0]

#Add second and third principal components
imputed_df['Second_Principal_Component'] = principal_components[:, 1]
imputed_df['Third_Principal_Component'] = principal_components[:, 2]

#Sort by first principal component
imputed_df = imputed_df.sort_values(by = 'First_Principal_Component')

#Export the imputed and transformed data
imputed_df.to_excel(os.path.join(config['processed_data'],'Model Output',model,'Overall_Index.xlsx'))

##

# Calculate contributions for each variable to each PC
contributions = np.array([normalized_data * loadings[i] for i in range(loadings.shape[0])])

res = {}

for i in range(0,3):
    df = pd.DataFrame(contributions[i], columns=imputed_df.columns[:-3], index = imputed_df.index)
    df['Index'] = df.sum(axis=1)
    df['CENSUS_ID_PID6'] = df.index.to_series().apply(lambda x: x.split('#')[1])
    #Sort by index
    df = df.sort_values(by = 'Index')

    res[i] = df

#Export the results
for i in range(0,3):
    res[i].to_excel(os.path.join(config['processed_data'],'Model Output',model,f'PC{i+1}_Contributions.xlsx'))
