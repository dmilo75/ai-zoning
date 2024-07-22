import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the data
df = pd.read_excel('/Users/david_dzz/Dropbox/Inclusionary Zoning/Github/ai-zoning/processed data/Model Output/Full Run/Overall_Index.xlsx')

# Extract state abbreviations
df['State'] = df['Muni'].apply(lambda x: x.split('#')[-1])

# Define regions and their states
regions = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
    'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
}

# Correctly assign states to regions, accounting for lowercase abbreviations
def map_state_to_region(state):
    state = state.upper()  # Convert to uppercase to match the region dictionaries
    for region, states in regions.items():
        if state in states:
            return region
    return 'Other'

df['Region'] = df['State'].apply(map_state_to_region)

# Assign colors to each region
region_colors = {
    'Northeast': 'blue',
    'Midwest': 'green',
    'South': 'red',
    'West': 'purple',
    'Other': 'gray'
}

df['Color'] = df['Region'].map(region_colors)

# Scatter plot of the first and second principal components
plt.figure(figsize=(12, 8))
for region, color in region_colors.items():
    region_df = df[df['Region'] == region]
    plt.scatter(region_df['First_Principal_Component'], region_df['Second_Principal_Component'],
                color=color, label=region, alpha=0.6)

plt.title('First and Second Principal Components by Region')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the data for T-SNE
features = ['First_Principal_Component', 'Second_Principal_Component']
tsne_data = df[features]

# Apply T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(tsne_data)

# Plot T-SNE results
plt.figure(figsize=(12, 8))
for region, color in region_colors.items():
    region_df = df[df['Region'] == region]
    plt.scatter(tsne_results[region_df.index, 0], tsne_results[region_df.index, 1],
                color=color, label=region, alpha=0.6)

plt.title('T-SNE Visualization of Municipalities by Region')
plt.xlabel('T-SNE 1')
plt.ylabel('T-SNE 2')
plt.legend()
plt.grid(True)
plt.show()
