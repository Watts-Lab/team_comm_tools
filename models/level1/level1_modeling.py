import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

# Define the base directory
base_dir = "/Users/evanrowbotham/Dev/data"
visualization_dir = os.path.join(base_dir, "level1_data/modeling/visualizations")
os.makedirs(visualization_dir, exist_ok=True)

# Load the feature importances from Level 0
feature_importances_opp = np.load(os.path.join(base_dir, 'feature_importances_opp.npy'))
feature_importances_dir = np.load(os.path.join(base_dir, 'feature_importances_dir.npy'))
feature_names = np.load(os.path.join(base_dir, 'feature_names.npy'), allow_pickle=True)

# Load the Level 1 dataset
data = pd.read_csv(os.path.join(base_dir, "larger_level1_dataset_cleaned_chat_level.csv"))

# Ensure the feature columns are in the same order as feature_importances
numeric_features = data[feature_names]

# Calculate the weighted averages for oppositional intensity and directness scores
data['oppositional_intensity_score'] = numeric_features.dot(feature_importances_opp).round().abs()
data['directness_score'] = numeric_features.dot(feature_importances_dir).round()

# Normalize the scores to be between 0 and 1
data['oppositional_intensity_score'] = (data['oppositional_intensity_score'] - data['oppositional_intensity_score'].min()) / (data['oppositional_intensity_score'].max() - data['oppositional_intensity_score'].min())
data['directness_score'] = (data['directness_score'] - data['directness_score'].min()) / (data['directness_score'].max() - data['directness_score'].min())

# Add message order for plotting
data['message_order'] = data.groupby('conversation_num').cumcount() + 1

# Add a conversation_outcome label based on the first word of the conversation_num
data['conversation_outcome'] = data['conversation_num'].apply(lambda x: 'deescalatory' if x.split('_')[0] == 'deescalatory' else 'escalatory')

# Calculate the sum of oppositional intensity and directness scores for each conversation
conversation_sums = data.groupby('conversation_num').agg({
    'oppositional_intensity_score': 'sum',
    'directness_score': 'sum'
}).reset_index()

# Merge with the main data to get conversation outcomes
conversation_sums = conversation_sums.merge(data[['conversation_num', 'conversation_outcome']].drop_duplicates(), on='conversation_num')

# Select the conversation with the highest OI and highest Directness for escalatory conversations
most_escalatory_conv = conversation_sums[conversation_sums['conversation_outcome'] == 'escalatory'].sort_values(by=['oppositional_intensity_score', 'directness_score'], ascending=[False, False]).iloc[0]['conversation_num']

# Select the three conversations with the lowest OI and highest Directness for de-escalatory conversations
most_deescalatory_convs = conversation_sums[conversation_sums['conversation_outcome'] == 'deescalatory'].sort_values(by=['directness_score', 'oppositional_intensity_score'], ascending=[False, True]).head(3)['conversation_num'].tolist()

# Filter the data for the selected conversations and cut off the last two messages
conv1 = data[data['conversation_num'] == most_escalatory_conv].iloc[:-2]
conv2_list = [data[data['conversation_num'] == conv].iloc[:-2] for conv in most_deescalatory_convs]

# Function to plot the conversation with color-coded Directness
def plot_conversation(conv, title, filename):
    norm = Normalize(vmin=0, vmax=1)  # Normalized range
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot points
    sc = plt.scatter(conv['message_order'], conv['oppositional_intensity_score'], c=conv['directness_score'], cmap=cmap, norm=norm, s=100, edgecolors="w", linewidth=0.5)
    plt.colorbar(sc, label='Directness Score')
    
    plt.xlabel('Message Number')
    plt.ylabel('Oppositional Intensity (Normalized)')
    plt.title(title)
    plt.ylim(0, 1)  # Normalized range
    plt.xticks(conv['message_order'])
    plt.tight_layout()
    
    # Annotate significant changes
    threshold_change = 0.1  # This threshold can be adjusted based on the dataset
    for idx in range(1, len(conv)):
        if abs(conv.iloc[idx]['oppositional_intensity_score'] - conv.iloc[idx-1]['oppositional_intensity_score']) >= threshold_change:
            plt.annotate(f'{conv.iloc[idx]["oppositional_intensity_score"]:.2f}', 
                         (conv.iloc[idx]['message_order'], conv.iloc[idx]['oppositional_intensity_score']),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

    plt.savefig(os.path.join(visualization_dir, filename))
    plt.show()

# Plot the most extreme escalatory conversation
plot_conversation(conv1, f'Extreme Escalatory Conversation {most_escalatory_conv}', f'extreme_escalatory_conversation_{most_escalatory_conv}_color_coded.png')

# Plot the three extreme de-escalatory conversations
for i, conv2 in enumerate(conv2_list):
    plot_conversation(conv2, f'Extreme De-escalatory Conversation {most_deescalatory_convs[i]}', f'extreme_deescalatory_conversation_{most_deescalatory_convs[i]}_color_coded.png')
