"""
file: calculate_user_level_features.py
---
This file defines the UserLevelFeaturesCalculator class using the modules defined in "features".

The intention behind this class is to use these modules and define any and all user level features here. 
"""

# Importing modules from features
from utils.summarize_chat_level_features import get_count_dataframe
from features.user_centroids import *


class UserLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame, user_data: pd.DataFrame, vect_data: pd.DataFrame, input_columns:list) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
            @param user_data (pd.DataFrame): This is a pandas dataframe of the user level features derived from the chat level dataframe.
            @param vect_data (pd.DataFrame): This is a pandas dataframe of the message embeddings correlating with each instance of the chat csv. 
            @param input_columns (list): This is a list containing all the columns in the chat level features dataframe that 
                                         should not be summarized.
        """
        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.vect_data = vect_data

        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.columns_to_summarize = [column for column in self.chat_data.columns \
                                     if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]

    def calculate_user_level_features(self) -> pd.DataFrame:
        """
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The conversation level dataset given to this class during initialization along with 
							new columns for each conv level feature.
        """

        # Get total counts by aggregating chat level features
        self.get_user_level_summary_statistics_features()

        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        # self.get_centroids()

        return self.user_data

    def get_user_level_summary_statistics_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to conversation level features.
            Specifically, it looks at the mean and standard deviations at message and word level.
        """
        # For each summarizable feature
        for column in self.columns_to_summarize:
            # Average/Mean of feature across the Conversation
            self.user_data = pd.merge(
                left=self.user_data,
                right=get_count_dataframe(self.chat_data, column),
                on=['conversation_num', 'speaker_nickname'],
                how="inner"
            )

    def get_centroids(self) -> None:
        """
        This function is used to get the centroid of each user's chats in a given conversation to be used for future discursive metric calculations. 
        
        """
        self.user_data['mean_embedding'] = get_user_centroids(self.chat_data, self.vect_data)



# FEATURES FOR USER LEVEL

# 1. Semantic modulation
# MAXIMUM shift in speech style within a transition between chunks
# AVERAGE shift in speech style across chunks
# VARIANCE shift across all chunks (high variance --> inconsistent shifts vs consistent shifts)

# 2. Summarize chat level features for speakers (get count dataframe)
# words, characters, LIWC, ConvoKit, BERT
# GETTING CENTROIDS FOR CONVERSATION --> discursive diversity (per conversation)
# --> may have to store this information in separate file for readability?

# 3. Inputs for a model
# Currently, all the inputs are in the conversational-level output. All chat level features are aggregated into min, average, max, and stdev across all speakers in a given conversation. For example, MAX positivity represents the speaker with the highest positivity score across all chats in its conversation.

 
