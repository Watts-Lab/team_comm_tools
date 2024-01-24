"""
file: calculate_conversation_level_features.py
---
This file defines the ConvLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all conv level features here. 
"""

# Importing modules from features
from utils.gini_coefficient import *
from features.basic_features import *
# from utils.summarize_chat_level_features import *
from utils.summarize_features import *
from utils.preprocess import *
from features.get_all_DD_features import *


class ConversationLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame, user_data: pd.DataFrame, conv_data: pd.DataFrame, vect_data: pd.DataFrame, input_columns:list) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
            @param conv_data (pd.DataFrame): This is a pandas dataframe of the conversation level features derived from the 
                                             chat level dataframe.
            @param input_columns (list): This is a list containing all the columns in the chat level features dataframe that 
                                         should not be summarized.
        """
        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.conv_data = conv_data
        self.vect_data = vect_data
        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.columns_to_summarize = [column for column in self.chat_data.columns \
                                     if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]
        self.summable_columns = ["num_words", "num_chars", "num_messages"]
        
    def calculate_conversation_level_features(self) -> pd.DataFrame:
        """
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The conversation level dataset given to this class during initialization along with 
							new columns for each conv level feature.

        """
        # Get gini based features by aggregating user-level totals, pass in USER LEVEL FEATURES
        self.get_gini_features()
        print("Generated gini features.")

        # Get summary statistics by aggregating chat level features, pass in CHAT LEVEL FEATURES
        self.get_conversation_level_aggregates()
        print("Generated chat aggregates.")

        # Get summary statistics by aggregating user level features, pass in USER LEVEL FEATURES 
        self.get_user_level_aggregates()
        print("Generated user aggregates.")
        
        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        self.get_discursive_diversity_features()

        return self.conv_data

    def get_gini_features(self) -> None:
        """
            This function is used to calculate the gini index for features in the conversation.

            Note that Gini matters only when "amount" is involved. Thus, we should only calculate this for:
            - word count (e.g., total number of words)
            - character count
            - message count
            - function_word_accommodation

            (In essence, these all represent _counts_ of something, for which it makes sense to get a "total")
        """

        for column in self.summable_columns:
            
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_gini(self.user_data.copy(), "sum_"+column), # this applies to the summed columns in user_data, which matches the above
                on=['conversation_num'],
                how="inner"
            )

    def get_conversation_level_aggregates(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to conversation level features.

            Specifically, it looks at 4 aggregation functions: Max, Min, Mean, Standard Deviation.
        """

        # For each summarizable feature
        for column in self.columns_to_summarize:
            
            # Average/Mean of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.chat_data.copy(), column, 'average_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.chat_data.copy(), column, 'stdev_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.chat_data.copy(), column, 'min_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.chat_data.copy(), column, 'max_'+column),
                on=['conversation_num'],
                how="inner"
            )

        # Do this only for the columns that make sense (e.g., countable things)
        for column in self.summable_columns:
            # Sum for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_sum(self.chat_data.copy(), column, 'sum_'+column),
                on=['conversation_num'],
                how="inner"
            )
    
    def get_user_level_aggregates(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to conversation level features.
            Specifically, it looks at the mean and standard deviations at message and word level.
        """

        # Sum Columns were created using self.get_user_level_summed_features()
        for column in self.columns_to_summarize:
            
            # Average/Mean of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.user_data.copy(), "sum_"+column, 'average_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.user_data.copy(), "sum_"+column, 'stdev_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.user_data.copy(), "sum_"+column, 'min_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.user_data.copy(), "sum_"+column, 'max_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

        # Average Columns were created using self.get_user_level_averaged_features()
        for column in self.columns_to_summarize:
            
            # Average/Mean of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.user_data.copy(), "average_"+column, 'average_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.user_data.copy(), "average_"+column, 'stdev_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.user_data.copy(), "average_"+column, 'min_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.user_data.copy(), "average_"+column, 'max_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

    
    def get_discursive_diversity_features(self) -> None:
        """
            This function is used to calculate the discursive diversity for each conversation 
            based on the word embeddings (SBERT) and chat level information.
        """
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_DD_features(self.chat_data, self.vect_data),
            on=['conversation_num'],
            how="inner"
        )