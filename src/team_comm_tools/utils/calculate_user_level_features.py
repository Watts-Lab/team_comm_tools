# Importing modules from features
from team_comm_tools.utils.summarize_features import get_user_sum_dataframe, get_user_mean_dataframe, get_user_max_dataframe, get_user_min_dataframe, get_user_stdev_dataframe, get_user_median_dataframe
from team_comm_tools.features.get_user_network import *
from team_comm_tools.features.user_centroids import *
from fuzzywuzzy import process

class UserLevelFeaturesCalculator:
    """
    Initialize variables and objects used by the UserLevelFeaturesCalculator class.

    This class uses various feature modules to define user- (speaker) level features. It reads input data and
    initializes variables required to compute the features.

    :param chat_data: Pandas dataframe of chat-level features read from the input dataset
    :type chat_data: pd.DataFrame
    :param user_data: Pandas dataframe of user-level features derived from the chat-level dataframe
    :type user_data: pd.DataFrame
    :param vect_data: Pandas dataframe of message embeddings corresponding to each instance of the chat data
    :type vect_data: pd.DataFrame
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID. Defaults to "conversation_num".
    :type conversation_id_col: str
    :param speaker_id_col: A string representing the column name that should be selected as the speaker ID. Defaults to "speaker_nickname".
    :type speaker_id_col: str
    :param input_columns: List of columns in the chat-level features dataframe that should not be summarized
    :type input_columns: list
    :param user_aggregation: If true, will aggregate features at the user level
    :type user_aggregation: bool
    :param user_methods: Specifies which functions users want to aggregate with (e.g., mean, stdev...) at the user level
    :type user_methods: list
    :param user_columns: Specifies which columns (at the chat level) users want aggregated for the user level
    :type user_columns: list
    """
    def __init__(self, chat_data: pd.DataFrame, 
                        user_data: pd.DataFrame, 
                        vect_data: pd.DataFrame, 
                        conversation_id_col: str, 
                        speaker_id_col: str, 
                        input_columns:list,
                        user_aggregation: bool,
                        user_methods: list,
                        user_columns: list) -> None:

        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.vect_data = vect_data
        self.conversation_id_col = conversation_id_col
        self.speaker_id_col = speaker_id_col
        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.user_aggregation = user_aggregation
        self.user_methods = user_methods
        
        if user_columns is None:
            self.columns_to_summarize = [column for column in self.chat_data.columns \
                                        if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]
        else:
            if user_aggregation == True and len(user_columns) == 0:
                print("Warning: user_aggregation is True but no user_columns specified. Defaulting user_aggregation to False.")
                self.user_aggregation = False
            else:
                user_columns_in_data = list(set(user_columns).intersection(set(self.chat_data.columns)))
                if(len(user_columns_in_data) != len(user_columns)):
                    print(
                        "Warning: One or more requested user columns are not present in the data. Ignoring them."
                    )
                    
                    # print(user_columns_in_data, user_columns)
                    
                    for i in user_columns:
                        matches = process.extract(i, self.chat_data.columns, limit=3)
                        best_match, similarity = matches[0]
                        
                        if similarity == 100:
                            continue
                        elif similarity >= 80:
                            print("Did you mean", best_match, "instead of", i, "?")
                        else:
                            print(i, "not found in data and no close match.")

                self.columns_to_summarize = user_columns_in_data

        self.summable_columns = ["num_words", "num_chars", "num_messages"]
        
        # ensure all lowercase
        self.user_methods = [col.lower() for col in self.user_methods]
        self.columns_to_summarize = [col.lower() for col in self.columns_to_summarize]
        
        # replace interchangable words in columns_to_summarize
        for i in range(len(self.user_methods)):
            if self.user_methods[i] == "average":
                self.user_methods[i] = "mean"
            elif self.user_methods[i] == "maximum":
                self.user_methods[i] = "max"
            elif self.user_methods[i] == "minimum":
                self.user_methods[i] = "min"
            elif self.user_methods[i] == "standard deviation":
                self.user_methods[i] = "stdev"
            elif self.user_methods[i] == "sd":
                self.user_methods = "stdev"

    def calculate_user_level_features(self) -> pd.DataFrame:
        """
        Main driver function for creating user-level features.

        This function computes various user-level features by aggregating chat-level data,
        and appends them as new columns to the input user-level data.

        :return: The user-level dataset with new columns for each user-level feature
        :rtype: pd.DataFrame
        """

        # Get mean features for all features
        # self.get_user_level_mean_features()
        
        # Get total counts for all features
        self.get_user_level_summed_features()
        
        # Get user summary statistics for all features (e.g. mean, min, max, stdev)
        self.get_user_level_summary_statistics_features()
        
        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        # self.get_centroids()

        # Get list of other users in a given conversation
        self.get_user_network()

        return self.user_data

    def get_user_level_summary_statistics_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to user level features.
            
            There are many possible ways to aggregate user level features, e.g.:
            - Mean of all chats by a given user;
            - Max of all chats by a given user;
            - Weighted mean (e.g., looking at different time points?)
            ... and so on.

            This is an open question, so we are putting a TODO here.
        """
                
        if self.user_aggregation == True:
            # For each summarizable feature
            for column in self.columns_to_summarize:
                
                # Average/Mean of feature across the User
                if 'mean' in self.user_methods:
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_mean_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )
                    
                # Maxima for the feature across the User
                if 'max' in self.user_methods:
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_max_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )
                    
                # Minima for the feature across the User
                if 'min' in self.user_methods:
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_min_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )
                    
                # Standard Deviation of feature across the User
                if 'stdev' in self.user_methods:
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_stdev_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )
                    
                # Median of feature across the User
                if 'median' in self.user_methods:
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_median_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )

    def get_user_level_summed_features(self) -> None:
        """
        Aggregate summary statistics from chat-level features that need to be summed together.

        Features for which summing makes sense include:
        - Word count (total number of words)
        - Character count
        - Message count
        - Function word accommodation

        This function calculates and merges the summed features into the user-level data.

        :return: None
        :rtype: None
        """
        
        # For each summarizable feature
        for column in self.summable_columns:
                
            # Sum of feature across the Conversation
            self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_sum_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                on=[self.conversation_id_col, self.speaker_id_col],
                how="inner"
            )

        # if self.user_aggregation == True:

        #     # For each summarizable feature
        #     for column in self.summable_columns:
                
        #         # Sum of feature across the Conversation
        #         self.user_data = pd.merge(
        #             left=self.user_data,
        #             right=get_user_sum_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
        #             on=[self.conversation_id_col, self.speaker_id_col],
        #             how="inner"
        #         )

            # for column in self.columns_to_summarize: # TODO --- Gini depends on the summation happening; something is happening here where it's causing Gini to break.
            #     if column not in self.summable_columns:
            #         # Sum of feature across the Conversation
            #         self.user_data = pd.merge(
            #             left=self.user_data,
            #             right=get_user_sum_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
            #             on=[self.conversation_id_col, self.speaker_id_col],
            #             how="inner"
            #         )

    def get_user_level_mean_features(self) -> None:
        """
        Aggregate summary statistics by calculating mean user-level features from chat-level features.

        This function calculates and merges the mean features into the user-level data.

        :return: None
        :rtype: None
        """
        
        if self.user_aggregation == True:
            # For each summarizable feature
            for column in self.columns_to_summarize:

                if 'mean' in self.user_methods:
                    # Average/Mean of feature across the User
                    self.user_data = pd.merge(
                        left=self.user_data,
                        right=get_user_mean_dataframe(self.chat_data, column, self.conversation_id_col, self.speaker_id_col),
                        on=[self.conversation_id_col, self.speaker_id_col],
                        how="inner"
                    )
                

    def get_centroids(self) -> None:
        """
        Calculate the centroid of each user's chats in a given conversation for future discursive metric calculations.

        This function computes and appends the mean embedding (centroid) of each user's chats to the user-level data.

        :return: None
        :rtype: None
        """
        self.user_data['mean_embedding'] = get_user_centroids(self.chat_data, self.vect_data, self.conversation_id_col, self.speaker_id_col)

    def get_user_network(self) -> None:
        """
        Get the user list per user per conversation.

        This function calculates and appends the list of other users in a given conversation to the user-level data.

        :return: None
        :rtype: None
        """
        self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_network(self.user_data, self.conversation_id_col, self.speaker_id_col),
                on=[self.conversation_id_col, self.speaker_id_col],
                how="inner"
            )