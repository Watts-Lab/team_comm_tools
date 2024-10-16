# Importing modules from features
from team_comm_tools.features.basic_features import *
from team_comm_tools.features.get_all_DD_features import *
from team_comm_tools.features.turn_taking_features import*
from team_comm_tools.features.burstiness import *
from team_comm_tools.features.information_diversity import *
from team_comm_tools.utils.summarize_features import *
from team_comm_tools.utils.gini_coefficient import *
from team_comm_tools.utils.preprocess import *
from fuzzywuzzy import process

class ConversationLevelFeaturesCalculator:
    """
    Initialize variables and objects used by the ConversationLevelFeaturesCalculator class.

    This class uses various feature modules to define conversation-level features. It reads input data and
    initializes variables required to compute the features.

    :param chat_data: Pandas dataframe of chat-level features read from the input dataset
    :type chat_data: pd.DataFrame
    :param user_data: Pandas dataframe of user-level features derived from the chat-level dataframe
    :type user_data: pd.DataFrame
    :param conv_data: Pandas dataframe of conversation-level features derived from the chat-level dataframe
    :type conv_data: pd.DataFrame
    :param vect_data: Pandas dataframe of processed vectors derived from the chat-level dataframe
    :type vect_data: pd.DataFrame
    :param vector_directory: Directory where vector files are stored
    :type vector_directory: str
    :param input_columns: List of columns in the chat-level features dataframe that should not be summarized
    :type input_columns: list
    :param convo_aggregation: If true, will aggregate features at the conversational level
    :type convo_aggregation: bool
    :param convo_methods: Specifies which functions users want to aggregate with (e.g., mean, stdev...)
    :type convo_methods: list
    :param convo_columns: Specifies which columns (at the chat level) users want aggregated
    :type convo_columns: list
    :param user_aggregation: If true, will aggregate features at the user level
    :type convo_aggregation: bool
    :param user_methods: Specifies which functions users want to aggregate with (e.g., mean, stdev...) at the user level
    :type user_methods: list
    :param user_columns: Specifies which columns (at the chat level) users want aggregated for the user level
    :type user_columns: list
        """
    def __init__(self, chat_data: pd.DataFrame, 
                        user_data: pd.DataFrame, 
                        conv_data: pd.DataFrame, 
                        vect_data: pd.DataFrame, 
                        vector_directory: str, 
                        conversation_id_col: str,
                        speaker_id_col: str,
                        message_col: str,
                        timestamp_col: str,
                        input_columns:list,
                        convo_aggregation: bool,
                        convo_methods: list,
                        convo_columns: list,
                        user_aggregation: bool,
                        user_methods: list,
                        user_columns: list
                        ) -> None:

        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.conv_data = conv_data
        self.vect_data = vect_data
        self.vector_directory = vector_directory
        self.conversation_id_col = conversation_id_col
        self.speaker_id_col = speaker_id_col
        self.message_col = message_col
        self.timestamp_col = timestamp_col
        self.convo_aggregation = convo_aggregation
        self.convo_methods = convo_methods
        self.user_aggregation = user_aggregation
        self.user_methods = user_methods
        self.user_columns = user_columns
        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        if 'conversation_num' not in self.input_columns:
            self.input_columns.append('conversation_num')

        # check if user inputted convo_columns is None
        if convo_columns is None:
            self.columns_to_summarize = [column for column in self.chat_data.columns \
                                        if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]
        else:
            if convo_aggregation == True and (len(convo_columns) == 0 or len(convo_methods) == 0):
                print(
                    "Warning: convo_aggregation is True but no convo_columns specified. Defaulting convo_aggregation to False."
                )
                self.convo_aggregation = False
            else:
                convo_columns_in_data = list(set(convo_columns).intersection(set(self.chat_data.columns)))

                if(len(convo_columns_in_data) != len(convo_columns)):
                    print(
                        "Warning: One or more requested user columns are not present in the data. Ignoring them."
                    )
                    
                    for i in convo_columns:
                        matches = process.extract(i, self.chat_data.columns, limit=3)
                        best_match, similarity = matches[0]
                        
                        if similarity == 100:
                            continue
                        elif similarity >= 80:
                            print("Did you mean", best_match, "instead of", i, "?")
                        else:
                            print(i, "not found in data and no close match.")

                    
                self.columns_to_summarize = convo_columns_in_data
                
                # ensure all lowercase
                self.convo_methods = [col.lower() for col in self.convo_methods]
                self.columns_to_summarize = [col.lower() for col in self.columns_to_summarize]
                
                # replace interchangable words in columns_to_summarize
                for i in range(len(self.convo_methods)):
                    if self.convo_methods[i] == "average":
                        self.convo_methods[i] = "mean"
                    if self.convo_methods[i] == "maximum":
                        self.convo_methods[i] = "max"
                    if self.convo_methods[i] == "minimum":
                        self.convo_methods[i] = "min"
                    if self.convo_methods[i] == "standard deviation":
                        self.convo_methods[i] = "stdev"
                    if self.convo_methods[i] == "sd":
                        self.convo_methods[i] = "stdev"
                    if self.convo_methods[i] == "std":
                        self.convo_methods[i] = "stdev"

                
        # check if user inputted user_columns is None 
        if user_columns is None:
            self.user_columns = [column for column in self.chat_data.columns \
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
                    
                    print(user_columns_in_data, user_columns)
                    
                    for i in user_columns:
                        matches = process.extract(i, self.chat_data.columns, limit=3)
                        best_match, similarity = matches[0]
                        
                        if similarity == 100:
                            continue
                        elif similarity >= 80:
                            print("Did you mean", best_match, "instead of", i, "?")
                        else:
                            print(i, "not found in data and no close match.")

                self.user_columns = user_columns_in_data
                
                # ensure all lowercase
                self.user_methods = [col.lower() for col in self.user_methods]
                self.user_columns = [col.lower() for col in self.user_columns]
                
                # replace interchangable words in columns_to_summarize
                for i in range(len(self.user_methods)):
                    if self.user_methods[i] == "average":
                        self.user_methods[i] = "mean"
                    if self.user_methods[i] == "maximum":
                        self.user_methods[i] = "max"
                    if self.user_methods[i] == "minimum":
                        self.user_methods[i] = "min"
                    if self.user_methods[i] == "standard deviation":
                        self.user_methods[i] = "stdev"
                    if self.user_methods[i] == "sd":
                        self.user_methods[i] = "stdev"
                    if self.user_methods[i] == "std":
                        self.user_methods[i] = "stdev"

        self.summable_columns = ["num_words", "num_chars", "num_messages"]
        
                
        
    def calculate_conversation_level_features(self, feature_methods: list) -> pd.DataFrame:
        """
        Main driver function for creating conversation-level features.

        This function computes various conversation-level features by aggregating chat-level and user-level features,
        and appends them as new columns to the input conversation-level data.

        :return: The conversation-level dataset with new columns for each conversation-level feature
        :rtype: pd.DataFrame
        """

        for method in feature_methods:
            method(self)

        return self.conv_data

    def get_turn_taking_features(self) -> None:
        """
        Calculate the turn-taking index in the conversation.

        This function merges turn-taking features into the conversation-level data.

        :return: None
        :rtype: None
        """

        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_turn(self.chat_data.copy(), self.conversation_id_col, self.speaker_id_col),
            on=[self.conversation_id_col],
            how="inner"
        )

    def get_gini_features(self) -> None:
        """
        Calculate the Gini index for relevant features in the conversation.

        This function computes the Gini index for features involving counts, such as:
        - Word count
        - Character count
        - Message count
        - Function word accommodation

        The Gini index is then merged into the conversation-level data.

        :return: None
        :rtype: None
        """
        for column in self.summable_columns:
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_gini(self.user_data.copy(), "sum_"+column, self.conversation_id_col), # this applies to the summed columns in user_data, which matches the above
                on=[self.conversation_id_col],
                how="inner"
            )

    def get_conversation_level_aggregates(self) -> None:
        """
        Aggregate summary statistics from chat-level features to conversation-level features.

        This function calculates and merges the following aggregation functions for each summarizable feature:
        - Average/Mean
        - Standard Deviation
        - Minimum
        - Maximum

        For countable features (e.g., num_words, num_chars, num_messages), it also calculates and merges the sum.

        :return: None
        :rtype: None
        """

        if self.convo_aggregation == True:
            # For each summarizable feature
            for column in self.columns_to_summarize:
                
                # Average/Mean of feature across the Conversation
                if 'mean' in self.convo_methods:
                    self.conv_data = pd.merge(
                        left=self.conv_data,
                        right=get_mean(self.chat_data.copy(), column, 'mean_'+column, self.conversation_id_col),
                        on=[self.conversation_id_col],
                        how="inner"
                    )

                # Standard Deviation of feature across the Conversation
                if 'stdev' in self.convo_methods:
                    self.conv_data = pd.merge(
                        left=self.conv_data,
                        right=get_stdev(self.chat_data.copy(), column, 'stdev_'+column, self.conversation_id_col),
                        on=[self.conversation_id_col],
                        how="inner"
                    )

                # Minima for the feature across the Conversation
                if 'min' in self.convo_methods:
                    self.conv_data = pd.merge(
                        left=self.conv_data,
                        right=get_min(self.chat_data.copy(), column, 'min_'+column, self.conversation_id_col),
                        on=[self.conversation_id_col],
                        how="inner"
                    )

                # Maxima for the feature across the Conversation
                if 'max' in self.convo_methods:
                    self.conv_data = pd.merge(
                        left=self.conv_data,
                        right=get_max(self.chat_data.copy(), column, 'max_'+column, self.conversation_id_col),
                        on=[self.conversation_id_col],
                        how="inner"
                    )
                    
                # Median for the feature across the Conversation
                if 'median' in self.convo_methods:
                    self.conv_data = pd.merge(
                        left=self.conv_data,
                        right=get_median(self.chat_data.copy(), column, 'median_'+column, self.conversation_id_col),
                        on=[self.conversation_id_col],
                        how="inner"
                    )

        # Do this only for the columns that make sense (e.g., countable things); we do this regardless of aggregation, as it's necessary for gini.
        for column in self.summable_columns:
            # Sum for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_sum(self.chat_data.copy(), column, 'sum_'+column, self.conversation_id_col),
                on=[self.conversation_id_col],
                how="inner"
            )

    def get_user_level_aggregates(self) -> None:
        """
        Aggregate summary statistics from user-level features to conversation-level features.

        This function calculates and merges the following aggregation functions for each user-level feature:
        - Average/Mean of summed user-level features
        - Standard Deviation of summed user-level features
        - Minimum of summed user-level features
        - Maximum of summed user-level features
        - Average/Mean of averaged user-level features
        - Standard Deviation of averaged user-level features
        - Minimum of averaged user-level features
        - Maximum of averaged user-level features


        :return: None
        :rtype: None
        """

        if self.convo_aggregation == True and self.user_aggregation == True:
            
            # aggregates from the user level based on conversation methods
            if 'mean' in self.convo_methods:
                for user_column in self.user_columns:
                    for user_method in self.user_methods:
                         # Average/Mean of User-Level Feature
                        self.conv_data = pd.merge(
                            left=self.conv_data,
                            right=get_mean(self.user_data.copy(), user_method + "_" +user_column, "mean_user_" + user_method + "_" +user_column, self.conversation_id_col),
                            on=[self.conversation_id_col],
                            how="inner"
                        )

            if 'stdev' in self.convo_methods:
                for user_column in self.user_columns:
                    for user_method in self.user_methods:
                        # Standard Deviation of User-Level Feature
                        self.conv_data = pd.merge(
                            left=self.conv_data,
                            right=get_stdev(self.user_data.copy(), user_method + "_" + user_column, 'stdev_user_' + user_method + "_" + user_column, self.conversation_id_col),
                            on=[self.conversation_id_col],
                            how="inner"
                        )

            if 'min' in self.convo_methods:
                for user_column in self.user_columns:
                    for user_method in self.user_methods:
                        # Minima of User-Level Feature
                        self.conv_data = pd.merge(
                            left=self.conv_data,
                            right=get_min(self.user_data.copy(), user_method + "_" + user_column, 'min_user_' + user_method + "_" + user_column, self.conversation_id_col),
                            on=[self.conversation_id_col],
                            how="inner"
                        )
                
            if 'max' in self.convo_methods:
                for user_column in self.user_columns:
                    for user_method in self.user_methods:
                        # Maxima of User-Level Feature
                        self.conv_data = pd.merge(
                            left=self.conv_data,
                            right=get_max(self.user_data.copy(), user_method + "_" + user_column, 'max_user_' + user_method + "_" + user_column, self.conversation_id_col),
                            on=[self.conversation_id_col],
                            how="inner"
                        )
                        
            if 'median' in self.convo_methods:
                for user_column in self.user_columns:
                    for user_method in self.user_methods:
                        # Median of User-Level Feature
                        self.conv_data = pd.merge(
                            left=self.conv_data,
                            right=get_median(self.user_data.copy(), user_method + "_" + user_column, 'median_user_' + user_method + "_" + user_column, self.conversation_id_col),
                            on=[self.conversation_id_col],
                            how="inner"
                        )


    def get_discursive_diversity_features(self) -> None:
        """
        Calculate discursive diversity features for each conversation.

        This function computes discursive diversity based on the word embeddings (SBERT) 
        and chat-level information, and merges the features into the conversation-level data.

        :return: None
        :rtype: None
        """
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_DD_features(self.chat_data, self.vect_data, self.conversation_id_col, self.speaker_id_col, self.timestamp_col),
            on=[self.conversation_id_col],
            how="inner"
        )
            
      
    def calculate_team_burstiness(self) -> None:
        """
        Calculate the team burstiness coefficient.

        This function computes the team burstiness coefficient by looking at the differences 
        in standard deviation and mean of the time intervals between chats, and merges the 
        results into the conversation-level data.

        :return: None
        :rtype: None
        """
        if {'time_diff'}.issubset(self.chat_data.columns):
            self.conv_data = pd.merge(
            left = self.conv_data,
            right = get_team_burstiness(self.chat_data, "time_diff", self.conversation_id_col),
            on = [self.conversation_id_col],
            how = "inner"
        )
    
    def calculate_info_diversity(self) -> None:
        """
        Calculate an information diversity score for the team.

        This function computes the information diversity score by looking at the cosine 
        similarity between the mean topic vector of the team and each message's topic vectors, 
        and merges the results into the conversation-level data.

        :return: None
        :rtype: None
        """
        self.conv_data = pd.merge(
            left = self.conv_data,
            right = get_info_diversity(self.chat_data, self.conversation_id_col, self.message_col),
            on = [self.conversation_id_col],
            how = "inner"
        )