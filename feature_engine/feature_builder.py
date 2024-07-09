# feature_builder.py

# 3rd Party Imports
import pandas as pd
pd.options.mode.chained_assignment = None 
import re
import numpy as np
from pathlib import Path
import time

# Imports from feature files and classes
from utils.calculate_chat_level_features import ChatLevelFeaturesCalculator
from utils.calculate_user_level_features import UserLevelFeaturesCalculator
from utils.calculate_conversation_level_features import ConversationLevelFeaturesCalculator
from utils.preprocess import *
from utils.check_embeddings import *

class FeatureBuilder:
    """The FeatureBuilder is the main engine that reads in the user's inputs and specifications and generates 
    conversational features. The FeatureBuilder separately calls the classes (the ChatLevelFeaturesCalculator,
    ConversationLevelFeaturesCalculator, and UserLevelFeaturesCalculator) to generate conversational features at
    different levels.

    :param input_df: A pandas DataFrame containing the conversation data that you wish to featurize.
    :type input_df: pd.DataFrame 
    
    :param vector_directory: Directory path where the vectors are to be cached.
    :type vector_directory: str
    
    :param output_file_path_chat_level: Path where the output csv file is to be generated (assumes that the '.csv' suffix is added).
    :type output_file_path_chat_level: str
    
    :param analyze_first_pct: Analyze the first X% of the data. This parameter is useful because the earlier stages of the conversation may be more predictive than the later stages. Thus, researchers may wish to analyze only the first X% of the conversation data and compare the performance with using the full dataset. Defaults to [1.0].
    :type analyze_first_pct: list(float), optional
    
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID. Defaults to "conversation_num".
    :type conversation_id_col: str, optional

    :param speaker_id_col: A string representing the column name that should be selected as the speaker ID. Defaults to "speaker_nickname".
    :type speaker_id_col: str, optional

    :param message: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message: str, optional
    
    :param cumulative_grouping: If true, uses a cumulative way of grouping chats (not just looking within a single ID, but also at what happened before.) 
        NOTE: This parameter and the following one (`within_grouping`) was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica). 
        It assumes that there are exactly 3 nested columns at different levels: a High, Mid, and Low level; further, it assumes that these levels are temporally nested: that is, each
        group/conversation has one High-level identifier, which contains one or more Mid-level identifiers, which contains one or more Low-level identifiers.
        Defaults to False.
    :type cumulative_grouping: bool, optional
    
    :param within_task: If true, groups cumulatively in such a way that we only look at prior chats that are of the same "task" (Mid-level identifier). Defaults to False.
    :type within_task: bool, optional
    
    :param ner_training_df: This is a pandas dataframe of training data for named entity recognition feature
    :type ner_training_df: pd.DataFrame
    
    :param ner_cutoff: This is the cutoff value for the confidence of prediction for each named entity
    :type ner_cutoff: int

    :return: The FeatureBuilder doesn't return anything; instead, it writes the generated features to files in the specified paths. It will also print out its progress, so you should see "All Done!" in the terminal, which will indicate that the features have been generated.
    :rtype: None

    """
    def __init__(
            self, 
            input_df: pd.DataFrame, 
            vector_directory: str,
            output_file_path_chat_level: str, 
            output_file_path_user_level: str,
            output_file_path_conv_level: str,
            analyze_first_pct: list = [1.0], 
            turns: bool=True,
            conversation_id_col: str = "conversation_num",
            speaker_id_col: str = "speaker_nickname",
            message_col: str = "message",
            timestamp_col: str | tuple[str, str] = "timestamp",
            grouping_keys: list = [],
            cumulative_grouping = False, 
            within_task = False,
            ner_cutoff: int = 0.9,
            ner_training_df: pd.DataFrame = None
        ) -> None:

        #  Defining input and output paths.
        self.chat_data = input_df.copy()
        self.orig_data = input_df.copy()
        self.ner_training = ner_training_df
        self.vector_directory = vector_directory
        print("Initializing Featurization...")
        self.output_file_path_conv_level = output_file_path_conv_level
        self.output_file_path_user_level = output_file_path_user_level

        # Basic error detetection
        # user didn't specify a file name, or specified one with only nonalphanumeric chars
        if not bool(self.output_file_path_conv_level) or not bool(re.sub('[^A-Za-z0-9_]', '', self.output_file_path_conv_level)):
            raise ValueError("ERROR: Improper conversation-level output file name detected.")
        if not bool(self.output_file_path_user_level) or not bool(re.sub('[^A-Za-z0-9_]', '', self.output_file_path_user_level)):
            raise ValueError("ERROR: Improper user (speaker)-level output file name detected.")

        # Set first pct of conversation you want to analyze
        assert(all(0 <= x <= 1 for x in analyze_first_pct)) # first, type check that this is a list of numbers between 0 and 1
        self.first_pct = analyze_first_pct

        # Parameters for preprocessing chat data
        self.turns = turns
        self.conversation_id_col = conversation_id_col
        self.speaker_id_col = speaker_id_col
        self.message_col = message_col
        self.timestamp_col = timestamp_col
        self.column_names = {
            'conversation_id_col': conversation_id_col,
            'speaker_id_col': speaker_id_col,
            'message_col': message_col,
            'timestamp_col': timestamp_col
        }
        self.grouping_keys = grouping_keys
        self.cumulative_grouping = cumulative_grouping # for grouping the chat data
        self.within_task = within_task
        self.ner_cutoff = ner_cutoff

        # check grouping rules
        if self.conversation_id_col not in self.chat_data.columns and len(self.grouping_keys)==0:
            if(self.conversation_id_col == "conversation_num"):
                raise ValueError("Conversation identifier not present in data. Did you perhaps forget to pass in a `conversation_id_col`?")
            raise ValueError("Conversation identifier not present in data.")
        if self.cumulative_grouping and len(grouping_keys) == 0:
            print("WARNING: No grouping keys provided. Ignoring `cumulative_grouping` argument.")
            self.cumulative_grouping = False
        if self.cumulative_grouping and len(grouping_keys) != 3:
            print("WARNING: Can only perform cumulative grouping for three-layer nesting. Ignoring cumulative command and grouping by unique combinations in the grouping_keys.")
            self.cumulative_grouping = False
            self.conversation_id_col = "conversation_num"
        if self.cumulative_grouping and self.conversation_id_col not in self.grouping_keys:
            raise ValueError("Conversation identifier for cumulative grouping must be one of the grouping keys.")
        if self.grouping_keys and not self.cumulative_grouping and self.conversation_id_col != "conversation_num":
            print("WARNING: When grouping by the unique combination of a list of keys (`grouping_keys`), the conversation identifier must be auto-generated (`conversation_num`) rather than a user-provided column. Resetting conversation_id.")
            self.conversation_id_col = "conversation_num"
        
        self.preprocess_chat_data()

        # set new identifier column for cumulative grouping.
        if self.cumulative_grouping and len(grouping_keys) == 3:
            print("NOTE: User has requested cumulative grouping. Auto-generating the key `conversation_num` as the conversation identifier for cumulative convrersations.")
            self.conversation_id_col = "conversation_num"

        # Input columns are the columns that come in the raw chat data
        self.input_columns = self.chat_data.columns

        # Set all paths for vector retrieval (contingent on turns)
        df_type = "turns" if self.turns else "chats"
        if(self.cumulative_grouping): # create special vector paths for cumulative groupings
            if(self.within_task):
                df_type = df_type + "/cumulative/within_task/"
            df_type = df_type + "/cumulative/"

        """
        File path cleanup and assumptions:
        -----
        - By design, we save data into a folder called 'output/' (and add it if not already present in the path)
        - Within 'output/', we save data within the following subfolders:
            - chat/ for chat-level data
            - turn/ for turn-level data
            - conv/ for convesation-level data
            - user/ for user-level data
        - We always output files as a csv (and add '.csv' if not present)
        - We consider the "base file name" to be the file name of the chat-level data, and we use this to name the file
            containing the vector encodings
        - The inputted file name must be a valid, non-empty string
        - The inputted file name must not contain only special characters with no alphanumeric component
        """
        # We assume that the base file name is the last item in the output path; we will use this to name the stored vectors.
        try:
            base_file_name = output_file_path_chat_level.split("/")[-1]
        except:
            raise ValueError("ERROR: Improper chat-level output file name detected.") 

        if not bool(base_file_name) or not bool(re.sub('[^A-Za-z0-9_]', '', base_file_name)): # user didn't specify a file name, or specified one with only nonalphanumeric chars
            raise ValueError("ERROR: Improper chat-level output file name detected.")

        try:
            folder_type_name = output_file_path_chat_level.split("/")[-2]
        except IndexError: # user didn't specify a folder, so we will have to append it for them
            folder_type_name = "turn" if self.turns else "chat"
            output_file_path_chat_level = '/'.join(output_file_path_chat_level.split("/")[:-1]) + '/' + folder_type_name + '/' + base_file_name

        # We check whether the second to last item is a "folder type": either chat or turn.
        if folder_type_name not in ["chat", "turn"]: # user didn't specify the folder type, so we will append it for them
            folder_type_name = "turn" if self.turns else "chat"
            output_file_path_chat_level = '/'.join(output_file_path_chat_level.split("/")[:-1]) + '/' + folder_type_name + '/' + base_file_name

        # Set file paths, ensuring correct subfolder type is added.
        self.output_file_path_chat_level = re.sub('chat', 'turn', output_file_path_chat_level) if self.turns else output_file_path_chat_level
        if self.output_file_path_chat_level.split(".")[-1] != "csv": self.output_file_path_chat_level = self.output_file_path_chat_level + ".csv"
        if not re.match("(.*\/|^)conv\/", self.output_file_path_conv_level):
            self.output_file_path_conv_level = "/".join(self.output_file_path_conv_level.split("/")[:-1]) + "/conv/" + self.output_file_path_conv_level.split("/")[-1]
        if self.output_file_path_conv_level.split(".")[-1] != "csv": self.output_file_path_conv_level = self.output_file_path_conv_level + ".csv"
        if not re.match("(.*\/|^)user\/", self.output_file_path_user_level):
            self.output_file_path_user_level = "/".join(self.output_file_path_user_level.split("/")[:-1]) + "/user/" + self.output_file_path_user_level.split("/")[-1]
        if self.output_file_path_user_level.split(".")[-1] != "csv": self.output_file_path_user_level = self.output_file_path_user_level + ".csv"

        # Ensure output/ is added before the subfolder.
        if not re.match("(.*\/|^)output\/", self.output_file_path_chat_level):
            self.output_file_path_chat_level = re.sub('/' + folder_type_name + '/', '/output/' + folder_type_name + '/', self.output_file_path_chat_level)
        if not re.match("(.*\/|^)output\/", self.output_file_path_conv_level):
            self.output_file_path_conv_level = re.sub('/conv/', '/output/conv/', self.output_file_path_conv_level)
        if not re.match("(.*\/|^)output\/", self.output_file_path_user_level):
            self.output_file_path_user_level = re.sub('/user/', '/output/user/', self.output_file_path_user_level)

        self.vect_path = vector_directory + "sentence/" + ("turns" if self.turns else "chats") + "/" + base_file_name
        self.bert_path = vector_directory + "sentiment/" + ("turns" if self.turns else "chats") + "/" + base_file_name

        # Check + generate embeddings
        check_embeddings(self.chat_data, self.vect_path, self.bert_path, self.message_col)

        self.vect_data = pd.read_csv(self.vect_path, encoding='mac_roman')

        self.bert_sentiment_data = pd.read_csv(self.bert_path, encoding='mac_roman')

        # Deriving the base conversation level dataframe.
        self.conv_data = self.chat_data[[self.conversation_id_col]].drop_duplicates()

    def set_self_conv_data(self) -> None:
        """
        Derives the base conversation level dataframe.

        Set Conversation Data around `conversation_num` once preprocessing completes.
        We need to select the first TWO columns, as column 1 is the 'index' and column 2 is 'conversation_num'.

        :return: None
        :rtype: None
        """     
        self.conv_data = self.chat_data[[self.conversation_id_col]].drop_duplicates()

    def merge_conv_data_with_original(self) -> None:
        """
        Merge conversation-level data with the original dataset, so that we retain all desired columns, both original and generated.

        If the conversational identifier was generated by our system (as opposed to an existing column in the original dataset),
        the function uses the preprocessed data, which contains the generated ID. Otherwise, it uses the original dataset.

        The function groups the original conversation data by "conversation_num" and merges it with the
        conversation-level data (`conv_data`). It drops duplicate rows and removes the 'index' column if present.

        :return: None
        :rtype: None
        """

        if(self.conversation_id_col == "conversation_num" and "conversation_num" not in self.orig_data.columns):
            # This indicates that the user asked us to generate a conversation_num, as it wasn't in the original
            orig_conv_data = self.preprocessed_data # we therefore use the preprocessed data instead of the original
        else:
            orig_conv_data = self.orig_data

        # Use the 1st item in the row, as they are all the same at the conv level
        orig_conv_data = orig_conv_data.groupby([self.conversation_id_col]).nth(0).reset_index()

        final_conv_output = pd.merge(
            left= self.conv_data,
            right = orig_conv_data,
            on=[self.conversation_id_col],
            how="left"
        ).drop_duplicates()

        self.conv_data = final_conv_output

        # drop index column, if present
        if {'index'}.issubset(self.conv_data.columns):
            self.conv_data = self.conv_data.drop(columns=['index'])

    def featurize(self, col: str="message") -> None:
        """
        Main driver function for feature generation.

        This function creates chat-level features, generates features for different 
        truncation percentages of the data if specified, and produces user-level and 
        conversation-level features. Finally, the features are saved into the 
        designated output files.

        :param col: Column to preprocess, defaults to "message"
        :type col: str, optional

        :return: None
        :rtype: None
        """

        # Step 1. Create chat level features.
        print("Chat Level Features ...")
        self.chat_level_features()

        # Things to store before we loop through truncations
        self.chat_data_complete = self.chat_data # store complete chat data
        self.output_file_path_user_level_original = self.output_file_path_user_level
        self.output_file_path_chat_level_original = self.output_file_path_chat_level
        self.output_file_path_conv_level_original = self.output_file_path_conv_level

        # Step 2.
        # Run the chat-level features once, then produce different summaries based on 
        # user specification.
        for percentage in self.first_pct: 
            # Reset chat, conv, and user objects
            self.chat_data = self.chat_data_complete
            self.user_data = self.chat_data[[self.conversation_id_col, self.speaker_id_col]].drop_duplicates()
            self.set_self_conv_data()

            print("Generating features for the first " + str(percentage*100) + "% of messages...")
            self.get_first_pct_of_chat(percentage)
            
            # update output paths based on truncation percentage to save in a designated folder
            if percentage != 1: # special folders for when the percentage is partial
                self.output_file_path_user_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_user_level_original)
                self.output_file_path_chat_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_chat_level_original)
                self.output_file_path_conv_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_conv_level_original)
            else:
                self.output_file_path_user_level = self.output_file_path_user_level_original
                self.output_file_path_chat_level = self.output_file_path_chat_level_original
                self.output_file_path_conv_level = self.output_file_path_conv_level_original
            
            # Make it possible to create folders if they don't exist
            Path(self.output_file_path_user_level).parent.mkdir(parents=True, exist_ok=True)
            Path(self.output_file_path_chat_level).parent.mkdir(parents=True, exist_ok=True)
            Path(self.output_file_path_conv_level).parent.mkdir(parents=True, exist_ok=True)

            # Step 3a. Create user level features.
            print("Generating User Level Features ...")
            self.user_level_features()

            # Step 3b. Create conversation level features.
            print("Generating Conversation Level Features ...")
            self.conv_level_features()
            self.merge_conv_data_with_original()
            
            # Step 4. Write the feartures into the files defined in the output paths.
            print("All Done!")
            self.save_features()

    def preprocess_chat_data(self) -> None:
        """
        Call all preprocessing modules needed to clean the chat text.

        This function groups the chat data as specified, verifies column presence, creates original and lowercased columns, preprocesses text, and optionally processes chat turns.

        :param turns: Whether to preprocess naive turns, defaults to False
        :type turns: bool, optional
        :param col: Columns to preprocess, including conversation_id, speaker_id and message, defaults to None
        :type cumulative_grouping: bool, optional
        :param within_task: Whether to group within tasks, defaults to False
        :type within_task: bool, optional
        
        :return: None
        :rtype: None
        """

        # create the appropriate grouping variables and assert the columns are present
        self.chat_data = preprocess_conversation_columns(self.chat_data, self.conversation_id_col, self.timestamp_col, self.grouping_keys, self.cumulative_grouping, self.within_task)
        assert_key_columns_present(self.chat_data, self.column_names)

        # save original column with no preprocessing
        self.chat_data[self.message_col + "_original"] = self.chat_data[self.message_col]

        # create new column that retains punctuation
        self.chat_data["message_lower_with_punc"] = self.chat_data[self.message_col].astype(str).apply(preprocess_text_lowercase_but_retain_punctuation)
    
        # Preprocessing the text in `col` and then overwriting the column `col`.
        # TODO: We should probably use classes to abstract preprocessing module as well?
        self.chat_data[self.message_col] = self.chat_data[self.message_col].astype(str).apply(preprocess_text)

        if (self.turns):
            self.chat_data = preprocess_naive_turns(self.chat_data, self.column_names)

        # Save the preprocessed data (so we don't have to do this again)
        self.preprocessed_data = self.chat_data

    def chat_level_features(self) -> None:
        """
        Instantiate and use the ChatLevelFeaturesCalculator to create chat-level features.

        This function creates chat-level features using the ChatLevelFeaturesCalculator 
        and adds them to the `self.chat_data` dataframe. It also removes special 
        characters from the column names.

        :return: None
        :rtype: None
        """
        # Instantiating.
        chat_feature_builder = ChatLevelFeaturesCalculator(
            chat_data = self.chat_data,
            vect_data = self.vect_data,
            bert_sentiment_data = self.bert_sentiment_data,
            ner_training = self.ner_training,
            ner_cutoff = self.ner_cutoff,
            conversation_id_col = self.conversation_id_col,
            message_col = self.message_col,
            timestamp_col = self.timestamp_col
        )
        # Calling the driver inside this class to create the features.
        self.chat_data = chat_feature_builder.calculate_chat_level_features()
        # Remove special characters in column names
        self.chat_data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in self.chat_data.columns]

    def get_first_pct_of_chat(self, percentage) -> None:
        """
        Truncate each conversation to the first X% of rows.

        This function groups the chat data by `conversation_num` and retains only 
        the first X% of rows for each conversation.

        :param percentage: Percentage of rows to retain in each conversation
        :type percentage: float

        :return: None
        :rtype: None
        """
        chat_grouped = self.chat_data.groupby(self.conversation_id_col)
        num_rows_to_retain = pd.DataFrame(np.ceil(chat_grouped.size() * percentage)).reset_index()
        chat_truncated = pd.DataFrame()
        for conversation_num, num_rows in num_rows_to_retain.itertuples(index=False):
            chat_truncated = pd.concat([chat_truncated,chat_grouped.get_group(conversation_num).head(int(num_rows))], ignore_index = True)

    def user_level_features(self) -> None:
        """
        Instantiate and use the UserLevelFeaturesCalculator to create user-level features.

        This function creates user-level features using 
        the UserLevelFeaturesCalculator, and adds them to the `self.user_data` dataframe.
        It also removes special characters from the column names.

        :return: None
        :rtype: None
        """
        user_feature_builder = UserLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            user_data = self.user_data,
            vect_data= self.vect_data,
            conversation_id_col = self.conversation_id_col,
            speaker_id_col = self.speaker_id_col,
            input_columns = self.input_columns
        )
        self.user_data = user_feature_builder.calculate_user_level_features()
        # Remove special characters in column names
        self.user_data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in self.user_data.columns]

    def conv_level_features(self) -> None:
        """
        Instantiate and use the ConversationLevelFeaturesCalculator to create conversation-level features.

        This function creates conversation-level features using 
        the ConversationLevelFeaturesCalculator, and adds them to the `self.conv_data` dataframe.

        :return: None
        :rtype: None
        """
        conv_feature_builder = ConversationLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            user_data = self.user_data,
            conv_data = self.conv_data,
            vect_data = self.vect_data,
            vector_directory = self.vector_directory,
            conversation_id_col = self.conversation_id_col,
            speaker_id_col = self.speaker_id_col,
            message_col = self.message_col,
            timestamp_col = self.timestamp_col,
            input_columns = self.input_columns
        )
        self.conv_data = conv_feature_builder.calculate_conversation_level_features()

    def save_features(self) -> None:
        """
        Save the feature dataframes to their respective output file paths.

        This function saves the `chat_data`, `user_data`, and `conv_data` dataframes 
        to the respective CSV files specified in the output file paths provided during initialization.

        :return: None
        :rtype: None
        """
        self.chat_data.to_csv(self.output_file_path_chat_level, index=False)
        self.user_data.to_csv(self.output_file_path_user_level, index=False)
        self.conv_data.to_csv(self.output_file_path_conv_level, index=False)