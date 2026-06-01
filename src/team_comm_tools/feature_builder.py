# 3rd Party Imports
import pandas as pd
pd.options.mode.chained_assignment = None 
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from time import perf_counter
import itertools
import warnings

# Imports from feature files and classes
from team_comm_tools.utils.download_resources import download
download()
from team_comm_tools.utils.calculate_chat_level_features import ChatLevelFeaturesCalculator
from team_comm_tools.utils.calculate_user_level_features import UserLevelFeaturesCalculator
from team_comm_tools.utils.calculate_conversation_level_features import ConversationLevelFeaturesCalculator
from team_comm_tools.utils.preprocess import *
from team_comm_tools.utils.check_embeddings import *
from team_comm_tools.feature_dict import feature_dict

class FeatureBuilder:
    """
    The FeatureBuilder is the main engine that reads in the user's inputs and specifications and generates 
    conversational features. The FeatureBuilder separately calls the classes 
    (ChatLevelFeaturesCalculator, ConversationLevelFeaturesCalculator, and 
    UserLevelFeaturesCalculator) to generate conversational features at different levels.

    :param input_df: A pandas DataFrame containing the conversation data that you wish to featurize.
    :type input_df: pd.DataFrame 
    :param vector_directory: Directory path where the vectors are to be cached. Defaults to "./vector_data/".
    :type vector_directory: str
    :param output_file_base: Base name for the output files, used to auto-generate filenames for each 
        of the three levels. Defaults to "output."
    :type output_file_base: str
    :param output_file_path_chat_level: Path where the chat (utterance)-level output csv file is 
        to be generated. This parameter will override the base name.
    :type output_file_path_chat_level: str
    :param output_file_path_user_level: Path where the user (speaker)-level output csv file is 
        to be generated. This parameter will override the base name.
    :type output_file_path_user_level: str
    :param output_file_path_conv_level: Path where the conversation-level output csv file is to be 
        generated. This parameter will override the base name.
    :type output_file_path_conv_level: str
    :param custom_features: A list of additional features outside of the default features that should 
        be calculated. Defaults to an empty list (i.e., no additional features beyond the defaults will 
        be computed).
    :type custom_features: list, optional
    :param turns: If true, collapses multiple "chats"/messages by the same speaker in a row into a 
        single "turn." Defaults to False.
    :type turns: bool, optional
    :param conversation_id_col: A string representing the column name that should be selected as 
        the conversation ID. Defaults to "conversation_num".
    :type conversation_id_col: str, optional
    :param speaker_id_col: A string representing the column name that should be selected as the speaker ID. 
        Defaults to "speaker_nickname".
    :type speaker_id_col: str, optional
    :param message_col: A string representing the column name that should be selected as the message. 
        Defaults to "message".
    :type message_col: str, optional
    :param timestamp_col: A timestamp column name, or a tuple of (start_timestamp_col, end_timestamp_col).
        Defaults to "timestamp".
    :type timestamp_col: str | tuple[str, str], optional
    :param timestamp_unit: A string representing the unit of the timestamp (if the timestamp is numeric). 
        Defaults to 'ms' (milliseconds). Other options (D, s, ms, us, ns) can be found on the Pandas 
        reference: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
    :type timestamp_unit: str, optional
    :param grouping_keys: A list of multiple identifiers that collectively identify a conversation. If 
        non-empty, the data will be grouped by all keys in the list and use the grouped key as the unique 
        "conversational identifier."
    :type grouping_keys: list, optional
    :param cumulative_grouping: If true, uses a cumulative way of grouping chats (looking not just within 
        a single ID, but also at what happened before). NOTE: This parameter and the following one 
        (`within_task`) were created in the context of a multi-stage Empirica game (see: 
        https://github.com/Watts-Lab/multi-task-empirica). Assumes exactly 3 nested columns at different 
        levels: a High, Mid, and Low level; that are temporally nested. Defaults to False.
    :type cumulative_grouping: bool, optional
    :param within_task: If true, groups cumulatively such that only prior chats of the same "task" 
        (Mid-level identifier) are considered. Defaults to False.
    :type within_task: bool, optional
    :param ner_training_df: A pandas DataFrame of training data for named entity recognition features. 
        Defaults to None and will not generate named entity features if it does not exist.
    :type ner_training_df: pd.DataFrame, optional
    :param ner_cutoff: The cutoff value for the confidence of prediction for each named entity. 
        Defaults to 0.9.
    :type ner_cutoff: float
    :param regenerate_vectors: If true, regenerates vector data even if it already exists. Defaults to False.
    :type regenerate_vectors: bool, optional
    :param compute_vectors_from_preprocessed: If true, computes vectors using preprocessed text (with 
        capitalization and punctuation removed). Defaults to False.
    :type compute_vectors_from_preprocessed: bool, optional
    :param custom_liwc_dictionary_path: This is the path of the user's own LIWC dictionary file (.dic). Defaults to empty string.
    :type custom_liwc_dictionary_path: str, optional
    :param convo_aggregation: If true, aggregates features at the conversational level. Defaults to True.
    :type convo_aggregation: bool, optional
    :param convo_methods: Specifies which aggregation functions (e.g., mean, stdev) to use at the 
        conversational level. Defaults to ['mean', 'max', 'min', 'stdev'].
    :type convo_methods: list, optional
    :param convo_columns: Specifies which columns (at the utterance/chat level) to aggregate for the 
        conversational level. Defaults to all numeric columns.
    :type convo_columns: list, optional
    :param user_aggregation: If true, aggregates features at the speaker/user level. Defaults to True.
    :type user_aggregation: bool, optional
    :param user_methods: Specifies which functions to aggregate with (e.g., mean, stdev) at the user level. 
        Defaults to ['mean', 'max', 'min', 'stdev'].
    :type user_methods: list, optional
    :param user_columns: Specifies which columns (at the utterance/chat level) to aggregate for the 
        speaker/user level. Defaults to all numeric columns.
    :type user_columns: list, optional
    :param use_gpu: Specifies whether to use GPU for vert/bert model. Defaults to False.
    :type use_gpu: bool, optional
    :param corr_thresh: Minimum absolute Spearman correlation used to treat two numeric
        columns as redundant during summary reduction. Defaults to 0.9.
    :type corr_thresh: float, optional
    :param min_na_ratio: Threshold for dropping numeric columns with high missing-value
        ratio during summary reduction. Defaults to 0.3.
    :type min_na_ratio: float, optional
    :param min_zero_ratio: Threshold for dropping numeric columns with high zero ratio
        during summary reduction. Defaults to 0.9.
    :type min_zero_ratio: float, optional
    :param min_group_size: Minimum connected-component size to treat a correlated set
        of columns as a redundancy group. Defaults to 2.
    :type min_group_size: int, optional
    :param treat_zero_as_na: If true, zeros are treated as missing values when computing
        redundancy metrics and selecting representative columns. Defaults to True.
    :type treat_zero_as_na: bool, optional
    :param drop_redundant_columns: If true, chat/user/conversation outputs are reduced to
        representative numeric columns based on summary statistics. Defaults to False.
    :type drop_redundant_columns: bool, optional
    :return: The FeatureBuilder writes the generated features to files in the specified paths. The progress 
        will be printed in the terminal, indicating completion with "All Done!".
    :rtype: None
    """
    def __init__(
            self, 
            input_df: pd.DataFrame, 
            vector_directory: str = "./vector_data/",
            output_file_base: str = "output",
            output_file_path_chat_level: str = None, 
            output_file_path_user_level: str = None,
            output_file_path_conv_level: str = None,
            custom_features: list = [],
            # analyze_first_pct: list = [1.0],
            turns: bool = False,
            conversation_id_col: str = "conversation_num",
            speaker_id_col: str = "speaker_nickname",
            message_col: str = "message",
            timestamp_col: str | tuple[str, str] = "timestamp",
            timestamp_unit = "ms",
            grouping_keys: list = [],
            cumulative_grouping = False, 
            within_task = False,
            ner_training_df: pd.DataFrame = None,
            ner_cutoff: float = 0.9,
            regenerate_vectors: bool = False,
            compute_vectors_from_preprocessed: bool = False,
            custom_liwc_dictionary_path: str = '',
            convo_aggregation = True,
            convo_methods: list = ['mean', 'max', 'min', 'stdev'],
            convo_columns: list = None,
            user_aggregation = True,
            user_methods: list = ['mean', 'max', 'min', 'stdev'],
            user_columns: list = None,
            use_gpu: bool = False,
            corr_thresh: float = 0.9, 
            min_na_ratio: float = 0.3, 
            min_zero_ratio: float = 0.9, 
            min_group_size: int = 2,
            treat_zero_as_na: bool = True,
            drop_redundant_columns: bool = False
        ) -> None:

        ###### Initialization ######
        # Ensure output_file_base only contains alphanumeric characters and underscores.
        self.output_file_base = re.sub('[^A-Za-z0-9_]', '', output_file_base)
        if self.output_file_base != output_file_base:
            output_file_base = re.sub('[^A-Za-z0-9_]', '', output_file_base)
            warnings.warn("WARNING: Special characters detected in output_file_base. These characters have been automatically removed.")
        # Determine a human-readable identifier for this run (used in log headers).
        # Prefer the distinct output file name; fall back to output_file_base if no path is given.
        if output_file_path_chat_level:
            self.file_base_name = re.sub(r'(_chat_level|_level_chat|_turn_level|_level_turn|\.csv)', '', output_file_path_chat_level.split("/")[-1])
        else:
            self.file_base_name = self.output_file_base
        # Set up logging
        self.logger = setup_logger(name="feature_builder_logger", log_file_path=f"./{self.output_file_base}/logs/feature_builder.log")
        self.summ_logger = setup_logger(name="summary_details_logger", log_file_path=f"./{self.output_file_base}/logs/summary_details.log")
        # Check that input is a dataframe
        if not isinstance(input_df, pd.DataFrame):
            self.logger.error(f"Expected a Pandas DataFrame as input_df, but got {type(input_df).__name__}")
            raise TypeError(f"Expected a Pandas DataFrame as input_df, but got {type(input_df).__name__}")
        input_df = input_df.reset_index(drop=True) # reset index to avoid issues with indexing later on
        print("Initializing Featurization...")
        self.logger.info(f"=== Start Initializing FeatureBuilder for {self.file_base_name}.csv ===")
        
        ###### Set all parameters ######
        # assert(all(0 <= x <= 1 for x in analyze_first_pct)) # first, type check that this is a list of numbers between 0 and 1
        # self.first_pct = analyze_first_pct # Set first pct of conversation you want to analyze
        self.turns = turns
        self.conversation_id_col = conversation_id_col
        self.speaker_id_col = speaker_id_col
        self.message_col = message_col
        self.timestamp_col = timestamp_col
        self.timestamp_unit = timestamp_unit
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
        self.regenerate_vectors = regenerate_vectors
        self.convo_aggregation = convo_aggregation
        self.convo_methods = convo_methods
        self.convo_columns = convo_columns
        self.user_aggregation = user_aggregation
        self.user_methods = user_methods
        self.user_columns = user_columns
        self.use_gpu = use_gpu
        self.corr_thresh = corr_thresh
        self.min_na_ratio = min_na_ratio
        self.min_zero_ratio = min_zero_ratio
        self.min_group_size = min_group_size
        self.treat_zero_as_na = treat_zero_as_na
        self.drop_redundant_columns = drop_redundant_columns
        # Defining input and output paths.
        self.chat_data = input_df.copy()
        self.orig_data = input_df.copy()
        self.ner_training = ner_training_df
        self.vector_directory = vector_directory
        self.custom_liwc_dictionary = self.load_custem_liwc_dict(custom_liwc_dictionary_path)
        # Set features to generate
        # TODO --- think through more carefully which ones we want to exclude and why
        self.feature_dict = feature_dict
        self.feature_names = [
            ### Chat Level
            "Named Entity Recognition",
            "Sentiment (RoBERTa)",
            "Message Length",
            "Message Quantity",
            "Information Exchange",
            "LIWC and Other Lexicons",
            "Questions",
            "Conversational Repair",
            "Word Type-Token Ratio",
            "Proportion of First-Person Pronouns",
            "Function Word Accommodation",
            "Content Word Accommodation",
            "Hedge",
            "TextBlob Subjectivity",
            "TextBlob Polarity",
            "Positivity Z-Score",
            "Dale-Chall Score",
            "Time Difference",
            "Politeness Strategies",
            "Politeness / Receptiveness Markers",
            "Certainty",
            "Online Discussion Tags",
            ### Conversation Level
            "Turn-Taking Index",
            "Equal Participation",
            "Team Burstiness", #TODO add dependencies
            "Conversation Level Aggregates",
            "User Level Aggregates",
            "Information Diversity",
            "Conversation Level Aggregates",
            "User Level Aggregates"
        ]
        # warning if user added invalid custom/exclude features
        invalid_features = set()
        for feat in custom_features:
            if feat in self.feature_dict: # TODO: check dependencies
                self.feature_names.append(feat)
            else:
                invalid_features.add(feat)
        if invalid_features:
            invalid_features_str = ', '.join(invalid_features)
            print(f"WARNING: Invalid custom features provided. Ignoring `{invalid_features_str}`.")
            self.logger.warning(f"WARNING: Invalid custom features provided. Ignoring `{invalid_features_str}`.")
        # remove named entities if we didn't pass in the column
        if self.ner_training is None:
            self.feature_names.remove("Named Entity Recognition")
        # remove timestamp-related features if we didn't pass in the column
        timestamp_features = ['Time Difference', "Team Burstiness"]
        if isinstance(self.timestamp_col, str):
            if self.timestamp_col not in self.chat_data.columns:
                for feat in timestamp_features:
                    self.feature_names.remove(feat)
            else:
                # verify timestamp format
                self.verify_timestamp_format(self.timestamp_col)
        elif isinstance(self.timestamp_col, tuple):
            timestamp_start, timestamp_end = self.timestamp_col
            if not {timestamp_start, timestamp_end}.issubset(self.chat_data.columns):
                for feat in timestamp_features:
                    self.feature_names.remove(feat)
            else:
                # verify timestamp format
                self.verify_timestamp_format(timestamp_start)
                self.verify_timestamp_format(timestamp_end)
            
        # deduplicate functions and append them into a list for calculation
        self.feature_methods_chat = []
        self.feature_methods_conv = []
        need_sentence = False
        need_sentiment = False
        for feature in self.feature_names:
            if(not need_sentence and feature_dict[feature]["vect_data"]):
                need_sentence = True
            if(not need_sentiment and feature_dict[feature]["bert_sentiment_data"]):
                need_sentiment = True
            level, func = self.feature_dict[feature]["level"], self.feature_dict[feature]['function']
            if level == 'Chat':
                if func not in self.feature_methods_chat:
                    self.feature_methods_chat.append(func)
            elif level == 'Conversation':
                if func not in self.feature_methods_conv:
                    self.feature_methods_conv.append(func)

        # drop all columns that are in our generated feature set --- we don't want to create confusion!
        chat_features = list(itertools.chain(*[self.feature_dict[feature]["columns"] for feature in self.feature_dict.keys() if self.feature_dict[feature]["level"] == "Chat"]))
        if self.custom_liwc_dictionary:
            chat_features += [lexicon_type + "_lexical_wordcount_custom" for lexicon_type in self.custom_liwc_dictionary.keys()]
        columns_to_drop = [col for col in chat_features if col in self.chat_data.columns]
        self.chat_data = self.chat_data.drop(columns=columns_to_drop)
        self.orig_data = self.orig_data.drop(columns=columns_to_drop)

        if compute_vectors_from_preprocessed:
            self.vector_colname = self.message_col # because the message col will eventually get preprocessed
        else:
            self.vector_colname = self.message_col + "_original" # because this contains the original message

        self.preprocess_chat_data()

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

        # Use the output_file_base parameter to auto-generate paths (since we have a lot of assumptions in how the output path looks)
        self.output_file_path_chat_level = output_file_path_chat_level
        self.output_file_path_conv_level = output_file_path_conv_level
        self.output_file_path_user_level = output_file_path_user_level

        

        if self.output_file_path_chat_level is None:
            self.output_file_path_chat_level = "./" + self.output_file_base + "_chat_level.csv"
        if self.output_file_path_conv_level is None:
            self.output_file_path_conv_level = "./" + self.output_file_base + "_conv_level.csv"
        if self.output_file_path_user_level is None:
            self.output_file_path_user_level = "./" + self.output_file_base + "_user_level.csv"

        # Basic error detetection
        if not bool(self.output_file_path_conv_level) or not bool(re.sub('[^A-Za-z0-9_]', '', self.output_file_path_conv_level)):
            self.logger.error("ERROR: Improper conversation-level output file name detected.")
            raise ValueError("ERROR: Improper conversation-level output file name detected.")
        if not bool(self.output_file_path_user_level) or not bool(re.sub('[^A-Za-z0-9_]', '', self.output_file_path_user_level)):
            self.logger.error("ERROR: Improper user (speaker)-level output file name detected.")
            raise ValueError("ERROR: Improper user (speaker)-level output file name detected.")

        # We assume that the base file name is the last item in the output path; we will use this to name the stored vectors.
        if ('/' not in self.output_file_path_chat_level or 
            '/' not in self.output_file_path_conv_level or 
            '/' not in self.output_file_path_user_level):
            self.logger.error(
                "We expect you to pass a path in for your output files "
                "(output_file_path_chat_level, output_file_path_user_level, and "
                "output_file_path_conv_level). If you would like the output to be "
                "the current directory, please append './' to the beginning of your "
                "filename(s). Your filename should be in the format: "
                "path/to/output_name.csv or ./output_name.csv for the current working directory."
            )
            raise ValueError(
                "We expect you to pass a path in for your output files "
                "(output_file_path_chat_level, output_file_path_user_level, and "
                "output_file_path_conv_level). If you would like the output to be "
                "the current directory, please append './' to the beginning of your "
                "filename(s). Your filename should be in the format: "
                "path/to/output_name.csv or ./output_name.csv for the current working directory."
            )

        try:
            base_file_name = self.output_file_path_chat_level.split("/")[-1]
        except:
            self.logger.error("ERROR: Improper chat-level output file name detected.")
            raise ValueError("ERROR: Improper chat-level output file name detected.") 

        if not bool(base_file_name) or not bool(re.sub('[^A-Za-z0-9_]', '', base_file_name)): # user didn't specify a file name, or specified one with only nonalphanumeric chars
            self.logger.error("ERROR: Improper chat-level output file name detected.")
            raise ValueError("ERROR: Improper chat-level output file name detected.")

        try:
            folder_type_name = self.output_file_path_chat_level.split("/")[-2]
        except IndexError: # user didn't specify a folder, so we will have to append it for them
            folder_type_name = "turn" if self.turns else "chat"
            self.output_file_path_chat_level = '/'.join(self.output_file_path_chat_level.split("/")[:-1]) + '/' + folder_type_name + '/' + base_file_name

        # We check whether the second to last item is a "folder type": either chat or turn.
        if folder_type_name not in ["chat", "turn"]: # user didn't specify the folder type, so we will append it for them
            folder_type_name = "turn" if self.turns else "chat"
            self.output_file_path_chat_level = '/'.join(self.output_file_path_chat_level.split("/")[:-1]) + '/' + folder_type_name + '/' + base_file_name

        # Set file paths, ensuring correct subfolder type is added.
        self.output_file_path_chat_level = re.sub(r'chat', r'turn', self.output_file_path_chat_level) if self.turns else self.output_file_path_chat_level
        if self.output_file_path_chat_level.split(".")[-1] != "csv": 
            self.output_file_path_chat_level = self.output_file_path_chat_level + ".csv"
        if not re.match(r"(.*\/|^)conv\/", self.output_file_path_conv_level):
            self.output_file_path_conv_level = "/".join(self.output_file_path_conv_level.split("/")[:-1]) + "/conv/" + self.output_file_path_conv_level.split("/")[-1]
        if self.output_file_path_conv_level.split(".")[-1] != "csv": 
            self.output_file_path_conv_level = self.output_file_path_conv_level + ".csv"
        if not re.match(r"(.*\/|^)user\/", self.output_file_path_user_level):
            self.output_file_path_user_level = "/".join(self.output_file_path_user_level.split("/")[:-1]) + "/user/" + self.output_file_path_user_level.split("/")[-1]
        if self.output_file_path_user_level.split(".")[-1] != "csv": 
            self.output_file_path_user_level = self.output_file_path_user_level + ".csv"

        # Ensure output/ is added before the subfolder.
        if not re.match(r"(.*\/|^)output\/", self.output_file_path_chat_level):
            self.output_file_path_chat_level = re.sub(r'/' + folder_type_name + r'/', r'/output/' + folder_type_name + r'/', self.output_file_path_chat_level)
        if not re.match(r"(.*\/|^)output\/", self.output_file_path_conv_level):
            self.output_file_path_conv_level = re.sub(r'/conv/', r'/output/conv/', self.output_file_path_conv_level)
        if not re.match(r"(.*\/|^)output\/", self.output_file_path_user_level):
            self.output_file_path_user_level = re.sub(r'/user/', r'/output/user/', self.output_file_path_user_level)

        # Logic for processing vector cache
        self.vect_path = vector_directory + "sentence/" + ("turns" if self.turns else "chats") + "/" + base_file_name        
        self.bert_path = vector_directory + "sentiment/" + ("turns" if self.turns else "chats") + "/" + base_file_name

        check_embeddings(self.chat_data, self.vect_path, self.bert_path, need_sentence, need_sentiment, self.regenerate_vectors, self.use_gpu, self.vector_colname, self.logger)

        if(need_sentence):
            self.vect_data = pd.read_csv(self.vect_path, encoding='mac_roman')
        else:
            self.vect_data = None

        if(need_sentiment):
            self.bert_sentiment_data = pd.read_csv(self.bert_path, encoding='mac_roman')
        else:
            self.bert_sentiment_data = None

        # Deriving the base conversation level dataframe.
        self.conv_data = self.chat_data[[self.conversation_id_col]].drop_duplicates()
        print("Initialization Complete.")
        self.logger.info(f"=== Initialization Complete for {self.file_base_name}.csv ===")
        self.logger.info("")
    
    
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

    def featurize(self) -> None:
        """
        Main driver function for feature generation.

        This function creates chat-level features, generates features for different 
        truncation percentages of the data if specified, and produces user-level and 
        conversation-level features. Finally, the features are saved into the 
        designated output files.

        :return: None
        :rtype: None
        """
        # Log start of run
        start_time = perf_counter()
        self.logger.info(f"=== Team Communication Toolkit FeatureBuilder Run initiated for {self.file_base_name}.csv ===")
        self.logger.info(f"Featurize started at {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        num_lines = self.chat_data.shape[0]
        num_speakers = self.chat_data[self.speaker_id_col].nunique()
        num_conversations = self.chat_data[self.conversation_id_col].nunique()
        self.logger.info(f"Data file has {num_lines} lines (chats), {num_speakers} unique speakers, {num_conversations} unique conversations.")
        
        # Step 1. Create chat level features.
        print("Chat Level Features ...")
        self.logger.info("--- Chat Level Features ---")
        self.chat_level_features()

        # Things to store before we loop through truncations
        self.chat_data_complete = self.chat_data # store complete chat data
        self.output_file_path_user_level_original = self.output_file_path_user_level
        self.output_file_path_chat_level_original = self.output_file_path_chat_level
        self.output_file_path_conv_level_original = self.output_file_path_conv_level

        # Step 2.
        # Run the chat-level features once, then produce different summaries based on 
        # user specification.
        # for percentage in self.first_pct: 
            # Reset chat, conv, and user objects
        self.chat_data = self.chat_data_complete
        self.user_data = self.chat_data[[self.conversation_id_col, self.speaker_id_col]].drop_duplicates()
        self.set_self_conv_data()

            # print("Generating features for the first " + str(percentage*100) + "% of messages...")
            # self.logger.info("Generating features for the first " + str(percentage*100) + "% of messages...")
            # self.get_first_pct_of_chat(percentage)
            
            # update output paths based on truncation percentage to save in a designated folder
            # if percentage != 1: # special folders for when the percentage is partial
            #     self.output_file_path_user_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_user_level_original)
            #     self.output_file_path_chat_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_chat_level_original)
            #     self.output_file_path_conv_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_conv_level_original)
            # else:
        self.output_file_path_user_level = self.output_file_path_user_level_original
        self.output_file_path_chat_level = self.output_file_path_chat_level_original
        self.output_file_path_conv_level = self.output_file_path_conv_level_original
            
        # Make it possible to create folders if they don't exist
        Path(self.output_file_path_user_level).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_file_path_chat_level).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_file_path_conv_level).parent.mkdir(parents=True, exist_ok=True)
            
        # Store column names of what we generated, so that the user can easily access them
        self.chat_features = list(itertools.chain(*[feature_dict[feature]["columns"] for feature in self.feature_names if feature_dict[feature]["level"] == "Chat"]))
        if self.custom_liwc_dictionary:
            self.chat_features += [lexicon_type + "_lexical_wordcount_custom" for lexicon_type in self.custom_liwc_dictionary.keys()]
        self.conv_features_base = list(itertools.chain(*[feature_dict[feature]["columns"] for feature in self.feature_names if feature_dict[feature]["level"] == "Conversation"]))
            
        # Step 3a. Create user level features.
        print("Generating User Level Features ...")
        self.logger.info("--- User Level Features ---")
        self.user_level_features()

        # Step 3b. Create conversation level features.
        print("Generating Conversation Level Features ...")
        self.logger.info("--- Conversation Level Features ---")
        self.conv_level_features()
        self.merge_conv_data_with_original()
        
        # Step 4. Write the features into the files defined in the output paths.
        self.conv_features_all =  [col for col in self.conv_data if col not in list(self.orig_data.columns) + ["conversation_num", self.message_col + "_original", "message_lower_with_punc"]] # save the column names that we generated!
        end_time = perf_counter()
        print("All Done!")
        self.logger.info(f"=== Featurization Completed for {self.file_base_name}.csv in {end_time - start_time:.2f} seconds! ===")
        self.logger.info("")
        
        self.logger.info(f"=== Feature Output Summary for {self.file_base_name}.csv (Please see summary_details.log for all the details) ===")
        self.logger.info("--- Chat Level ---")
        chat_data_reduced = self.generate_summary_stats(self.chat_data)
        if self.drop_redundant_columns:
            self.chat_data = chat_data_reduced
        self.logger.info("--- Conversation Level ---")
        conv_data_reduced = self.generate_summary_stats(self.conv_data)
        if self.drop_redundant_columns:
            self.conv_data = conv_data_reduced
        self.logger.info("--- User Level ---")
        user_data_reduced = self.generate_summary_stats(self.user_data)
        if self.drop_redundant_columns:
            self.user_data = user_data_reduced

        self.save_features()

    def preprocess_chat_data(self) -> None:
        """
        Call all preprocessing modules needed to clean the chat text.

        This function groups the chat data as specified, verifies column presence, creates original and lowercased columns, preprocesses text, and optionally processes chat turns.
        
        :return: None
        :rtype: None
        """
        # check grouping rules and assert the columns are present
        for role, col in self.column_names.items():
            if col not in self.chat_data.columns:
                if role == 'conversation_id_col':
                    if len(self.grouping_keys) == 0:
                        if self.conversation_id_col == "conversation_num":
                            raise KeyError("Conversation identifier not present in data. Did you perhaps forget to pass in a `conversation_id_col`?")
                        raise KeyError("Conversation identifier not present in data.")
                elif role == 'timestamp_col':
                    if self.cumulative_grouping and len(self.grouping_keys) == 3:
                        raise KeyError(f"Timestamp column is required for cumulative grouping. Please provide a valid timestamp column.")
                else:
                    raise KeyError(f"Missing required columns in DataFrame: '{col}' (expected for {role})")
            else:
                print(f"Confirmed that data has {role} column: {col}!")
                self.chat_data[col] = self.chat_data[col].fillna('')

        if self.cumulative_grouping and len(self.grouping_keys) == 0:
            warnings.warn("WARNING: No grouping keys provided. Ignoring `cumulative_grouping` argument.")
            self.cumulative_grouping = False
        if self.cumulative_grouping and len(self.grouping_keys) != 3:
            warnings.warn("WARNING: Can only perform cumulative grouping for three-layer nesting. Ignoring cumulative command and grouping by unique combinations in the grouping_keys.")
            self.cumulative_grouping = False
            self.conversation_id_col = "conversation_num"
        if self.cumulative_grouping and self.conversation_id_col not in self.grouping_keys:
            raise ValueError("Conversation identifier for cumulative grouping must be one of the grouping keys.")
        if self.grouping_keys and not self.cumulative_grouping and self.conversation_id_col != "conversation_num":
            warnings.warn("WARNING: When grouping by the unique combination of a list of keys (`grouping_keys`), the conversation identifier must be auto-generated (`conversation_num`) rather than a user-provided column. Resetting conversation_id.")
            self.conversation_id_col = "conversation_num"

        # create the appropriate grouping variables and assert the columns are present
        self.chat_data = preprocess_conversation_columns(self.chat_data, self.column_names, self.grouping_keys, self.cumulative_grouping, self.within_task)
        self.chat_data = remove_unhashable_cols(self.chat_data, self.column_names)
        self.orig_data = remove_unhashable_cols(self.orig_data, self.column_names, warning=False) # remove unhashable columns from the original data too to avoid issues with drop_duplicates

        # save original column with no preprocessing
        self.chat_data[self.message_col + "_original"] = self.chat_data[self.message_col]

        # create new column that retains punctuation
        self.chat_data["message_lower_with_punc"] = self.chat_data[self.message_col].astype(str).apply(preprocess_text_lowercase_but_retain_punctuation)
    
        # Preprocessing the text in `message_col` and then overwriting the column `message_col`.
        # TODO: We should probably use classes to abstract preprocessing module as well?
        self.chat_data[self.message_col] = self.chat_data[self.message_col].astype(str).apply(preprocess_text)

        if (self.turns):
            self.chat_data = preprocess_naive_turns(self.chat_data, self.column_names)

        # Save the preprocessed data (so we don't have to do this again)
        self.preprocessed_data = self.chat_data

        # set new identifier column for cumulative grouping.
        if self.cumulative_grouping and len(self.grouping_keys) == 3:
            warnings.warn("NOTE: User has requested cumulative grouping. Auto-generating the key `conversation_num` as the conversation identifier for cumulative conversations.")
            self.conversation_id_col = "conversation_num"

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
            timestamp_col = self.timestamp_col,
            timestamp_unit = self.timestamp_unit,
            custom_liwc_dictionary = self.custom_liwc_dictionary,
            logger = self.logger
        )
        # Calling the driver inside this class to create the features.
        self.chat_data = chat_feature_builder.calculate_chat_level_features(self.feature_methods_chat)
        # Remove special characters in column names
        self.chat_data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in self.chat_data.columns]

    # def get_first_pct_of_chat(self, percentage) -> None:
    #     """
    #     Truncate each conversation to the first X% of rows.

    #     This function groups the chat data by `conversation_num` and retains only 
    #     the first X% of rows for each conversation.

    #     :param percentage: Percentage of rows to retain in each conversation
    #     :type percentage: float

    #     :return: None
    #     :rtype: None
    #     """
    #     chat_grouped = self.chat_data.groupby(self.conversation_id_col)
    #     num_rows_to_retain = pd.DataFrame(np.ceil(chat_grouped.size() * percentage)).reset_index()
    #     chat_truncated = pd.DataFrame()
    #     for conversation_num, num_rows in num_rows_to_retain.itertuples(index=False):
    #         chat_truncated = pd.concat([chat_truncated,chat_grouped.get_group(conversation_num).head(int(num_rows))], ignore_index = True)

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
            user_aggregation = self.user_aggregation,
            user_methods = self.user_methods,
            user_columns = self.user_columns,
            chat_features = self.chat_features,
            logger=self.logger
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
            convo_aggregation = self.convo_aggregation,
            convo_methods = self.convo_methods,
            convo_columns = self.convo_columns,
            user_aggregation = self.user_aggregation,
            user_methods = self.user_methods,
            user_columns = self.user_columns,
            chat_features = self.chat_features,
            logger=self.logger
        )
        # Calling the driver inside this class to create the features.
        self.conv_data = conv_feature_builder.calculate_conversation_level_features(self.feature_methods_conv)

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
    
    def load_custem_liwc_dict(self, custom_liwc_dictionary_path: str) -> dict:
        """
        Load the custom LIWC dictionary from the provided path.

        This function reads the custom LIWC dictionary from the provided path and returns
        the parsed dictionary. If the path is empty/invalid, returns an empty dict.

        :param custom_liwc_dictionary_path: Path to the custom LIWC dictionary file.
        :type custom_liwc_dictionary_path: str

        :return: Custom LIWC dictionary
        :rtype: dict
        """
        if not custom_liwc_dictionary_path:
            return {}
        else:
            # Read .dic file if the path is provided
            custom_liwc_dictionary_path = Path(custom_liwc_dictionary_path)
            if not custom_liwc_dictionary_path.exists():
                warnings.warn(f"WARNING: The custom LIWC dictionary file does not exist: {custom_liwc_dictionary_path}")
                return {}
            elif not custom_liwc_dictionary_path.suffix == '.dic':
                warnings.warn(f"WARNING: The custom LIWC dictionary file is not a .dic file: {custom_liwc_dictionary_path}")
                return {}
            else:
                with open(custom_liwc_dictionary_path, 'r', encoding='utf-8-sig') as file:
                    dicText = file.read()
                    try:
                        return load_liwc_dict(dicText)
                    except Exception as e:
                        warnings.warn(f"WARNING: Failed loading custom liwc dictionary: {e}")
                        return {}
    
    def verify_timestamp_format(self, timestamp_col) -> None:
        """
        Verifies that a column in a DataFrame is composed of values that can be parsed
        either as datetime or as numeric values suitable for time difference calculations.

        :param timestamp_col: The name of the column to verify.
        :type timestamp_col: str

        :return: None
        :rtype: None
        :raises ValueError: If the column contains values that cannot be parsed as datetime or numeric.
        """
        series = self.chat_data[timestamp_col]
        if series.isnull().any():
            raise ValueError(f"Timestamp column '{timestamp_col}' contains null values")
        
        try:
            pd.to_datetime(series)
            return
        except Exception:
            pass

        try:
            pd.to_numeric(series)
            return
        except Exception:
            pass

        raise ValueError(
            f"Column '{timestamp_col}' contains values that are neither parseable as datetime "
            f"nor convertible to numeric format."
        )
    
    def log_column_groups(self, groups, max_groups, max_cols_per_group):
        """
        Log correlated feature groups to standard and detailed loggers.

        :param groups: Correlated column groups.
        :type groups: list[list[str]]
        :param max_groups: Maximum number of groups to print to the standard logger.
        :type max_groups: int
        :param max_cols_per_group: Maximum number of columns shown per group in
            the standard logger.
        :type max_cols_per_group: int

        :return: None
        :rtype: None
        """
        total_groups = len(groups)
        self.logger.info("Found %s correlated feature groups", total_groups)
        for i, group in enumerate(groups[:max_groups], 1):
            size = len(group)
            if size > max_cols_per_group:
                shown = ", ".join(group[:max_cols_per_group])
                self.logger.info(
                    "[Group %02d | size=%d] %s ... (+%d more)",
                    i, size, shown, size - max_cols_per_group
                )
            else:
                self.logger.info(
                    "[Group %02d | size=%d] %s",
                    i, size, ", ".join(group)
                )
        if total_groups > max_groups:
            self.logger.info(
                "... (%d more groups not shown)",
                total_groups - max_groups
            )
        self.summ_logger.info("Full correlated feature groups output:")
        for i, group in enumerate(groups, 1):
            self.summ_logger.info(
                "[Group %02d | size=%d] %s",
                i, len(group), ", ".join(group)
            )

    def keep_one_column_per_group(self, df, groups):
        """
        Select one representative column from each correlated group.

        Non-grouped columns are preserved, and grouped columns are reduced to the
        best-scoring representative based on valid-count and variance.

        :param df: Original dataframe.
        :type df: pd.DataFrame
        :param groups: Groups of similar columns.
        :type groups: list[list[str]]

        :return: Final list of columns to keep.
        :rtype: list[str]
        """
        grouped_cols = set()
        representative_map = {}
        kept_group_cols = []

        for group in groups:
            grouped_cols.update(group)

            def score(col):
                s = df[col]
                if self.treat_zero_as_na:
                    valid_count = ((~s.isna()) & (s != 0)).sum()
                else:
                    valid_count = s.notna().sum()

                variance = s.replace(0, pd.NA).dropna().var() if self.treat_zero_as_na else s.dropna().var()
                variance = 0 if pd.isna(variance) else variance

                return (valid_count, variance)

            best_col = max(group, key=score)
            kept_group_cols.append(best_col)
            representative_map[best_col] = [c for c in group if c != best_col]

        ungrouped_cols = [c for c in df.columns if c not in grouped_cols]

        kept_columns = ungrouped_cols + kept_group_cols
        return kept_columns

    def generate_summary_stats(self, df) -> pd.DataFrame:
        """
        Log and optionally reduce redundant numeric feature columns.

        The method identifies numeric columns with high missingness and zero rates,
        discovers highly correlated feature groups, and retains one representative
        per group. Non-numeric columns are preserved and reattached before return.

        :param df: Input dataframe to summarize and optionally reduce.
        :type df: pd.DataFrame

        :return: Dataframe with non-numeric columns plus filtered numeric columns.
        :rtype: pd.DataFrame
        """
        # only analyze columns generated by the FeatureBuilder (i.e., not in the original input data);
        # original columns and non-numeric columns are preserved untouched.
        original_cols = [c for c in self.orig_data.columns if c in df.columns]
        df_reduced = df.drop(columns=original_cols).select_dtypes(include=[np.number])
        df_other = df.drop(columns=df_reduced.columns)

        # 1. list columns with lots of NAs
        na_ratio = df_reduced.isna().mean()
        cols_with_many_nas = na_ratio[na_ratio > self.min_na_ratio].index.tolist()
        drop_str = " were dropped" if self.drop_redundant_columns else ""
        self.logger.info(
            f"{len(cols_with_many_nas)} columns with more than {self.min_na_ratio * 100}% NA's{drop_str}"
        )
        self.summ_logger.info(
            f"Columns with more than {self.min_na_ratio * 100}% NA's{drop_str}:\n"\
            + "\n".join(" "*30 + f"- {str(col)}" for col in cols_with_many_nas))
        df_reduced = df_reduced.drop(columns=cols_with_many_nas)

        # 2. list columns with lots of zeros
        zero_ratio = (df_reduced == 0).mean()
        cols_with_many_zeros = zero_ratio[zero_ratio > self.min_zero_ratio].index.tolist()
        self.logger.info(
            f"{len(cols_with_many_zeros)} columns with more than {self.min_zero_ratio * 100}% zeros{drop_str}"
        )
        self.summ_logger.info(
            f"Columns with more than {self.min_zero_ratio * 100}% zeros{drop_str}:\n"\
            + "\n".join(" "*30 + f"- {str(col)}" for col in cols_with_many_zeros))
        df_reduced = df_reduced.drop(columns=cols_with_many_zeros)

        # 3. cluster similar columns
        if self.treat_zero_as_na:
            df_reduced = df_reduced.replace(0, np.nan)
        corr = df_reduced.corr(method="spearman", min_periods=max(10, int(0.05 * len(df_reduced)))).abs()
        cols = corr.columns.tolist()
        graph = {col: set() for col in cols}
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and r >= self.corr_thresh:
                    a, b = cols[i], cols[j]
                    graph[a].add(b)
                    graph[b].add(a)
        visited = set()
        groups = []
        for col in cols:
            if col in visited:
                continue
            stack = [col]
            group = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.append(node)
                stack.extend(graph[node] - visited)
            if len(group) >= self.min_group_size:
                groups.append(sorted(group))
        groups.sort(key=lambda g: (-len(g), g))
        self.log_column_groups(groups, max_groups=10, max_cols_per_group=8)

        kept_columns = self.keep_one_column_per_group(df_reduced, groups)
        df_reduced = df_reduced[kept_columns]
        if self.drop_redundant_columns:
            self.logger.info("For each group of similar columns, one representative with the most valid data and highest variance was retained")
        df_final = pd.concat([df_other, df_reduced], axis=1)
        return df_final