"""
file: feature_builder.py
---
This file defines the FeatureBuilder class using the modules defined in "utils" and "features".
The intention behind this class is to use these modules and:
- Preprocess the incoming dataset defined in an input file path.
- Create chat level features -> Use the moduled in "utils" and "features" to create features 
                                on each chat message in the dataset (like word count, character count etc.).
- Create conversation level features -> These can come from 2 sources:
                                        - By aggregating the chat level features
                                        - By defining new features specifically applicable for conversations
- Save the chat and conversation level features in the output path specified.
"""

# 3rd Party Imports
import pandas as pd

# Imports from feature files and classes
from features.basic_features import *
from features.gini_coefficient import *
from features.info_exchange_zscore import *
from features.lexical_features import *

from utils.summarize_chat_level_features import *
from utils.calculate_chat_level_features import ChatLevelFeaturesCalculator
from utils.calculate_conversation_level_features import ConversationLevelFeaturesCalculator
from utils.preprocess import *

class FeatureBuilder:
    def __init__(
            self, 
            input_file_path: str, 
            output_file_path_chat_level: str, 
            output_file_path_conv_level: str
        ) -> None:
        """
            This function is used to define variables used throughout the class.

        PARAMETERS:
            @param input_file_path (str): File path of the input csv dataset (assumes that the '.csv' suffix is added)
            @param output_file_path_chat_level (str): Path where the output csv file is to be generated 
                                                      (assumes that the '.csv' suffix is added)
            @param output_file_path_conv_level (str): Path where the output csv file is to be generated 
                                                      (assumes that the '.csv' suffix is added)
        """
        #  Defining input and output paths.
        self.input_file_path = input_file_path
        self.output_file_path_chat_level = output_file_path_chat_level
        self.output_file_path_conv_level = output_file_path_conv_level

        # Reading chat level data (this is available in the input file path directly).
        self.chat_data = pd.read_csv(self.input_file_path)
        # Deriving the base conversation level dataframe.
        # This is simply the unique rows across "batch_num", and "round_num".
        # Assume that "batch_num", and "round_num" together form a primary key for this table.
        self.conv_data = self.chat_data.groupby(["batch_num", "round_num"]).sum(numeric_only = True).reset_index().iloc[: , :2]

    def featurize(self, col: str="message") -> None:
        """
            This is the main driver function of this class.
        
        PARAMETERS:
            @param col (str): (Default value: "message")
                              This is a parameter passed onto the preprocessing modules 
                              so as to identify the columns to preprocess.
        """
        # Step 1. Preprocess the relevant column (the column that has the text used to create the features).
        self.preprocess_chat_data(col=col)
        # Step 2. Create chat level features.
        self.chat_level_features()
        # Step 3. Create conversation level features.
        self.conv_level_features()
        # Step 4. Write the feartures into the files defined in the output paths.
        self.save_features()

    def preprocess_chat_data(self, col: str="message") -> None:
        """
            This function is used to call all the preprocessing modules needed to clean the text.
        
        PARAMETERS:
            @param col (str): (Default value: "message")
                              This is used to identify the columns to preprocess.
        """
        # create new column that retains punctuation
        self.chat_data["message_lower_with_punc"] = self.chat_data[col].astype(str).apply(preprocess_text_lowercase_but_retain_punctuation)
    
        # Preprocessing the text in `col` and then overwriting the column `col`.
        # TODO: We should probably use classes to abstract preprocessing module as well?
        self.chat_data[col] = self.chat_data[col].astype(str).apply(preprocess_text)

    def chat_level_features(self) -> None:
        """
            This function instantiates and uses the ChatLevelFeaturesCalculator to create the chat level features 
            and add them into the `self.chat_data` dataframe.
        """
        # Instantiating.
        chat_feature_builder = ChatLevelFeaturesCalculator(
            chat_data = self.chat_data
        )
        # Calling the driver inside this class to create the features.
        self.chat_data = chat_feature_builder.calculate_chat_level_features()

    def conv_level_features(self) -> None:
        """
            This function instantiates and uses the ConversationLevelFeaturesCalculator to create the 
            conversation level features and add them into the `self.conv_data` dataframe.
        """
        # Instantiating.
        conv_feature_builder = ConversationLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            conv_data = self.conv_data
        )
        # Calling the driver inside this class to create the features.
        self.conv_data = conv_feature_builder.calculate_conversation_level_features()

    def save_features(self) -> None:
        """
            This function simply saves the files in the respective output file paths provided during initialization.
        """
        # TODO: For now this function is very trivial. We will finilize the output formats (with date-time info etc) 
        # and control the output mechanism through this function.
        self.chat_data.to_csv(self.output_file_path_chat_level, index=False)
        self.conv_data.to_csv(self.output_file_path_conv_level, index=False)