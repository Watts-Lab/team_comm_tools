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
    def __init__(self, input_file_path, output_file_path_chat_level, output_file_path_conv_level):
        self.input_file_path = input_file_path
        self.output_file_path_chat_level = output_file_path_chat_level
        self.output_file_path_conv_level = output_file_path_conv_level

        self.chat_data = pd.read_csv(self.input_file_path)
        self.conv_data = self.chat_data.groupby(["batch_num", "round_num"]).sum(numeric_only = True).reset_index().iloc[: , :2]
        pass

    def featurize(self, col="message"):
        self.preprocess_chat_data(col=col)
        self.chat_level_features()
        self.conv_level_features()
        self.save_features()

    def preprocess_chat_data(self, col="message"):
        self.chat_data[col] = self.chat_data[col].astype(str).apply(preprocess_text)
    
    def chat_level_features(self):
        chat_feature_builder = ChatLevelFeaturesCalculator(
            chat_data = self.chat_data
        )
        self.chat_data = chat_feature_builder.calculate_chat_level_features()

    def conv_level_features(self):
        conv_feature_builder = ConversationLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            conv_data = self.conv_data
        )
        self.conv_data = conv_feature_builder.calculate_conversation_level_features()

    def save_features(self):
        self.chat_data.to_csv(self.output_file_path_chat_level, index=False)
        self.conv_data.to_csv(self.output_file_path_conv_level, index=False)