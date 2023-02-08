from feature_engine.features.gini_coefficient import *
from feature_engine.features.basic_features import *
from feature_engine.utils.summarize_chat_level_features import *

class ConversationLevelFeaturesCalculator:
    def __init__(self, chat_level_data):
        self.chat_level_data = chat_level_data
        self.conversation_level_data = self.chat_level_data\
                                           .groupby(['batch_num', 'round_num'])\
                                           .sum(numeric_only=True).reset_index().iloc[:, :2]

    def calculate_conversation_level_features(self):
        self.get_gini_features()
        self.get_conversation_level_summary_statistics_features()
        self.get_talkative_member_features()
        return self.conversation_level_data

    def get_gini_features(self):
        # Gini for #Words
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_gini(self.chat_level_data, "num_words"),
            on=['batch_num', 'round_num'],
            how="inner"
        )
        # Gini for #Characters
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_gini(self.chat_level_data, "num_chars"),
            on=['batch_num', 'round_num'],
            how="inner"
        )

    def get_conversation_level_summary_statistics_features(self):
        # Message mean and std
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_average(self.chat_level_data, 'num_messages', 'average_message_count'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_stdev(self.chat_level_data, 'num_messages', 'average_message_count'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        # Word mean and std
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_average(self.chat_level_data, 'num_words', 'average_word_count'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_stdev(self.chat_level_data, 'num_words', 'std_word_count'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

    def get_talkative_member_features(self):
        # Message level talkative_member_features
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_max(self.chat_level_data, 'num_messages', 'max_messages'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_min(self.chat_level_data, 'num_messages', 'min_messages'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        # Word level talkative_member_features
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_max(self.chat_level_data, 'num_words', 'max_words'),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=get_min(self.chat_level_data, 'num_words', 'min_words'),
            on=['batch_num', 'round_num'],
            how="inner"
        )