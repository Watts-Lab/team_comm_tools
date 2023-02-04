from feature_engine.features.gini_coefficient import *
from feature_engine.features.basic_features import *


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
            right=average_message_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=std_message_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        # Word mean and std
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=average_word_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=std_word_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

    def get_talkative_member_features(self):
        # Message level talkative_member_features
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=most_talkative_member_message_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=least_talkative_member_message_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        # Word level talkative_member_features
        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=most_talkative_member_word_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )

        self.conversation_level_data = pd.merge(
            left=self.conversation_level_data,
            right=least_talkative_member_word_count(self.chat_level_data),
            on=['batch_num', 'round_num'],
            how="inner"
        )