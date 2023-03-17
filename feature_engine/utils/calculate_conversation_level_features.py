"""
file: calculate_conversation_level_features.py
---
This file defines the ConvLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all conv level features here. 
"""

# Importing modules from features
from features.gini_coefficient import *
from features.basic_features import *
from utils.summarize_chat_level_features import *

class ConversationLevelFeaturesCalculator:
	def __init__(self, chat_data: pd.DataFrame, conv_data: pd.DataFrame) -> None:
		"""
			This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
			@param conv_data (pd.DataFrame): This is a pandas dataframe of the conversation level features derived from the 
											 chat level dataframe.
		"""
		# Initializing variables
		self.chat_data = chat_data
		self.conv_data = conv_data

	def calculate_conversation_level_features(self) -> pd.DataFrame:
		"""
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The conversation level dataset given to this class during initialization along with 
							new columns for each conv level feature.
		"""
		# Get gini based features
		self.get_gini_features()
		# Get summary statistics by aggregating chat level features
		self.get_conversation_level_summary_statistics_features()
		self.get_talkative_member_features()

		# TODO - currently, lexical features are not being summarized!

		return self.conv_data

	def get_gini_features(self) -> None:
		"""
			This function is used to calculate the gini index for each conversation 
			based on the word level and character level information.
		"""
		# Gini for #Words
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_gini(self.chat_data, "num_words"),
			on=['batch_num', 'round_num'],
			how="inner"
		)
		# Gini for #Characters
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_gini(self.chat_data, "num_chars"),
			on=['batch_num', 'round_num'],
			how="inner"
		)

	def get_conversation_level_summary_statistics_features(self) -> None:
		"""
			This function is used to aggregate the summary statistics from 
			chat level features to conversation level features.
			Specifically, it looks at the mean and standard deviations at message and word level.
		"""
		# Message mean and std
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'num_messages', 'average_message_count'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'num_messages', 'std_message_count'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		# Word mean and std
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'num_words', 'average_word_count'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'num_words', 'std_word_count'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
		# Info Exchange (Z-Scores)
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'info_exchange_zscore_chats', 'average_info_exchange_zscore_chats'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'info_exchange_zscore_chats', 'std_info_exchange_zscore_chats'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'info_exchange_zscore_conversation', 'average_info_exchange_zscore_conversation'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'info_exchange_zscore_conversation', 'std_info_exchange_zscore_conversation'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	   # Number of questions
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'Qnum', 'average_Qnum'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'Qnum', 'std_Qnum'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	   # Proportion of clarification questions
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'NTRI', 'average_NTRI'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'NTRI', 'std_NTRI'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	   # Word type-to-token ratio
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'word_TTR', 'average_word_TTR'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'NTRI', 'std_word_TTR'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	
	   # Proportion of first person pronouns
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'first_pronouns_proportion', 'average_first_pronouns_proportion'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'first_pronouns_proportion', 'std_first_pronouns_proportion'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	   # Function word mimicry
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'FuncWordAcc', 'average_FuncWordAcc'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'FuncWordAcc', 'std_FuncWordAcc'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	
	   # Content word mimicry
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_average(self.chat_data, 'ContWordAcc', 'average_ContWordAcc'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_stdev(self.chat_data, 'ContWordAcc', 'std_ContWordAcc'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
	

	def get_talkative_member_features(self) -> None:
		"""
			This function is used to aggregate the summary statistics from 
			chat level features to conversation level features.
			Specifically, it looks at the maximum and minimum messages and words sent out in the conversation.
		"""
		# Message level talkative_member_features
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_max(self.chat_data, 'num_messages', 'max_messages'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_min(self.chat_data, 'num_messages', 'min_messages'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		# Word level talkative_member_features
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_max(self.chat_data, 'num_words', 'max_words'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_min(self.chat_data, 'num_words', 'min_words'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		# Min and Max for Information Exchange
		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_max(self.chat_data, 'info_exchange_zscore_chats', 'max_info_exchange_zscore_chats'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_min(self.chat_data, 'info_exchange_zscore_chats', 'min_info_exchange_zscore_chats'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_max(self.chat_data, 'info_exchange_zscore_conversation', 'max_info_exchange_zscore_conversation'),
			on=['batch_num', 'round_num'],
			how="inner"
		)

		self.conv_data = pd.merge(
			left=self.conv_data,
			right=get_min(self.chat_data, 'info_exchange_zscore_conversation', 'min_info_exchange_zscore_conversation'),
			on=['batch_num', 'round_num'],
			how="inner"
		)
