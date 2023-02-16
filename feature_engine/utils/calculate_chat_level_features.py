"""
file: calculate_chat_level_features.py
---
This file defines the ChatLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all chat level features here. 

The steps needed to add a feature would be to:
- First define any building blocks that the feature would need in the appropriate "features" module (like word counter).
- Define a function within the class that uses these building blocks to build the feature and appends it 
  to the chat level dataframe as columns.
- Call the feature defining function in the driver function.
"""

# Importing modules from features
from features.basic_features import *
from features.info_exchange_zscore import *
from features.lexical_features import *

class ChatLevelFeaturesCalculator:
	def __init__(self, chat_data: pd.DataFrame) -> None:
		"""
			This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
		"""
		self.chat_data = chat_data
		
	def calculate_chat_level_features(self) -> pd.DataFrame:
		"""
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The chat level dataset given to this class during initialization along with 
							new columns for each chat level feature.
		"""
		# Text-Based Basic Features
		self.text_based_features()

		# Info Exchange Feature
		self.info_exchange_feature()
		
		# lexical features
		self.lexical_features()

		# Return the input dataset with the chat level features appended (as columns)
		return self.chat_data
		
	def text_based_features(self) -> None:
		"""
			This function is used to implement the common text based featuers.
		"""
		# Count Words
		self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
		
		# Count Characters
		self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
		
		# Count Messages		
		self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
		
	def info_exchange_feature(self) -> None:
		"""
			This function helps in extracting the different types of z-scores from the chats 
			(see features/info_exchange_zscore.py to learn more about how these features are calculated).
		"""
		# Get Modified Wordcount: Total word count - first_singular pronouns
		self.chat_data["info_exchange_wordcount"] = self.chat_data["message"].apply(get_info_exchange_wordcount)
		
		# Get the z-score of each message across all chats
		self.chat_data = get_zscore_across_all_chats(self.chat_data, "info_exchange_wordcount")
		
		# Get the z-score within each conversation
		self.chat_data = get_zscore_across_all_conversations(self.chat_data, "info_exchange_wordcount")

	def lexical_features(self) -> None:
		"""
			This is a driver function that calls relevant functions in features/lexical_features.py to implement the lexical features.
		"""
		self.chat_data = pd.concat([self.chat_data, self.chat_data["message"].apply(lambda x: pd.Series(liwc_features(str(x))))], axis = 1)
