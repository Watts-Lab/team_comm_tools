from features.basic_features import *
from features.info_exchange_zscore import *
from features.lexical_features import *

class ChatLevelFeaturesCalculator:
	def __init__(self, chat_data):
		self.chat_data = chat_data
		
	
	def calculate_chat_level_features(self):
		# Text-Based Basic Features
		self.text_based_featurs()

		# Info Exchange Feature
		self.info_exchange_feature()
		
		# lexical features
		self.lexical_features()

		# Return the input dataset with the chat level features appended (as columns)
		return self.chat_data
		

	def text_based_featurs(self):
		# Count Words
		self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
		
		# Count Characters
		self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
		
		# Count Messages		
		self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
		

	def info_exchange_feature(self):
		# Get Modified Wordcount: Total word count - first_singular pronouns
		self.chat_data["info_exchange_wordcount"] = self.chat_data["message"].apply(get_info_exchange_wordcount)
		
		# Get the z-score of each message across all chats
		self.chat_data = get_zscore_across_all_chats(self.chat_data, "info_exchange_wordcount")
		
		# Get the z-score within each conversation
		self.chat_data = get_zscore_across_all_conversations(self.chat_data, "info_exchange_wordcount")

	def lexical_features(self):
		self.chat_data = pd.concat([self.chat_data, self.chat_data["message"].apply(lambda x: pd.Series(liwc_features(str(x))))], axis = 1)
