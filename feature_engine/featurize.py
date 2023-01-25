import csv
import pandas as pd

"""
file: featurize.py
---
This file is the main "engine" that takes as input a cleaned 
data file (as CSV) and outputs a bunch of generated conversational 
features (as another CSV, in the output/ folder).

Primarily, it adds columns to the input CSV, in which every column
transforms the message/conversation using one of the functions in
the features/ folder.
"""

# import our feature files
from features.basic_features import *
from features.gini_coefficient import *
from features.info_exchange_zscore import *
from features.lexical_features import *

'''
@param df = the name of the dataframe on which the operation is being applied.
	- assumes that the df is a chat-by-chat setup in which each row is 1 chat.
	- assumes that the chat is stored in a column called 'message'
@param feature_name = the name of the column you want the feature to be named
@param function_name = the name of the function used to create the feature
'''
def create_chat_level_feature(df, feature_name, function_name):
	df[feature_name] = df['message'].apply(lambda x: function_name(str(x)))
	return(df)


# Main Function
if __name__ == "__main__":

	# import the data from the data file
	# TODO: See GitHub Issue #56 - the LIWC lexicons run way too slowly.
	# playing with a tiny version for testing, as otherwise it hangs.
	#INPUT_FILE_PATH = './data/raw_data/jury_conversations_with_outcome_var.csv'
	INPUT_FILE_PATH = './data/raw_data/juries_tiny_for_testing.csv'
	OUTPUT_FILE_PATH_CHAT_LEVEL = './output/jury_TINY_output_chat_level.csv'

	conversation_data = pd.read_csv(INPUT_FILE_PATH)
	output_data_chats = conversation_data


	### CHAT-LEVEL FEATURES --------------------------------------------------------------
	'''
	chat-level features take place on the single-utterance level. Each person will make
	many utterances throughout a conversation.
	'''
	# text-based basic features
	output_data_chats = create_chat_level_feature(output_data_chats, "num_words", count_words)
	output_data_chats = create_chat_level_feature(output_data_chats, "num_chars", count_characters)
	output_data_chats = create_chat_level_feature(output_data_chats, "info_exchange_wordcount", get_info_exchange_wordcount)
	
	#Info Exchange Feature: get zscore for each message/utterance for the 'info exchange wordcount' feature; first across the whole dataset, then within the group (batch, round)
	output_data_chats = get_zscore_across_all_chats(output_data_chats,"info_exchange_wordcount")
	output_data_chats = get_zscore_across_all_conversations(output_data_chats,"info_exchange_wordcount")
  
	# lexical features
	output_data_chats = pd.concat([output_data_chats, output_data_chats.message.apply(lambda x: pd.Series(liwc_features(str(x))))], axis = 1)


	# generate output file
	output_data_chats.to_csv(OUTPUT_FILE_PATH_CHAT_LEVEL)


	### CONVERSATION-LEVEL FEATURES --------------------------------------------------------------
	OUTPUT_FILE_PATH_CONVERSATION_LEVEL = './output/jury_TINY_output_conversation_level.csv'

	'''
	conversation-level features take place at the level of the entire conversation; 
	we may need to pass the chat-level features into additional functions in order 
	to calculate conversation-level features. As an example, Gini is a conversational-
	level summary of how many chats each person in the conversation made.
	'''
	# basic output data on the conversation level -- just the batch and round nums
	output_data_conversations = conversation_data.groupby(['batch_num', 'round_num']).sum(numeric_only = True).reset_index().iloc[: , :2]
	
	# generate all conversation level features here
	output_data_conversations = pd.merge(output_data_conversations, get_gini(output_data_chats, "num_words"), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, get_gini(output_data_chats, "num_chars"), on=['batch_num', 'round_num'])

	output_data_conversations = pd.merge(output_data_conversations, average_message_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, most_talkative_member_message_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, least_talkative_member_message_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, std_message_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, average_word_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, most_talkative_member_word_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, least_talkative_member_word_count(output_data_chats), on=['batch_num', 'round_num'])
	output_data_conversations = pd.merge(output_data_conversations, std_word_count(output_data_chats), on=['batch_num', 'round_num'])



	# generate output file
	output_data_conversations.to_csv(OUTPUT_FILE_PATH_CONVERSATION_LEVEL)

