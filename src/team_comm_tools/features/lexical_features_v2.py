"""
file: lexical_features_v2.py
---
A faster version of the lexical_features.py file.
"""
# Importing packages
import pickle
import re
import pandas as pd
import os
from pathlib import Path

def get_liwc_count(regex, chat):
	""""
	Count the number of LIWC lexicon words

	Args:
		regex (str): The regular expression for the lexicon.
		chat(str): The message (utterance) being analyzed.

	Returns:
		float: The number of lexicon words present in the message
	"""
	if(len(chat) > 0):
		return len(re.findall(regex, chat))
	else:
		return 0

def liwc_features(chat_df: pd.DataFrame, message_col_original: str, custom_liwc_dictionary: dict={}) -> pd.DataFrame:
	"""
		This function takes in the chat level input dataframe and computes lexical features 
		(the number of words from a given lexicon, such as LIWC).
			  
	Args:
		chat_df (pd.DataFrame): This is a pandas dataframe of the chat level features. Should contain 'message' column.
		message_col (str): This is a string with the name of the column containing the message / text.
		custom_liwc_dictionary (dict): This is a dictionary of the user's custom LIWC dic. 

	Returns:
		pd.DataFrame: Dataframe of the lexical features stacked as columns.
	"""
	# Load the preprocessed lexical regular expressions
	try:
		current_dir = os.path.dirname(__file__)
		lexicon_pkl_file_path = os.path.join(current_dir, './assets/lexicons_dict.pkl')
		lexicon_pkl_file_path = os.path.abspath(lexicon_pkl_file_path)
		with open(lexicon_pkl_file_path, "rb") as lexicons_pickle_file:
			lexicons_dict = pickle.load(lexicons_pickle_file)
		
		# Return the lexical features stacked as columns
		df_lst = [pd.DataFrame(chat_df[message_col_original].apply(lambda chat: get_liwc_count(regex, chat)))\
											.rename({message_col_original: lexicon_type + "_lexical_wordcount"}, axis=1)\
				for lexicon_type, regex in lexicons_dict.items()]

		if custom_liwc_dictionary:
			df_lst += [pd.DataFrame(chat_df[message_col_original].apply(lambda chat: get_liwc_count(regex, chat)))\
											.rename({message_col_original: lexicon_type + "_lexical_wordcount_custom"}, axis=1)\
				for lexicon_type, regex in custom_liwc_dictionary.items()]
		return pd.concat(df_lst, axis=1)
	except FileNotFoundError:
		print("WARNING: Lexicons not found. Skipping feature...")
	except Exception as e:
		print(f'WARNING: Failed to generate lexicons due to unexpected error: {e}')
