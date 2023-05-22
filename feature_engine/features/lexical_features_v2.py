"""
file: lexical_features_v2.py
---
A faster version of the lexical_features.py file.
"""
# Importing packages
import pickle
import re
import pandas as pd

def liwc_features(chat_df: pd.DataFrame) -> pd.DataFrame:
	"""
		This function takes in the chat level input dataframe and computes the lexicon feautres for the 'message' column.
		TODO: The current implementation assumes presence of 'message' column. 
		      Might need to abstract this in order to make this generalizable for all datasets.
		      
	PARAMETERS:
		@param chat_df (pd.DataFrame): This is a pandas dataframe of the chat level features. Should contain 'message' column.

	RETURNS:
		(pd.DataFrame): Dataframe of the lexical features stacked as columns.
	"""
	# Load the preprocessed lexical regular expressions
	with open("../feature_engine/features/lexicons_dict.pkl", "rb") as lexicons_pickle_file:
		lexicons_dict = pickle.load(lexicons_pickle_file)
	
	# Return the lexical features stacked as columns
	return pd.concat(
		# Finding the # of occurances of lexicons of each type for all the messages.
		[pd.DataFrame(chat_df["message"].apply(lambda chat: len(re.findall(regex, chat))))\
			  							.rename({"message": lexicon_type}, axis=1)\
			for lexicon_type, regex in lexicons_dict.items()], 
		axis=1
	)
