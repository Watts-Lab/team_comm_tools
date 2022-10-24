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


if __name__ == "__main__":

	# import the data from the data file
	INPUT_FILE_PATH = './data/raw_data/jury_conversations_with_outcome_var.csv'
	OUTPUT_FILE_PATH = './output/jury_output.csv'

	conversation_data = pd.read_csv(INPUT_FILE_PATH)
	output_data = conversation_data

	# call features here
	# single chat-level features are applied per *message*
	# in the future, we can also generate per-conversation features -- but may need to format the input data differently
	output_data['num_words'] = output_data['message'].apply(lambda x: count_words(str(x)))
	output_data['num_chars'] = output_data['message'].apply(lambda x: count_characters(str(x)))
	

	# generate output file
	output_data.to_csv(OUTPUT_FILE_PATH)