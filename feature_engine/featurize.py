"""
file: featurize.py
---
This file is the main driver of the feature generating pipeline. 
It instantiates and calls the FeatureBuilder class which defines the logic used for feature creation.
"""

# Importing the Feature Generating Class
from feature_builder import FeatureBuilder

# Main Function
if __name__ == "__main__":
	# Instantiating the Feature Generating Class
	# feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/juries_tiny_for_testing.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/jury_TINY_output_chat_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/jury_TINY_output_conversation_level.csv"
	# )
	# # Calling the "engine"/"driver" function of the FeatureBuilder class 
	# # that creates the features, and writes them in output.
	
	# feature_builder.featurize(col="message")

	# calling feature builder for csop
	feature_builder_csop = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/csop_withblanks_output_chat_level.csv",
		output_file_path_conv_level = "../feature_engine/output/csop_withblanks_output_conversation_level.csv"
	)
	feature_builder_csop.featurize(col="message")