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
	# Calling the "engine"/"driver" function of the FeatureBuilder class 
	# that creates the features, and writes them in output.
	# Defines one class for each dataset.

	# TINY Test sets --- just two conversations each
	# Tiny Juries
	# feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/juries_tiny_for_testing.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/jury_TINY_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/jury_TINY_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/jury_TINY_output_conversation_level.csv",
	# 	turns = False,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# feature_builder.featurize(col="message")

	# Tiny CSOP
	# tiny_csop_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csop_conversations_TINY.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csop_TINY_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csop_TINY_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csop_TINY_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# tiny_csop_feature_builder.featurize(col="message")

	#####

	# FULL DATASETS BELOW

	# Juries
	jury_feature_builder = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
		output_file_path_chat_level = "../feature_engine/output/chat/jury_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/user/jury_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/conv/jury_output_conversation_level.csv",
		turns = True,
		analyze_first_pct = [0.25, 0.5, 0.75, 1]
	)
	jury_feature_builder.featurize(col="message")


	# CSOP (Abdullah)
	# csop_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csop_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csop_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csop_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# csop_feature_builder.featurize(col="message")


	# CSOP II (Nak Won Rim)
	# csopII_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csopII_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# csopII_feature_builder.featurize(col="message")

	

	# DAT - Divergent Association Task
	# dat_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/DAT_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# dat_feature_builder.featurize(col="message")


	# PGG (Small)
	# pgg_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/pgg_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/pgg_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/user/pgg_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/pgg_output_conversation_level.csv"
	# )
	# pgg_feature_builder.featurize(col="message")


	# Estimation (Gurcay)
	# gurcay_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/gurcay2015estimation_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# gurcay_estimation_feature_builder.featurize(col="message")

 	# Estimation (Becker)
	# becker_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/beckerestimation_output_conversation_level.csv",
	# 	turns = True,
	# 	analyze_first_pct = [0.25, 0.5, 0.75, 1]
	# )
	# becker_estimation_feature_builder.featurize(col="message")