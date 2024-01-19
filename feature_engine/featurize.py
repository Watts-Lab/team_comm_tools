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
	feature_builder = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/juries_tiny_for_testing.csv",
		output_file_path_chat_level = "../feature_engine/output/chat/jury_TINY_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/user/jury_TINY_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/conv/jury_TINY_output_conversation_level.csv",
		turns = False,
		analyze_first_pct = [0.25, 0.5, 0.75, 1]
	)
	feature_builder.featurize(col="message")

	# Tiny CSOP
	tiny_csop_feature_builder = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_TINY.csv",
		output_file_path_chat_level = "../feature_engine/output/chat/csop_TINY_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/user/csop_TINY_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/conv/csop_TINY_output_conversation_level.csv",
		turns = True,
		analyze_first_pct = [0.25, 0.5, 0.75, 1]
	)
	tiny_csop_feature_builder.featurize(col="message")

	# Tiny multi-task
	# tiny_multi_task_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/multi_task_TINY.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_TINY_output_chat_level_stageId_cumulative.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_TINY_output_user_level_stageId_cumulative.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_TINY_output_conversation_level_stageId_cumulative.csv",
	# 	turns = False,
	# 	conversation_id = "stageId",
	# 	cumulative_grouping = True
	# )
	# tiny_multi_task_feature_builder.featurize(col="message")

	#####

	# FULL DATASETS BELOW

	# Multi-Task Data
	# multi_task_feature_builder_cumulative = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/multi_task_conversations_with_dv_and_composition.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_output_chat_level_stageId_cumulative.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_output_user_level_stageId_cumulative.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_output_conversation_level_stageId_cumulative.csv",
	# 	turns = False,
	# 	conversation_id = "stageId",
	# 	cumulative_grouping = True
	# )
	# multi_task_feature_builder_cumulative.featurize(col="message")

	# multi_task_feature_builder_cumulative_task = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/multi_task_conversations_with_dv_and_composition.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_output_chat_level_stageId_cumulative_within_task.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_output_user_level_stageId_cumulative_within_task.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_output_conversation_level_stageId_cumulative_within_task.csv",
	# 	turns = False,
	# 	conversation_id = "stageId",
	# 	cumulative_grouping = True,
	# 	within_task = True
	# )
	# multi_task_feature_builder_cumulative_task.featurize(col="message")

	# multi_task_feature_builder_roundId_last = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/multi_task_conversations_with_dv_and_composition_dv_last_by_stage.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_output_chat_level_roundId_last.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_output_user_level_roundId_last.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_output_conversation_level_roundId_last.csv",
	# 	turns = False,
	# 	conversation_id = "roundId",
	# 	cumulative_grouping = False

	# )
	# multi_task_feature_builder_roundId_last.featurize(col="message")

	# multi_task_feature_builder_roundId_last_cumulative = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/multi_task_conversations_with_dv_and_composition_dv_last_by_stage.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_output_chat_level_roundId_last_cumulative.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_output_user_level_roundId_last_cumulative.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_output_conversation_level_roundId_last_cumulative.csv",
	# 	turns = False,
	# 	conversation_id = "roundId",
	# 	cumulative_grouping = True

	# )
	# multi_task_feature_builder_roundId_last_cumulative.featurize(col="message")

	# Juries
	# jury_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/jury_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/jury_output_conversation_level.csv",
	# 	turns = True
	# )
	# jury_feature_builder.featurize(col="message")


	# CSOP (Abdullah)
	# csop_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csop_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csop_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csop_output_conversation_level.csv",
	# 	turns = True
	# )
	# csop_feature_builder.featurize(col="message")


	# CSOP II (Nak Won Rim)
	# csopII_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csopII_output_conversation_level.csv",
	# 	turns = True
	# )
	# csopII_feature_builder.featurize(col="message")

	

	# DAT - Divergent Association Task
	# dat_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/DAT_output_conversation_level.csv",
	# 	turns = True
	# )
	# dat_feature_builder.featurize(col="message")


	# PGG (Small)
	# pgg_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/pgg_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/pgg_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/pgg_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/pgg_output_conversation_level.csv"
	# )
	# pgg_feature_builder.featurize(col="message")


	# Estimation (Gurcay)
	# gurcay_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/gurcay2015estimation_output_conversation_level.csv",
	# 	turns = True
	# )
	# gurcay_estimation_feature_builder.featurize(col="message")

 	# Estimation (Becker)
	# becker_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/beckerestimation_output_conversation_level.csv",
	# 	turns = True
	# )
	# becker_estimation_feature_builder.featurize(col="message")
