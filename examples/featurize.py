"""
file: featurize.py
---
This is an example file that declares a FeatureBuilder constructor for several empirical datasets.
"""

from team_comm_tools import FeatureBuilder
import pandas as pd

# Main Function
if __name__ == "__main__":
	
	# These two are small datasets for empirical purposes ("are the lights on?")
	tiny_juries_df = pd.read_csv("./example_data/tiny_data/juries_tiny_for_testing.csv", encoding='utf-8')
	tiny_multi_task_df = pd.read_csv("./example_data/tiny_data/multi_task_TINY.csv", encoding='utf-8')

	# These three are full datasets from published papers
	juries_df = pd.read_csv("./example_data/full_empirical_datasets/jury_conversations_with_outcome_var.csv", encoding='utf-8')
	csop_df = pd.read_csv("./example_data/full_empirical_datasets/csop_conversations_withblanks.csv", encoding='utf-8')
	csopII_df = pd.read_csv("./example_data/full_empirical_datasets/csopII_conversations_withblanks.csv", encoding='utf-8')
 
	"""
	TINY / TEST DATASETS -------------------------------
	
	These are smaller versions of (real) empirical datasets for the purpose of testing and demonstration.

	Note that the three output path parameters (output_file_path_chat_level, output_file_path_user_level, and 
	output_file_path_conv_level) expect a *path*, in addition to a filename. The format should be:

	/path/to/output_filename.csv

	If you would like the output to be the current working directory, you should pass in:

	./output_filename.csv.

	Failure to provide a valid path or a valid filename will result in an error.

	"""

	# Tiny Juries
	print("Tiny Juries Example...")
	tiny_juries_feature_builder = FeatureBuilder(
		input_df = tiny_juries_df,
		grouping_keys = ["batch_num", "round_num"],
		output_file_base = "jury_TINY_output", # Naming output files using the output_file_base parameter (recommended)
		turns = False,
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"]
	)
	tiny_juries_feature_builder.featurize()

	# Tiny Juries with custom aggregations
	print("Tiny Juries with Custom Aggregation...")
	tiny_juries_feature_builder_custom_agg = FeatureBuilder(
		input_df = tiny_juries_df,
		grouping_keys = ["batch_num", "round_num"],
		output_file_base = "jury_TINY_output_custom_agg", # Naming output files using the output_file_base parameter (recommended)
		turns = False,
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"],
		convo_methods = ['max', 'median'], # This will aggregate ONLY the "positive_bert" at the conversation level, using max and median.
		convo_columns = ['positive_bert'],
		user_methods = ['mean'], # This will aggregate ONLY "negative_bert" at the speaker/user level, using mean.
		user_columns = ['negative_bert'],
	)
	tiny_juries_feature_builder_custom_agg.featurize()
 
	# Tiny multi-task
	tiny_multi_task_feature_builder = FeatureBuilder(
		input_df = tiny_multi_task_df,
		conversation_id_col = "stageId",
		# alternatively, you can name each output file separately. NOTE, however, that we don't directly use this path;
		# we modify the path to place outputs within the `output/chat`, `output/conv`, and `output/user` folders.
		output_file_path_chat_level = "./multi_task_TINY_output_chat_level_stageId_cumulative.csv",
		output_file_path_user_level = "./multi_task_TINY_output_user_level_stageId_cumulative.csv",
		output_file_path_conv_level = "./multi_task_TINY_output_conversation_level_stageId_cumulative.csv",
		turns = False
	)
	tiny_multi_task_feature_builder.featurize()

	# FULL DATASETS BELOW ------------------------------------- #
	
	# Juries
	# jury_feature_builder = FeatureBuilder(
	# 	input_df = juries_df,
	# 	grouping_keys = ["batch_num", "round_num"],
	# 	vector_directory = "./vector_data/",
	# 	output_file_path_chat_level = "./jury_output_chat_level.csv",
	# 	output_file_path_user_level = "./jury_output_user_level.csv",
	# 	output_file_path_conv_level = "./jury_output_conversation_level.csv",
	# 	turns = True,
	# 	custom_features = [
	# 		"(BERT) Mimicry",
	# 		"Moving Mimicry",
	# 		"Forward Flow",
	# 		"Discursive Diversity"]
	# )
	# jury_feature_builder.featurize()

	# CSOP (Abdullah)
	# csop_feature_builder = FeatureBuilder(
	# 	input_df = csop_df,
	# 	vector_directory = "./vector_data/",
	# 	output_file_path_chat_level = "./csop_output_chat_level.csv",
	# 	output_file_path_user_level = "./csop_output_user_level.csv",
	# 	output_file_path_conv_level = "./csop_output_conversation_level.csv",
	# 	turns = True
	# )
	# csop_feature_builder.featurize()


	# CSOP II (Nak Won Rim)
	# csopII_feature_builder = FeatureBuilder(
	# 	input_df = csopII_df,
	# 	vector_directory = "./vector_data/",
	# 	output_file_path_chat_level = "./csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "./csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "./csopII_output_conversation_level.csv",
	# 	turns = True
	# )
	# csopII_feature_builder.featurize()