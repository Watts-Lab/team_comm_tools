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
 
	# test vector dataset
	test_vector_df = pd.read_csv("../tests/data/cleaned_data/test_vector.csv", encoding='utf-8')
 
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

	# Test Vectors
	# print("Testing vectors valid ...")
	# test_vector_feature_builder = FeatureBuilder(
	# 	input_df = test_vector_df,
  	# 	output_file_base = "test_vector",
	# 	custom_vect_path = "../tests/vector_data/sentence/chats/test_vector_valid.csv",
	# 	turns = False,
	# )
	# test_vector_feature_builder.featurize()
	
	valid_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vector_row_mismatch_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vector_data_mismatch_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	no_message_embedding_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	no_turn_level_data_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vect_diff_length_df = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vect_null = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vect_nan = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	vect_no_one_to_one = pd.read_csv("../tests/vector_data/sentence/chats/test_vector_valid.csv", encoding='utf-8')
	test_convo_num_issue = pd.read_csv("../tests/vector_data/sentence/chats/test_turns_convo_num_issue.csv", encoding='utf-8')
 
	# test number of rows mismatch
	vector_row_mismatch_df = vector_row_mismatch_df.iloc[:-1]
 
	# test chat data not equal to vector data (message)
	vector_data_mismatch_df.loc[0, 'message'] = 'goodbye'
 
	# test no message_embedding column
	no_message_embedding_df.rename(columns={'message_embedding': 'temp'}, inplace=True)
 
	# test vectors not same length
	vector_data_mismatch_df.loc[0, 'message_embedding'] = '[0.9]'
 
	# test null vectors
	vect_null.loc[0, 'message_embedding'] = '[]'
 
	# test nan vectors
	vect_nan.loc[0, 'message_embedding'] = '[np.nan, np.nan]'
 
	# test no 1-1 mapping
	vect_no_one_to_one.loc[0, 'message_embedding'] = '[0.1, 0.2]'
 
	test_cases = {
		"Valid DataFrame": valid_df,
  		"Vector Row Mismatch": vector_row_mismatch_df,
		"Vector Data Mistmatch": vector_data_mismatch_df,
		"No Message Embedding Column": no_message_embedding_df,
		"No Turn-Level Data (turns=True)": no_turn_level_data_df,
		"Vectors Not of Same Length": vector_data_mismatch_df,
		"Vectors Null": vect_null,
		"Vectors Nan": vect_nan,
  		# "Custom File Equals Default Dir": valid_df,
		"No 1-1 Mapping": vect_no_one_to_one,
	}
 
	for name, df in test_cases.items():
		custom_vect_path = "../tests/vector_data/sentence/chats/test_vector.csv"
		print(name)
		df.to_csv(custom_vect_path, index=False, encoding='utf-8')

		if name == "No Turn-Level Data (turns=True)":
			turns = True
		else:
			turns = False
   
		if name == "Custom File Equals Default Dir":
			custom_vect_path = "./vector_data/sentence/chats/test_vector_chat_level.csv"
      
		test_vector_feature_builder = FeatureBuilder(
			input_df=test_vector_df,
			output_file_base="test_vector",
			custom_vect_path=custom_vect_path,
			# Simulate turns=True for "No Turn-Level Data" case
			turns=turns,
		)
		test_vector_feature_builder.featurize()

	# Tiny Juries
	# tiny_juries_feature_builder = FeatureBuilder(
	# 	input_df = tiny_juries_df,
	# 	grouping_keys = ["batch_num", "round_num"],
  	# 	vector_directory = "./vector_data/",
    # 	# custom_vect_path = "C:/Users/amyta/Documents/GitHub/team_comm_tools/examples/vector_data/sentence/turns/jury_TINY_output_chat_level.csv", # testing turns = False but data mismatch
	# 	output_file_base = "jury_TINY_output", # Naming output files using the output_file_base parameter (recommended)
	# 	# turns = False, # want to turn turns off but then test with data that have turns -- CHECK THIS
	# 	turns = True, # this is og
	# 	custom_features = [
	# 		"(BERT) Mimicry",
	# 		"Moving Mimicry",
	# 		"Forward Flow",
	# 		"Discursive Diversity"]
	# )
	# tiny_juries_feature_builder.featurize()

	# Tiny multi-task
	# tiny_multi_task_feature_builder = FeatureBuilder(
	# 	input_df = tiny_multi_task_df,
	# 	conversation_id_col = "stageId",
  	# 	vector_directory = "./vector_data/",
	# 	# alternatively, you can name each output file separately. NOTE, however, that we don't directly use this path;
	# 	# we modify the path to place outputs within the `output/chat`, `output/conv`, and `output/user` folders.
	# 	output_file_path_chat_level = "./multi_task_TINY_output_chat_level_stageId_cumulative.csv",
	# 	output_file_path_user_level = "./multi_task_TINY_output_user_level_stageId_cumulative.csv",
	# 	output_file_path_conv_level = "./multi_task_TINY_output_conversation_level_stageId_cumulative.csv",
	# 	# turns = False
	# 	turns = True
	# )
	# tiny_multi_task_feature_builder.featurize()

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