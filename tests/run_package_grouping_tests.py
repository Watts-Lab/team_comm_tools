"""
file: run_package_grouping_tests.py
---
This file runs the feature builder pipeline on a multi-task dataset to confirm the package works
on the proper input format and performs grouping correctly.
"""

# Importing the Feature Generating Class
from team_comm_tools import FeatureBuilder
import pandas as pd

# Main Function
if __name__ == "__main__":

	tiny_multi_task_renamed_df = pd.read_csv("data/cleaned_data/multi_task_TINY_cols_renamed.csv", encoding='utf-8')
	package_agg_df = pd.read_csv("data/cleaned_data/test_package_aggregation.csv", encoding='utf-8')

	"""
	Testing Package Task 1
	---
	In this test, we simply test the functionaality of everything after we rename everything ("Case 1").
	Here, we use a test dataset that has a different conversation ID, speaker ID, message column, and timestamp
	column compared to the defaults, and ensure that nothing breaks.
	"""
	print("TESTING CASE 1 + FILE PATH ROBUSTNESS ......")
	testing_package_task_1 = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		conversation_id_col = "roundId",
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./tiny_multi_task_PT1_level_chat",
		output_file_path_user_level = "./tiny_multi_task_PT1_level_user",
		output_file_path_conv_level = "./tiny_multi_task_PT1_level_conv",
		turns = False,
	)
	testing_package_task_1.featurize()

	"""
	Testing Package Task 1 Advanced Features
	---
	In this test, we test the functionality of the advanced grouping features.
	
	"Case 2": .ngroup() feature
	- Group by ["gameId", "roundId", "stageId"] and assert that the number of groupings matches
		the stageId (which will confirm that it worked)

	"Case 3": Complex hieararchical grouping
	- (3a) ID: stageID; cumulative: True, within_task: False
	- (3b) ID: stageID; cumulative: True; within_task: True
	- (3c) ID: roundID; cumulative: True, within_task: True

	Improper examples:
	- grouping keys: ["roundID", "stageID"], ID: "gameID"
	"""
	print("TESTING CASE 2 ....")
	testing_case_2 = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		grouping_keys = ["roundId", "stageId"],
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/tiny_multi_task_case2_level_chat.csv",
		output_file_path_user_level = "./output/user/tiny_multi_task_case2_level_user.csv",
		output_file_path_conv_level = "./output/conv/tiny_multi_task_case2_level_conv.csv",
		turns = False
	)
	testing_case_2.featurize()

	print("TESTING CASE 3A .....")
	testing_case_3_a = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		conversation_id_col = "stageId",
		grouping_keys = ["gameId", "roundId", "stageId"],
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		cumulative_grouping = True, 
        within_task = False,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/tiny_multi_task_case3a_level_chat.csv",
		output_file_path_user_level = "./output/user/tiny_multi_task_case3a_level_user.csv",
		output_file_path_conv_level = "./output/conv/tiny_multi_task_case3a_level_conv.csv",
		turns = False
	)
	testing_case_3_a.featurize()

	print("TESTING CASE 3B .....")
	testing_case_3_b = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		conversation_id_col = "stageId",
		grouping_keys = ["gameId", "roundId", "stageId"],
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		cumulative_grouping = True, 
        within_task = True,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/tiny_multi_task_case3b_level_chat.csv",
		output_file_path_user_level = "./output/user/tiny_multi_task_case3b_level_user.csv",
		output_file_path_conv_level = "./output/conv/tiny_multi_task_case3b_level_conv.csv",
		turns = False
	)
	testing_case_3_b.featurize()

	print("TESTING CASE 3C .....")
	testing_case_3_c = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		conversation_id_col = "roundId",
		grouping_keys = ["gameId", "roundId", "stageId"],
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		cumulative_grouping = True, 
        within_task = True,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/tiny_multi_task_case3c_level_chat.csv",
		output_file_path_user_level = "./output/user/tiny_multi_task_case3c_level_user.csv",
		output_file_path_conv_level = "./output/conv/tiny_multi_task_case3c_level_conv.csv",
		turns = False
	)
	testing_case_3_c.featurize()

	print("TESTING IMPROPER CASE .....")
	testing_case_improper = FeatureBuilder(
		input_df = tiny_multi_task_renamed_df,
		conversation_id_col = "gameId",
		grouping_keys = ["roundId", "stageId"],
		speaker_id_col = "speakerId",
		message_col = "text",
		timestamp_col = "time",
		cumulative_grouping = False, 
        within_task = True,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/tiny_multi_task_improper_level_chat.csv",
		output_file_path_user_level = "./output/user/tiny_multi_task_improper_level_user.csv",
		output_file_path_conv_level = "./output/conv/tiny_multi_task_improper_level_conv.csv",
		turns = False
	)
	testing_case_improper.featurize()

	"""
	Test robustness of the FeatureBuilder to taking in an input that contains existing feature names.
	This was reported as bug here: https://github.com/Watts-Lab/team-comm-tools/issues/256
	"""

	chat_df_existing_output = pd.read_csv("./output/chat/test_chat_level_chat.csv")

	testing_chat_existing = FeatureBuilder(
		input_df = chat_df_existing_output,
		vector_directory = "./vector_data/",
		message_col = "message_original",
		output_file_path_chat_level = "./output/chat/test_chat_level_existing_chat.csv",
		output_file_path_user_level = "./output/user/test_chat_level_existing_user.csv",
		output_file_path_conv_level = "./output/conv/test_chat_level_existing_conv.csv",
		custom_features = [
            "(BERT) Mimicry",
            "Moving Mimicry",
            "Forward Flow",
            "Discursive Diversity"
        ],
		turns = False
	)
	testing_chat_existing.featurize()

	"""
	Test robustness of the vector pipeline to weird inputs:
	- Super long input
	- Input containing only symbols (e.g,. ":-)")
	- Empty input
	- Input with many spaces
	"""
	vector_testing_input = pd.read_csv("data/cleaned_data/test_vector_edge_cases.csv", encoding='latin-1')

	test_vectors = FeatureBuilder(
		input_df = vector_testing_input,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_vectors_chat.csv",
		output_file_path_user_level = "./output/user/test_vectors_user.csv",
		output_file_path_conv_level = "./output/conv/test_vectors_conv.csv",
		custom_features = [
            "(BERT) Mimicry",
            "Moving Mimicry",
            "Forward Flow",
            "Discursive Diversity"
        ],
		turns = False,
		regenerate_vectors = True
	)
	test_vectors.featurize()

	"""
	Test correctness of the custom aggregation pipeline:

	- Aggregate with all the functions for conversation level: [mean, max, min, stdev, median, sum]
	- Specify 'mean' as 'average' instead and ensure it shows up correctly
	- Aggregate with "mean" for the user level + a fake method (e.g., "foo")
	- Aggregate only "second_person_lexical_wordcount" at the conversation level
	- Aggregate "positive_bert" at the user level + a fake column (e.g., "bar") + a non-numeric column (e.g., "dale_chall_classification")
	"""

	print("Testing custom aggregation...")
	custom_agg_fb = FeatureBuilder(
        input_df = package_agg_df,
        grouping_keys = ["batch_num", "round_num"],
        vector_directory = "./vector_data/",
        output_file_base = "custom_agg_test" ,
        convo_methods = ['average', 'max', 'min', 'stdev', 'median', 'sum'],
        convo_columns = ['second_person_lexical_wordcount'], # testing functionality in case of typo
        user_methods = ['mean', 'foo'],
        user_columns = ['positive_bert', 'bar', 'dale_chall_classification'], # testing functionality in case of typo
	)
	custom_agg_fb.featurize()


	"""
	Test aggregation piepline when we switch aggregation to false

	(We should only get the default num words, num chars, and num messages aggregated).
	"""

	print("Testing aggregation turned off...")
	custom_agg_fb_no_agg = FeatureBuilder(
        input_df = package_agg_df,
        grouping_keys = ["batch_num", "round_num"],
        vector_directory = "./vector_data/",
        output_file_base = "custom_agg_test_no_agg" ,
        convo_aggregation = False,
        user_aggregation = False,
	)
	custom_agg_fb_no_agg.featurize()
