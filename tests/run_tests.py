"""
file: run_tests.py
---
This file runs the feature builder pipeline on our custom testing datasets.
"""

# Importing the Feature Generating Class
from team_comm_tools import FeatureBuilder
import pandas as pd
import chardet

# Main Function
if __name__ == "__main__":

	# detects CSV encoding of our datasets
	with open("data/cleaned_data/test_chat_level.csv", 'rb') as file:
		chat_encoding = chardet.detect(file.read())

	with open("data/cleaned_data/test_conv_level.csv", 'rb') as file:
		conv_encoding = chardet.detect(file.read())

	chat_df = pd.read_csv("data/cleaned_data/test_chat_level.csv", encoding=chat_encoding['encoding'])
	info_exchange_df = pd.read_csv("data/cleaned_data/info_exchange_zscore_chats.csv", encoding=chat_encoding['encoding'])
	conv_df = pd.read_csv("data/cleaned_data/test_conv_level.csv", encoding=conv_encoding['encoding'])
	test_ner_df = pd.read_csv("data/cleaned_data/test_named_entity.csv", encoding='utf-8')
	test_ner_training_df = pd.read_csv("data/cleaned_data/train_named_entity.csv")
	chat_complex_df = pd.read_csv("data/cleaned_data/test_chat_level_complex.csv", encoding=chat_encoding['encoding'])
	conv_complex_df = pd.read_csv("data/cleaned_data/test_conv_level_complex.csv", encoding=chat_encoding['encoding'])
	test_forward_flow_df = pd.read_csv("data/cleaned_data/fflow.csv", encoding=chat_encoding['encoding'])
	conv_complex_timestamps_df = pd.read_csv("data/cleaned_data/test_conv_level_complex_timestamps.csv", encoding=chat_encoding['encoding'])
	timediff_datetime  = pd.read_csv("data/cleaned_data/test_timediff_datetime.csv", encoding=chat_encoding['encoding'])
	timediff_numeric = pd.read_csv("data/cleaned_data/test_timediff_numeric.csv", encoding=chat_encoding['encoding'])
	timediff_numeric_unit = pd.read_csv("data/cleaned_data/test_timediff_numeric_unit.csv", encoding=chat_encoding['encoding'])
	time_pairs_datetime  = pd.read_csv("data/cleaned_data/test_time_pairs_datetime.csv", encoding=chat_encoding['encoding'])
	time_pairs_numeric = pd.read_csv("data/cleaned_data/test_time_pairs_numeric.csv", encoding=chat_encoding['encoding'])
	time_pairs_numeric_unit = pd.read_csv("data/cleaned_data/test_time_pairs_numeric_unit.csv", encoding=chat_encoding['encoding'])
	positivity_zscore = pd.read_csv("data/cleaned_data/positivity_zscore_chats.csv", encoding=chat_encoding['encoding'])

	# TESTING DATASETS -------------------------------

	# testing positivity zscore
	test_positivity = FeatureBuilder(
		input_df = positivity_zscore,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_positivity_chat_level.csv",
		output_file_path_user_level = "./output/user/test_positivity_user_level.csv",
		output_file_path_conv_level = "./output/conv/test_positivity_conv_level.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
	)
	test_positivity.featurize()
	
	# testing timediff datetime
	testing_timediff_datetime = FeatureBuilder(
		input_df = timediff_datetime,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_timediff_dt_level_chat.csv",
		output_file_path_user_level = "./output/user/test_timediff_dt_user.csv",
		output_file_path_conv_level = "./output/conv/test_timediff_dt_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
	)
	testing_timediff_datetime.featurize()

	# testing timediff numeric
	testing_timediff_numeric = FeatureBuilder(
		input_df = timediff_numeric,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_timediff_num_level_chat.csv",
		output_file_path_user_level = "./output/user/test_timediff_num_user.csv",
		output_file_path_conv_level = "./output/conv/test_timediff_num_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
	)
	testing_timediff_numeric.featurize()


	# testing timediff numeric with unit parameter
	testing_timediff_numeric_unit = FeatureBuilder(
		input_df = timediff_numeric_unit,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_timediff_num_unit_level_chat.csv",
		output_file_path_user_level = "./output/user/test_timediff_num_unit_user.csv",
		output_file_path_conv_level = "./output/conv/test_timediff_num_unit_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
		timestamp_unit = 'h'
	)
	testing_timediff_numeric_unit.featurize()

	# testing time pairs datetime
	testing_time_pairs_datetime = FeatureBuilder(
		input_df = time_pairs_datetime,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_time_pairs_dt_level_chat.csv",
		output_file_path_user_level = "./output/user/test_time_pairs_dt_user.csv",
		output_file_path_conv_level = "./output/conv/test_time_pairs_dt_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
		timestamp_col=("timestamp_start", "timestamp_end")
	)
	testing_time_pairs_datetime.featurize()

	# testing time pairs numeric
	testing_time_pairs_numeric = FeatureBuilder(
		input_df = time_pairs_numeric,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_time_pairs_num_level_chat.csv",
		output_file_path_user_level = "./output/user/test_time_pairs_num_user.csv",
		output_file_path_conv_level = "./output/conv/test_time_pairs_num_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
		timestamp_col=("timestamp_start", "timestamp_end")
	)
	testing_time_pairs_numeric.featurize()

	# testing time pairs numeric unit
	testing_time_pairs_numeric_unit = FeatureBuilder(
		input_df = time_pairs_numeric_unit,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_time_pairs_num_unit_level_chat.csv",
		output_file_path_user_level = "./output/user/test_time_pairs_num_unit_user.csv",
		output_file_path_conv_level = "./output/conv/test_time_pairs_num_unit_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
		timestamp_col=("timestamp_start", "timestamp_end"),
		timestamp_unit = 's'
	)
	testing_time_pairs_numeric_unit.featurize()

	# general chat level features
	testing_chat = FeatureBuilder(
		input_df = chat_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_chat_level_chat.csv",
		output_file_path_user_level = "./output/user/test_chat_level_user.csv",
		output_file_path_conv_level = "./output/conv/test_chat_level_conv.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	testing_chat.featurize()

	testing_info_exchange = FeatureBuilder(
		input_df = info_exchange_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/info_exchange_zscore_chats.csv",
		output_file_path_user_level = "./output/user/info_exchange_zscore_chats.csv",
		output_file_path_conv_level = "./output/conv/info_exchange_zscore_chats.csv",
		custom_features = [ # these require vect_data, so they now need to be explicitly included in order to calculate them
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	testing_info_exchange.featurize()

	testing_conv = FeatureBuilder(
		input_df = conv_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_conv_level_chat.csv",
		output_file_path_user_level = "./output/user/test_conv_level_user.csv",
		output_file_path_conv_level = "./output/conv/test_conv_level_conv.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True,
		timestamp_col="timestamp"
	)
	testing_conv.featurize()

	test_ner_feature_builder = FeatureBuilder(
		input_df = test_ner_df,
		ner_training_df = test_ner_training_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_named_entity_chat_level.csv",
		output_file_path_user_level = "./output/user/test_named_entity_user_level.csv",
		output_file_path_conv_level = "./output/conv/test_named_entity_conversation_level.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	test_ner_feature_builder.featurize()

	# testing perturbed chat level features
	testing_chat_complex = FeatureBuilder(
		input_df = chat_complex_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_chat_level_chat_complex.csv",
		output_file_path_user_level = "./output/user/test_chat_level_user_complex.csv",
		output_file_path_conv_level = "./output/conv/test_chat_level_conv_complex.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	testing_chat_complex.featurize()

	# testing conv features
	testing_conv_complex = FeatureBuilder(
		input_df = conv_complex_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_conv_level_chat_complex.csv",
		output_file_path_user_level = "./output/user/test_conv_level_user_complex.csv",
		output_file_path_conv_level = "./output/conv/test_conv_level_conv_complex.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	testing_conv_complex.featurize()

	testing_conv_complex_ts = FeatureBuilder(
		input_df = conv_complex_timestamps_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_conv_level_chat_complex_ts.csv",
		output_file_path_user_level = "./output/user/test_conv_level_user_complex_ts.csv",
		output_file_path_conv_level = "./output/conv/test_conv_level_conv_complex_ts.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)
	testing_conv_complex_ts.featurize()

	# testing forward flow
	testing_forward_flow = FeatureBuilder(
		input_df = test_forward_flow_df,
		vector_directory = "./vector_data/",
		output_file_path_chat_level = "./output/chat/test_forward_flow_chat.csv",
		output_file_path_user_level = "./output/user/test_forward_flow_user.csv",
		output_file_path_conv_level = "./output/conv/test_forward_flow_conv.csv",
		custom_features = [
			"(BERT) Mimicry",
			"Moving Mimicry",
			"Forward Flow",
			"Discursive Diversity"
		],
		turns = False,
		regenerate_vectors = True
	)

	testing_forward_flow.featurize()