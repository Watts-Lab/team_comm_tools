from feature_builder import FeatureBuilder

# Main Function
if __name__ == "__main__":
	feature_builder = FeatureBuilder(
		input_file_path = "feature_engine/data/raw_data/juries_tiny_for_testing.csv", 
		output_file_path_chat_level = "feature_engine/output/jury_TINY_output_chat_level.csv", 
		output_file_path_conv_level = "feature_engine/output/jury_TINY_output_conversation_level.csv" 
	)
	feature_builder.featurize(col="message")
