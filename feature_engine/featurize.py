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
	feature_builder = FeatureBuilder(
		input_file_path = "feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
		output_file_path_chat_level = "feature_engine/output/jury_output_chat_level.csv",
		output_file_path_conv_level = "feature_engine/output/jury_output_conversation_level.csv"
	)
	# Calling the "engine"/"driver" function of the FeatureBuilder class 
	# that creates the features, and writes them in output.
	
	feature_builder.featurize(col="message")
