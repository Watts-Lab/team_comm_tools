import re

def preprocess_text(text):
  	# For each individual message: preprocess to remove anything that is not an alphabet or number from the string
	return(re.sub(r"[^a-zA-Z0-9 ]+", '',text).lower())


## TODO: Decide "TURN"

## TODO: we don't always have a batch_num and round_num. We need to add preprocessing code that perhaps condenses each conversation ID into 1 column (rather than 2).