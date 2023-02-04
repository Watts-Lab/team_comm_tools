import re

def preprocess_text(text):
  	# For each individual message: preprocess to remove anything that is not an alphabet or number from the string
	return(re.sub(r"[^a-zA-Z0-9 ]+", '',text).lower())


## TODO: Decide "TURN"
