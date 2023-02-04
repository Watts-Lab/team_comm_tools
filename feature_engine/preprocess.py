import re

def preprocess_text(text):

	# preprocess to remove special characters
	return(re.sub('[^a-zA-Z ]+', '', text).lower())
