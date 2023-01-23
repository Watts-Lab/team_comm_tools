import itertools
import re
import os,glob

"""
file: lexical_features.py
---
Defines features that involve bag-of-words counts from a lexicon.
"""

'''
function: pad_with_space

Pads each regex with a space, as these occur on the word level 
(and we don't want to detect partial matches)
'''
def pad_with_space(text):
	return(" " + text + " ")

'''
function: get_lexicon_list_from_txt

Takes in a .txt file, in which each line is a lexicon term, and reads it into a list.

@param txt_file: name of the text file
'''
def get_lexicon_list_from_txt(txt_file):
	with open(txt_file) as lexicon:
		# return list of each word
		# replace instances in which "**" occurs, as this breaks python's regex
		return([pad_with_space(re.sub("\*\*", "\*", line.rstrip())) for line in lexicon])

'''
function: get_lexical_value_from_text

Takes in a lexicon list, and returns the number of matches within a given message or string.

@param text: the message/text that we are searching for lexicon words in.
@param lexicon_list: output of `get_lexicon_list_from_text`; a list of regexes or words that 
we are searching for inside the text.
'''
def get_lexical_value_from_text(text, lexicon_list):

	# preprocess to remove special characters
	text = re.sub('[^a-zA-Z ]+', '', text).lower()

	# Finds all matches from the lexicon, and flattens into a single list
	matches = list(itertools.chain(*[re.findall(regex, text) for regex in lexicon_list]))
	return(len(matches))

"""
LIWC Features

Create features drawn from the LIWC lexicons.

@ param text: the text being evaluated.
@ return value: a dictionary, in which each key is the name of the feature, and each value
is the leixcal value (count) within the text.
"""
def liwc_features(text):

	lexical_feature_dictionary = {}

	# Open every file in the folder
	directory = './features/lexicons/liwc_lexicons/'
	for filename in os.listdir(directory):
		lexicon_list = get_lexicon_list_from_txt(directory + filename)
		lexical_value = get_lexical_value_from_text(text, lexicon_list)
		lexical_feature_dictionary[filename] = lexical_value

	return(lexical_feature_dictionary)
