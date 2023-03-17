import re

'''
    This function takes the dataset WITH all punctuations as input
'''

from features.basic_features import *



## Get the number of question marks in one message (TODO)
def num_question(text):
  return len([x for x in text if x in ["?"]])


## Classify whether the message contains clarification questions
NTRI_list = ("what?","sorry","excuse me","huh?","who?","pardon?","say again?","say it again?","what's that","what is that")
def classify_NTRI(text):
  if len([x for x in NTRI_list if x in text]) > 0:
    return 1
  else:
    return 0
  

## Calculate the word type-to-token ration
def get_word_TTR(text):
  # remove punctuations
  new_text = re.sub(r"[^a-zA-Z0-9 ]+", '',text)
  # calculate the number of unique words
  num_unique_words = len(set(new_text.split()))
  # calculate the word type-to-token ratio
  if count_words(new_text) == 0:
    return 0
  else:
    return num_unique_words/count_words(new_text)   


## Proportion of first person pronouns
first_pronouns = ["i",'me','mine','myself','my','we','our','ours','ourselves','lets']
def get_proportion_first_pronouns(text):
  num_first_prononouns = len([x for x in text.split() if x in first_pronouns])
  if count_words(text) == 0:
    return 0
  else:
    return num_first_prononouns/count_words(text)