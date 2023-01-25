import numpy as np
import pandas as pd
import scipy.stats as stats

# source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html


# Get the wordcount used to calculate z-scores: total word count minus first_singular pronouns
def info_exchange_wordcount(text):
  # Count the first singular pronouns in the text.
  first_singular_wordcount = len([x for x in text.split() if x.lower() in ["i", "me", "my", "myself", "mine"]])
  # count_words() function was defined in basic_features.py
  return (count_words(text) - first_singular_wordcount)


# Get the z-score of each message at the chat level: compute z-score for each message
'''
@param chats_data = a dataframe of the chat, in which each row is one message.
@param on_column = the name of the numeric column on which the z-score is to be calculated. Should be info_exchange_wordcount
'''
def get_zscore_chats(chats_data, on_column):
  chats_data['zscore_chats'] = stats.zscore(chats_data[on_column])
  return (chats_data)


# Get the z-score of each conversation
'''
@param chats_data = a dataframe of the chat, in which each row is one message.
@param on_column = the name of the numeric column on which the z-score is to be calculated. Should be info_exchange_wordcount
'''
def get_zscore_conversation(chats_data, on_column):
  chats_data['zscore_conversation'] = chats_data[["batch_num", "round_num", on_column]].groupby(["batch_num", "round_num"])[on_column].transform(lambda x : stats.zscore(x))
  return chats_data
