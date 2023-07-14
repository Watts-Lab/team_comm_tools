import numpy as np
import pandas as pd
import re

'''
function: get_info_exchange_wordcount
(Chat-level function)

Get the wordcount used to calculate z-scores: total word count minus first_singular pronouns
'''
def get_info_exchange_wordcount(df, first_person):
  first_person_regex = " | ".join(first_person)
  df['first_person_raw'] = df['message'].apply(lambda chat: len(re.findall(first_person_regex, chat)))
  return (df["num_words"] - df["first_person_raw"])