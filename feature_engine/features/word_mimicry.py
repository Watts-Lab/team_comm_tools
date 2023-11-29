import numpy as np
import pandas as pd
from string import punctuation
import re
from .get_all_DD_features import *
from sklearn.metrics.pairwise import cosine_similarity

'''
    To compute word mimicry, we use the dataset that removed all the punctuations
    This is a *chat-level* feature in which order matters.
'''

####### Extract the function words & non-functions words from a message
## Get the function words in a given message
def get_function_words_in_message(text, function_word_reference):
   return [x for x in text.split() if x in function_word_reference]
# OUTPUT column: function_words

## Get the non-function words in a given message
def get_content_words_in_message(text, function_word_reference):
  return [x for x in text.split() if x not in function_word_reference]
# OUTPUT column: content_words


####### Return a list of words that are also used in other's previous turn
'''
@param df: the dataset that removed all punctuations
@param on_column: the column that we want to find mimicry on
                  For function words: input == `function_words`
                  For content words: input == `content_words`
'''
def mimic_words(df, on_column):
  word_mimic = [[]]
  for i in range(1, len(df)):
    if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
      word_mimic.append([x for x in df.loc[i, on_column] if x in df.loc[(i-1),on_column]])
    else:
      word_mimic.append([])
  return word_mimic
# OUTPUT: function_word_mimicry, content_word_mimicry

####### Compute the number of mimic words

## Function word mimicry: simply count the number of mimic words by using len()
'''
@param function_mimic_words: input is each entry under `function_word_mimicry` column.
'''
def function_mimicry_score(function_mimic_words):
  return len(function_mimic_words)
# OUTPUT column: function_word_accommodation 

## Content word mimicry: Compute the inverse frequency of each content word that also occurred in the other’s immediately preceding turn, then sum them up
# Step 1: compute the frequency of each content word across the whole dataset)
# --> ContWordFreq
'''
@param on_column: the column with which we calculate content word frequency. 
                  Input == 'content_words'
'''
def compute_frequency(df, on_column):
  return(dict(pd.Series(np.concatenate(df[on_column])).value_counts()))

# Step 2: compute the term frequency of each content mimic word, then sum them up
'''
This function happens on the individual level. 
@param column_mimic: the column that we want to compute term frequency on. 
                  Input is each entry under `content_word_mimicry` column
@param frequency_list: the dictionary of content word frequency across the dataset. 
                  Input is the result from step 1
'''
def computeTF(column_mimc, frequency_dict):
  tfdict = {}
  wf = pd.Series(column_mimc, dtype = 'str').value_counts()
  for i in wf.index:
    tfdict[i] = wf[i]/frequency_dict[i]
  return sum(tfdict.values())

# Step 3: Combine them
def Content_mimicry_score(df, column_count_frequency, column_count_mimic):
  # Compute the frequency of each content word across the whole dataset
  ContWordFreq = compute_frequency(df, column_count_frequency)
  # Compute the content_mimicry_score
  return df[column_count_mimic].apply(lambda x:computeTF(x, ContWordFreq))
# OUTPUT column: content_word_accommodation


# WITH BERT SENTENCE VECTORS 

def get_mimicry_bert(chat_data, vect_data):
  
  chat_df = chat_data.copy()
  chat_df['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())
  # print(chat_df.head())

  mimicry = []

  for num, conv in chat_df.groupby(['conversation_num'],  sort=False):

      # first chat has no zero mimicry score, nothing previous to compare it to 
      mimicry.append(0)
      prev_embedding = conv.iloc[0]['message_embedding']
      
      for index, row in conv[1:].iterrows():
          
          # last "pair" has only one element, safeguard against this
          cos_sim_matrix = cosine_similarity([row['message_embedding'], prev_embedding])
          cosine_sim = cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)][0]
          
          mimicry.append(cosine_sim)

          prev_embedding = row['message_embedding'] 

  return mimicry


def get_moving_mimicry(chat_data, vect_data):

  chat_df = chat_data.copy()
  chat_df['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

  moving_mimicry = []

  for num, conv in chat_df.groupby(['conversation_num'], sort = False):

      moving_mimicry.append(0)
      prev_embedding = conv.iloc[0]["message_embedding"]
      cached_pairwise_sims = 0
      chat_count = 1
      prev_mimicry = 0
      
      for index, row in conv[1:].iterrows():
      
          # find cosine similarity between current pair
          cos_sim_matrix = cosine_similarity([row['message_embedding'], prev_embedding])
          cosine_sim = cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)][0]

          # average this distance with the previous average
          
          moving_mimicry.append((cosine_sim + prev_mimicry)/2)

          # update values
          cached_pairwise_sims += cosine_sim
          chat_count += 1
          prev_mimicry = cached_pairwise_sims/chat_count
              
      
  return moving_mimicry
