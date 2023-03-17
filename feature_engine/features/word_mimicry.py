import numpy as np
import pandas as pd
from string import punctuation
import re

'''
    To compute word mimicry, we use the dataset that removed all the punctuations
'''



## Create function words reference
function_word_reference = []
for x in ["able am are aren't be been being can can't cannot could couldn't did didn't do don't get got gotta", 
          "had hadn't hasn't have haven't is isn't may should should've shouldn't was were will won't would would've wouldn't",
         "although and as because 'cause but if or so then unless whereas while a an each every all lot lots the this those",
         "anybody anything anywhere everybody's everyone everything everything's everywhere he he'd he's her him himself herself", 
         "his I I'd I'll I'm I've it it'd it'll it's its itself me my mine myself nobody nothing nowhere one one's ones our ours", 
         "she she'll she's she'd somebody someone someplace that that'd that,ll that,s them themselves these they they,d they'll they're they've",
         "us we we'd we'll we're we've what what'd what's whatever when where where'd where's wherever which who who's whom whose why", 
         "you you'd you'll you're you've your yours yourself ah hi huh like mm-hmm oh okay right uh uh-huh um well yeah yup",
         "about after against at before by down for from in into near of off on out over than to until up with without just no not really too very"]:
  function_word_reference += re.sub(r"[^a-zA-Z0-9 ]+", '', x).lower().split()



####### Extract the function words & non-functions words from a message
## Get the function words in a given message
def function_word(text):
   return [x for x in text.split() if x in function_word_reference]
# OUTPUT column: function_words

## Get the non-function words in a given message
def content_word(text):
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
    word_mimic.append([x for x in df.loc[i, on_column] if x in df.loc[(i-1),on_column]])
  return word_mimic
# OUTPUT: function_word_mimicry, content_word_mimicry




####### Compute the number of mimic words

## Function word mimicry: simply count the number of mimic words by using len()
'''
@param function_mimic_words: input is each entry under `function_word_mimicry` column.
'''
def Function_mimicry_score(function_mimic_words):
  return len(function_mimic_words)
# OUTPUT column: FuncWordAcc 

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
# OUTPUT column: ContWordAcc
