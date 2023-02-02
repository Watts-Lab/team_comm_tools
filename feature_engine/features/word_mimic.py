import numpy as np
import pandas as pd
from string import punctuation

######## Create function words reference
function_word_reference = []
for x in ["able am are aren’t be been being can can’t cannot could couldn’t did didn’t do don’t get got gotta had hadn’t hasn’t have haven’t is isn’t may should should’ve shouldn’t was were will won’t would would’ve wouldn’t",
         "although and as because ’cause but if or so then unless whereas while",
         "a an each every all lot lots the this those",
         "anybody anything anywhere everybody’s everyone everything everything’s everywhere he he’d he’s her him himself herself his I I’d I’ll I’m I’ve it it’d it’ll it’s its itself me my mine myself nobody nothing nowhere one one’s ones our ours she she’ll she’s she’d somebody someone someplace that that’d that’ll that’s them themselves these they they’d they’ll they’re they’ve us we we’d we’ll we’re we’ve what what’d what’s whatever when where where’d where’s wherever which who who’s whom whose why you you’d you’ll you’re you’ve your yours yourself",
         "about after against at before by down for from in into near of off on out over than to until up with without",
         "ah hi huh like mm-hmm oh okay right uh uh-huh um well yeah yup",
         "just no not really too very"]:
  function_word_reference += x.lower().split()



######## Aggregate data by per turn
'''
@param df = original dataframe of the chat, in which each row is one message.
'''
def newdf_perturn(df):
  # Merge subsequent rows if their "speaker_hash" are the same as the preceding rows
  newdf = df.dropna(subset=['message']).groupby((df.speaker_hash != df.speaker_hash.shift(1)).cumsum()).agg({'batch_num':min, 'round_num':min, 'speaker_hash':"first", 
                                                                                                         'speaker_nickname': "first","timestamp":"first", "message":'. '.join}).reset_index(drop=True)
  # Replace "'" with "’" to be consistent with the function word reference
  newdf['message'] = newdf['message'].transform(lambda x:x.replace("'","’"))
  return newdf


######## Differentiate the function words and content words

## Get the function words in a given message: if they are in the function_word_reference
def function_word(text):
   return [x.lower().strip(punctuation) for x in text.split() if x.lower().strip(punctuation) in function_word_reference]


## Get the non-function words in a given message: any word that is not a function word is defined as a content word
def content_word(text):
  return [x.lower().strip(punctuation) for x in text.split() if x.lower().strip(punctuation) not in function_word_reference]


######## Find words that are also used in other's previous turn
'''
@param df = the result of newdf_perturn(), in which each row represents one turn
@param on_column: which column we want to find mimic words in.
'''
def mimic_words(df, on_column):
  word_mimic = [[]]
  for i in range(1, len(df)):
    word_mimic.append([x for x in df.loc[i, on_column] if x in df.loc[(i-1),on_column]])
  return word_mimic


######## Compute the word_mimicry
## Get number of function words also used in other’s prior turn - simply count the function_words_mimicry
# FuncWordAcc: Included in the featurize.py

## Compute the inverse frequency of each content word that also occurred in the other’s immediately preceding turn, then sum them up
# Step 1: compute the frequency of each content word in the corpus (frequency across the whole dataset)
# --> ContWordFreq
'''
@param df = the dataframe contains column content_words
@param on_column: which column we want to count word frequency from
'''
def compute_frequency(df, on_column):
  return(dict(pd.Series(np.concatenate(df[on_column])).value_counts()))

# Step 2: compute the term frequency of each content mimic word, then sum them up
# --> ContWordAcc
def computeTF(content_mimic):
  tfdict = {}
  wf = pd.Series(content_mimic).value_counts()
  for i in wf.index:
    tfdict[i] = wf[i]/ContWordFreq[i]   #ContWordFreq is the result from step 1
  return sum(tfdict.values())
