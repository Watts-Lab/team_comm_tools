import numpy as np
import pandas as pd
import scipy.stats as stats
import nltk
import csv
from nltk.corpus import stopwords

# Get the z-score of each message at the chat level: compute z-score for each message
'''
@param chat_row = one row/utterance of data
@param col_name = the name of the col for z score calculation.It will be the positive words count
@return z score of positive words at a chat level
'''
def chat_pos_zscore(df,on_column):
    #Trying to group real positive words by each chat
    df['pos_words_counts'] = df['positive_words'] - df['nltk_english_stopwords']
    return stats.zscore(df['pos_words_counts'])
