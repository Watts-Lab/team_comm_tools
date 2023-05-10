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

    #get the list of positive words
    with open("./features/lexicons/liwc_lexicons/positive_affect", "r") as f1:
        poswordslist = [line.rstrip() for line in f1]

    with open("./features/lexicons/nltk_english_stopwords.txt", "r") as f2:
        stop_words = [line.rstrip() for line in f2]


    # print(poswordslist)
    # print(stop_words)
    
    num_stop_words = df[on_column].apply(lambda x: len([word for word in x.split() if word.lower() in stopwords]))
    df['total_pos_words'] = df[on_column].apply(lambda x: len([word for word in x.split() if word.lower() in poswordslist]))

    #Trying to group real positive words by each chat
    df['pos_words_counts'] = df['total_pos_words']-df['stop_words']
    return stats.zscore(df['pos_words_counts'])
