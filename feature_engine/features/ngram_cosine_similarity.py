import pandas as pd
import scipy as sp
import nltk
import numpy as np
from nltk import ngrams
from collections import Counter
import math
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# define a function to create bigrams from a given text
def get_ngrams_list(text,n):
    tokens = nltk.word_tokenize(text)
    return list(ngrams(tokens,n))

# get tokens of length n
def get_ngrams_df(df,on_column,n):
   
   #tokenize the text into words
   df['ngrams'] = df[on_column].apply(lambda x: get_ngrams_list(x,n))
   return(df)

def ngram_cosine_similarity(df,on_column,n):

    #create ngrams and store it into a column
    df = get_ngrams_df(df,on_column,n)

    #create text from the ngrams
    df['text'] = df['ngrams'].apply(lambda x: [' '.join(ng) for ng in x])

    # Define CountVectorizer with n-gram range and fit to the text
    vectors = CountVectorizer(ngram_range=(0, n))
    vectors.fit(df['text'].apply(lambda x: ' '.join(x)))

    # Transform the ngrams into vectors
    ngram_vectors = vectors.transform(df['text'].apply(lambda x: ' '.join(x)))

    # Compute the cosine similarities and average them out
    # get the matrix
    cosine_sim_matrix = np.matrix(cosine_similarity(ngram_vectors))

    # Create a new column in the input dataframe with the "average pairwise cosine similarity of these vectors"
    return(np.matrix.mean(cosine_sim_matrix, axis=1))
