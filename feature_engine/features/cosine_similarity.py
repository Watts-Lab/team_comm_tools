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

def ngram_cosine_similarity(df,on_column,n):

    #create ngrams and store it into a column
    get_ngrams_df(df,on_column,n)

    #create text from the ngrams
    df['text'] = df['ngrams'].apply(lambda x: [' '.join(ng) for ng in x])

    # Define CountVectorizer with n-gram range and fit to the text
    vectors = CountVectorizer(ngram_range=(0, n))
    vectors.fit(df['text'].apply(lambda x: ' '.join(x)))

    # Transform the ngrams into vectors
    ngram_vectors = vectors.transform(df['text'].apply(lambda x: ' '.join(x)))

    # Compute the cosine similarities and average them out
    # get the matrix
    matrix = cosine_similarity(ngram_vectors)
    
    # Create a new column in the input dataframe with the cosine similarity matrix
    return pd.Series(cosine_sim_matrix.tolist())
    
    #Test for cosine similarity
    data = {
        "name": ["I am sort of","I guess I am crazy"]
    }

    df = pd.DataFrame(data)

    ngram_cosine_similarity(df,"name",2)
