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

    # Create an empty list to store the cosine similarities
    cosine_similarities = []

    # Loop through each row of the ngram_vectors matrix
    for i in range(ngram_vectors.shape[0]):
        # Get the i-th row of the ngram_vectors matrix
        vector_i = ngram_vectors.getrow(i)

        # Compute the cosine similarities between the i-th row and all other rows
        cosine_similarities_i = cosine_similarity(vector_i, ngram_vectors)[0]

        # Exclude the diagonal value (cosine similarity of a row with itself)
        cosine_similarities_i = cosine_similarities_i[np.arange(cosine_similarities_i.shape[0]) != i]

        # Compute the average cosine similarity for the i-th row
        average_i = np.mean(cosine_similarities_i)

        # Append the average cosine similarity to the list
        cosine_similarities.append(average_i)

    # Add the cosine similarities to the dataframe
    df['cosine_similarity'] = cosine_similarities

