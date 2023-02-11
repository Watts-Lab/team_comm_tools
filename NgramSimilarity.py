import pandas as pd
import scipy as sp
import nltk
from nltk import ngrams
from collections import Counter
import math
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# get tokens of length n
def get_ngrams(df,on_column,n):
   
   #tokenize the text into words
   df['ngrams'] = df[on_column].apply(lambda x: list(ngrams(x.split(), n)))



def ngram_cosine_similarity(df,on_column,n):

    #create ngrams and store it into a column
    get_ngrams(df,on_column,n)

    #create text from the ngrams
    df['text'] = df['ngrams'].apply(lambda x: [' '.join(ng) for ng in x])

    # Define CountVectorizer with n-gram range and fit to the text
    vectors = CountVectorizer(ngram_range=(0, n))
    vectors.fit(df['text'].apply(lambda x: ' '.join(x)))

    # Transform the ngrams into vectors
    ngram_vectors = vectors.transform(df['text'].apply(lambda x: ' '.join(x)))

    # Compute the cosine similarities and write them to the file
    return cosine_similarity(ngram_vectors)

#Test for cosine similarity
data = {
    "name": ["I am sort of","I guess I am crazy"]
}

df = pd.DataFrame(data)

ngram_cosine_similarity(df,"name",2)