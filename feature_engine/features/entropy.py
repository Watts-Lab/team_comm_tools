import pandas as pd
import scipy as sp
import nltk
import numpy as np
from nltk import ngrams
from collections import Counter
import math
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize

#Get priors using the previous conversations. and try to check how 

def ngram_dialog_act_entropy(df,on_column,n,set1,set2,set1_label,set2_label):

    # Initialize count vectorizer - this will extract ngrams from 0 to n (this covers maximum words/combinations possible)
    vectorizer = CountVectorizer(vocabulary=set1+set2)

    # Count occurrences of words in each bag
    X = vectorizer.fit_transform(df[on_column])

    # Count occurrences of words in each bag
    #X = vectorizer.fit_transform([row[on_column]])
    
    X_dense = X.toarray()
    set1_counts = np.sum(X_dense[:, 0:len(set1)], axis=0)
    set2_counts = np.sum(X_dense[:, len(set2):], axis=1)    

    # Normalize counts to obtain probabilities
    set1_probs = set1_counts / np.sum(set1_counts)
    set2_probs = set2_counts / np.sum(set2_counts)

    # Calculate entropy for each bag
    set1_entropy = -np.sum(set1_probs * np.log2(set1_probs))
    set2_entropy = -np.sum(set2_probs * np.log2(set2_probs))

    #return the result
    if set1_entropy > set2_entropy:
        print("case1")
        print(set1_entropy)
        print(set2_entropy)
        return set1_label
    elif set1_entropy < set2_entropy:
        print("case2")
        print(set1_entropy)
        print(set2_entropy)
        return set2_label
    else:
        print("case3")
        print(set1_entropy)
        print(set2_entropy)
        return "neutral"
        
def create_sets(filepath):
    with open(filepath, 'r') as file:
        set = set(file.read().split())
    return set
