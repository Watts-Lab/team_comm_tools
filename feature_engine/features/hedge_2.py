"""
import pandas as pd
import numpy as np
import requests
import sklearn
import nltk
import stanfordnlp
from itertools import combinations
from features.ngram_similarity import *
from nltk import ngrams
from nltk import jaccard_distance

# TODO - need to change this local path!
MODELS_DIR = '/Users/priyadcosta/Desktop/Positivity/Random/stanford-corenlp-4.5.2'

def jaccard_distance(set1, set2):
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

# The function to calculate the Jaccard distance for discourse markers and sets the status accordingly
def discourse_markers_check(df, on_column, threshold, n, discourse_markers):
    
    # Iterate over each row of the dataframe
    for i, row in df.iterrows():
        max_jaccard = 0

        # Tokenize the words and create ngrams
        words = nltk.word_tokenize(row[on_column])
        words_set = nltk.ngrams(words, n)
        
        # Create a list to hold the ngram sets for each row
        ngram_sets = []

        # Iterate over each ngram in the set of ngrams
        for ngram in words_set:
            # Create a set of ngrams, frozenset is used to create an immutable set of each n-gram so that it can be added to the set.
            #if we do not use frozen set, we will not be able to add a set object to a set. Hence, we use frozen set
            ngram = set([frozenset(ngram[j:j+n]) for j in range(len(ngram)-n+1)])
            # Add the set of ngrams to the list
            ngram_sets.append(ngram)

        # Iterate over each set in the list
        for ngram_set in ngram_sets:
            # Calculate the Jaccard distance with the universal set i.e discourse markers
            jaccard = jaccard_distance(set(ngram_set), set(discourse_markers))
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                
        # Store the maximum Jaccard distance for the sets in the list
        df.at[i, 'Max_Jaccard'] = max_jaccard
        
        # Mark the row as a hedged sentence if the maximum Jaccard distance is below the threshold
        if max_jaccard >= threshold:
            df.at[i, 'DM_output'] = True
        else:
            df.at[i, 'DM_output'] = False
            
 
def booster_helper(words_list, booster_words, negative_words=["not", "without"]):

    #iterate throught the list of words in the text and check if a booster word is preceeded by "not" or "without"
    for i in range(1, len(words_list)):
        if words_list[i] in set(booster_words)and words_list[i-1] not in negative_words:
          return True
    return False

def booster_check(df, on_column, booster_words, negative_words=["not", "without"]):
    # Apply the check_word function to each row of the DataFrame, after splitting the contents of each data frame into a list
    df['booster_output'] = df[on_column].apply(lambda x: booster_helper(x.split(), booster_words, negative_words=["not", "without"]))
    
# Load the English language model
nlp = stanfordnlp.Pipeline(processors='tokenize,pos', models_dir=MODELS_DIR, treebank='en_ewt', use_gpu=True)

def is_true_hedged_term(token,tree):
    '''   
    Rule 1: If token t is (i) a root word, (ii) has the part-of-speech
    VB* and (iii) has an nsubj (nominal subject) dependency
    with the dependent token being a first person pronoun (i,
    we), t is a hedge, otherwise, it is a non-hedge.
   
    Rule 2: If token t is followed by a token with part-of-speech
    IN (preposition/subordinating conjunction), t is a non-hedge, otherwise, hedge
 
    Rule 3: If token t has a ccomp (clausal complement) dependent, t is a hedge, otherwise, non-hedge.
   
    Rule 4: If token t has an xcomp (open clausal complement)
    dependent d and d has a mark dependent to, t is a nonhedge, otherwise, it is a hedge.

    Rule 5: If token t has an xcomp (open clausal complement)
    dependent, t is a hedge, otherwise, it is a non-hedge.
 
    Rule 6: If token t has a ccomp (clausal complement) or xcomp
    (open clausal complement) dependent, t is a hedge, otherwise, it is a non-hedge.

    Rule 7: If token t has relation amid with its head h and h has
    part of speech N (Noun)*, t is a non-hedge, otherwise, it is a hedge.
  
    Rule 8: If token t has relation aux (auxiliary verb) with its head h and h has
    dependent have, t is a non-hedge, otherwise, it is a hedge

    Rule 9: If token t is followed by token than, t is a non-hedge,
    otherwise, it is a hedge.
    '''


    if token.upos.startswith('VB') and 'nsubj' in token.dependency_relation:
        nsubj = tree[token.governor-1].get('word')
        if nsubj in ['i', 'we']:
            return True
    elif token.upos == 'IN':
        return False
    elif 'ccomp' in token.dependency_relation:
        return True
    elif 'xcomp' in token.dependency_relation:
        xcomp = tree[token.dependents[0]-1]
        if 'mark' in xcomp.dependency_relation and xcomp.get('word') == 'to':
            return False
        else:
            return True
    elif 'amod' in token.dependency_relation:
        head = tree[token.governor-1]
        if head.upos.startswith('N'):
            return False
        else:
            return True
    elif 'aux' in token.dependency_relation:
        head = tree[token.governor-1]
        if 'have' in [dep.get('lemma') for dep in head.dependents]:
            return False
        else:
            return True
    elif token.get('word') == 'than':
        return False
    else:
        return True


def hedge_check_helper(sentence,hedge_words):

    #creates a parsed sentence. Eg. Priya ate a Pizza will yield an output of [('nsubj', 1, 0), ('ROOT', 0, 2), ('det', 4, 3), ('obj', 2, 4)]
    # ('nsubj', 1, 0) - Priya is the subject of the word "ate"
    # ('ROOT', 0, 2) - "ate is the root of the sentence"
    # ('det', 4, 3) - "a" is the determiner of the pizza
    # ('obj', 2, 4) - Pizza is the object of the verb "ate"
    parsed_sentence = nlp(sentence)

    #creates a tree using the dependencies described above. The root is the root of the tree. We use 0 as we do not know how long the sentence is 
    tree = parsed_sentence.sentences[0].dependencies

    # Determine if each value in the tree is a hedge or a non-hedge
    for node in tree.nodes.values():
        #if the node has a valid word in the sentence 
        if node.get('word'):

            #check if it is a hedged term as per the 8 rules above and it is present in the list of hedged words
            if is_true_hedged_term(node, tree) and node in set(hedge_words) :
                return True
    return False        

#applies the function to the dataframe
def hedge_check(df,on_column,hedge_words):
    df['hedge_output'] = df[on_column].apply(lambda x: booster_helper(x,hedge_words))
    
def is_hedged_sentence2(df,on_column):
    
    #get the lists of discourse markers
    dm_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/booster_words.txt'
    discourse_markers = requests.get(dm_path).text.strip().split('\n')

    #create a list of booster words
    booster_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/booster_words.txt'
    booster_words = requests.get(booster_path).text.strip().split('\n')

    #create a list of hedge words
    hedge_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/hedge_words.txt'
    hedge_words = requests.get(hedge_path).text.strip().split('\n')
    
    #run discourse marker check
    discourse_markers_check(df, on_column, 0.75, 1, discourse_markers)
    
    #run booster check
    booster_check(df, on_column,booster_words, negative_words=["not", "without"])

    #run hedge check
    hedge_check(df,on_column,hedge_words)

    if df['booster_output'] is True or df['DM_output'] is True or df['hedge_output'] is True:
        return True
    else:
        return False
"""

'''
#Testing
# create a DataFrame with text data
df = pd.DataFrame({'text': ["Im not actually sure if that is true, maybe we can skip that for now?",
                            "Ill take the fifth.",
                            "My name is Priya",
                            "The efficacy of the new drug has not been confirmed, and there mifght be potential side effects on the liver"]})

is_hedged_sentence2(df,'text')
print(df['booster_output'])
print(df['DM_output'])
print(df['hedge_output'])
print(df['is_hedged_sentence2'])
'''
