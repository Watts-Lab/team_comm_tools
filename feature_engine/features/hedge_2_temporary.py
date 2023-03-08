import pandas as pd
import numpy as np
import requests
import sklearn
import nltk
import stanfordnlp
nlp = stanfordnlp.Pipeline(lang='en', dir='/Users/priyadcosta/Desktop/Positivity/stanford-corenlp-4.5.2')
from itertools import combinations
import ngram_similarity_functions as nsf
from nltk import ngrams
from nltk import jaccard_distance

#create a list of discourse markers
dm_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/booster_words.txt'
discourse_markers = requests.get(dm_path).text.strip().split('\n')

#create a list of booster words
booster_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/booster_words.txt'
booster_words = requests.get(booster_path).text.strip().split('\n')

#create a list of hedge words
hedge_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/hedge_words.txt'
hedge_words = requests.get(hedge_path).text.strip().split('\n')


#DISCOURSE MARKERS

#The function to calculate the jaccard distance for discourse markers and sets the status accordingly
def discourse_markers_check(df, on_column, threshold, n, discourse_markers):
    # Create a column to hold the maximum Jaccard distance for each list of sets
    df['Max_Jaccard'] = 0
    
    # Iterate over each row of the dataframe
    for i, row in df.iterrows():
        max_jaccard = 0
        
        # Iterate over each set in the list
        for word_set in row[on_column]:
            # Calculate the Jaccard distance with the universal set
            jaccard = jaccard_distance(word_set, set(discourse_markers))
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                
        # Store the maximum Jaccard distance for the sets in the list
        df.at[i, 'Max_Jaccard'] = max_jaccard
        
        # Mark the row as a hedged sentence if the maximum Jaccard distance is below the threshold
        if max_jaccard <= threshold:
            df.at[i, 'DM_output'] = True
        else:
            df.at[i, 'DM_output'] = False


#BOOSTER WORDS
def booster_helper(words_list, booster_list, negative_words=["not", "without"]):
    #intially set the status to false
    status = False

    #iterate through the wordlist
    for i in range(1, len(words_list)):

        #if a word is present in the booster word list and is not preceed by 'not' or 'without'
        if words_list[i] == set(booster_list)and words_list[i-1] not in negative_words:
            #it is a booster word i.e it is hedged
            status = True
            break
    return status

#apply the booster helper to each row in the data frame
def booster_check(df, on_column,words_list, booster_list, negative_words=["not", "without"]):
    # Apply the check_word function to each row of the DataFrame
    df['booster_output'] = df[on_column].apply(lambda x: booster_helper(x, booster_list, negative_words=["not", "without"]))


#HEDGES

# Load the English language model
nlp = stanfordnlp.Pipeline(processors='tokenize,pos,lemma,depparse', lang='en')

#Hedge Rules as mentioned in the paper
def is_true_hedged_term(token,tree):
    '''   
    Rule 1: If token t is (i) a root word, (ii) has the part-of-speech
    VB* and (iii) has an nsubj (nominal subject) dependency
    with the dependent token being a first person pronoun (i,
    we), t is a hedge, otherwise, it is a non-hedge.
   
    Rule 2: If token t is followed by a token with part-of-speech
    IN, t is a non-hedge, otherwise, hedge
 
    Rule 3: If token t has a ccomp (clausal complement) dependent, t is a hedge, otherwise, non-hedge.
   
    Rule 4: If token t has an xcomp (open clausal complement)
    dependent d and d has a mark dependent to, t is a nonhedge, otherwise, it is a hedge.

    Rule 5: If token t has an xcomp (open clausal complement)
    dependent, t is a hedge, otherwise, it is a non-hedge.
 
    Rule 6: If token t has a ccomp (clausal complement) or xcomp
    (open clausal complement) dependent, t is a hedge, otherwise, it is a non-hedge.

    Rule 7: If token t has relation amid with its head h and h has
    part of speech N*, t is a non-hedge, otherwise, it is a hedge.
  
    Rule 8: If token t has relation aux with its head h and h has
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

#if 
def hedge_check_helper(sentence,hedge_words):

    doc = nlp(sentence)
    tree = doc.sentences[0].dependencies

     # Parse the sentence and get the dependency parse tree
    '''
    nlp(sentence) creates an nlp object, which is a container for processing the input text
    (in this case, sentence). The nlp object uses various components, such as a tokenizer, 
    a part-of-speech tagger, and a parser, to analyze the input text and create a structured 
    representation of it.

    doc.sentences[0].dependencies retrieves the dependency parse tree for the first sentence 
    in the parsed document. A dependency parse tree is a way of representing the grammatical 
    structure of a sentence in terms of the relationships between words. 
    Each word in the sentence is represented as a node in the tree, and the edges between the 
    nodes indicate the grammatical relationships between the words.

    So, the overall effect of this code is to parse the input sentence and generate a dependency 
    parse tree for it, which can be used for further analysis or processing.
    '''

    # Determine if each token in the tree is a hedge or a non-hedge
    for token in tree.nodes.values():
        if token.get('word'):
            #if any of the 9 rules above are satisfied and the term is in the hedge set, it is hedged
            if is_true_hedged_term(token, tree) and token in set(hedge_words) :
                return True
    return False        

#applies the hedge_helper function to the dataframe
def hedge_check(df,on_colum):
    df['hedge_output'] = df[on_colum].apply(hedge_check_helper)

#FINAL FUNCTION TO BE CALLED TO CHECK IF THE SENTENCE IS HEDGED
def is_hedged_sentence2(df):
    #iterate through each row in the data frame
    for i, row in df.iterrows():

        #if the status by the booster check or the discourse marker check or the hedge check is true, the sentence is hedged
        if df['booster_output'] is True or df['DM_output'] is True or df['hedge_output'] is True:
            df.at[i, 'is_hedged_sentence2'] = True
        else:
            df.at[i, 'DM_output'] = False