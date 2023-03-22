import pandas as pd
import re
import nltk

from basic_features import count_words

'''
Because of the need to split up sentences, this function requires the version of pre-process
WITH punctuation.
'''

# Define the function to calculate the Dale-Chall score
def count_syllables(word):
    # count the number of syllables in a word
    return(sum(list(map(word.lower().count, "aeiou"))))

def count_difficult_words(text):
    # count the number of difficult words in a text
    difficult_words = 0
    # recall that words are already pre-processed; substitute punctuation here for words only
    words = re.sub(r"[^a-zA-Z0-9 ]+", '',text).split()

    #get the list of dale-chall words
    with open('./features/lexicons/dale_chall.txt', 'r') as file:
        easy_word_list = [line.strip() for line in file]

    remaining_words = words - easy_word_list

    for word in remaining_words:
        # words with more than 3 syllables are difficult
        if(count_syllables(word) >= 3):
            difficult_words += 1
    
    return difficult_words

def dale_chall_helper(text):
    # calculate the Dale-Chall readability score of a text
    num_words = count_words(text)
    num_sentences = len(re.split(r'[.?!]\s*', text)) 
    avg_sentence_length = num_words/num_sentences
    num_difficult_words = count_difficult_words(text)

    #get the percentage of difficult words(odw)
    if num_words == 0:
        pdw = 0
    else:
        pdw = num_difficult_words/num_words*100
        
    raw_score = (0.1579*pdw) + (0.0496*avg_sentence_length)
    if pdw > 5:
        raw_score += 3.6365
    return raw_score

def classify_text(score):
    if score <= 4.9:
        return "easy"
    elif score <= 5.9:
        return "medium"
    else:
        return "difficult"

def get_dale_chall_score_and_classfication(df,on_column):
    df['dale_chall_score'] = df[on_column].apply(dale_chall_helper)
    df['dale_chall_classification'] = df['dale_chall_score'].apply(classify_text)
