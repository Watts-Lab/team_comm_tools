import pandas as pd
import re
import nltk

# Define the function to calculate the Dale-Chall score
def count_syllables(word):
    # count the number of syllables in a word
    return(sum(list(map(word.lower().count, "aeiou"))))

def count_words(text):
    # count the number of words in a text
    words = text.split()
    return len(words)

def count_difficult_words(text):
    # count the number of difficult words in a text
    difficult_words = 0
    words = text.split()

    #get the list of dale-chall words
    with open('./features/lexicons/dale_chall.txt', 'r') as file:
        word_list = [line.strip() for line in file]

    for word in words:
        word = word.lower().strip(".:;?!")
        if word not in word_list:
            count = count_syllables(word)
            if count >= 3:
                difficult_words += 1
    return difficult_words

def dale_chall_helper(text):
    # calculate the Dale-Chall readability score of a text
    words = count_words(text)
    sentences = len(text.split("."))
    avg_sentence_length = words/sentences
    difficult_words = count_difficult_words(text)

    #get the percentage of difficult words(odw)
    if words == 0:
        pdw = 0
    else:
        pdw = difficult_words/words*100
        
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
