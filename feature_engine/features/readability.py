import pandas as pd
import re
import nltk

# Define the function to calculate the Dale-Chall score
def count_syllables(word):
    # count the number of syllables in a word
    count = 0
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


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

data = {
    "name": ["Society is intrinsically impossible,” says Debord; however, according to Sargeant[1] , it is not so much society that is intrinsically impossible, but rather the absurdity, and some would say the collapse, of society. Presemantic narrative holds that culture is dead. If one examines cultural discourse, one is faced with a choice: either accept dialectic postsemioticist theory or conclude that narrativity may be used to oppress the underprivileged, but only if the premise of subtextual capitalism is valid. In a sense, the example of Derridaist reading which is a central theme of Pynchon’s The Crying of Lot 49 is also evident in V, although in a more self-falsifying sense. The subject is contextualised into a cultural discourse that includes truth as a reality. “Narrativity is part of the genre of sexuality,” says Baudrillard. But Bataille’s model of material subcultural theory states that academe is capable of significance. The subject is interpolated into a cultural discourse that includes reality as a paradox. Thus, deconstructivist feminism holds that sexual identity has objective value. If cultural discourse holds, we have to choose between presemantic narrative and neopatriarchial deappropriation. But Lacan’s critique of cultural discourse implies that the task of the reader is significant form. Bataille uses the term ‘presemantic narrative’ to denote the economy, and subsequent paradigm, of textual class. However, the premise of Sontagist camp states that art is capable of deconstruction, but only if narrativity is interchangeable with reality; otherwise, reality comes from the masses. A number of discourses concerning not deconstruction, but predeconstruction exist. Therefore, the subject is contextualised into a cultural discourse that includes culture as a whole. Many discourses concerning presemantic narrative may be discovered. However, Derrida’s model of cultural discourse suggests that consciousness serves to entrench the status quo. The subject is interpolated into a subtextual capitalism that includes reality as a totality."
    ,"Hi I am Priya"]
}

df = pd.DataFrame(data)

get_dale_chall_score_and_classfication(df,'name')
print(df['dale_chall_score'])
print(df['dale_chall_classification'])