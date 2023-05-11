import os
"""
Returns the list of easy words according to Dale-Chall readability.
"""
def get_dale_chall_easy_words():
    #get the list of dale-chall words
    with open('./features/lexicons/dale_chall.txt', 'r') as file:
        easy_word_list = [line.strip() for line in file]
    return easy_word_list