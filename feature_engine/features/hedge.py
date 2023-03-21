import pandas as pd
import re

#returns the total number of hedge words present in the data set (We can add more hedge words)
def is_hedged_sentence_1(df,on_column):
    
    with open("hedge_words.txt", "r") as f1:
        hedge_words = f1.read().split()
    
    pattern = "|".join(hedge_words)
    return df[on_column].apply(lambda x: bool(re.search(pattern, x, re.IGNORECASE)))
