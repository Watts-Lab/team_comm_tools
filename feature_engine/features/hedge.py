import pandas as pd
import re

#returns the total number of hedge words present in the data set (We can add more hedge words)
def is_hedged_sentence_1(df,on_column):
        
    pattern = "|".join(hedge_words)
    return df[on_column].apply(lambda x: int(bool(re.search(pattern, x, re.IGNORECASE))))
