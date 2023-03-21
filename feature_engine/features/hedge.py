import pandas as pd
import re

#returns the total number of hedge words present in the data set (We can add more hedge words)
def is_hedged_sentence_1(df,on_column):
    hedge_words = ["sort of", "kind of", "I guess", "I think", "a little", "maybe", "possibly", "probably"]
    pattern = "|".join(hedge_words)
    return df[on_column].apply(lambda x: bool(re.search(pattern, x, re.IGNORECASE)))



#Testing
# create a DataFrame with text data
df = pd.DataFrame({'text': ["Im not actually sure if that is true, maybe we can skip that for now?",
                            "Ill take the fifth.",
                            "My name is Priya",
                            "The efficacy of the new drug has not been confirmed, and there mifght be potential side effects on the liver"]})

print(is_hedged_sentence_1(df,'text'))
