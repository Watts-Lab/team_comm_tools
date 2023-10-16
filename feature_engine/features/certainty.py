import pandas as pd
import re

def get_certainty(chat): 
    
    # parse certainty lexicon, compile into master regex, delimited by | 
    certainty = pd.read_csv("./features/lexicons/certainty.txt").sort_values(["NumWords", "NumCharacters"], ascending=False)
    master_regex = certainty["Word"].str.cat(sep='|')
   
    # pattern match via re library
    certainty_score = 0
    matches = re.findall(master_regex, chat)

    for match in matches:
        certainty_score += certainty.loc[certainty['Word'] == match]["Certainty"].iloc[0]

    # safeguard against division by zero error
    if (len(matches) == 0):
        return certainty_score
    return (certainty_score / len(matches))