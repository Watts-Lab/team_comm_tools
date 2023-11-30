import pandas as pd
import re

# Note: This feature requires the message WITH punctuation.

# parse certainty lexicon, compile into master regex, delimited by | 
certainty = pd.read_csv("./features/lexicons/certainty.txt").sort_values(["NumWords", "NumCharacters"], ascending=False)
master_regex = certainty["Word"].str.cat(sep='\\b|') + "\\b"


def get_certainty(chat): 
    
    # default certainty value is 4.5; aka a "neutral" statement in the event we don't find anything
    DEFAULT_CERTAINTY = 4.5

    # pattern match via re library
    certainty_score = 0
    matches = re.findall(master_regex, chat)

    for match in matches:
        certainty_score += certainty.loc[certainty['Word'] == match]["Certainty"].iloc[0]

    # safeguard against division by zero error
    if (len(matches) == 0):
        return DEFAULT_CERTAINTY 
    return (certainty_score / len(matches))