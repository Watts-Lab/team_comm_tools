import pandas as pd
import re

def get_certainty(chat): 
    certainty = pd.read_csv("./features/lexicons/certainty.txt").sort_values(["NumWords", "NumCharacters"], ascending=False)
   
    certainty_score = 0

    # sub every instance with the associated normative certainty score 
    for index, row in certainty.iterrows():
        
        certainty_pattern = row["Word"]
        num_matches = len(re.findall(certainty_pattern, chat))

        # so that we don't double count substrings
        # i.e. to prevent counting both "i think that" AND "i think"
        chat = re.sub(certainty_pattern, '', chat)
        certainty_score += (row["Certainty"] * num_matches)
       
    
    return certainty_score