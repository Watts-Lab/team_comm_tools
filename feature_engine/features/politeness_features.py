import convokit
import spacy
import pandas as pd
import re

from convokit import Corpus, Speaker, Utterance
from convokit import download
from convokit import TextParser

from convokit import PolitenessStrategies

ps = PolitenessStrategies()

# Note: if you get an error in which `en_core_web_sm` is not found, do the following: python3 -m spacy download en_core_web_sm

spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])

def get_politeness_strategies(text):
    """
    Using the ConvoKit politeness package, obtains politeness annotations of each message, with some fields 
    including HASHEDGE, Factuality, Deference, Gratitude, Apologizing, etc.

    Source: https://convokit.cornell.edu/documentation/politenessStrategies.html

    Args:
       text(str): The text of the utterance to be analyzed.

    Returns:
        dict: A dictionary containing the politeness strategies extracted, in a format as follows.
        These names are then cleaned up downstream in the ChatLevelFeaturesCalculator before being appended
        to the output dataframe.

        ```
        {'feature_politeness_==Please==': 1,
         'feature_politeness_==Please_start==': 0,
         'feature_politeness_==HASHEDGE==': 0,
         'feature_politeness_==Indirect_(btw)==': 0,
         'feature_politeness_==Hedges==': 0,
         'feature_politeness_==Factuality==': 0,
         'feature_politeness_==Deference==': 0,
         'feature_politeness_==Gratitude==': 0,
         'feature_politeness_==Apologizing==': 0,
         'feature_politeness_==1st_person_pl.==': 0,
         'feature_politeness_==1st_person==': 0,
         'feature_politeness_==1st_person_start==': 0,
         'feature_politeness_==2nd_person==': 1,
         'feature_politeness_==2nd_person_start==': 0,
         'feature_politeness_==Indirect_(greeting)==': 1,
         'feature_politeness_==Direct_question==': 0,
         'feature_politeness_==Direct_start==': 0,
         'feature_politeness_==HASPOSITIVE==': 0,
         'feature_politeness_==HASNEGATIVE==': 0,
         'feature_politeness_==SUBJUNCTIVE==': 1,
         'feature_politeness_==INDICATIVE==': 0} 
        ```
    """
    if pd.isnull(text):
        text = ""
    utt = ps.transform_utterance(
        text, spacy_nlp=spacy_nlp
    )
    return(utt.meta["politeness_strategies"])