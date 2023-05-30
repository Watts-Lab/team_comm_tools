import convokit
import spacy
import pandas as pd
import re

from convokit import Corpus, Speaker, Utterance
from convokit import download
from convokit import TextParser

from convokit import PolitenessStrategies

'''
function: politeness_features
(Chat-level function)

This gets the politeness annotations of each message, with some fields 
including HASHEDGE, Factuality, Deference, Gratitude, Apologizing, etc.

'''

ps = PolitenessStrategies()
spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])

def get_politeness_strategies(text):
    if pd.isnull(text):
        text = ""
    utt = ps.transform_utterance(
        text, spacy_nlp=spacy_nlp
    )
    return(utt.meta["politeness_strategies"])

