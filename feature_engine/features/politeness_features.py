
import convokit
import spacy
import pandas as pd
import re

from convokit import Corpus, Speaker, Utterance
from convokit import download
from convokit import TextParser

from convokit import PolitenessStrategies

ps = PolitenessStrategies()
# full_jury_data = pd.read_csv('../../data/raw_data/jury_conversations_with_outcome_var.csv')
spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])

# full_jury_data.head()

def get_politeness_strategies(text):
    if pd.isnull(text):
        text = ""
    utt = ps.transform_utterance(
        text, spacy_nlp=spacy_nlp
    )
    return(utt.meta["politeness_strategies"])

