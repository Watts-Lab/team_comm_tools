import pandas as pd
import numpy as np

from .discursive_diversity import *

"""Computes the variance between discursive diversity scores for discrete chunks across a conversation. Returns this variance as a conversation-level metric.

The Variance in Discursive Diversity feature leverage the Discursive Diversity feature, alongside the chunking utility, that computed chunks temporally or chat-wise. 

Args:
    chat_data (pd.DataFrame): The utterance (chat)-level dataframe.
    conversation_id_col (str): The name of the column containing the conversation identifiers.
    speaker_id_col (str): The name of the column containing the speaker identifiers.

Returns:
    pd.DataFrame: A grouped dataframe that contains the conversation identifier as the key, and contains a new column ("variance_in_DD") for each conversation's variance in discursive diversity score.

"""


def get_variance_in_DD(chat_data, conversation_id_col, speaker_id_col):
    dd_results = chat_data.groupby([conversation_id_col, 'chunk_num']).apply(lambda x: get_DD(x, conversation_id_col, speaker_id_col))
    dd_results = dd_results.reset_index(drop=True)
    results = dd_results.groupby(conversation_id_col, as_index=False).var()[[conversation_id_col, 'discursive_diversity']]
    return results.rename(columns={'discursive_diversity': 'variance_in_DD'})
