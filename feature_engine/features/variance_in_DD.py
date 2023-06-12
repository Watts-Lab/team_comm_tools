import pandas as pd
import numpy as np

from .discursive_diversity import *

# Conversation level feature !

def get_variance_in_DD(chat_data):
    dd_results = chat_data.groupby(['conversation_num', 'chunk_num']).apply(get_DD)
    dd_results = dd_results.reset_index(drop=True)
    results = dd_results.groupby("conversation_num", as_index=False).mean()[['conversation_num', 'discursive_diversity']]
    return results.rename(columns={'discursive_diversity': 'variance_in_DD'})
