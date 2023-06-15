import pandas as pd
import numpy as np
from functools import reduce

from .discursive_diversity import *
from .variance_in_DD import *
from .within_person_discursive_range import *
from utils.assign_chunk_nums import *

def conv_to_float_arr(df):
    if isinstance(df['message_embedding'][0], str):
            df['message_embedding'] = [val[1:-1] for val in df['message_embedding']]
            df['message_embedding'] = [[float(e) for e in embedding.split(',')] for embedding in df['message_embedding']]
            df['message_embedding'] = [np.array(e) for e in df['message_embedding']]
    return df

def get_DD_features(chat_data, vect_data):
    
    chats = chat_data.copy()

    # Format data
    chats['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

    # Get discursive diversity
    disc_div = get_DD(chats)
    disc_div = disc_div.replace(np.nan, 0)

    num_chunks = 3

    # Split into chunks 
    chats_chunked = assign_chunk_nums(chats, num_chunks)

    # Get variance in discursive diversity 
    var_disc_div = get_variance_in_DD(chats_chunked)

    # Get within-person discursive range metrics
    modulation_metrics = get_within_person_disc_range(chats_chunked, num_chunks)

    dd_features = [disc_div, var_disc_div, modulation_metrics]
    return reduce(lambda x, y: pd.merge(x, y, on = 'conversation_num'), dd_features)
