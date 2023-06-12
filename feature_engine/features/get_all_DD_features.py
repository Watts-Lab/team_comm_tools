import pandas as pd
import numpy as np
import itertools
from functools import reduce

from .discursive_diversity import *
from .variance_in_DD import *
from .within_person_discursive_range import *

def conv_to_float_arr(df):
    if isinstance(df['message_embedding'][0], str):
            df['message_embedding'] = [val[1:-1] for val in df['message_embedding']]
            df['message_embedding'] = [[float(e) for e in embedding.split(',')] for embedding in df['message_embedding']]
            df['message_embedding'] = [np.array(e) for e in df['message_embedding']]
    return df

def assign_chunk_nums(chat_data, num_chunks):
    list_conv_df = [tpl[1] for tpl in list(chat_data.groupby('conversation_num'))]
    split_conv_df = [assign_chunk_nums_helper(df, num_chunks) for df in list_conv_df]
    return reduce(lambda x, y: pd.concat([x, y], axis=0), split_conv_df)

def assign_chunk_nums_helper(chat_data, num_chunks):
    list_df = np.array_split(chat_data, num_chunks)
    for x in range(num_chunks):
        list_df[x]['chunk_num'] = str(x)
    return reduce(lambda x, y: pd.concat([x, y], axis=0), list_df)

def get_DD_features(chat_data, vect_data):
     
    # Format data
    chat_data['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

    # Get discursive diversity
    disc_div = get_DD(chat_data)

    # Split into chunks 
    chat_data = assign_chunk_nums(chat_data, 10)

    # Get variance in discursive diversity 
    var_disc_div = get_variance_in_DD(chat_data)

    # Get within-person discursive range metrics
    modulation_metrics = get_within_person_disc_range(chat_data)

    dd_features = [disc_div, var_disc_div, modulation_metrics]
    return reduce(lambda x, y: pd.merge(x, y, on = 'conversation_num'), dd_features)
