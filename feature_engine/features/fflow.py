import pandas as pd
import numpy as np
from .get_all_DD_features import *
# from sklearn.metrics.pairwise import cosine_similarity


# CHAT LEVEL FEATURE

def get_forward_flow(chat_data, vect_data):
    
    chat_df = chat_data.copy()
    chat_df['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

    forward_flow = []

    for num, conv in chat_df.groupby(['conversation_num'],  sort=False):

        forward_flow.append(0)
        cached_embedding = conv.iloc[0]["message_embedding"]
        chat_count = 1
        avg_embedding = cached_embedding / chat_count

        for index, row in conv[1:].iterrows():
            
            # determine distance from that and prev average, append to the list
            cos_sim_matrix = cosine_similarity([row['message_embedding'], avg_embedding])
            cosine_sim = cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)][0]

            forward_flow.append(cosine_sim)

            # add to cache, increment count
            cached_embedding += row["message_embedding"]
            chat_count += 1
            
            # calculate new average
            avg = cached_embedding / chat_count
    
    return forward_flow