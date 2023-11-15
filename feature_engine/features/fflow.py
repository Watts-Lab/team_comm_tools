import pandas as pd
import numpy as np
from .get_all_DD_features import *
from sklearn.metrics.pairwise import cosine_similarity


def fflow(chat_data, vect_data):

    chat_df = chat_data.copy()
    chat_df['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

    lex_cohesion_adj_chats = []
    lex_cohesion_cumulative = []
    fflow_1 = []
    fflow_2 = []
    cached_cohesion = 0
    pair_index = 1

    for num, conv in chat_df.groupby(['batch_num', 'round_num']):

        for i, pair in conv.groupby(conv.index // 2):
            
            # last "pair" has only one element, safeguard against this
            if (len(pair) == 2):
                
                cos_sim_matrix = cosine_similarity([pair.iloc[0]['message_embedding'], pair.iloc[1]['message_embedding']])
                lex_cohesion_adj_chats.append(cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)][0])
                cached_cohesion += cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)][0]
                lex_cohesion_cumulative.append(cached_cohesion/pair_index)
                pair_index += 1
        
        fflow_1.append(sum(lex_cohesion_adj_chats) / len(conv))
        fflow_2.append(sum(lex_cohesion_cumulative) / len(conv))
    
    final = chat_df[['conversation_num']].drop_duplicates()
    final['lex_cohesion_pairs'] = fflow_1
    final['lex_cohesion_cumul'] = fflow_2
    return final