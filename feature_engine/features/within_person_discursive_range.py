import pandas as pd
import numpy as np
from .discursive_diversity import get_cosine_similarity

def get_within_person_disc_range(chat_data):

    #calculate mean vector per speaker per chunk
    mean_vec_speaker_chunks = pd.DataFrame(chat_data.groupby(['conversation_num', 'speaker_nickname', 'chunk_num']).message_embedding.apply(np.mean)).unstack('chunk_num').rename(columns={'message_embedding': 'mean_chunk_vec'})

    #collapse multi-index
    mean_vec_speaker_chunks.columns = ["_c".join(col).strip() for col in mean_vec_speaker_chunks.columns.values]

    #each element in inter_chunk_range is a list of cosine distances BETWEEN each pair of consecutive chunks
    num_chunks = 10
    inter_chunk_range = [ [] for i in range(num_chunks - 1)]

    for conv_idx, row in mean_vec_speaker_chunks.iterrows():
        for index in range(num_chunks - 1):
            tpl = [row['mean_chunk_vec_c' + str(index)], row['mean_chunk_vec_c'+ str(index+1)]]
            value = 0
            if (pd.isnull(tpl)).any():
                value = np.nan
            else:
                value = 1 - get_cosine_similarity(tpl)[0]
            inter_chunk_range[index].append(value)

    # hard coded column names....
    range_df = pd.DataFrame(inter_chunk_range, index=['c0_c1', 'c1_c2', 'c2_c3', 'c3_c4', 'c4_c5', 'c5_c6', 'c6_c7', 'c7_c8', 'c8_c9']).T
    range_df['conversation_num'] = mean_vec_speaker_chunks.reset_index()['conversation_num']
    range_df = range_df.set_index('conversation_num')

    # variance within person discursive range 
    var_disc_range = range_df.groupby('conversation_num').apply(lambda x: np.nanmean(np.nanvar(x, axis=0))).to_frame().rename(columns={0:'var_disc_range'})
    
    # average within person discursive range 
    avg_disc_range = range_df.groupby('conversation_num').apply(lambda x: np.nanmean(np.nanmean(x, axis=0))).to_frame().rename(columns={0:'mean_disc_range'})

    return pd.merge(
                left=var_disc_range,
                right=avg_disc_range,
                on=['conversation_num'],
                how="inner"
            )
