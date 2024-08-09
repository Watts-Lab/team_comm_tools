import pandas as pd
import numpy as np
from .discursive_diversity import get_cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore') # We get empty slice warnings for short conversations

def get_nan_vector():
    current_dir = os.path.dirname(__file__)
    nan_vector_file_path = os.path.join(current_dir, './assets/nan_vector.txt')
    nan_vector_file_path = os.path.abspath(nan_vector_file_path)

    f = open(nan_vector_file_path, "r")
    str_vec = f.read()
    nan_vector_list = [float(e) for e in str_vec[1:-1].split(',')]
    return np.array(nan_vector_list)

"""Computes the degree to which a speaker's diction changes between each chunk in a conversation. Performs an aggregation function on these distances this across all speakers per chunk interval to return either incongruent modulation or within person discursive range conversation-level metrics.

The Incongruent Modulation feature measures the variance of the rate of speaker shifts. It computes the distance between every speaker's mean embeddings across pairs of consecutive chunks in a conversation. The variance of these cosine distances across speakers per interval is summed.

The Within Person Discursive Range feature measures the average degree of speaker shifts. It computes the distance between every speaker's mean embeddings across pairs of consecutive chunks in a conversation. The average of these cosine distances across speakers per interval is summed.

Args:
    chat_data (pd.DataFrame): The utterance (chat)-level dataframe.
    num_chunks (int): The number of chunks this conversation is split into.
    conversation_id_col (str): The name of the column containing the conversation identifiers.
    speaker_id_col (str): The name of the column containing the speaker identifiers.

Returns:
    pd.DataFrame: A grouped dataframe that contains the conversation identifier as the key, and contains new columns ("incongruent_modulation") and ("within_person_discursive_range").

"""

def get_within_person_disc_range(chat_data, num_chunks, conversation_id_col, speaker_id_col):

    # Get nan vector 
    nan_vector = get_nan_vector()

    #calculate mean vector per speaker per chunk
    mean_vec_speaker_chunks = pd.DataFrame(chat_data.groupby([conversation_id_col, speaker_id_col, 'chunk_num']).message_embedding.apply(np.mean)).unstack('chunk_num').rename(columns={'message_embedding': 'mean_chunk_vec'})

    #collapse multi-index
    mean_vec_speaker_chunks.columns = ["_c".join(col).strip() for col in mean_vec_speaker_chunks.columns.values]

    actual_num_chunks = len(mean_vec_speaker_chunks[2:].columns) # omit the first two, which is conversation_num and speaker_nickname

    #each element in inter_chunk_range is a list of cosine distances BETWEEN each pair of consecutive chunks
    inter_chunk_range = [ [] for i in range(actual_num_chunks - 1)]

    for conv_idx, row in mean_vec_speaker_chunks.iterrows():
        for index in range(actual_num_chunks - 1):
            tpl = [row['mean_chunk_vec_c' + str(index)], row['mean_chunk_vec_c'+ str(index+1)]]
            value = 0
            if (pd.isnull(tpl)).all():
                value = np.nan
            elif (pd.isnull(tpl)).any():
                # Compare with Nanvector 
                if type(tpl[0]) == float:     
                    tpl = [nan_vector, tpl[1]]
                else:
                    tpl = [nan_vector, tpl[0]]
                value = 1 - get_cosine_similarity(tpl)[0]
            else:
                value = 1 - get_cosine_similarity(tpl)[0]
            inter_chunk_range[index].append(value)

    index = []
    for i in range(actual_num_chunks - 1):
        index.append("c" + str(i) + "_c" + str(i + 1))
    range_df = pd.DataFrame(inter_chunk_range, index=index).T
    range_df[conversation_id_col] = mean_vec_speaker_chunks.reset_index()[conversation_id_col]
    range_df = range_df.set_index(conversation_id_col)

    # variance within person discursive range 
    var_disc_range = range_df.groupby(conversation_id_col).apply(lambda x: np.nanvar(x, axis=0).sum()).to_frame().rename(columns={0:'incongruent_modulation'})
    
    # average within person discursive range 
    avg_disc_range = range_df.groupby(conversation_id_col).apply(lambda x: np.nanmean(x, axis=0).sum()).to_frame().rename(columns={0:'within_person_disc_range'})

    return pd.merge(
                left=var_disc_range,
                right=avg_disc_range,
                on=[conversation_id_col],
                how="inner"
            )
