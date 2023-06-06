import pandas as pd
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity


#from given code 
def get_unique_pairwise_combos(lst):
    '''Computes all unique pairwise combinations of the elements in a list.
    input: array or list
    output: list of unique pairwise combinations of elements of the input list'''
    return list(itertools.combinations(lst, 2))


def get_cosine_similarity(vecs):
    '''computes cosine similarity between a list of vectors
    input: list of vectors (vecs) (this has to be a pair!)
    output: cosine similarity
    '''
    if len(vecs) > 1:
        cos_sim_matrix = cosine_similarity(vecs)
        return cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)]
    else:
        return np.nan


def get_DD(conversation_data):

    #convert to float 

    # vec_arrays = []
    # for index, r in vector_data.iterrows():
    #     if (pd.isnull(r.iloc[1])):
    #         vec_arrays.append(np.nan)
    #     else:
    #         vec_arrays.append(np.array(r, dtype="float64"))
    
    # conversation_data['mean_vec'] = vec_arrays

    # get user average vector per conversation
    user_centroid_per_conv = pd.DataFrame(conversation_data.groupby(['conversation_num','speaker_nickname']).apply(lambda x: np.mean(x['mean_vec'].to_frame().dropna()))).reset_index().rename(columns={0:'day_mean_vec'})

    # For each team(conversation) get all unique pairwise combinations of members' centroids:
    user_pairs = pd.DataFrame(user_centroid_per_conv.groupby(['conversation_num'])['day_mean_vec'].\
    apply(get_unique_pairwise_combos)).reset_index().\
    rename(columns={'day_mean_vec':'user_pairs_per_conv'})

    # get cosine distances between each pair for every conversation, average all the distances to get DD per conversation

    cos_dists_mean_widay_btwu = []

    for lst in user_pairs.user_pairs_per_conv:

        # Make sure list isn't empty:
        if lst:
            # Store the cosine distances for the person's list of tuples
            cos_dists = []
            for tpl in lst:
                try:
                    cos_d = 1 - get_cosine_similarity([tpl[0], tpl[1]])
                    cos_dists.append(cos_d)
                except ValueError as e:
                    #np.nan in tuple
                    pass

            # Compute mean of cos dists
            cos_dists_mean_widay_btwu.append(np.nanmean(cos_dists, dtype="float64"))
        else:
            cos_dists_mean_widay_btwu.append(np.nan)

    user_pairs['discursive_diversity'] =  cos_dists_mean_widay_btwu

    return user_pairs[['conversation_num', 'discursive_diversity']]

