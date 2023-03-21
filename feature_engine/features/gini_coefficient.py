import numpy as np
import csv
import pandas as pd

from utils.summarize_chat_level_features import get_count_dataframe

# source: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
def gini_coefficient(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

'''
@param conversation_data = a dataframe of the conversations, in which each row is one chat.
@param on_column = the name of the numeric column on which the Gini coefficient is to be calculated.
'''
def get_gini(conversation_data, on_column):
	grouped_conversation_data = get_count_dataframe(conversation_data, on_column)

	# for all speakers per {batch, round}: apply Gini
	gini_calculated = grouped_conversation_data.groupby(["conversation_num"]).apply(lambda df : gini_coefficient(np.asarray(df[on_column]))).reset_index().rename(columns={0: "gini_coefficient_" + on_column})
	return(gini_calculated)