import numpy as np
import csv
import pandas as pd

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
	grouped_conversation_data = conversation_data[["batch_num", "round_num", "speaker_nickname", on_column]].groupby(["batch_num", "round_num", "speaker_nickname"]).sum().reset_index() 
    # gets this dataframe:
	# Batch# Round# Speaker  Total Number of Words
	# 0 	 1      Priya    100
	# 0      1      Yuluan   90

	# for all speakers per {batch, round}: apply Gini
	gini_calculated = grouped_conversation_data.groupby(["batch_num", "round_num"]).apply(lambda df : gini_coefficient(np.asarray(df[on_column]))).reset_index().rename(columns={0: "gini_coefficient_" + on_column})
	return(gini_calculated)