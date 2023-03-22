import numpy as np
import pandas as pd
import utils.summarize_chat_level_features as summary

def time_to_datetime(df, on_column):
    # TODO - this assume that the time can be converted correctly to datetime!
    # what if it is a Unix time?
    return(pd.to_datetime(df[on_column]))

def get_time_diffs(df, on_column):
    #convert timestamp column to datetime type (in minutes)

    """
    TODO -- this group needs to happen at the *conversation* level; otherwise, we get the difference across all conversations, which throws off our feature.
    """
    datetimes = time_to_datetime(df, on_column)
    return(datetimes.diff().dt.total_seconds()/60)

def get_mean_msg_duration(df,on_column):
    # calculate difference between consecutive timestamps
    mean_msg_duration_calculated = df.groupby(["conversation_num"]).apply(lambda df : np.mean(np.asarray(df['time_diffs']))).reset_index().rename(columns={0: "mean_msg_duration"})
    return(mean_msg_duration_calculated)

def get_stddev_msg_duration(df,on_column):
    # calculate difference between consecutive timestamps
    df['time_diffs'] = get_time_diffs(df,on_column)
    mean_msg_duration_calculated = df.groupby(["conversation_num"]).apply(lambda df : np.std(np.asarray(df['time_diffs']))).reset_index().rename(columns={0: "stdev_msg_duration"})
    return(mean_msg_duration_calculated)