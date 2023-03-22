import numpy as np
import pandas as pd
import utils.summarize_chat_level_features as summary

def time_to_datetime(df, on_column):
    # TODO - this assume that the time can be converted correctly to datetime!
    # what if it is a Unix time?

    df['datetime'] = pd.to_datetime(df[on_column])
    return(df)

def get_time_diffs(df, on_column):
    #convert timestamp column to datetime type (in minutes)
    df_datetime = time_to_datetime(df, on_column)[["conversation_num", "datetime"]]
    df_datetime['time_diffs'] = df_datetime.groupby(["conversation_num"]).diff()
    df_datetime = df_datetime.dropna() # the first person's time is always NA, as no message precedes them
    df_datetime['time_diffs'] = df_datetime['time_diffs'].apply(lambda x: x.total_seconds()/60)
    return(df_datetime)

def get_mean_msg_duration(df,on_column):
    # calculate difference between consecutive timestamps
    df_datetime = get_time_diffs(df, on_column)
    mean_msg_duration_calculated = df_datetime.groupby(["conversation_num"]).apply(lambda df : np.mean(np.asarray(df['time_diffs']))).reset_index().rename(columns={0: "mean_msg_duration"})
    return(mean_msg_duration_calculated)

def get_stddev_msg_duration(df,on_column):
    # calculate difference between consecutive timestamps
    df_datetime = get_time_diffs(df, on_column)
    mean_msg_duration_calculated = df_datetime.groupby(["conversation_num"]).apply(lambda df : np.std(np.asarray(df['time_diffs']))).reset_index().rename(columns={0: "stdev_msg_duration"})
    return(mean_msg_duration_calculated)