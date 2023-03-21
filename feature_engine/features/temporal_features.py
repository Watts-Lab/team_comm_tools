import pandas as pd
import utils.summarize_chat_level_features as summary

def mean_msg_duration(df,on_column):

    #convert timestamp column to datetime type (in minutes)
    df[on_column] = pd.to_datetime(df[on_column])

    # calculate difference between consecutive timestamps
    df['time_diff'] = df[on_column].diff().dt.total_seconds()/60
    
    return summary.get_average(df, 'time_diff', 'avg_time')

def stddev_msg_duration(df,on_column):

    #convert timestamp column to datetime type (in minutes)
    df[on_column] = pd.to_datetime(df[on_column])

    # calculate difference between consecutive timestamps
    df['time_diff'] = df[on_column].diff().dt.total_seconds()/60

    return summary.get_stdev(df, 'time_diff', 'std_dev_time')

