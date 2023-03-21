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

'''
df = pd.DataFrame({'timestamp': ['2023-02-20 09:00:00', '2023-02-20 09:05:00', '2023-02-20 09:10:00'],
                   'conversation_num': ['1', '2', '3'],
                   'speaker_nickname': ['a', 'b', 'c']})
df1 = mean_msg_duration(df,'timestamp')   
df1['avg_time']

df = stddev_msg_duration(df,'timestamp')   
df['std_dev_time']    
'''