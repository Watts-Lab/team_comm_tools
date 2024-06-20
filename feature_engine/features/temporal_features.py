import pandas as pd
import numpy as np

def coerce_to_date_or_number(value):
    try:
        pd.to_datetime(value)
        return value
    except (ValueError, TypeError):
        try:
            pd.to_numeric(value)
            return value
        except (ValueError, TypeError):
            return None

def get_time_diff(df,on_column):
    # Replace instances in which the time is a string that cannot be coerced into a date or number 
    df[on_column] = df[on_column].apply(coerce_to_date_or_number)

    #convert timestamp column to datetime type (in minutes)
    try:
        if(isinstance(df[on_column][0], str)): # String datetime, e.g., '2023-02-20 09:00:00'
            df[on_column] = pd.to_datetime(df[on_column])
        elif(isinstance(df[on_column][0], np.int64)): 
            df[on_column] = pd.to_datetime(df[on_column], unit='ms')
        
        # set and zero time_diff column
        df["time_diff"] = np.zeros(len(df))

        for i in range(1, len(df)):
            if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
                df.loc[i, "time_diff"] = (df.loc[i, on_column] - df.loc[(i-1), on_column]) / pd.Timedelta(seconds=1)
    except TypeError:
        # dateTime conversion failed, which means that we can likely treat it as just an int representing # seconds elapsed
        for i in range(1, len(df)):
            if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
                df.loc[i, "time_diff"] = (df.loc[i, on_column] - df.loc[(i-1), on_column])

    return df['time_diff']

def get_time_diff_startend(df):
    # set and zero time_diff column
    df["time_diff"] = np.zeros(len(df))

    for i in range(1, len(df)):
        if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
            df.loc[i, "time_diff"] = (df.loc[i, "timestamp_start"] - df.loc[(i-1), "timestamp_end"])

    return df['time_diff']