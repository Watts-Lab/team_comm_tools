import pandas as pd
import numpy as np

def coerce_to_date_or_number(value):
    """
    Helper function in which we check that the timestamp column contains either a datetime value or a number
    that can be interpreted as a time elapsed; otherwise, sets it equal to none.

    Args:
        value: The value to check; type can be anything
    Returns:
        Either the value itself (if it is a valid timestamp value) or None otherwise

    """
    try:
        pd.to_datetime(value)
        return value
    except (ValueError, TypeError):
        try:
            pd.to_numeric(value)
            return value
        except (ValueError, TypeError):
            raise ValueError("Incorrect timestamp format: " + str(value))

def get_time_diff(df, on_column, conversation_id_col, timestamp_unit):
    """
    Obtains the time difference between messages, assuming there is only a *single* timestamp column
    representing the time of each utterance.

    Args:
        df (pd.DataFrame): This is a pandas dataframe of the chat level features.
        on_column (str): The column name for the timestamp columns.
        conversation_id_col(str): A string representing the column name that should be selected as the unique conversation identifier.
        timestamp_unit (str): A string representing the unit of a timestamp. Defaults to 'ms'.

    Returns:
        pd.Series: A column representing the time difference between messages.
    """

    # Replace instances in which the time is a string that cannot be coerced into a date or number with None
    df[on_column] = df[on_column].apply(coerce_to_date_or_number)        

    #convert timestamp column to datetime type (in minutes)
    try:
        if(isinstance(df[on_column][0], str)): # String datetime, e.g., '2023-02-20 09:00:00'
            df[on_column] = pd.to_datetime(df[on_column])
        elif(isinstance(df[on_column][0], np.int64)):             
            unit = timestamp_unit
            df[on_column] = pd.to_datetime(df[on_column], unit=unit)
        
        # set and zero time_diff column
        df["time_diff"] = np.zeros(len(df))

        for i in range(1, len(df)):
            if df.loc[i, conversation_id_col] == df.loc[i-1, conversation_id_col]: # only do this if they're in the same conversation
                df.loc[i, "time_diff"] = (df.loc[i, on_column] - df.loc[(i-1), on_column]) / pd.Timedelta(seconds=1)
    
    except TypeError:

        # throw an error
        raise ValueError("Error parsing timestamp column. Please refer to documentation for correct format.")

    return df['time_diff']

def get_time_diff_startend(df, timestamp_start, timestamp_end, conversation_id_col, timestamp_unit):
    """
    Obtains the time difference between messages, assuming there are *two* timestamp columns, one representing
    the start of a message and one representing the end of a message.

    Currently assumes that the start and end columns are named "timestamp_start" and "timestamp_end", although
    this should be made more generalizable in a future commit.

    Args:
        df (pd.DataFrame): This is a pandas dataframe of the chat level features.
        timestamp_start(str): A string representing the column name that should be selected as the start timestamp.
        timestamp_end(str): A string representing the column name that should be selected as the end timestamp.
        conversation_id_col(str): A string representing the column name that should be selected as the conversation ID.
        timestamp_unit (str): A string representing the unit of a timestamp. Defaults to 'ms'.

    Returns:
        pd.Series: A column representing the time difference between messages.
    """

    df[timestamp_start] = df[timestamp_start].apply(coerce_to_date_or_number)
    df[timestamp_end] = df[timestamp_end].apply(coerce_to_date_or_number)

    #convert timestamp column to datetime type (in minutes)
    try:
        if(isinstance(df[timestamp_start][0], str)): # String datetime, e.g., '2023-02-20 09:00:00'
            df[timestamp_start] = pd.to_datetime(df[timestamp_start])
            df[timestamp_end] = pd.to_datetime(df[timestamp_end])
        elif(isinstance(df[timestamp_start][0], np.int64)):             
            unit = timestamp_unit
            df[timestamp_start] = pd.to_datetime(df[timestamp_start], unit=unit)
            df[timestamp_end] = pd.to_datetime(df[timestamp_end], unit=unit)
        # set and zero time_diff column
        df["time_diff"] = np.zeros(len(df))

        for i in range(1, len(df)):
            if df.loc[i, conversation_id_col] == df.loc[i-1, conversation_id_col]: # only do this if they're in the same conversation
                df.loc[i, "time_diff"] = (df.loc[i, timestamp_start] - df.loc[(i-1), timestamp_end]) / pd.Timedelta(seconds=1)

    except TypeError:
        
        # throw an error
        raise ValueError("Error parsing timestamp column. Please refer to documentation for correct format.")


    return df['time_diff']