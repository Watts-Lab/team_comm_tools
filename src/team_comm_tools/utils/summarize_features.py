import numpy as np

'''
These functions are for the purpose of calculating aggregates of various user level features at the conversation level.

'''

def get_user_sum_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame by summing a specified column per individual.

    This function groups chat-level data by user and conversation, sums the values of a 
    specified numeric column for each user (speaker), and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to sum for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the total sum of the specified column per individual.
    :rtype: pandas.DataFrame
    """
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).sum().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "sum_"+on_column})
    # gets this dataframe:
    # Batch# Round# Speaker  Total Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)


def get_user_mean_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame by averaging a specified column per individual.

    This function groups chat-level data by user and conversation, calculates the mean values
    of a specified numeric column for each user, and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to mean for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the mean of the specified column per individual.
    :rtype: pandas.DataFrame
    """
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).mean().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "mean_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Mean Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)

def get_user_max_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame by maximizing a specified column per individual.

    This function groups chat-level data by user and conversation, calculates the max values
    of a specified numeric column for each user, and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to max for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the max of the specified column per individual.
    :rtype: pandas.DataFrame
    """
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).max().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "max_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Max Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)

def get_user_min_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame by minmizing a specified column per individual.

    This function groups chat-level data by user and conversation, calculates the min values
    of a specified numeric column for each user, and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to max for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the min of the specified column per individual.
    :rtype: pandas.DataFrame
    """
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).min().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "min_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Min Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)

def get_user_stdev_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame with the standard deviation a specified column per individual.

    This function groups chat-level data by user and conversation, calculates the standard deviation values
    of a specified numeric column for each user, and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to standard deviation for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the standard deviation of the specified column per individual.
    :rtype: pandas.DataFrame
    """   
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).std().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "stdev_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Standard Deviation of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data) 

def get_user_median_dataframe(chat_level_data, on_column, conversation_id_col, speaker_id_col):
    """Generate a user-level summary DataFrame with the median of a specified column per individual.

    This function groups chat-level data by user and conversation, calculates the median values
    of a specified numeric column for each user, and returns the resulting DataFrame.

    :param chat_level_data: The DataFrame in which each row represents a single chat.
    :type chat_level_data: pandas.DataFrame
    :param on_column: The name of the numeric column to median for each user.
    :type on_column: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :param speaker_id: The column name representing the user identifier.
    :type speaker_id: str
    :return: A grouped DataFrame with the median of the specified column per individual.
    :rtype: pandas.DataFrame
    """   
    grouped_conversation_data = chat_level_data[[conversation_id_col, speaker_id_col, on_column]].groupby([conversation_id_col, speaker_id_col]).median().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "median_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Median of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data) 

def get_mean(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the mean of a specified column per conversation.

    This function calculates the mean of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated mean.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be averaged.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the mean of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform(lambda x: np.mean(x))
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

def get_max(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the maximum value of a specified column per conversation.

    This function calculates the maximum value of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated maximum value.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be aggregated for maximum value.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the maximum value of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform("max")
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

def get_min(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the minimum value of a specified column per conversation.

    This function calculates the minimum value of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated minimum value.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be aggregated for minimum value.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the minimum value of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform("min")
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

def get_stdev(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the standard deviation of a specified column per conversation.

    This function calculates the standard deviation of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated standard deviation.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be aggregated for standard deviation.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the standard deviation of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform(lambda x: np.std(x))
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

def get_median(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the median of a specified column per conversation.

    This function calculates the median of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated median.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be aggregated for median.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID.
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the median of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform(lambda x: np.median(x))
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

def get_sum(input_data, column_to_summarize, new_column_name, conversation_id_col):
    """Generate a summary DataFrame with the sum of a specified column per conversation.

    This function calculates the sum of a specified column for each conversation in the input data,
    and returns a DataFrame containing the conversation number and the calculated sum.

    :param input_data: The DataFrame containing data at the chat or user level.
    :type input_data: pandas.DataFrame
    :param column_to_summarize: The name of the column to be aggregated for the sum.
    :type column_to_summarize: str
    :param new_column_name: The desired name for the new summary column.
    :type new_column_name: str
    :param conversation_id_col: A string representing the column name that should be selected as the conversation ID
    :type conversation_id_col: str
    :return: A DataFrame with the conversation number and the sum of the specified column.
    :rtype: pandas.DataFrame
    """
    input_data[new_column_name] = input_data.groupby([conversation_id_col], sort=False)[column_to_summarize].transform(lambda x: np.sum(x))
    return(input_data[[conversation_id_col, new_column_name]].drop_duplicates())

