import re
import pandas as pd
import warnings



def preprocess_conversation_columns(df: pd.DataFrame, column_names: dict, grouping_keys: list, 
                                    cumulative_grouping: bool = False, within_task: bool = False) -> pd.DataFrame:
    """
    Preprocesses conversation data by removing special characters from column names and assigning a conversation number.

    :param df: The DataFrame containing conversation data.
    :type df: pd.DataFrame
    :param conversation_id: The column name to use for assigning conversation numbers.
    :type conversation_id: str, optional
    :param timestamp_col: The name of the column containing the timestamp
    :type timestamp_col: str
    :param grouping_keys: A list of multiple identifier keys for a conversation.
    :type grouping_keys: list
    :param cumulative_grouping: Whether to group data cumulatively based on the conversation_id.
           This option was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica).
    :type cumulative_grouping: bool, optional
    :param within_task: Used only if cumulative_grouping is True, to specify if grouping is within the "task." 
        This option was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica).
    :type within_task: bool, optional
    :return: The preprocessed DataFrame with a conversation number column.
    :rtype: pd.DataFrame
    """
    # remove all special characters from df
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
    if not grouping_keys: # case 1: single identifier
        return df
    if not set(grouping_keys).issubset(df.columns):
        raise ValueError("One or more grouping keys does not exist in the column set.")
    if cumulative_grouping and len(grouping_keys) == 3: # case 3: cumulative grouping
        df = create_cumulative_rows(df, column_names['conversation_id_col'], column_names['timestamp_col'], grouping_keys, within_task)
    else: # case 2: grouping multiple keys, or case 3 but not 3 layers
        df['conversation_num'] = df.groupby(grouping_keys).ngroup()
        df = df[df.columns.tolist()[-1:] + df.columns.tolist()[0:-1]] # make the new column first
    return df

def assert_key_columns_present(df: pd.DataFrame, column_names: dict) -> None:
    """Ensure that the DataFrame has essential columns and handle missing values.
    
    This function  if the essential columns `conversation_id_col`, `speaker_id_col`, and 
    `message_col` are present. If any of these columns are missing, a 
    KeyError is raised. 

    :param df: The DataFrame to check and process.
    :type df: pandas.DataFrame
    :param column_names: Columns to preprocess.
    :type column_names: dict
    :raises KeyError: If one of `conversation_id_col`, `speaker_id_col`, and `message_col` columns is missing.
    """
    
    # Assert that key columns are present
    for role, col in column_names.items():
        if role == 'timestamp_col':
            continue # skip timestamp column
        if col not in df.columns:
            raise KeyError(f"Missing required columns in DataFrame: '{col}' (expected for {role})\n Columns available: {df.columns}")
        else:
            print(f"Confirmed that data has {role} column: {col}!")
            df[col] = df[col].fillna('')

def remove_unhashable_cols(df: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    """
    If a required column contains unhashable types, raise an error.
    Otherwise, remove those columns from the DataFrame and print a warning message.

    :param df: Pandas DataFrame to validate
    :type df: pd.DataFrame
    :param column_names: Dictionary of 4 required columns that must not contain unhashable types
    :type column_names: dict

    :return: Cleaned DataFrame (if columns were removed)
    :rtype: pd.DataFrame
    :raises ValueError: if a required column contains unhashable types
    """
    # Check cols with unhashable types
    unhashable_cols = {}

    def is_unhashable(obj):
        """
        Small function to test whether a data type is hashable, without storing a hard-coded list of hashable types.
        
        :param obj: an object to test hashability
        :return: Whether or not the object is unhashable
        :rtype: bool
        """
        try:
            hash(obj)
            return False
        except TypeError:
            return True

    for col in df.columns:
        unhashable_values = df[col].apply(lambda x: is_unhashable(x))
        if unhashable_values.any():
            unique_types = df[col][unhashable_values].apply(lambda x: type(x)).unique()
            unhashable_cols[col] = unique_types
    # Check if any required columns contain unhashable types
    removable_cols = []
    for col, types in unhashable_cols.items():
        error_message = f"Column '{col}' has unhashable data types: {[t.__name__ for t in types]}"
        if col in column_names.values():
            raise ValueError(error_message)
        else:
            warnings.warn(f"WARNING: {error_message}. Removing '{col}' from the DataFrame.")
            removable_cols.append(col)
    if removable_cols:
        df = df.drop(columns=removable_cols)
    return df

def preprocess_text_lowercase_but_retain_punctuation(text):
    """Convert the input text to lowercase while retaining punctuation.

    This function takes a string and converts all characters to lowercase,
    keeping any punctuation marks intact.

    :param text: The input text to process.
    :type text: str
    :return: The processed text with all characters in lowercase.
    :rtype: str
    """
    return(text.lower())

def preprocess_text(text: str) -> str:
    """Preprocess text by removing non-alphanumeric characters and converting to lowercase.

    This function takes a string, removes any characters that are not letters, numbers, or spaces,
    except certain emojis, and converts the remaining text to lowercase.

    :param text: The input text to process.
    :type text: str
    :return: The processed text containing only alphanumeric characters and spaces in lowercase.
    :rtype: str
    """
    emojis_to_preserve = {
        "(:", "(;", "):", "/:", ":(", ":)", ":/", ";)"
    }

    emoji_placeholders = {}
    # Replace each emoji with a unique placeholder
    for i, emoji in enumerate(emojis_to_preserve):
        placeholder = f"EMOJI_{i}"
        emoji_placeholders[placeholder] = emoji
        text = text.replace(emoji, placeholder)

    # Clean the text by removing unwanted characters, except placeholders
    text = re.sub(r"[^a-zA-Z0-9 EMOJI_]+", '', text)
    # Restore the preserved emojis by replacing placeholders back to original emojis
    for placeholder, emoji in emoji_placeholders.items():
        text = text.replace(placeholder, emoji)

    return text.lower()

def preprocess_naive_turns(chat_data, column_names):
    """Combine adjacent rows of the same speaker in the same conversation and compress messages into a "turn".

    This function first generates a 'turn_id' for each chat message within the same conversation,
    indicating turns taken by the active speaker. It then combines messages with the same 'turn_id' 
    within each conversation to compress repeated messages from the same speaker.

    :param chat_data: The chat data to process.
    :type chat_data: pandas.DataFrame
    :param column_names: Columns to preprocess.
    :type column_names: dict
    :return: The processed chat data with combined message turns.
    :rtype: pandas.DataFrame
    """
    conversation_id_col = column_names['conversation_id_col']
    message_col = column_names['message_col']
    speaker_id_col = column_names['speaker_id_col']
    turn_id_per_conv = (
        chat_data.groupby([conversation_id_col], sort=False)
        .apply(lambda df : get_turn_id(df, speaker_id_col))
        .reset_index(level=0, drop=True)  # Ensures long format
    )
    if len(chat_data[conversation_id_col].unique()) == 1:
        turn_id_per_conv = turn_id_per_conv.T
    else:
        turn_id_per_conv = turn_id_per_conv.to_frame()
    turn_id_per_conv = turn_id_per_conv.rename(columns={0:'turn_id'})
    turn_id_per_conv["turn_id"] = turn_id_per_conv["turn_id"].astype(str)
    # chat_data = pd.concat([chat_data, turn_id_per_conv["turn_id"]], axis=1)
    chat_data = chat_data.merge(turn_id_per_conv, left_index=True, right_index=True, how="left") # merge with index to preserve order
    
    # Use turn_id to compress messages with the same turn id per conversation
    chat_data = chat_data.groupby(conversation_id_col, sort=False).apply(
        lambda df : df.groupby('turn_id', as_index=False).apply(lambda df : compress(df, message_col))).reset_index(drop=True)
    
    return chat_data

def get_turn_id(df, speaker_id_col):
    """Generate turn IDs for a conversation to identify turns taken by speakers.

    This function compares the current speaker with the previous one to identify when a change in speaker occurs, 
    and then assigns a unique 'turn_id' that increments whenever the speaker changes within the conversation.

    :param df: The DataFrame containing chat data for a single conversation.
    :type df: pandas.DataFrame
    :param speaker_id_col: A string representing the column name that should be selected as the speaker ID.
    :type speaker_id_col: str
    :return: A Series containing the turn IDs.
    :rtype: pandas.Series
    """
    df[f"{speaker_id_col}_x"] = df[speaker_id_col].shift()
    return (df[speaker_id_col] != df[f"{speaker_id_col}_x"]).cumsum()
    
def compress(turn_df, message_col):
    """Combine messages in the same turn into a single message.

    This function takes a DataFrame representing messages in a single turn and
    concatenates their 'message' and 'message_lower_with_punc' columns into
    single strings if there are multiple messages in the same turn.

    :param turn_df: The DataFrame containing messages in a single turn.
    :type turn_df: pandas.DataFrame
    :param message_col: A string representing the column name that should be selected as the message.
    :type message_col: str
    :return: A Series with combined messages for the turn.
    :rtype: pandas.Series
    """
    result = turn_df.iloc[0]
    if (len(turn_df) > 1):
        result[message_col] = turn_df[message_col].str.cat(sep=' ')
        result['message_lower_with_punc'] = turn_df['message_lower_with_punc'].str.cat(sep=' ')
        result[message_col + "_original"] = turn_df[message_col + "_original"].str.cat(sep=' ')
    return result

def create_cumulative_rows(input_df, conversation_id, timestamp_col, grouping_keys, within_task = False):
    """Generate cumulative rows for chat data to analyze conversations in context.

    This function takes chat-level data and duplicates rows to facilitate the analysis of conversations
    in the context of preceding chats. It enables the inclusion of chats from previous stages or tasks within
    the same conversation.

    NOTE: This function was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica).
    
    It assumes that there are exactly 3 nested columns at different levels: a High, Mid, and Low level; further, it assumes that these levels are temporally nested: that is, each
    group/conversation has one High-level identifier, which contains one or more Mid-level identifiers, which contains one or more Low-level identifiers.

    This is specifically applicable to a hierachical conversation in which the same group of pairing does a series of different activities, each of which
    may have one or more subparts. Thus, the group as a whole will have a "high-level" identiifer; each activity will have a "mid-level" identifier, and each
    sub-part will have a "low-level" identifier.

    :param input_df: The DataFrame containing chat data.
    :type input_df: pandas.DataFrame

    :param conversation_id: The ID (e.g., stage or round) used for grouping the data.
    :type conversation_id: str

    :param timestamp_col: The column containing the timestamp. Since we assume that the conversation is evolving over time, we use the timestamp column to
        make the analysis of conversation "cumulative" (that is, to include in our analysis prior discussions for other activities).
    :type timestamp_col: str

    :param grouping_keys: A list of three hierarchical keys, which must be passed in the order of (highest level, mid level, lowest level). 
        We assume that, for a given item at the highest level, there are one or more items at the mid level; for each item at the mid level, there
        are one or more items at the lowest level.
    :type grouping_keys: list

    :param within_task: Flag to determine whether to restrict the analysis to the same activity or "task" (assumed to be the Mid-Level Identifier), defaults to False.
    :type within_task: bool, optional

    :return: The processed DataFrame with cumulative rows added.
    :rtype: pandas.DataFrame
    """
    level_high, level_mid, level_low = grouping_keys[0], grouping_keys[1], grouping_keys[2] # In Empirica data: ['gameId', 'roundId', 'stageId']

    # If the conversation_id is the highest level ID (gameId), return as is -- no changes requred
    if(conversation_id == level_high): return input_df

    # print a warning in case user gave incompatible instructions:
    if(conversation_id == level_mid and within_task):
        print("WARNING: Cumulative grouping with the mid-level identifier is incompatible with the `within_task` parameter. Ignoring `within_task` parameter.")

    result_df = pd.DataFrame(columns=input_df.columns)

    # prev stageId
    prev_low_level_id = None

    # Iterate through rows
    for _, current_row in input_df.iterrows():
            
        # current stageId
        if current_row[level_low] != prev_low_level_id: # we have transitioned to a new low-level identifier (subactivity)

            prev_low_level_id = current_row[level_low]

            if(conversation_id == level_low):
                # Duplicate rows from all previous stageId's with the same high-level identifier
                if(within_task): # ensure the mid-level identifier is the same
                    previous_rows = input_df.loc[(input_df[level_low] != current_row[level_low]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high]) & (input_df[level_mid] == current_row[level_mid])].copy()
                else:
                    previous_rows = input_df.loc[(input_df[level_low] != current_row[level_low]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high])].copy()
                if(not previous_rows.empty):
                    previous_rows['conversation_num'] = current_row[level_low]
                    result_df = pd.concat([result_df, previous_rows], ignore_index=True)
            if(conversation_id == level_mid):
                # Duplicate rows from all previous mid-level identifiers with the same high-level identifier
                previous_rows = input_df.loc[(input_df[level_mid] != current_row[level_mid]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high])].copy()
                if(not previous_rows.empty):
                    previous_rows['conversation_num'] = current_row[level_mid]
                    result_df = pd.concat([result_df, previous_rows], ignore_index=True)

            cur_Id_rows = input_df.loc[(input_df[conversation_id] == current_row[conversation_id])].copy()
            cur_Id_rows['conversation_num'] = current_row[conversation_id]
            # Concatenate the current row to the result DataFrame

            result_df = (result_df.copy() if cur_Id_rows.empty else cur_Id_rows.copy() if result_df.empty
               else pd.concat([result_df, cur_Id_rows], ignore_index=True).drop_duplicates() # to silence FutureWarning
              )

    return result_df
