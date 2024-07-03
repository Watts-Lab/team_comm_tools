import re
import pandas as pd

def preprocess_conversation_columns(df, conversation_id, timestamp_col, grouping_keys, cumulative_grouping = False, within_task = False):
	"""
    Preprocesses conversation data by removing special characters from column names and assigning a conversation number.

    :param df: The DataFrame containing conversation data.
    :type df: pd.DataFrame
    :param conversation_id: The column name to use for assigning conversation numbers.
    :type conversation_id: str, optional
    :param cumulative_grouping: Whether to group data cumulatively based on the conversation_id.
    :type cumulative_grouping: bool, optional
    :param within_task: Used only if cumulative_grouping is True, to specify if grouping is within the task.
    :type within_task: bool, optional
    :return: The preprocessed DataFrame with a conversation number column.
    :rtype: pd.DataFrame
    """

	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
	
	if not grouping_keys: # case 1: single identifier
		return df
	if not set(grouping_keys).issubset(df.columns):
		print(df)
		print(grouping_keys)
		raise ValueError("One or more grouping keys does not exist in the column set.")
	if cumulative_grouping and len(grouping_keys) == 3: # case 3: cumulative grouping
		df = create_cumulative_rows(df, conversation_id, timestamp_col, grouping_keys, within_task)
	else: # case 2: grouping multiple keys, or case 3 but not 3 layers
		df['conversation_num'] = df.groupby(grouping_keys).ngroup()
		df = df[df.columns.tolist()[-1:] + df.columns.tolist()[0:-1]] # make the new column first

	return df

def assert_key_columns_present(df, column_names):
	"""Ensure that the DataFrame has essential columns and handle missing values.
    
    This function removes all special characters from the DataFrame column names
    and checks if the essential columns `conversation_num`, `message`, and 
    `speaker_nickname` are present. If any of these columns are missing, a 
    KeyError is raised. It also fills missing values in the essential columns 
    with default values.

    :param df: The DataFrame to check and process.
    :type df: pandas.DataFrame
	:param column_names: Columns to preprocess.
    :type column_names: dict
    :raises KeyError: If one of `conversation_num`, `message`, or `speaker_nickname` columns is missing.
    """

	conversation_id_col = column_names['conversation_id_col']
	speaker_id_col = column_names['speaker_id_col']
	message_col = column_names['message_col']

	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

	# Assert that key columns are present
	if {conversation_id_col, speaker_id_col, message_col}.issubset(df.columns):
		print(f"Confirmed that data has conversation_id: {conversation_id_col}, speaker_id: {speaker_id_col} and message: {message_col} columns!")
		# ensure no NA's in essential columns
		df[conversation_id_col] = df[conversation_id_col].fillna(0)
		df[speaker_id_col] = df[speaker_id_col].fillna(0)
		df[message_col] = df[message_col].fillna('')
	else:
		print("One or more of conversation_id, speaker_id or message columns are missing! Raising error...")
		print("Columns available: ")
		print(df.columns)
		raise KeyError

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

def preprocess_text(text):
	"""Preprocess text by removing non-alphanumeric characters and converting to lowercase.

    This function takes a string, removes any characters that are not letters, numbers, or spaces,
    and converts the remaining text to lowercase.

    :param text: The input text to process.
    :type text: str
    :return: The processed text containing only alphanumeric characters and spaces in lowercase.
    :rtype: str
    """
	return(re.sub(r"[^a-zA-Z0-9 ]+", '',text).lower())

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
	turn_id_per_conv = chat_data.groupby([conversation_id_col], sort=False).apply(lambda df : get_turn_id(df, speaker_id_col))
	turn_id_per_conv = turn_id_per_conv.to_frame().reset_index().rename(columns={0:'turn_id'})
	chat_data = pd.concat([chat_data, turn_id_per_conv["turn_id"]], axis=1)
	
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
	return result

def create_cumulative_rows(input_df, conversation_id, timestamp_col, grouping_keys, within_task = False):
	"""Generate cumulative rows for chat data to analyze conversations in context.

    This function takes chat-level data and duplicates rows to facilitate the analysis of conversations
    in the context of preceding chats. It enables the inclusion of chats from previous stages or tasks within
    the same conversation.

    NOTE: This function was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica).
    
	It assumes that there are exactly 3 nested columns at different levels: a High, Mid, and Low level; further, it assumes that these levels are temporally nested: that is, each
	group/conversation has one High-level identifier, which contains one or more Mid-level identifiers, which contains one or more Low-level identifiers.

    :param input_df: The DataFrame containing chat data.
    :type input_df: pandas.DataFrame
    :param conversation_id: The ID (e.g., stage or round) used for grouping the data.
    :type conversation_id: str
    :param within_task: Flag to determine whether to restrict the analysis to the same task (assumed to be the Mid-Level Identifier), defaults to False.
    :type within_task: bool, optional
    :return: The processed DataFrame with cumulative rows added.
    :rtype: pandas.DataFrame
    """
	level_high, level_mid, level_low = grouping_keys[0], grouping_keys[1], grouping_keys[2] # In Empirica data: ['gameId', 'roundId', 'stageId']

	# If the conversation_id is the highest level ID (gameId), return as is -- no changes requred
	if(conversation_id == level_high): return input_df

	result_df = pd.DataFrame(columns=input_df.columns)

	# prev stageId
	prev_stageId = None

	# Iterate through rows
	for _, current_row in input_df.iterrows():
			
		# current stageId
		if current_row[level_low] != prev_stageId: # we have transitioned to a new stageId

			prev_stageId = current_row[level_low]

			if(conversation_id == level_low):
				# Duplicate rows from all previous stageId's with the same 'gameId'
				if(within_task): # ensure roundId's are the same
					previous_rows = input_df.loc[(input_df[level_low] != current_row[level_low]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high]) & (input_df[level_mid] == current_row[level_mid])].copy()
				else:
					previous_rows = input_df.loc[(input_df[level_low] != current_row[level_low]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high])].copy()
				if(not previous_rows.empty):
					previous_rows['conversation_num'] = current_row[level_low]
					result_df = pd.concat([result_df, previous_rows], ignore_index=True)
			if(conversation_id == level_mid):
				# Duplicate rows from all previous roundId's with the same gameId
				previous_rows = input_df.loc[(input_df[level_mid] != current_row[level_mid]) & (input_df[timestamp_col] < current_row[timestamp_col]) & (input_df[level_high] == current_row[level_high])].copy()
				if(not previous_rows.empty):
					previous_rows['conversation_num'] = current_row[level_mid]
					result_df = pd.concat([result_df, previous_rows], ignore_index=True)

			cur_Id_rows = input_df.loc[(input_df[conversation_id] == current_row[conversation_id])].copy()
			cur_Id_rows['conversation_num'] = current_row[conversation_id]
			# Concatenate the current row to the result DataFrame
			result_df = pd.concat([result_df, cur_Id_rows], ignore_index=True).drop_duplicates()

	return result_df
