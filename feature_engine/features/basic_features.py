import numpy as np

"""
file: basic_features.py
---
Defines simple text features on a text or conversation.
"""

"""
function: count_words

Returns the number of words in a message.
"""
def count_words(text):
	"""Counts the number of words in the chat."""
	return len(text.split())

"""
function: count_characters

Returns the number of characters in a message.
"""
def count_characters(text):
	"""Counts the number of words in the chat."""
	return len(str(text))

'''
function: get_wordcount_dataframe

Returns a dataframe that gets the total number of X per individual, where X could be # words, etc.

@param conversation_data = a dataframe of the conversations, in which each row is one chat.
@param on_column = the name of the numeric column, X, which is summed per individual
'''
def get_count_dataframe(conversation_data, on_column):
	grouped_conversation_data = conversation_data[["batch_num", "round_num", "speaker_nickname", on_column]].groupby(["batch_num", "round_num", "speaker_nickname"]).sum().reset_index() 
    # gets this dataframe:
	# Batch# Round# Speaker  Total Number of Words
	# 0 	 1      Priya    100
	# 0      1      Yuluan   90
	return(grouped_conversation_data)

'''
function: get_message_counts

Returns a dataframe that counts the total number of messages sent per individual.
'''
def get_message_counts(conversation_data):
	conversation_data['num_messages'] = 1
	return(get_count_dataframe(conversation_data, 'num_messages'))


"""
MESSAGE COUNT FUNCTIONS
---
This section implements functions related to the message counts per member.

- Average message count: average_message_count()
- Talkative member message count: most_talkative_member_message_count()
- Laconic member message count: least_talkative_member_message_count()
- Variation in message counts: std_message_count()

"""

'''
function: average_message_count()

Returns the average number of messages sent per member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def average_message_count(conversation_data):
	grouped_conversation_data = get_message_counts(conversation_data)
	grouped_conversation_data['average_message_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_messages'].transform(lambda x: np.mean(x))
	return(grouped_conversation_data[['batch_num', 'round_num', 'average_message_count']].drop_duplicates())

'''
function: most_talkative_member_message_count()

Returns the number of messages of the most talkative member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def most_talkative_member_message_count(conversation_data):
	grouped_conversation_data = get_message_counts(conversation_data)
	grouped_conversation_data['most_talkative_member_message_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_messages'].transform(max)
	return(grouped_conversation_data[['batch_num', 'round_num', 'most_talkative_member_message_count']].drop_duplicates())

'''
function: least_talkative_member_message_count()

Returns the number of messages of the least talkative member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def least_talkative_member_message_count(conversation_data):
	grouped_conversation_data = get_message_counts(conversation_data)
	grouped_conversation_data['least_talkative_member_message_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_messages'].transform(min)
	return(grouped_conversation_data[['batch_num', 'round_num', 'least_talkative_member_message_count']].drop_duplicates())

'''
function: std_message_count()

Returns the standard deviation of the number of messages sent per member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def std_message_count(conversation_data):
	grouped_conversation_data = get_message_counts(conversation_data)
	grouped_conversation_data['std_message_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_messages'].transform(lambda x: np.std(x))
	return(grouped_conversation_data[['batch_num', 'round_num', 'std_message_count']].drop_duplicates())


"""
WORD COUNT FUNCTIONS
---
This section implements functions related to the word counts per member.

- Average word count: average_word_count()
- Talkative member word count: most_talkative_member_word_count()
- Laconic member word count: least_talkative_member_word_count()
- Variation in word counts: std_word_count()

"""

'''
function: average_word_count()

Returns the average number of words sent per member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def average_word_count(conversation_data):
	grouped_conversation_data = get_count_dataframe(conversation_data, "num_words")
	grouped_conversation_data['average_word_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_words'].transform(lambda x: np.mean(x))
	return(grouped_conversation_data[['batch_num', 'round_num', 'average_word_count']].drop_duplicates())

'''
function: most_talkative_member_word_count()

Returns the number of words of the most talkative member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def most_talkative_member_word_count(conversation_data):
	grouped_conversation_data = get_count_dataframe(conversation_data, "num_words")
	grouped_conversation_data['most_talkative_member_word_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_words'].transform(max)
	return(grouped_conversation_data[['batch_num', 'round_num', 'most_talkative_member_word_count']].drop_duplicates())

'''
function: least_talkative_member_word_count()

Returns the number of words of the least talkative member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def least_talkative_member_word_count(conversation_data):
	grouped_conversation_data = get_count_dataframe(conversation_data, "num_words")
	grouped_conversation_data['least_talkative_member_word_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_words'].transform(min)
	return(grouped_conversation_data[['batch_num', 'round_num', 'least_talkative_member_word_count']].drop_duplicates())

'''
function: std_word_count()

Returns the standard deviation of the number of messages sent per member.

Note: this is a team-level metric, so it is the same for all members in a team.
'''
def std_word_count(conversation_data):
	grouped_conversation_data = get_count_dataframe(conversation_data, "num_words")
	grouped_conversation_data['std_word_count'] = grouped_conversation_data.groupby(["batch_num", "round_num"], sort=False)['num_words'].transform(lambda x: np.std(x))
	return(grouped_conversation_data[['batch_num', 'round_num', 'std_word_count']].drop_duplicates())
