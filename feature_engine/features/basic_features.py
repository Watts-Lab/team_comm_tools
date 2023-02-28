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


"""
function: count_messages

This is trivial; by definition, each message counts as 1.
(Useful for grouping later.)
"""
def count_messages(text):
	return(1)