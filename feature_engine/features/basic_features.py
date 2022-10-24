import numpy as np

"""
file: basic_features.py
---
Defines simple text features on a text or conversation.
"""

def count_words(text):
	"""Counts the number of words in the chat."""
	return len(text.split())

def count_characters(text):
	"""Counts the number of words in the chat."""
	return len(str(text))