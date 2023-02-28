import pytest
# import our feature files
from features.basic_features import *

"""
file: test_featurize.py
---
This file uses pytest to define unit testing functions for
the feature-generating functions (in features/).

When run, this file will run tests to confirm that the feature-
generating functions generate the correct expected output.

To run all test in this file run: `python3 -m pytest`

For pytest documentation: https://docs.pytest.org/en/7.1.x/getting-started.html
"""

def test_count_words_basic():
	assert count_words("hello world") == 2

def test_count_characters_basic():
	assert count_characters("hello world") == 11