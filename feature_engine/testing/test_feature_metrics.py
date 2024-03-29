import pytest
import pandas as pd
import numpy as np
from numpy import nan
import logging

test_chat_df =  pd.read_csv("../output/chat/reddit_test_chat_level.csv")

# test_df['test_pass'] = test_df.apply(lambda row: row[row['expected_column']] == row['expected_value'], axis=1)
# test_df['obtained_value'] = test_df.apply(lambda row: row[row['expected_column']], axis=1)
# test_df[["message", "expected_column", "expected_value", "obtained_value", "test_pass"]]

# Set up logging to capture only error messages
logger = logging.getLogger("pytest_prints")
logger.setLevel(logging.ERROR)

# Add a console handler to the logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)

@pytest.mark.parametrize("row", test_chat_df.iterrows())
def test_chat_unit_equality(row):
    actual = row[1][row[1]['expected_column']]
    expected = row[1]['expected_value']
    
    try:
        assert actual == expected
    except AssertionError:
        logger.error("")
        logger.error("------TEST FAILED------")
        logger.error("Testing %s for message: %s ", row[1]['expected_column'], row[1]['message_original'])
        logger.error("Expected value: %s ", expected)
        logger.error("Actual value: %s", actual)

        raise  # Re-raise the AssertionError to mark the test as failed
