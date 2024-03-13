import pytest
import pandas as pd
import numpy as np
from numpy import nan

test_df =  pd.read_csv("../output/chat/test_num_words.csv")
# test_df['test_pass'] = test_df.apply(lambda row: row[row['expected_column']] == row['expected_value'], axis=1)
# test_df['obtained_value'] = test_df.apply(lambda row: row[row['expected_column']], axis=1)
# test_df[["message", "expected_column", "expected_value", "obtained_value", "test_pass"]]

@pytest.mark.parametrize("row", test_df.iterrows())
def test_unit_equality(row):
    assert row[1][row[1]['expected_column']] == row[1]['expected_value']
