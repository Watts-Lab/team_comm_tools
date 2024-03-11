import pytest
import pandas as pd
import numpy as np
from numpy import nan

df =  pd.read_csv("../output/chat/csopII_output_chat_level.csv")
df['test_pass'] = df.apply(lambda row: row["message"] == row["num_words"], axis=1)
print(df['test_pass'])

# @pytest.mark.parametrize("row", df.head(5).iterrows())
# def test_col_exists(row):
#     assert row[1]["message"] == "i think 504 would still be ok"

def test_col_exists():
    assert False in df['test_pass'].values
