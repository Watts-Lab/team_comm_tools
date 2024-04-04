import numpy as np
import csv
import pandas as pd

"""
file: turn-taking_features.py
---
Calculates the turn-taking index for each speaker
"""

"""
function: count_turns

Returns the total number of turns for each speaker
"""

"""
@param input_date = a dataframe of conversations, in which each row is one chat
"""

def count_turns(input_data):
    temp_turn_count = 1
    consolidated_actions = []

    start_index = input_data.index[0] + 1
    end_index = len(input_data) + input_data.index[0]

    for turn_index in range(start_index, end_index):

        if input_data["speaker_nickname"][turn_index] == input_data["speaker_nickname"][turn_index - 1]:
            temp_turn_count += 1
        else:
            consolidated_actions.append((input_data["speaker_nickname"][turn_index - 1], temp_turn_count))
            temp_turn_count = 1

        if turn_index == max(range(start_index, end_index)):
            consolidated_actions.append((input_data["speaker_nickname"][turn_index], temp_turn_count))

    assert sum(x[1] for x in consolidated_actions) == len(input_data)

    df_consolidated_actions = pd.DataFrame(columns=["speaker_nickname", "turn_count"], data=consolidated_actions)

    return df_consolidated_actions

"""
function: count_turn_taking_index


Returns the turn-taking index for each speaker
"""

def count_turn_taking_index(input_data):
    return (len(count_turns(input_data)) - 1) / (len(input_data) - 1)

def get_turn(input_data):
    turn_calculated_2 = input_data.groupby("conversation_num").apply(lambda x : count_turn_taking_index(x)).reset_index().rename(columns={0: "turn_taking_index"})
    return turn_calculated_2



# import unittest
# class TestTurnTakingFeatures(unittest.TestCase):
#
#     def test_count_turns(self):
#         input_data = pd.DataFrame({
#             "conversation_num": ["ab", "ab", "ab", "ab", "ab"],
#             "speaker_nickname": ["A", "A", "B", "A", "B"]
#         })
#
#         expected_output = pd.DataFrame({
#             "speaker_nickname": ["A", "B", "A", "B"],
#             "turn_count": [2, 1, 1, 1]
#         })
#
#         self.assertTrue(count_turns(input_data).equals(expected_output))
#
#     def test_count_turn_taking_index(self):
#         input_data = pd.DataFrame({
#             "conversation_num": ["ab", "ab", "ab", "ab", "ab"],
#             "speaker_nickname": ["A", "A", "B", "A", "B"]
#         })
#
#         expected_output = 0.75  # (4 - 1) / (5 - 1)
#
#         self.assertEqual(count_turn_taking_index(input_data), expected_output)
#
#     def test_get_turn(self):
#         input_data = pd.DataFrame({
#             "conversation_num": ["ab", "ab", "ab", "ab", "ab", "cd", "cd"],
#             "speaker_nickname": ["A", "A", "B", "A", "B", "C", "C"]
#         })
#
#         expected_output = pd.DataFrame({
#             "conversation_num": ["ab", "cd"],
#             "turn_taking_index": [0.75, 0]
#         })
#
#         self.assertTrue(get_turn(input_data).equals(expected_output))
#
#
# if __name__ == '__main__':
#     unittest.main()
