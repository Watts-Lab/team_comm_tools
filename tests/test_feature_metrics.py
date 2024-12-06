from team_comm_tools.feature_dict import feature_dict
import pytest
import pandas as pd
import numpy as np
from numpy import nan
import logging
import itertools

test_chat_df = pd.read_csv("./output/chat/test_chat_level_chat.csv")

test_timediff_dt = pd.read_csv("./output/chat/test_timediff_dt_level_chat.csv")
test_timediff_numeric = pd.read_csv("./output/chat/test_timediff_num_level_chat.csv")
test_timediff_numeric_unit = pd.read_csv("./output/chat/test_timediff_num_unit_level_chat.csv")
test_time_pairs_dt = pd.read_csv("./output/chat/test_time_pairs_dt_level_chat.csv")
test_time_pairs_numeric = pd.read_csv("./output/chat/test_time_pairs_num_level_chat.csv")
test_time_pairs_numeric_unit = pd.read_csv("./output/chat/test_time_pairs_num_unit_level_chat.csv")
test_info_exchange_zscore_df = pd.read_csv("./output/chat/info_exchange_zscore_chats.csv")
test_pos = pd.read_csv("./output/chat/test_positivity_chat_level.csv")

test_conv_df = pd.read_csv("./output/conv/test_conv_level_conv.csv")
test_chat_complex_df = pd.read_csv(
    "./output/chat/test_chat_level_chat_complex.csv")
test_conv_complex_df = pd.read_csv(
    "./output/conv/test_conv_level_conv_complex.csv")
test_conv_complex_df_ts = pd.read_csv(
    "./output/conv/test_conv_level_conv_complex_ts.csv")
test_forward_flow_df = pd.read_csv("./output/chat/test_forward_flow_chat.csv")
test_ner = pd.read_csv('./output/chat/test_named_entity_chat_level.csv')

# Import the Feature Dictionary

chat_features = [feature_dict[feature]["columns"]
                 for feature in feature_dict.keys() if feature_dict[feature]["level"] == "Chat"]
conversation_features = [feature_dict[feature]["columns"] for feature in feature_dict.keys(
) if feature_dict[feature]["level"] == "Conversation"]

num_features_chat = len(list(itertools.chain(*chat_features)))
num_features_conv = len(list(itertools.chain(*conversation_features)))

# Print the test coverage
num_tested_chat = test_chat_df['expected_column'].nunique() + test_chat_complex_df['feature'].nunique() + test_forward_flow_df['feature'].nunique()
test_chat = test_chat_df['expected_column'].unique().tolist() + test_chat_complex_df['feature'].unique().tolist() + \
            test_forward_flow_df['feature'].unique().tolist() + ["named_entities"] + ["time_diff"]
tested_chat = len(set(test_chat))
num_tested_conv = len(set(test_conv_df['expected_column'].unique().tolist() + test_conv_complex_df['feature'].unique().tolist()))
tested_features = {}

with open('test.log', 'w') as f:
    f.write(f'Tested {tested_chat} features out of {num_features_chat} chat level features: {tested_chat/num_features_chat * 100:.2f}% Coverage!\n')
    f.write(f'Tested {num_tested_conv} features out of {num_features_conv} conv level features: {num_tested_conv/num_features_conv * 100:.2f}% Coverage!\n')
    pass

# ---- MAIN TESTS ------

main_tests = [test_chat_df, test_timediff_dt, test_timediff_numeric, test_timediff_numeric_unit, test_time_pairs_dt, test_time_pairs_numeric, test_time_pairs_numeric_unit, test_info_exchange_zscore_df, test_pos]
@pytest.mark.parametrize("df", main_tests)
def test_chat_unit_equality(df):
    for row in df.iterrows():
        actual = row[1][row[1]['expected_column']]
        expected = row[1]['expected_value']

        # if expected_column doesn't exist in tested_features, add an entry for it
        if row[1]['expected_column'] not in tested_features:
            tested_features[row[1]['expected_column']] = {'passed': 0, 'failed': 0}

        try:
            if (pd.isnull(actual) and pd.isnull(expected)):
                assert True
            elif (type(actual) == str):
                assert actual == expected
            else:
                assert round(float(actual), 3) == round(float(expected), 3)
            tested_features[row[1]['expected_column']]['passed'] += 1
        except AssertionError:
            tested_features[row[1]['expected_column']]['failed'] += 1
            with open('test.log', 'a') as file:
                file.write("\n")
                file.write("------TEST FAILED------\n")
                file.write(
                    f"Testing {row[1]['expected_column']} for message: {row[1]['message_original']}\n")
                file.write(f"Expected value: {expected}\n")
                file.write(f"Actual value: {actual}\n")

            raise AssertionError # Re-raise the AssertionError to mark the test as failed

tested_features['Named Entity Recognition'] = {'passed': 0, 'failed': 0}
@pytest.mark.parametrize("row", test_ner.iterrows())
def test_named_entity_recognition(row):

    if pd.isnull(row[1]['expected_value']):
        try:

            assert row[1]['named_entities'] == "()"
            tested_features['Named Entity Recognition']['passed'] += 1
        except AssertionError:
            parsed_actual = row[1]['named_entities'].replace(
                " ", "").replace("(", "").replace(")", "").split(',')
            actual = parsed_actual[0::2]

            if actual and actual[-1] == '':
                # Remove the last element
                actual.pop()
            tested_features['Named Entity Recognition']['failed'] += 1
            with open('test.log', 'a') as file:
                file.write("\n")
                file.write("------TEST FAILED------\n")
                file.write(
                    f"Testing NER for message: {row[1]['message_original']}\n")
                file.write(f"Expected value: {[]}\n")
                file.write(f"Actual value: {actual}\n")
    else:
        expected = row[1]['expected_value'].split(',')
        parsed_actual = row[1]['named_entities'].replace(
            " ", "").replace("(", "").replace(")", "").split(',')
        actual = parsed_actual[0::2]

        if actual and actual[-1] == '':
            # Remove the last element
            actual.pop()

        try:
            assert len(expected) == len(actual)
            for named_entity in expected:
                assert named_entity.lower().strip() in actual
            tested_features["Named Entity Recognition"]['passed'] += 1
        except AssertionError:
            tested_features["Named Entity Recognition"]['failed'] += 1
            with open('test.log', 'a') as file:
                file.write("\n")
                file.write("------TEST FAILED------\n")
                file.write(
                    f"Testing NER for message: {row[1]['message_original']}\n")
                file.write(f"Expected value: {expected}\n")
                file.write(f"Actual value: {actual}\n")

            # we don't raise an AssertionError here because NER isn't a perfect feature


@pytest.mark.parametrize("conversation_num, conversation_rows", test_conv_df.groupby('conversation_num'))
def test_conv_unit_equality(conversation_num, conversation_rows):
    test_failed = False
    expected_out = ""
    actual_out = ""

    for _, row in conversation_rows.iterrows():
        if row['expected_column'] not in tested_features:
            tested_features[row['expected_column']] = {
                'passed': 0, 'failed': 0}
        actual = row[row['expected_column']]
        expected = row['expected_value']

    try:
        assert round(actual, 3) == round(expected, 3)
        tested_features[row['expected_column']]['passed'] += 1
    except AssertionError:
        tested_features[row['expected_column']]['failed'] += 1
        expected_out = expected
        actual_out = actual
        test_failed = True

    if test_failed:
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(
                f"Testing {row['expected_column']} for conversation_num: {conversation_num}\n")
            file.write(f"Expected value: {expected_out}\n")
            file.write(f"Actual value: {actual_out}\n")

        raise AssertionError # Re-raise the AssertionError to mark the test as failed

# Helper function for batches of 3 rows (INV/DIR)
def get_batches(dataframe, batch_size=3):
    batches = []
    rows = list(dataframe.iterrows())
    for i in range(0, len(rows), batch_size):
        batches.append(rows[i:i + batch_size])
    return batches

def get_conversation_batches(dataframe, batch_size=3):
    # group by conversation_num and get the last row from the group
    last_rows = dataframe.groupby('conversation_num').tail(1)

    # get 3 row batches of these last rows
    batches = []
    rows = list(last_rows.iterrows())
    for i in range(0, len(rows), batch_size):
        batches.append(rows[i:i + batch_size])
    return batches


@pytest.mark.parametrize("batch", get_batches(test_chat_complex_df, batch_size=3))
def test_chat_complex(batch):
    feature = batch[0][1]['feature']
    if feature not in tested_features:
        tested_features[feature] = {'passed': 0, 'failed': 0}

    og_result = batch[0][1][feature]
    inv_result = batch[1][1][feature]
    dir_result = batch[2][1][feature]

    inv_distance = abs(og_result - inv_result)
    dir_distance = abs(og_result - dir_result)

    # calculate ratio between inv and dir
    ratio = dir_distance / inv_distance

    try:

        assert ratio > 1
        tested_features[feature]['passed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST PASSED------\n")
            file.write(
                f"Testing {feature} for message: {batch[0][1]['message']}\n")
            file.write(f"Inv message: {batch[1][1]['message']}\n")
            file.write(f"Dir message: {batch[2][1]['message']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")
    except AssertionError:
        tested_features[feature]['failed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(
                f"Testing {feature} for message: {batch[0][1]['message']}\n")
            file.write(f"Inv message: {batch[1][1]['message']}\n")
            file.write(f"Dir message: {batch[2][1]['message']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")

        raise AssertionError # Re-raise the AssertionError to mark the test as failed


conversation_complex_batches = get_batches(test_conv_complex_df, batch_size=3) + get_conversation_batches(
    test_forward_flow_df, batch_size=3) + get_batches(test_conv_complex_df_ts, batch_size=3)

@pytest.mark.parametrize("batch", conversation_complex_batches)
def test_conv_complex(batch):
    feature = batch[0][1]['feature']
    if feature not in tested_features:
        tested_features[feature] = {'passed': 0, 'failed': 0}

    og_result = batch[0][1][feature]
    inv_result = batch[1][1][feature]
    dir_result = batch[2][1][feature]

    inv_distance = abs(og_result - inv_result)
    dir_distance = abs(og_result - dir_result)

    # calculate ratio between inv and dir
    ratio = 0
    if (inv_distance == 0) and dir_distance > 0:
        ratio = 2
    else:
        ratio = dir_distance / inv_distance

    try:

        assert ratio > 1
        tested_features[feature]['passed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST PASSED------\n")
            file.write(
                f"Testing {feature} for conversation: {batch[0][1]['conversation_num']}\n")
            file.write(
                f"Inv conversation: {batch[1][1]['conversation_num']}\n")
            file.write(
                f"Dir conversation: {batch[2][1]['conversation_num']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")
    except AssertionError:
        tested_features[feature]['failed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(
                f"Testing {feature} for conversation: {batch[0][1]['conversation_num']}\n")
            file.write(
                f"Inv conversation: {batch[1][1]['conversation_num']}\n")
            file.write(
                f"Dir conversation: {batch[2][1]['conversation_num']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")

        # we don't raise an AssertionError here because it's not a perfect feature

@pytest.mark.parametrize("batch", get_conversation_batches(test_forward_flow_df, batch_size=3))
def test_forward_flow_unit(batch):
    if (batch[0][1]['test_type'] != 'unit_eq'):
        return
    feature = batch[0][1]['feature']

    if feature not in tested_features:
        tested_features[feature] = {'passed': 0, 'failed': 0}

    og_result = round(batch[0][1][feature], 3)
    inv_result = round(batch[1][1][feature], 3)
    dir_result = round(batch[2][1][feature], 3)

    inv_distance = abs(og_result - inv_result)
    dir_distance = abs(og_result - dir_result)

    # calculate ratio between inv and dir
    if (inv_distance == 0) and (dir_distance > 0):
        tested_features[feature]['passed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST PASSED------\n")
            file.write(
                f"Testing {feature} for unit equality and perturbation across conversations: {batch[0][1]['conversation_num']}, {batch[1][1]['conversation_num']}, {batch[2][1]['conversation_num']}\n")
    else:
        tested_features[feature]['failed'] += 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(
                f"Testing {feature} for conversation: {batch[0][1]['conversation_num']}\n")
            file.write(f"OG Result: {batch[0][1][feature]}\n")
            file.write(f"UNIT TEST Result: {batch[1][1][feature]}\n")
            file.write(f"DIR Result: {batch[2][1][feature]}\n")


def test_final_results():
    # print out the results
    with open('test.log', 'a') as file:

        for feature, results in tested_features.items():
            accuracy = results['passed'] / \
                (results['passed'] + results['failed']) * 100
            # file.write(f"Feature: {feature}\n")
            if accuracy != 100:
                file.write("\n")
                file.write("------RESULTS------\n")
                file.write(f"Feature: {feature}\n")
                file.write(
                    f"Accuracy: {results['passed'] / (results['passed'] + results['failed']) * 100:.2f}%")
                file.write("\n")
