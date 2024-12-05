import pandas as pd

def calculate_lsm(chat_df):
    
    """ 
    This function calculates Language Style Matching (LSM) scores for the Team Communication Toolkit.
    
    Source: Language Style Matching as a Predictor of Social Dynamics in Small Groups by Amy L. Gonzales, Jeffrey T. Hancock, and James W. Pennebaker.

     Args:
        chat_df (pd.DataFrame): A pandas DataFrame with columns for conversation_id, speaker_id, 
            and various word-level counts (e.g., num_words, conjunction_lexical_wordcount, etc.).

    Returns:
        pd.DataFrame: A pandas DataFrame with additional columns for LSM scores and related calculations.
    """
    
    # Create a new column with the sum of all pronouns (first person singular, first person plural, second person, third person)
    chat_df['total_pronouns'] = (
        chat_df['first_person_singular_lexical_wordcount'] +
        chat_df['first_person_plural_lexical_wordcount'] +
        chat_df['second_person_lexical_wordcount'] +
        chat_df['third_person_lexical_wordcount']
    )

    # Group by conversation_id and speaker_id to prepare for LSM calculations
    grouped_df = chat_df.groupby(['conversation_id', 'speaker_id']).agg({
        'num_words': 'sum',
        'conjunction_lexical_wordcount': 'sum',
        'total_pronouns': 'sum',
        'adverbs_lexical_wordcount': 'sum',
        'article_lexical_wordcount': 'sum',
        'quantifier_lexical_wordcount': 'sum',
        'negation_lexical_wordcount': 'sum',
        'preposition_lexical_wordcount': 'sum',
        'indefinite_pronoun_lexical_wordcount': 'sum',
        'auxiliary_verbs_lexical_wordcount': 'sum',
    }).reset_index()
    # Resets the index so speaker is treated as a normal column
                     
    # Now start calculating LSM score (for each function word category for each person)
   
    # List of function word columns to calculate percentages for
    function_word_columns = [
        'total_pronouns',
        'conjunction_lexical_wordcount',
        'adverbs_lexical_wordcount',
        'article_lexical_wordcount',
        'quantifier_lexical_wordcount',
        'negation_lexical_wordcount',
        'preposition_lexical_wordcount',
        'indefinite_pronoun_lexical_wordcount',
        'auxiliary_verbs_lexical_wordcount'
    ]

    # Loop through each function word column and divide it by the total number of words (eg. x percent of total words were pronouns)
    for column in function_word_columns:
        grouped_df[f'{column}_percent'] = (grouped_df[column] / grouped_df['num_words']) * 100 

    # Compute group-level sums and counts for each conversation
    group_sums = grouped_df.groupby('conversation_id')[function_word_columns].transform('sum')
    group_counts = grouped_df.groupby('conversation_id')[function_word_columns].transform('count')

    # Calculate group averages excluding the current speaker
    for column in function_word_columns:
        grouped_df[f'{column}_group_avg'] = (group_sums[column] - grouped_df[column]) / (group_counts[column] - 1)

    # Calculate LSM score
    for column in function_word_columns:
        grouped_df[f'{column}_lsm'] = 1 - (abs(grouped_df[f'{column}_percent'] - grouped_df[f'{column}_group_avg']) / (grouped_df[f'{column}_percent'] + grouped_df[f'{column}_group_avg']))

    return grouped_df 
