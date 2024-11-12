import pandas as pd 

def calculate_lsm(chat_df):
    """ 
    This function calculates Language Style Matching (LSM) scores.
    
    Source: Language Style Matching as a Predictor of Social Dynamics in Small Groups by Amy L. Gonzales, Jeffrey T. Hancock, and James W. Pennebaker.
    """
    
    #Create a new column with the sum of all pronouns (first person singular, first person plural, second person, third person)
    chat_df['total_pronouns'] = (chat_df ['first_person_singular_lexical_wordcount'] + chat_df['first_person_plural_lexical_wordcount'] + chat_df['second_person_lexical_wordcount'] + chat_df['third_person_lexical_wordcount'])

    #Group by the speaker_id column and sum all the num_words and other function word counts
    grouped_df = df.groupby('speaker_id').agg({
        'num_words': 'sum',
        'conjunction_lexical_wordcount': 'sum',
        'total_pronouns' : 'sum',
        'adverbs_lexical_wordcount' : 'sum',
        'article_lexical_wordcount' : 'sum',
        'quantifier_lexical_wordcount' : 'sum',
        'negation_lexical_wordcount' : 'sum',
        'preposition_lexical_wordcount' : 'sum',
        'indefinite_pronoun_lexical_wordcount' : 'sum',
        'auxiliary_verbs_lexical_wordcount' : 'sum',
    }).reset_index()
    #Resets the index so speaker is treated as a normal column
                     
    #Now start calculating LSM score (for each function word category for each person)
   
    #List of function word columns to calculate percentages for
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

    #Loop through each function word column and divide it by the total number of words (eg. x percent of total words were pronouns)
    for column in function_word_columns:
        grouped_df[f'{column}_percent'] = (grouped_df[column] / grouped_df['num_words']) * 100 

#############UPDATED ABOVE ONLY#################################


#Extract group number from the 'speaker' column so for example 1.Blue becomes 1
grouped_df['group'] = grouped_df['speaker'].apply(lambda x: x.split('.')[0])

#Now calculate group sums and counts for each group (to later calculate average of REST of group members)
group_sums = grouped_df.groupby('group')[function_word_columns].transform('sum')
group_counts = grouped_df.groupby('group')[function_word_columns].transform('count')

#Calculate group averages and exclude the current speaker. Subtract the speaker's score from the group total and subtract 1 from the count to adjust the average. 
for column in function_word_columns:
    grouped_df[f'{column}_group_avg'] = (group_sums[column] - grouped_df[column]) / (group_counts[column] - 1)

#Use the LSM formula, example here for pronouns for person 1: 1 − (|pp1 − ppG|/(pp1 + ppG))
for column in function_word_columns:
    grouped_df[f'{column}_lsm'] = 1 - (abs(grouped_df[column] - grouped_df[f'{column}_group_avg']) / (grouped_df[column] + grouped_df[f'{column}_group_avg']))

print(grouped_df.columns)

#List all LSM scores for each function word group.
lsm_columns = [
    'total_pronouns_lsm',
    'conjunction_lexical_wordcount_lsm',
    'adverbs_lexical_wordcount_lsm',
    'article_lexical_wordcount_lsm',
    'quantifier_lexical_wordcount_lsm',
    'negation_lexical_wordcount_lsm',
    'preposition_lexical_wordcount_lsm',
    'indefinite_pronoun_lexical_wordcount_lsm',
    'auxiliary_verbs_lexical_wordcount_lsm'
]

#Export individual LSM scores to excel file (optional)
columns_to_export = ['speaker', 'group'] + lsm_columns
grouped_df[columns_to_export].to_csv('individual_lsm_scores', index=False)

#Now we have the individual LSM scores for each of the 9 function word groups for every person in every group. 
#Now we need to find the group LSM score for each of the 9 function word groups for each group. Example: Group ppLSM = (pp1 + pp2 + pp3 + pp4)/4,



#Calculate group average for each function word group.
group_lsm_scores = grouped_df.groupby('group')[lsm_columns].mean().reset_index()

#Print group-level LSM scores (9 scores for each group)
print(group_lsm_scores)

#Calculate overall group LSM score (average of each group's 9 function word group LSM scores)
group_lsm_scores['overall_lsm'] = group_lsm_scores[lsm_columns].mean(axis=1)
print (group_lsm_scores)

#Calculate Cronbach's alpha
import numpy as np

def my_cronbach_alpha(df):
    item_scores = df.T
    item_vars = item_scores.var(axis=1, ddof=1)
    total_score_var = item_scores.sum(axis=0).var(ddof=1)
    num_items = len(df.columns)
    alpha = num_items / (num_items - 1) * (1 - sum(item_vars) / total_score_var)
    return alpha

#Apply Cronbach's alpha to the 9 LSM columns for each group
alpha = my_cronbach_alpha(group_lsm_scores[lsm_columns])
group_lsm_scores['my_cronbach_alpha'] = alpha
print(f"Cronbach's a: {alpha}")

#Save the final group LSM scores and overall scores to a CSV file
group_lsm_scores.to_csv('group_lsm_scores.csv', index=False)
