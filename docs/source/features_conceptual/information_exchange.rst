.. _information_exchange:

Information Exchange
=====================

High-Level Intuition
*********************
Actual "information" exchanged, i.e. word count minus first-person singular pronouns,z-scored at both the chat and conversation levels.

Citation
*********
Improving Teamwork Using Real-Time Language Feedback, Tausczik and Pennebaker,2013: https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf

Implementation Basics 
**********************
Word count minus first-person singular pronouns was taken as a measure of information exchange. Then converted to z-scores.

1. Word count minus first-person singular pronouns --> "info_exchange_wordcount"
2. Compute the z-score at the chat level: compute z-score for each message across all conversations --> "zscore_chats"
3. Compute the z-score at the conversation level: group by batch and round, then compute the z-score for each conversation --> "zscore_conversation"

Implementation Notes/Caveats 
*****************************
1. Personal opinion acts as an important part in on-task communications, and this feature specifically excludes first person pronouns
that might be indicative of personal opinions, which might not be ideal in all cases.
2. This method does not capture the quality of the information itself because it solely relies on the quantity. A person might say a lot of words but none of the information is meaningful to the topic.


Interpreting the Feature
*************************

We are assuming this is a single conversation, and this is a dataset that consists of only one conversation.

Example:
Messages in a conversation:

1. "I went to the store."
   - info_exchange_wordcount: 4 (5 words minus 1 first person pronoun "I")

2. "Bought some groceries for dinner."
   - info_exchange_wordcount: 5 (5 words minus 0 first person pronouns)

3. "It's raining today."
   - info_exchange_wordcount: 3 (3 words minus 0 first person pronouns)

Mean = 4, Standard deviation ≈ 0.82

#### z-scores:
Read more about z-scores here: https://www.statology.org/z-score-python/

- **zscore_chats**:
  - Message 1: 0 
  - Message 2: 1.22
  - Message 3: -1.22

- **zscore_conversation**:
  - Same values as above since it's a single conversation.

### Interpretation **zscore_chats**:
  - 0: Average information exchange.
  - 1.22: Higher-than-average.
  - -1.22: Lower-than-average.

Related Features 
*****************

Generally strongly correlated with message length, especially in cases where there are not a ton of pronoun use.