.. _politeness_receptiveness_markers:

Politeness Receptiveness Markers
============

High-Level Intuition
*********************
A collection of conversational markers that indicates the use of politeness / receptiveness.

Citation
*********
`Yeomans et al., (2020) <https://www.mikeyeomans.info/papers/receptiveness.pdf>`_
`SECR Module (For computing features from Yeomans et al., 2020) <https://github.com/bbevis/SECR/tree/main>`_

Implementation Basics 
**********************

We follow a very similar framework to the SECR Module to compute a 39 politeness features for each chat in a conversation. The chats are first preprocessed in the following ways:

1. Convert all words to lowercase
2. Remove/expand contractions (i.e don’t to do not; can’t to cannot; let’s to let us)
3. Ensure all characters are legal traditional A-Z alphabet letters by using corresponding RegExs

We then calculate the general categories of features in different ways, following similar structure as the SECR module.

1. count_matches and Adverb_Limiter: calculates features using a standard bag-of-words approach, detecting the number of keywords from a pre-specified list stored in keywords.py.
2. get_dep_pairs/get_dep_pairs_noneg: use Spacy to get dependency pairs for relevant words, using token.dep_ to differentiate with negation.
3. Question: Question-related features are computed by counting the number of question words in a chat.
4. word_start: detect certain conjunctions/affirmation words using pre-specified dictionary

The corresponding counts are then returned concatenated to the original dataframe.


Implementation Notes/Caveats 
*****************************
NA

Interpreting the Feature 
*************************
Read the code associated with this feature and answer the following questions, if applicable:

1. What are the bounds of the score? What does a high versus low score mean? (How should you read this score?)
2. Give a concrete example (e.g., negative score versus positive score)
3. What DOESN’T the score measure? That is, what does the score take into account, and what are some ways that it might not capture the high-level social science concept?
4. Are there any edge cases that we should be aware of? (e.g., if the conversation contains only one chat?) To the best of your knowledge, how does the code handle it?

Related Features 
*****************
Politness Strategies