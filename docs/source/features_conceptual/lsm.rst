.. _LSM:

LANGUAGE STYLE MATCHING
============

High-Level Intuition
*********************
Language Style Matching (LSM) measures the degree to which individuals in a conversation align their linguistic styles. It reflects social dynamics like rapport, group cohesion, and interpersonal understanding by analyzing the similarity in function word usage (e.g., pronouns, conjunctions) between group members.

Citation
*********
Gonzales, A. L., Hancock, J. T., & Pennebaker, J. W. (2010). Language style matching as a predictor of social dynamics in small groups. Communication Research, 37(1), 3–19. https://doi.org/10.1177/0093650209351468

Implementation Basics 
**********************
The code computes LSM by analyzing the usage of specific function words (e.g., pronouns, conjunctions) by speakers in a conversation. It calculates the proportion of each function word type used by a speaker and compares it to the average usage of the same word type by other speakers in the same conversation. The formula for LSM reflects the similarity between these proportions, with higher values indicating greater alignment.

Implementation Notes/Caveats 
*****************************
This implementation adheres closely to the methodology described in the source paper, noting that our implementation does not include the calculation of Cronbach's alpha. However, there may be differences in how certain edge cases are handled, such as:
1. Conversations with only one speaker.
2. Instances where specific function word counts are zero, requiring safeguards to avoid division by zero errors.

Interpreting the Feature 
*************************
Read the code associated with this feature and answer the following questions, if applicable:

1. What are the bounds of the score? What does a high versus low score mean? (How should you read this score?) LSM scores range from 0 to 1, where 1 indicates perfect linguistic alignment and 0 indicates no alignment.
2. **Concrete Example:** 
2. Give a concrete example (e.g., negative score versus positive score) A high LSM score (e.g., 0.85) for pronouns suggests that a speaker's pronoun usage closely matches the average usage by other group members. A low LSM score (e.g., 0.2) suggests less alignment in pronoun usage.
3. What DOESN’T the score measure? That is, what does the score take into account, and what are some ways that it might not capture the high-level social science concept? LSM does not capture the content or context of conversations. It focuses purely on function word alignment and may not reflect deeper social or relational dynamics.
4. Are there any edge cases that we should be aware of? (e.g., if the conversation contains only one chat?) To the best of your knowledge, how does the code handle it? If a conversation has only one speaker, no LSM score is calculated.

Related Features 
*****************
Are there any related/similar features to this one? Is this part of an "umbrella" or group of features? Write about them here, and do your best to explain how they are different. Why would you use one implementation over the other? This might be similar to the mimicry score, but is different in that it hones in on the repetition of function words instead of overall mimicry.
