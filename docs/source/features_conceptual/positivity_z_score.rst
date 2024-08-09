.. _positivity_z_score:

Positivity Z-Score
===================

High-Level Intuition
*********************

The relative extent to which an utterance is more (or less) positive, compared to other messages. 

Citation
*********
`(Tausczik & Pennebaker, 2013) <https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf>`_

Implementation Basics 
**********************
A positivity score assined using  `RoBERTa; Liu et al., (2019) <https://arxiv.org/abs/1907.11692>`_ to calculate two flavors of the z-score: 
1. A score of the message with respect to other messages in the same conversation (positivity_zscore_chats)
2. A scores the messages with respect to all messages in the data. (positivity_zscore_conversations)

Implementation Notes/Caveats 
*****************************
NA

Interpreting the Feature 
*************************
Scores can be both negative and positive, with no bounds. 
A negative score indicates a the chat is less positive compared to other chats (in the conversation or dataset depending on the feature being used), while a positive score indicates a the chat is more positive compared to other chats.

Related Features 
*****************
This feature is one of several that measure sentiment. Other sentiment-related features include :ref:`positivity_bert`; Negative_Emotion and Positive_Emotion, which are attributes of :ref:`politeness_receptiveness_markers`; :ref:`textblob_polarity`; and LIWC (a relevant column name being positive_affect_lexical_per_100).