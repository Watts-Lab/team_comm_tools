.. _positivity_z_score:

Positivity Z-Score
===================

High-Level Intuition
*********************

This feature measures how positive an utterance is relative to other messages. This positivity is gauged by comparing each message against all messages in a dataset or within the same conversation.

Citation
********

`Tausczik & Pennebaker (2013) <https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf>`_

Implementation
**************

A positivity score is first assigned using `RoBERTa; Liu et al., (2019) <https://arxiv.org/abs/1907.11692>`_. We then compute two types of z-scores based on this positivity score:

1. **Positivity among all messages (`positivity_zscore_chats`)**:
   - This z-score measures how a message's positivity compares to all other messages in the entire dataset.

2. **Positivity within the same conversation (`positivity_zscore_conversations`)**:
   - This z-score measures how a message's positivity compares to other messages within the same conversation (grouping by the unique conversational identifier).

Interpreting the Feature
*************************

The positivity z-scores can be both negative and positive, with no fixed bounds.

- **Negative Score**: Indicates that the utterance is less positive compared to other messages (either in the entire dataset or within the same conversation, depending on the reference point of the z-score).
- **Zero Score**: Indicates that the utterance has a typical (average) level of positivity compared to other messages.
- **Positive Score**: Indicates that the utterance is more positive compared to other messages.

Related Features
****************

This feature is part of a broader category of sentiment analysis features. Other related features include:

- :ref:`positivity_bert`
- Negative_Emotion and Positive_Emotion (attributes of :ref:`politeness_receptiveness_markers`)
- :ref:`textblob_polarity`
- :ref:`liwc` (e.g., positive_affect_lexical_per_100)

These features collectively help analyze and interpret the sentiment conveyed in messages.