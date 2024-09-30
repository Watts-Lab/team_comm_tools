.. _gini_coefficient:

Gini Coefficient
=================

High-Level Intuition
*********************
This feature measures the extent to which airtime is shared equally in a conversation.

Citation
*********
`Tausczik and Pennebaker (2013) <https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf>`_

Implementation Basics 
**********************
Using speaker-based aggregation, we aggregate the total number of **words**, **characters**, and **utterances** spoken by each individual in a conversation. We then compute the standard Gini coefficient with respect to each type of aggregation. We use three different measures of the quantity of "airtime" in a conversation in order to give the user more optionality.

Interpreting the Feature 
*************************
The `Gini Coefficient <https://en.wikipedia.org/wiki/Gini_coefficient>`_, originally proposed by Corrado Gini, is commonly used in economics and human development to measure inequality. It is bounded between 0 and 1. According to the Wikipedia article: 

.. epigraph::

    A Gini coefficient of 0 reflects perfect equality, where all ... values are the same, while a Gini coefficient of 1 (or 100%) reflects maximal inequality among values, a situation where a single individual has all the [values] while all others have none."

In the case of conversations, we are measuring the Gini coefficient with respect to the values of the number of **words**, **characters**, and **messages** sent by each speaker in a conversation; thus, 0 would imply that all members share airtime equally, and 1 would imply that one speaker did all of the talking.

One limitation of our measure is that **if some speakers have NO utterance at all, they do not even show up in the data**. For example, if there are 20 speakers in a conversation, and only five individuals do the talking, the 15 silent individuals would not appear in the data and would not be detected. The Gini coefficient would therefore be computed with respect to only 5 speakers. rather than the 20 "true" participants. To adjust for this issue, you may consider pre-processing your data with the full list of expected participants, and giving non-speaking participants a filler utterance (e.g., "<FILLER>"), so that the silent speakers can be registered in the speaker dataframe.

Related Features 
*****************
While not a measure of equality, the :ref:`turn_taking_index` also measures, to some extent, whether individuals are sharing airtime --- or whether one individual is over-contributing by sending many messages in a row.