.. _forward_flow:

Forward Flow
=============

High-Level Intuition
*********************
This feature captures "stream of consciousness", i.e. the evolution of thoughts/chats over time.

Citation
*********
`Gray et. al (2019) <https://psycnet.apa.org/fulltext/2019-03039-001.pdf>`_

Implementation Basics 
**********************

This feature measures the "forward flow" of a conversation by measuring how divergent a given utterance is to all of its precending utterances.

This feature relies on the fact tht each utterance in a given conversation corresponds to some high dimensional vector, which is computed by default using sBERT, a transformer-based model that generates sentence embeddings. As this feature is computed at the utterance level, it computes the cosine distance (that is, 1 - cosine similarity) between the current vector embedding and the "average" vector embedding exhibited within the conversation thus far (i.e. the average vector amongst all preceding utterances). This captures how much the current utterance has evolved from the "average" conversation up to that point, i.e. the "forward flow" of the conversation.

Implementation Notes/Caveats 
*****************************
In order to determine forward flow, the original source computed the average cosine distance between the current utterance and every single precending chat, as opposed to a single "average" preceding chat (i.e. averaging all the preceding utterance vectors). This modification was made with runtime in mind, as computed every chat to its preceding chat would run on the order of O(n^2) time complexity, whereas the current implementation runs on the order of O(n) time complexity. This modification may affect the interpretation of the feature, as it may not capture the "forward flow" as granularly as the original implementation. 

Interpreting the Feature 
*************************
Forward flow hopes to capture the stream of consciousness in a conversation, i.e. how thoughts/chats are evolving throughout the length of a conversation. Each utterance has its own forward flow score, which measures how semantically divergent the current chat is compared to the preceding chats. The forward flow score ranges from 0 to 2, with a score of 0 indicating that the current utterance is identically aligned to the ideas exchanged within the conversation up to that point, and a score of 2 indicating that the current utterance is maximally divergent from the ideas exchanged within the conversation up to that point. 

**Note 1:** The cosine distance metric is theoretically bounded at [0, 2]. However, in practice, with sBERT, it is very rare to encounter negative cosine similarity scores; this constrains the boundaries of the metric to [0, 1].

**Note 2:** If a conversation contains a single utterance, the forward flow score will be 0, as there is no preceding chat to compare the current chat to. Similarly, the forward flow score of the first utterance is 0.

.. list-table:: Output File
   :widths: 40 20 20
   :header-rows: 1

   * - message
     - speaker
     - forward_flow
   * - Hi, my name is Shruti!
     - Speaker A
     - 0
   * - Hey, my name is Nathaniel, but I go by Nate.
     - Speaker B
     - 0.86
   * - What's the plan for today?
     - Speaker A
     - 0.12
   * - My name is Emily.
     - Speaker C
     - 0.78

Related Features 
*****************
This feature is somewhat similar to other measures of the extent to which speakers vary in a conversation, including :ref:`mimicry_bert`. In Mimicry, however, we compare each speaker's utterance to the immediately preceding utterance (rather than the average of all preceding utterances).