.. _forward_flow:

Forward Flow
=============

High-Level Intuition
*********************
This feature captures "stream of consciousness", i.e. the evolution of thoughts/chats over time.

Citation
*********
`Anderson et. al (2019) <https://psycnet-apa-org.proxy.library.upenn.edu/fulltext/2019-03039-001.pdf>`_

Implementation Basics 
**********************

This feature measures the forward flow of a conversation by measuring how divergent a given utterance is to all of its precending utterances

This feature relies on the fact tht each utterance in a given conversation corresponds to some high dimensional vector, which is computed using sBERT, a transformer-based model that generates sentence embeddings. As this feature is computed at the utterance level, it computes the cosine distance between the current vector embedding and the "average" vector embedding exhibited within the conversation thus far (i.e. the average vector amongst all preceding utterances). This captures how much the current utterance has evolved from the "average" conversation up to that point, i.e. the "forward flow" of the conversation.

Implementation Notes/Caveats 
*****************************
In order to determine forward flow, the original source computed the average cosine distance between the current utterance and every single precending chat, as opposed to a single "average" preceding chat (i.e. averaging all the preceding utterance vectors). This modification was made with runtime in mind, as computed every chat to its preceding chat would run on the order of O(n^2) time complexity, whereas the current implementation runs on the order of O(n) time complexity. This modification may affect the interpretation of the feature, as it may not capture the "forward flow" as granularly as the original implementation. 


Interpreting the Feature 
*************************
Forward flow hopes to capture the stream of consciousness in a conversation, i.e. how thoughts/chats are evolving throughout the length of a conversation. Each utterance has it's own forward flow score, which measure how semantically divergent the current chat is compared to the preceding chats. The forward flow score ranges from 0 to 1, with a score of 0 indicating that the current utterance is identically aligned to the ideas exchanged within the conversation up to that point, and a score of 1 indicating that the current utterance is maximally divergent from the ideas exchanged within the conversation up to that point. 

Note that if a conversation contains a single chat, the forward flow score will be 0, as there is no preceding chat to compare the current chat to.

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
N/A