.. _mimicry_bert:

Mimicry (BERT)
==============

High-Level Intuition
*********************
This feature measure how much the current utterance "mimicks" the previous utterance in a conversation.

Citation
*********
https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf

Implementation Basics 
**********************
Using BERT's Sentence Transfomers model (https://sbert.net/), utterances are represented as multidimensional embeddings. Stepping through each message in a conversation, this feature computes the cosine similarity of the current embedding and previous embedding to determine their degree of mimicry. 

Implementation Notes/Caveats 
*****************************
Note that the first utterance in a conversation cannot have a mimicry score, as there is no "previous utterance" to associate it with. In this case, we assign a value of 0 to this utterance, signalling that there is no mimicry involved, a completely original thought. 

Interpreting the Feature 
*************************
This feature generates a score between 0-1 for each utterance in a conversation, with scores closer to 0 representing a more original thought compared with the previous chat (lacking mimicry), while scores near 1 represent a higher degree of mimicry/similarity with the previous chat. 

It's important to note that this score doesn't measure the overall mimicry of the conversation. As an utterance-level feature, it computes the mimicry only between the selected chat and the previous. If a particular message is only similar to chats exchanged before it's direct previous chat, therefore, it won't have a high mimicry score (see below). In the same vein, high mimicry score for an individual chat does not signal that a conversation overall employed high mimicry.

.. list-table:: Output File
   :widths: 40 20
   :header-rows: 1

   * - message
     - speaker
     - mimicry_bert
   * - Hi, my name is Shruti!
     - Speaker A
     - 0
   * - Hey, my name is Nathaniel, but I go by Nate.
     - Speaker B
     - 0.89
   * - What's the plan for today?
     - Speaker A
     - 0.12
   * - My name is Emily.
     - Speaker C
     - 0.09

Related Features 
*****************
This toolkit incorporates a host of mimicry-related features, with others including Function Word Accommodation, Content Word Accommodation, and Moving Mimicry. The former two features use a more concrete bag-of-words approach to computing mimicry within two discrete categories. Moving Mimicry is similar in that it uses sBERT embeddings to compute similarity, but differs in that it  helps reason towards the overall flow of mimicry throughout a conversation, rather than at one instantaneous point.