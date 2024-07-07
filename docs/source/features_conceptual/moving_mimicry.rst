.. _moving_mimicry:

Moving Mimicry
===============

High-Level Intuition
*********************
This feature measure the degree of mimicry a conversation has displayed up until the current utterance. 

Citation
*********
https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf

Implementation Basics 
**********************
Using BERT's Sentence Transfomers model (https://sbert.net/), utterances are represented as multidimensional embeddings. Stepping through each message in a conversation, this feature first computes the cosine similarity of the current embedding and previous embedding to determine their degree of mimicry. It then computes the average of all mimicry scores computed thus far (up until this point in the conversation), including the mimicry metric just computed, and assigned the output to the current chat. The feature then this running average for the proceeding chat. 

Implementation Notes/Caveats 
*****************************
Note that the first utterance in a conversation cannot have a moving mimicry score, as there is no "previous utterance" to associate it with. In this case, we assign a value of 0 to this utterance, signalling that there is no mimicry involved, i.e. a completely original thought. 

Interpreting the Feature 
*************************
This feature generates a score between 0-1 for each utterance in a conversation, with scores closer to 0 representing a more original conversation (lacking mimicry) up until chat X, while scores near 1 represent a higher degree of mimicry/similarity within the conversation up until chat X. 

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
     - 0.68
   * - My name is Emily.
     - Speaker C
     - 0.58

Related Features 
*****************
This toolkit incorporates a host of mimicry-related features, with others including Function Word Accommodation, Content Word Accommodation, and Mimicry (BERT). The former two features use a more concrete bag-of-words approach to computing mimicry within two discrete categories. Mimicry (BERT) is similar in that it still uses sBERT embeddings to compute similarity, but differs in that it helps reason towards the mimicry at one instantaneous point in a conversation, rather than the overall flow of mimicry throughout a conversation up until a certain point.