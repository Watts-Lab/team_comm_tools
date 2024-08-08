.. _moving_mimicry:

Moving Mimicry
===============

High-Level Intuition
*********************
This feature measure the degree of mimicry a conversation has displayed up until the current utterance. 

Citation
*********
N/A; This is a novel measure that builds on related measures of mimicry/accommodation (see "Related Features").

Implementation Basics 
**********************
Using BERT's Sentence Transfomers model (https://sbert.net/), utterances are represented as multidimensional embeddings. Stepping through each message in a conversation, this feature first computes the cosine similarity of the current embedding and previous embedding to determine their degree of mimicry. It then computes the average of all mimicry scores computed thus far (up until this point in the conversation), including the mimicry metric just computed, and assigned the output to the current chat. The feature then this running average for the proceeding chat. 

Implementation Notes/Caveats 
*****************************
Note that the first utterance in a conversation cannot have a moving mimicry score, as there is no "previous utterance" to associate it with. In this case, we assign a value of 0 to this utterance. 

Interpreting the Feature 
*************************
This feature generates a score between 0-1 for each utterance in a conversation, with scores closer to 0 representing a more original conversation (lacking mimicry) up until chat X, while scores near 1 represent a higher degree of mimicry/similarity within the conversation up until chat X. 

.. list-table:: Output File
   :widths: 40 20 20
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
This toolkit incorporates a host of mimicry-related features, with others including :ref:`function_word_accommodation`, :ref:`content_word_accommodation`, and :ref:`mimicry_bert`. The former two features use a bag-of-words approach to compute mimicry within two discrete categories. Mimicry (BERT) is similar in that it still uses SBERT embeddings to compute similarity, but differs in that it helps reason towards the mimicry discretely between a single utterance and the previous utterance, rather than the overall flow of mimicry throughout a conversation up until a certain point.