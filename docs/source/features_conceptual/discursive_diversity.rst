.. _discursive_diversity:

Discursive Diversity
====================

High-Level Intuition
*********************
This feature measures how semantically divergent the speakers are in a conversation.

Citation
*********
`Lix, Katharina, et. al (2020) <https://osf.io/preprints/socarxiv/8pjga>`_`

Implementation Basics 
**********************
This feature relies on representing speakers as high dimensional vectors. These vectors are designed to capture the semantic content of each speaker's contribution in a conversation.

Every utterance in a given conversation corresponds to some high dimensional vector, which is computed using sBERT, a transformer-based model that generates sentence embeddings. Each speakers' utterances are then represented as a collection of vectors in this high-dimensional space. To map each speaker to a single high dimensional vector, we average the vectors of all their utterances (i.e. compute the centroid). 

After computing each speaker's corresponding vector, we compute the pairwise cosine distance between each pair of speakers. The cosine distance is a measure of similarity between two vectors, with a value of 0 indicating that the vectors are maximally dissimilar (i.e. the speakers contributed semantically divergent utterances), and a value of 1 indicating that the vectors are identical (i.e. the speakers contributed semantically identical utterances). This set of pairwise cosine distances is then averaged to produce a single discursive diversity score for the conversation.

Interpreting the Feature 
*************************
Read the code associated with this feature and answer the following questions, if applicable:

The vector embeddings, or centroids, that correspond to each speaker capture the average semantic meaning (sourced from sBERT embeddings) of their contributions to a conversation. The discursive diversity scores indicates how far or near these vector embeddings, or speakers, are from each other. 

The discursive diversity metric ranges from 0 to 1. A score of 0 indicates that the speakers are maximally semantically divergent, while a score of 1 indicates that the speakers are semantically identical. 

In this example, the first conversation displays a much higher discursive diversity score than the second conversation. This suggests that the speakers in the first conversation contributed more semantically divergent utterances than the speakers in the second conversation.

.. list-table:: Input File
   :widths: 20 40 20
   :header-rows: 1

   * - conversation
     - message
     - speaker
   * - Conversation 1
     - What's the plan looking like today?
     - Speaker A
   * - Conversation 1
     - I want to go study at the library.
     - Speaker B
   * - Conversation 1
     - Can we go eat at the new restaurant?
     - Speaker A
   * - Conversation 1
     - I'd rather go to the gym.
     - Speaker C
   * - conversation
     - message
     - speaker
   * - Conversation 2
     - What's the plan looking like today?
     - Speaker A
   * - Conversation 2
     - I want to go study at the library.
     - Speaker B
   * - Conversation 2
     - What homework do we have?
     - Speaker A
   * - Conversation 2
     - We have English and math homework due on Friday.
     - Speaker C


.. list-table:: Ouput File
   :widths: 20 20
   :header-rows: 1

   * - conversation
     - discursive diversity
   * - Conversation 1
     - 0.85
   * - Conversation 2
     - 0.32

Related Features 
*****************
Discursive diversity is the core feature within an umbrella group of discursive features. Other features within this umbrella include variance in discursive diversity, incongruent modulation, and within person discursive range, all of which rely conversation chunking. These features are designed to capture the diversity of semantic content within a conversation, and how this diversity changes over time, whereas discursive diversity alone captures the overall diversity of semantic content within a conversation. 
