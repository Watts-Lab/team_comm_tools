.. _discursive_diversity:

Discursive Diversity
====================

High-Level Intuition
*********************
This feature measures how semantically divergent the speakers are in a conversation.

Citation
*********
`Lix, Katharina, et. al (2022) <https://osf.io/preprints/socarxiv/8pjga>`_

Implementation Basics 
**********************
**Speaker-Level Vectors.** This feature relies on representing speakers as high dimensional vectors. These vectors are designed to capture the semantic content of each speaker's contribution in a conversation.

Every utterance in a given conversation corresponds to some high dimensional vector, which is computed by default using sBERT, a transformer-based model that generates sentence embeddings. Each speaker's utterances are then represented as a collection of vectors in this high-dimensional space. To map each speaker to a single high dimensional vector, we average the vectors of all their utterances (i.e. compute the centroid). 

**Discursive Diversity: Average Pairwise Cosine Distance Between Speakers.** After computing each speaker's corresponding vector, we compute the pairwise cosine distance between each pair of speakers. The cosine distance is a measure of difference or similarity between two vectors, defined as 1 - cosine similarity. 

Thus for each pair of speakers in a conversation, we obtain a measure ranging from 0 to 2. A value of 0 indicates that the vectors are identical (suggesting that these two speakers contributed the same content); 1 that the vectors are orthogonal; and 2 that the vectors are maximally different (pointing in opposite directions; this suggests that the two speakers contributed semantically opposing meanings). This set of pairwise cosine distances is then averaged to produce a single discursive diversity score for the conversation.

Note: In practice, with sBERT, it is very rare to encounter negative cosine similarity scores, even though they are possible in theory; thus, this would constrain the boundaries of the discursive diversity metric to [0, 1].

In addition to Discursive Diversity, the feature also generates three related features: Variance in Discursive Diversity, Incongruent Modulation, and Within-Person Discursive Range, which help convey the extent to which individuals change their semantic meaning over time. To capture change over time, all three of these features rely on **chunking a conversation into three equal 'periods'** --- capturing the "beginning," "middle," and "end" of a conversation. Currently, we rely on a naive metric that divides the total number of utterances evenly.

**Variance in Discursive Diversity.** This measures the extent to which Discursive Diversity varies arcross the course of a conversation. Here, we separately compute Discursive Diversity for the "beginning," "middle," and "end" of the conversation, then compute the variance in Discursive Diversity across the three chunks.

**Incongruent Modulation.** This measures the "group-level variance in members' within-person semantic shifts" (see Lix et al., 2022) across the three time periods. Put another way, this attribute measures the extent to which different members of the team are making shifts in meaning across time. Do only some team members make changes in their meaning (incongruent modulation), or do all members make semantic changes equally (congruent modulation)>? The higher the value, the higher the group-level variance in within-person semantic shifts, and the more "incongruent" the modulation.

For each speaker, we compute a centroid in each of the three time periods. We then compute the cosine distance between a given speaker's centroids from (Period 1, Period 2) and (Period 2, Period 3), and compute the variance in these two values. Finally we sum the total variance across all speakers.

**Within-Person Discursive Range.** This measures the total "semantic distance between an individual’s language use across the three milestone stages" (see Lix et al., 2022). The higher the value, the more individuals tend to vary their semantic meanings over time.

For each speaker, we again compute a centroid in each of the three time periods. We then compute the cosine distance between a given speaker's centroids from (Period 1, Period 2) and (Period 2, Period 3). We then take the average between these two values (distance betwen Periods 1-2, distance between Periods 2-3), and sum the averages across all speakers.

Implementation Notes/Caveats 
*****************************
In the event that a particular speaker or time period has no utterances, we use a default `null vector <https://github.com/Watts-Lab/team_comm_tools/blob/main/src/team_comm_tools/features/assets/nan_vector.txt>`_ (a vector representation of NaN) to represent that speaker/time period.

Interpreting the Feature 
*************************
The vector embeddings, or centroids, that correspond to each speaker capture the average semantic meaning (sourced from sBERT embeddings) of their contributions to a conversation. The discursive diversity scores indicates how far or near these vector embeddings, or speakers, are from each other. 

The discursive diversity metric ranges from 0 to 2 (in practice, we typically see values between 0 and 1; see note above). A score of 0 indicates that the speakers are semantically similar, while a score of 2 indicates that the speakers are semantically different.

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
Discursive Diversity is the core feature within an umbrella group of discursive features. The other three features in this umbrella --- Variance in Discursive Diversity, Incongruent Modulation, and Within-Person Discursive Range --- all use conversation chunking and build on the central feature. These features capture the diversity of semantic content within a conversation and how this diversity changes over time, whereas the primary Discursive Diversity metric captures the overall diversity of semantic content within a conversation. 
