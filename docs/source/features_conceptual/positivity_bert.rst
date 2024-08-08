.. _positivity_bert:

Positivity (BERT)
=================

High-Level Intuition
*********************
This feature measures the positivity of a message using BERT's generated valence sentiment markers.

Citation
*********
Twitter-roBERTa-base-sentiment model from the `Hugging Face Transformers library <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment>`_.

Implementation Basics 
**********************
The code runs inference on the the Twitter-roBERTa-base-sentiment model to predict how relatively positive, negative, and neutral a message is on a 0-1 scale.

Implementation Notes/Caveats 
*****************************
This feature precomputes these valence ratings in the data preprocessing step & stores them locally; this essentially "caches" the sentiment markers, preventing the case where a user spends extra time regenerating these ratings on subsequent requests.


Interpreting the Feature 
*************************
This feature returns 3 general sentiment markers: **positive_bert**, **negative_bert**, and **neutral_bert**. Each score ranges from 0-1, and all three scores add up to 1. This feature measures the extent to which a particular utterance aligns with each label, relative to the other labels. 

Below is an example output file:

.. list-table:: Output File
   :widths: 40 20 20 20
   :header-rows: 1

   * - message
     - positive_bert
     - negative_bert
     - neutral_bert
   * - The idea sounds great!
     - 0.97
     - 0.01
     - 0.02
   * - I disagree, this idea is terrible.
     - 0.02
     - 0.92
     - 0.06
   * - Who's idea was it?
     - 0.05
     - 0.35
     - 0.60


Related Features 
*****************
N/A