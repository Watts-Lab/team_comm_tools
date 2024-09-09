.. _certainty:

Certainty
=========

High-Level Intuition
*********************
This feature measures how "certain", i.e. confident or sure, a given utterance is, using the Certainty Lexicon published in Rocklage et al. (2023).

Citation
*********
`Rocklage et al. (2023) <https://journals.sagepub.com/doi/pdf/10.1177/00222437221134802?casa_token=teghxGBQDHgAAAAA:iby1S-4piT4bQZ6-1lPNGOKUJsx-Ep8DaURu1OGvjuRWDbOf5h6AyfbSLVUgHjyIv31D_aS6PPbT>`_

Implementation Basics 
**********************
The Certainty Lexicon provided a dictionary of words and phrases and their associated certainty metric. This feature uses a regular expression to identify these words and phrases within a given utterance, and subsequently averages the corresponding certainty metrics to get the overall certainty score. If no matches are found, the default value is 4.5 (indicating that an utterance is neither certain nor uncertain). 

Implementation Notes/Caveats 
*****************************
Several of the keys in the Certainty Lexicon are substrings of each other, i.e. "I know it" and "I know it is". In these cases, we match the utterance with the longer substring to avoid double counting.

Interpreting the Feature 
*************************
Each key in the Certainty Lexicon is associated with a certainty score between 0 (very uncertain) and 9 (very certain). The feature computes the average certainty score of all matched words/phrases in a given utterance, so the score remains within this 0-9 range. If there are no matches found, then the default value of 4.5 (no certainty indicators present, "neutral certainty") is returned.

.. list-table:: Output File
   :widths: 40 20

   * - message
     - certainty
   * - I'm confident that she is on her way.
     - 8.4
   * - I'm not too sure.
     - 0.7
   * - My name is Emily.
     - 4.5


Related Features 
*****************
Another measure of Certainty is contained within :ref:`liwc`. Additionally, :ref:`hedge` and associated features (see the measure of hedging within :ref:`politeness_strategies` also captures the degree of uncertainty for a given utterance.