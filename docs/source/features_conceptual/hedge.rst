.. _hedge:

Hedge
============

High-Level Intuition
*********************
Captures whether a speaker appears to “hedge” their statement and express lack of certainty.

Citation
*********
`Ranganath, et al. (2013) <https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf>`_

Implementation Basics 
**********************
The score is a binary representation of whether an utterance contains any hedges. A score of 1 is assigned if any hedge phrases (”I think,” “a little,” “maybe,” “possibly”) are present, and a score of 0 is assigned otherwise.

Implementation Notes/Caveats 
*****************************
This is a bag of words feature, which is a naive approach towards detecting hedges.

Interpreting the Feature 
*************************
A score of 1 is assigned if hedge phrases (e.g., ”I think,” “a little,” “maybe,” “possibly”) are present, and a score of 0 is assigned otherwise.


Related Features 
*****************
Please see :ref:`politeness_strategies`, which also contains a measure for hedging.