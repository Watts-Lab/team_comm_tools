.. _TEMPLATE:

Textblob Subjectivity
============

High-Level Intuition
*********************
Measures whether a message is subjective or objective

Citation
*********
`Cao, et al. (2020) <https://arxiv.org/pdf/2010.07292>`_

Implementation Basics 
**********************
To calculate subjectivity, we use the TextBlob Library in python. 
This library is implemented using the Naive Bayes Algorithm, `Textblob <https://textblob.readthedocs.io/en/dev/>`_ which is a "Bag of Words"-based classifier.

Implementation Notes/Caveats 
*****************************
This function uses a "Bag of Words"-based classifier, which is a naive way of measuring subjectivity.

Interpreting the Feature 
*************************

Scores are a continuous variable, ranging from -1 (extremely subjective) to 1 (extremely objective).


Related Features 
*****************
N/A