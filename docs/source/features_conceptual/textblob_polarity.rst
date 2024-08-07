.. _textblob_polarity:

Textblob Polarity
============

High-Level Intuition
*********************
Measures the polarity i.e. how positive or negative a message is

Citation
*********
`Cao, et al. (2020) <https://arxiv.org/pdf/2010.07292>`_

Implementation Basics 
**********************
To calculate polarity, we use the TextBlob Library in python. 
This library is implemented using the Naive Bayes Algorithm, `Textblob <https://textblob.readthedocs.io/en/dev/>`_ which is a "Bag of Words"-based classifier.

Implementation Notes/Caveats 
*****************************
This function uses a "Bag of Words"-based classifier, which is a naive way of measuring polarity.

For example, in the sentence "Everything in this restaurant was anything but lovely, amazing, wonderful, great!",
the sentence actually has a negative meaning as it means that nothing in the restaurant was good.
However, the algorithm will classify it as a positive sentence because it simply counts the number of positive and negative words 
(4 positive words in this case make the sentence positive for the algorithm).


Interpreting the Feature 
*************************

Scores are a continuous variable, ranging from -1 (extremely negative) to 1 (extremely positive)

Related Features 
*****************
N/A