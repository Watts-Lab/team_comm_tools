.. _word_ttr:

Word Type-Token Ratio
=====================

High-Level Intuition
*********************
The number of tokens refers to the number of words in a message. The number of types refers to the number of unique words in a message. Word Type-Token Ratio is defined as the number of unique words over the number of words.

Citation
*********
`Reichel, et al., 2015 <https://cpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/f/1603/files/2017/01/Reichel_etal_Interspeech_2015-2i4gnzk.pdf>`_
`Williamson, 2009 <https://www.sltinfo.com/wp-content/uploads/2014/01/type-token-ratio.pdf>`_

Implementation Basics 
**********************
Get the word type-token ratio, calculated as follows: Number of Unique Words / Number of Total Words.

Interpreting the Feature 
*************************
The function assumes that punctuation is retained when being inputted, but parses it out within the function.

Related Features 
*****************
N/A