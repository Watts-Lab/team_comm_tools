.. _questions:

Question (Naive)
=================

High*Level Intuition
*********************
This feature is a naive measure of counting the number of questions asked.

Citation
*********
`Ranganath et al. (2013) <https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf>`_

Implementation  
***************
The feature first tokenizes an utterance into sentences based on its punctuation. Any sentence ending with a question mark (?) or beginning with a question word is counted as a "question."

The list of question words used is:

* what
* why
* when
* where
* who
* which
* whom
* whose
* how
* is
* am
* are
* do
* does
* did
* can
* could
* shall
* should
* will
* would
* have
* has
* had
* don't

Interpreting the Feature 
*************************
The feature provides a rough sense of the number of questions asked; however, a key limitation is that this measure of question is naive --- there can be ways of phrasing questions without question words or question marks, and there can ber positions of question words other than the beginning of the sentence. For example, "the project should be updated, should it not" is a question, but it would not be detected (false negative). Additionally, there may be false positives in which a sentence begins with a question word but is not, in fact, a question (e.g., "where I live is a very nice neighborhood").

Related Features 
*****************
:ref:`politeness_strategies` (direct_question) and :ref:`politeness_receptiveness_markers` (WH_Questions, YesNo_Questions) both also have related measures of asking questions.