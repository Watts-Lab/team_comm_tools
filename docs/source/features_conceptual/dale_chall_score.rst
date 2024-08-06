.. _TEMPLATE:

Dale-Chall Score
============

High-Level Intuition
*********************
A score that provides a numeric gauge of the comprehension difficulty that readers come upon when reading a text.

Citation
*********
`Cao, et al. (2020) <https://dl.acm.org/doi/pdf/10.1145/3432929?casa_token=B5WlyazkwNIAAAAA:E-1nT55uQnGslAHCfO21sdeaXfaefJsT5ZpU2hq49eagiYaGSGpohlmTyUn4NslWtNOZuAl3XvcFXQ>`_

Implementation Basics 
**********************

The Dale–Chall readability formula is a readability test that provides a numeric gauge of the comprehension difficulty that readers come upon when reading a text. It uses a list of 3000 words that groups of fourth-grade American students could reliably understand, considering any word not on that list to be difficult.

The Formula for the Dale-Chall readability score is:

The formula for calculating the raw score of the Dale–Chall readability score (1948) is given below:

0.1579(difficult words ×100/words) + 0.0496(words/sentences)

Scores range from 0 - 10, details can be found below:

`Dale Chall Score <https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula>`_

Credits: Wikipedia

Implementation Notes/Caveats 
*****************************
NA

Interpreting the Feature 
*************************

Scores range from 0 to 10, and can be interpreted as:

=====  =============================================
Score  Notes
=====  =============================================
4.9    easily understood by an average 4th-grade student or lower
5.0–5.9  easily understood by an average 5th- or 6th-grade student
6.0–6.9  easily understood by an average 7th- or 8th-grade student
7.0–7.9  easily understood by an average 9th- or 10th-grade student
8.0–8.9  easily understood by an average 11th- or 12th-grade student
9.0–9.9  easily understood by an average college student
=====  =============================================

Related Features 
*****************
NA