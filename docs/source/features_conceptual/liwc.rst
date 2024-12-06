.. _liwc:

Linguistic Inquiry and Word Count (LIWC) and Other Lexicons
============================================================

High-Level Intuition
*********************
The Linguistic Inquiry and Word Count (LIWC) is a series of dictionaries representing various psychologically relevant concepts (for example, "Home" includes words such as "family," "apartmentment," and "kitchen;" "Exclusive" language refers to words such as "but," "without," and "exclude.") These create a very simple proxy for the content of language.

In addition to LIWC, we also incorporate additional lexicons where relevant, such as Hu and Liu's `Positive Word Lexicon <http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html>`_.

Citation
*********
By default, we use LIWC-2007; `Pennebaker et al. (2007) <https://www.liwc.net/LIWC2007LanguageManual.pdf>`_

Positive Word Lexicon: `Hu and Liu (2004) <https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf>`_

`NLTK's List of English Stowords <https://gist.github.com/sebleier/554280>`_

Implementation 
****************
For each word in the LIWC lexicon, we use a regular expression to count the number of times the word appears. The regular expression captures word stems where relevant; for example, "certain*" would capture "certainty," "certainly," etc.

**Note:** 

- **In v.1.0.3 and earlier:**  
  Word counts were presented as a scaled rate of **number of words per 100 words**, computed using the following formula:

  .. code-block:: text

      Rate of word use per 100 words = (count / utterance length) * (utterance length / 100)

- **In v.1.0.4 and later:**  
  Lexical values are represented as a **raw count** of the number of times they appear in the utteranc

Interpreting the Feature 
*************************
In general, the higher the value, the more that an utterance displays the concept "represented" by the lexicon. For example, a high value of the Home lexicon suggests that a speaker is discussing topics related to the home.

The scaled value is lower-bounded at 0 (if no words from the lexicon appear) and has no upper bound, as the value increases with the length of the utterance; for example, in a hypothetical utterance that is 1,000 words long, if all 1,000 words belong to a lexicon, the rate per 100 words would be (1000/1000 * 1000/100 = 10).

We note, however, that in general, the lexicon-based approach to measuring concepts has several limitations. **Lexicons use a bag-of-words-based approach**, which means that it does not take the ordering of words or the context into account. This can be particularly signifiant for issues such as positive language; the sentence "I am not happy" contains the word "happy," which would count towards the positive lexicon --- even though the statement is negative!

Finally, **LIWC typically recommends having an utterance of at least 100 words**, as measures for shorter pieces of text may not be reliable.

Related Features 
*****************
LIWC and related lexicons capture a variety of concepts that re-occur in other parts of the toolkit; for example, positivity is also measured using :ref:`positivity_bert`; measures of certainty appear both here and in :ref:`certainty`; :ref:`proportion_of_first_person_pronouns` uses first-person pronouns, which also appear in LIWC; :ref:`politeness_strategies` and :ref:`politeness_receptiveness_markers` also contain redundant measures (e.g., first person, second person).