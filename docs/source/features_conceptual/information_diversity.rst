.. _information_diversity:

FEATURE NAME
============

High-Level Intuition
*********************
This conversation-level feature measures the diversity of different topics discussed in a conversation.

Citation
*********
`Riedl and Woolley (2018) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2384068>`_

Implementation Basics 
**********************
**Preprocessing and Generating Topics.** We first preprocess the data by ensuring all utterance are in lowercase, lemmatized, and by removing stop words and words shorter than three caracters. We then use the `gensim <https://radimrehurek.com/gensim/>`_ package to create an `LDA model <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>` for each conversation, generating a corresponding topic space with its in which the number of dimensions equals the number of topics. 

**Determining Number of Topics.** To determine the number of topics used, we use the square root of the number of utterances (rows) in the conversation, rounded to the nearest integer.

**Computing the Measure.** A team's information diversity is then computed by examining the average cosine distance (where cosine distance is defined as 1 - cosine similarity) between the "topic vector" associated with a given utterance and the mean topic vector across the entire conversation.

Implementation Notes/Caveats 
*****************************
As of September 9, 2024, this feature uses a LDA-based topic model. However, because LDA is stochastic, **it does not generate consistent results**. An updated version of this feature with more stable topic extraction is currently under development.

Interpreting the Feature 
*************************
The output ranges between 0 and 1, with higher values indicating a higher level of diversity in the topics discussed throughout the conversation. As discussed in Riedl and Woolley (2018), typical information diversity values are quite small, with the paper having a mean score of 0.04 and standard deviation of 0.05.

Related Features 
*****************
This feature is related to other measures of the "diversity" of ideas exchanged in conversation, such as :ref:`discursive_diversity`.