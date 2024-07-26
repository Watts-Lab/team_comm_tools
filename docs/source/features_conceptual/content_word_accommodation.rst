.. _content_word_accommodation:

Content Word Accommodation
============================

High-Level Intuition
*********************
This feature measure how much the current utterance "mimicks" the previous utterance in a conversation, with respect to the content words in the message. 

Citation
*********
https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf

Implementation Basics 
**********************
Counts the number of shared content words between the current and previous utterance in a conversation. Have a reference list of content words that are considered in the computation. Content words are defined as words that "possess semantic content and contribute to the meaning of the sentence", i.e. any words that are NOT function words. Some examples of content words are "paper", "coffee", "Mr.Smith", etc.


Implementation Notes/Caveats 
*****************************
Note that the first utterance in a conversation cannot have a mimicry score, as there is no "previous utterance" to associate it with. In this case, we assign a value of 0 to this utterance, signalling that there is no mimicry involved, a completely original thought. 

Interpreting the Feature 
*************************
This feature generates a shared word count for each utterance in a conversation, with lower scores representing a more original thought compared with the previous chat (lacking content word mimicry), while higher scores represent a higher degree of content word mimicry with the previous chat. The bounds of this score range from 0 to the total number of words in the selected chat.

It's important to note that this score doesn't measure the overall mimicry of the conversation. As an utterance-level feature, it computes the content word mimicry only between the selected chat and the previous. It's also important to note that this feature doesn't measure mimicry in its entirety, but rather within the context of content words. Check Mimicry (BERT) for a more comprehensive mimicry feature.

Related Features 
*****************
This toolkit incorporates a host of mimicry-related features, with others including Function Word Accommodation, Mimicry (BERT), and Moving Mimicry. The former use a similar concrete bag-of-words approach to compute mimicry within the content word domain, rather than the function word domain. Mimicry (BERT) uses cosine similarity between sBERT embeddings to measure mimicry between a given chat and the previous. Moving Mimicry is similar to Mimicry (BERT) in that it uses sBERT embeddings to compute similarity, but differs in that it  helps reason towards the overall flow of mimicry throughout a conversation, rather than at one instantaneous point.