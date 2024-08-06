.. _function_word_accommodation:

Function Word Accommodation
============================

High-Level Intuition
*********************
This feature measures how much the current utterance "mimics" the previous utterance in a conversation, with respect to the function words in the message. Function words are often part of the grammatical structure or style of a piece of text, rather than part of its "substance;" thus, a high degree of function word accommodation can be loosely said to indicate mimicry of on another's speaking style (as distinct from the content).

Citation
*********
`Ranganath et al. (2013) <https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf>`_

Implementation Basics 
**********************
The feature countds the number of shared function words between the current and previous utterance in a conversation. Function words are defined as words that "express grammatical relationships among other words within a sentence." The list of 195 function words provided by the original authors is reproduced below.

**Auxiliary and copular verbs**
  able, am, are, aren’t, be, been, being, can, can’t, cannot, could, couldn’t, did, didn’t, do, don’t, get, got, gotta, had, hadn’t, hasn’t, have, haven’t, is, isn’t, may, should, should’ve, shouldn’t, was, were, will, won’t, would, would’ve, wouldn’t

**Conjunctions**
  although, and, as, because, ’cause, but, if, or, so, then, unless, whereas, while

**Determiners, Predeterminers, and Quantifiers**
  a, an, each, every, all, lot, lots, the, this, those

**Pronouns and Wh-words**
  anybody, anything, anywhere, everybody’s, everyone, everything, everything’s, everywhere, he, he’d, he’s, her, him, himself, herself, his, I, I’d, I’ll, I’m, I’ve, it, it’d, it’ll, it’s, its, itself, me, my, mine, myself, nobody, nothing, nowhere, one, one’s, ones, our, ours, she, she’ll, she’s, she’d, somebody, someone, someplace, that, that’d, that’ll, that’s, them, themselves, these, they, they’d, they’ll, they’re, they’ve, us, we, we’d, we’ll, we’re, we’ve, what, what’d, what’s, whatever, when, where, where’d, where’s, wherever, which, who, who’s, whom, whose, why, you, you’d, you’ll, you’re, you’ve, your, yours, yourself

**Prepositions**
  about, after, against, at, before, by, down, for, from, in, into, near, of, off, on, out, over, than, to, until, up, with, without

**Discourse Particles**
  ah, hi, huh, like, mm-hmm, oh, okay, right, uh, uh-huh, um, well, yeah, yup

**Adverbs and Negatives**
  just, no, not, really, too, very


Implementation Notes/Caveats 
*****************************
Note that the first utterance in a conversation cannot have a mimicry score, as there is no "previous utterance" to associate it with. In this case, we assign a value of 0 to this utterance.

Interpreting the Feature 
*************************
This feature generates a word count for each utterance in a conversation, with lower scores representing a lower degree of accommodation, and higher scores representing a higher degree of acommodation. The bounds of this score are 0 (no shared function words) to the total number of words in the selected chat (the entire utterance consists of function words, and all were shared between successive utterances).

It's important to note that this score doesn't measure the overall level of mimicry over the course of the conversation. As an utterance-level feature, it computes the function word mimicry only between the focal utterance and the previous one. It's also important to note that, because the feature is lexical, it is determined purely based on the number of shared words, and not based on the meaning of those words. Other measures of mimicry relying on transformer-based models (e.g., the Mimicry (BERT) features) can help to mitigate this issue.

Related Features 
*****************
Other mimicry-related features include :doc:`Content Word Accomodation <content_word_accomodation.rst>`, :doc:`Mimicry (BERT) <mimicry_bert.rst>`, and :doc:`Moving Mimicry <moving_mimicry.rst>`. Content Word Accommodation is a lexical feature that is the complement of this one; it measures the number of all words *other than* function words shared between successive utterances. The latter two use more advanced, transformer-based models to compute similarity between utterances. Mimicry (BERT) uses the cosine similarity between sBERT embeddings to measure mimicry between a given utterance and the previous one. Moving Mimicry is similar to Mimicry (BERT) in that it uses sBERT embeddings to compute similarity, but differs in that it helps reason towards the overall flow of mimicry throughout a conversation, rather than discretely between a single utterance and the previous utterance.