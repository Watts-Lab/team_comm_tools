.. _content_word_accommodation:

Content Word Accommodation
============================

High-Level Intuition
*********************
This feature measures how much the current utterance "mimics" the previous utterance in a conversation, with respect to the content words (that is, non-function words) in the message. Content words are those that possess semantic content, so this measure roughly estimates the extent to which individuals are echoing each other in a "substantive" way, as opposed to mimicking the speaking style.

Citation
*********
`Ranganath et al. (2013) <https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf>`_

Implementation Basics 
**********************
To compute the feature, we count the number of shared content words (defined as anything that is not on the function word list) between the current and previous utterance in a conversation, normalized by the frequency at which the word appears. This follows the original authors' method:

	Content words are defined as any word that is not a function word. For each content word w in a given speaker’s turn, if w also occurs in the immediately preceding turn of the other, we count w as an accommodated content word. The raw count of accommodated content words is be the total number of these accommodated content words over every turn in the conversation side. Because content words vary widely in frequency, we normalized our counts by the frequency of each word.

For completeness, we interprete "the frequency of each word" in two distinct ways:

1. **The frequency of each word across the entire dataset (`content_word_accommodation`)**: here, we normalize non-function words with respect to the language used across all conversations in the dataset. This version of accommodation is useful if the entire dataset consists of similar conversations, or conversations about the same topic. Normalizing with respect to a larger dataset will be useful in establishing better estimates in identifying (and appropriately weigting) whichs words carry meaningful content in a particular domain.

2. **The frequency of each word within a given conversation (`content_word_accommodation_per_conv`)**: here, we normalize non-function words with respect only to the language in a given conversation. This version of accommodation is useful if the dataset consists of very distinct conversations, for which it may not make sense to assume that the distribution of which words are "important" will hold across different domains.

The feature requires a reference list of function words, which are defined by the original authors as follows.

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
This feature generates a (normalized) shared word count for each utterance in a conversation, with lower scores (close to 0) representing utterances that discuss more "different" content compared to the previous utterance; in other words, there is a lack of content word mimicry. Higher scores represent utterances that discuss the "same" content compared to the previous utterance.

It's important to note that this score doesn't measure the overall level of mimicry over the course of the conversation. As an utterance-level feature, it computes the content word mimicry only between the focal utterance and the previous one. It's also important to note that, because the feature is lexical, it doesn't measure "mimicry" in the sense of having a similar opinion --- only in the sense of having a large number of shared words. For example, "I loved the movie last night" and "I hated the movie last night" express opposing sentiments, but share most of their content words ("movie", "last", "night"). Additionally, homonyms may create false positives; "I was present in class" and "I got you a present" share the content word "present," but it takes on a different meaning in each case. Other measures of mimicry relying on transformer-based models (e.g., the Mimicry (BERT) features) can help to mitigate this issue.

Related Features 
*****************
Other mimicry-related features include :ref:`function_word_accommodation`, :ref:`mimicry_bert`, and :ref:`moving_mimicry`. Function Word Accommodation is a lexical feature that is the complement of this one; it measures the number of function words shared between successive utterances. The latter two use more advanced, transformer-based models to compute similarity between utterances. Mimicry (BERT) uses the cosine similarity between sBERT embeddings to measure mimicry between a given utterance and the previous one. Moving Mimicry is similar to Mimicry (BERT) in that it uses sBERT embeddings to compute similarity, but differs in that it helps reason towards the overall flow of mimicry throughout a conversation, rather than discretely between a single utterance and the previous utterance.