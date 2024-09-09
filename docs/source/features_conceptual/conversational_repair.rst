.. _conversational_repair:

Conversational Repair
=======================

High-Level Intuition
*********************
This feature measures the presence of repair questions that indicate missing some information and requesting for clarification (for example, "what?"; "sorry"; "excuse me?"). These are also known as "next-turn repair indicators" (NTRI).

Citation
*********
`Ranganath et al. (2013) <https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf>`_

Implementation
****************
The feature is a binary indicator of whether or not there was request for conversational repair. It uses a regular expression to count the number of repair markers in an utterance. It returns 1 if there is at least one repair marker, and 0 otherwise.

The specific regular expression used is as follows:
.. code-block:: python

    "what\?+|sorry|excuse me|huh\??|who\?+|pardon\?+|say.*again\??|what'?s that|what is that"


The regular expression slightly modifies the original (from Ranganath et al., 2013) to accommodate additional variations (for example, the original included only "say again" and "say it again," while ours uses the ".*" syntax to account for other variations, such as "say that again?").

Interpreting the Feature 
*************************
This feature can be interpreted as a naive measure of whether there was a next-turn repair indicator, but since it is a lexical feature, it has associated limiations from using a bag-of-words approach (namely, ignoring important context, and being limited to these specific words). There are many ways of expressing a repair indicator that may not be captured in this particular regular expression (e.g., "come again?") or ways of using the repair indicator words that are not NTRI's (e.g., "excuse me for being late").

Additionally, NTRI's may be more common in in-person or verbal conversations (where ambient noise may cause people not to fully hear their conversational partner), and less common in text-based chats or online conversations. Depending on your data, this feature may be sparse.

Related Features 
*****************
Next-turn repair indicators are a specific category of questions. Other mechanisms of detecting questions include :ref:`questions`, :ref:`politeness_strategies` (direct_question), and :ref:`politeness_receptiveness_markers` (WH_Questions, YesNo_Questions).