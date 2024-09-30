.. _information_exchange:

Information Exchange
=====================

High-Level Intuition
*********************

This feature measures the actual "information" exchanged in a chat by calculating the word count minus the count of first-person singular pronouns (e.g., "I", "we", "me"). This value is then standardized (converted to z-scores) at two levels: (1) all utterances across the entire dataset ("chat") and (2) utterances within a specific conversation ("conversation").

Citation
********

`Tausczik and Pennebaker (2013) <https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf>`_

Implementation Basics
*********************

The method for calculating information exchange involves the following steps:

1. **Calculate "info_exchange_wordcount":**

   * For each message, count the total number of words.
   * Subtract the number of first-person singular pronouns (like "I") from the total word count.

2. **Compute z-scores for all utterances in the data (`zscore_chats`):**

   * Calculate the z-score of "info_exchange_wordcount" for all utterances, across all conversations.

3. **Compute z-scores within each conversation (`zscore_conversation`):**

   * Calculate the z-score of "info_exchange_wordcount" within a given conversation, grouping by the unique conversational identifier.

Implementation Notes/Caveats
****************************

1. **Exclusion of First-Person Pronouns:**

   * This feature specifically excludes first-person pronouns in its calculation, which is a simplistic assumption about what constitutes meaningful content.

2. **Focus on Quantity, Not Quality:**

   * This method measures the quantity of information, not the quality. In effect, the measure captures whether a message is longer or shorter relative to the typical message in the data ("chat") or relative to the typical message in a particular conversation ("conversation"). 
   * A message may have many words but still be meaningless. For measures that capture semantic meaning, see features such as :ref:`forward_flow`.

Interpreting the Feature
*************************

Let's assume there is a single conversation consisting of several messages. Here’s an example to illustrate:

Example:
--------

Messages in a conversation:

1. **"I went to the store."**
   
   * `info_exchange_wordcount`: 4 (5 words minus 1 first-person pronoun "I")

2. **"Bought some groceries for dinner."**
   
   * `info_exchange_wordcount`: 5 (5 words minus 0 first-person pronouns)

3. **"It's raining today."**
   
   * `info_exchange_wordcount`: 3 (3 words minus 0 first-person pronouns)

**Mean** of info_exchange_wordcount = 4, **Standard deviation** ≈ 0.82

z-scores:
---------

For more details on z-scores, visit: https://www.statology.org/z-score-python/

* **zscore** (Since the example consists of a single conversation, the "chat" and "conversation" levels are equivalent):
  
  * Message 1: 0 
  * Message 2: 1.22
  * Message 3: -1.22

Interpretation:
-----------------------------------

* `0`: Indicates average information exchange.
* `1.22`: Indicates higher-than-average information exchange.
* `-1.22`: Indicates lower-than-average information exchange.

Related Features
*****************

This feature is strongly correlated with message length, especially in cases where there is minimal use of first-person pronouns.