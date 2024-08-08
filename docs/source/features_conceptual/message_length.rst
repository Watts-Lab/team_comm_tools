.. _message_length:

Message Length
===============

High-Level Intuition
*********************
Returns the number of words in a message/utterance

Citation
*********
NA

Implementation Basics 
**********************

Returns the number of words in a message/utterance by splitting on the whitespace, after preprocessing to remove punctuation.

Implementation Notes/Caveats 
*****************************
This feature does not recognize successive punctuation marks as words. 
For example,for "?????", the message length will be 0.

Interpreting the Feature 
*************************

Analyzing word count can help in understanding the nature of the interaction—whether it’s more casual and quick-paced or detailed and thorough.
Longer messages may indicate more detailed explanations, more extensive engagement, or more complex topics being discussed. 
Conversely, shorter messages might be more direct, concise, or reflect quick interactions.

For example, a curt "Hi" has a message length of 1, whereas a more detailed "Hello, How are you doing today?" has a message length of 6.

Related Features 
*****************
NA