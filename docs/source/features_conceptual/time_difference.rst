.. _time_difference:

Time Difference
============

High-Level Intuition
*********************
The response time between successive utterances.

Citation
*********
`Reichel, et al. (2015) <https://cpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/f/1603/files/2017/01/Reichel_etal_Interspeech_2015-2i4gnzk.pdf>`_

Implementation Basics 
**********************
This feature counts the mean duration of a message in minutes by subtracting the timestamp of the next message from the current message. 
Mean and Standard Deviation for the message are subsequently calculated.

Implementation Notes/Caveats 
*****************************
NA

Interpreting the Feature 
*************************
The difference between the timestamp of the current utterance and the previous utterance in seconds.
We accept only DateTime objects, Unix timestamps, or the number of seconds elapsed from the beginning of the conversation.

Related Features 
*****************
This feature helps us understand other temporal-related features, such as Burstiness (which measures whether the messages come in at regular intervals or whether they come in all at once in a short "burst")