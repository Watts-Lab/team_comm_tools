.. _TEMPLATE:

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
The difference between the timestamp of the current utterance and the previous utterance in seconds

Related Features 
*****************
NA