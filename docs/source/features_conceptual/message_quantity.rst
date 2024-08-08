.. _message_quantity:

Message Quantity
============

High-Level Intuition
*********************
This function by itself is trivial; by definition, each message counts as 1. However, at the conversation level, we use this function to count the total number of messages/utterance via aggregation.

Citation
*********
NA

Implementation Basics 
**********************

This function is trivial; by definition, each message counts as 1. However, at the conversation level, we use this function to count the total number of messages/utterance via aggregation.

Implementation Notes/Caveats 
*****************************
This feature becomes relevant at the conversation level, but is trivial at the chat level.

Interpreting the Feature 
*************************

This feature provides a measure of the conversation's length and activity. 
A higher count indicates a more extensive, while a lower count may suggest a brief interaction. 
It is important to check this feature while comparing different conversations as the number of utterances can be a confounder and affect the outcomes of the conversation


Related Features 
*****************
NA