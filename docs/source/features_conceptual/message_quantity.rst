.. _message_quantity:

Message Quantity
=================

High-Level Intuition
*********************
This function by itself is trivial; by definition, each message counts as 1. However, at the conversation level, we use this function to count the total number of messages/utterance via aggregation.

Citation
*********
`Cao et al. (2021) <https://dl.acm.org/doi/pdf/10.1145/3432929>`_

`Marlow et al. (2018 as objective communication frequency <https://pdf.sciencedirectassets.com/272419/1-s2.0-S0749597817X00071/1-s2.0-S074959781630125X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjED4aCXVzLWVhc3QtMSJHMEUCIQDi1IPGu%2BOfutPPvJtxfJhG3SpY6wG1ZVsKytSbvNRWlgIga%2BZnmK4RYB2VU1lPtQ2upquMItqYUZRhmzVYOdOFCLoqsgUIZxAFGgwwNTkwMDM1NDY4NjUiDFlcRYIcwnjr762KPCqPBfC8ZSaZXVC5uflWJ5%2BqBH%2F2FLDstUTX2m7h1lFVUQq5f4OB0Kc9eGw%2BL%2F4iXjkYUDVSDSLtLIIbnBwByBJvIxmeobr%2BK2H9RlssfVto4HRXHHtH27TzGwzRRioTlY2rGLzn4Hz9qXYpLJkHK7OSMpovbxzxH67cCiJG1slj6YHgGv0%2BLQWvALaNGI9xG6QFovU1iMAwPMKbRA6MW1V7vNHkSBenmLires74TS%2FdQ1Kv1lnP0uQUfotnhNfXrePqD1dFN7ifUm1uHmWT6g6tvy7TKLaDsqyiPG1eNNgdjRkT0wpqsV6ICUTGRIn%2FSDIBiL9NbTqyvaJiXviIhRgm%2Fll0TtO5CM3niT5gOSB4WwioUgd6KUUVIw7yGTJEuXSdlSavOx2K%2Fb7tUQ4tMxn6%2FU0CJbfFBqaiLp9rOJPIevdpipdGxAptn9vRIEMzQs%2Blybs%2B6%2FbD5e92KoLoFlp9W%2FBf3MsO0UIihz1XGHPmpYnXGNzQO8qsrIFeD93VhzUgBDpWoH%2BWcg2XNw7T2%2BdTzz8ug%2FOcQQoTqrr2TwxeMfji6LvHY0KTWC0QzRjlXEXyAC1Cirui1R9DbX4WfO%2FACzSUfENJJ%2BwDdtgbm0i28xU8spZxuXsDPI9JI6%2BpEN3YBZlSmWnrseTKEL02V3LsENoKtrs1%2FDzSLUxfzhNALHGErSZDfFRWo%2BOeqV2GTV50IsvxitE3tq1spKg8TLI0Mg0LegEZazM45gBtK%2B8T5G7b%2FlF3904dM%2FE%2FWasBSavoF%2FNUtZ1yOdKxVcxDdfNFk2O%2BWvPzjhZHvDBN8u%2Fo0hUpqHOOQKvDJmNj4OnrJMgX8%2BHbLtNjel3P%2BZNoARwam7B7Qmq9wwSseQWI0yn0aGUwrtT9tgY6sQEhhrTIOBTTZQZs2KWIT2i0%2BEOGWIHv0PThtp%2FgCpnzmsaTnCs2%2F9etcnWQk0ckDozm58hU9Rhs3YxY08mrpcz9s5KypinRO64TnI%2Fv6R1B4%2B%2BdrJLE9AcS%2BYbn3Ddq7AYiskyqd44hPWv%2BE27xJqHiJ6sb2Vpx3B3G8UgtaQUzYPWxrO2xOaVoPSdyBVuUeOTwvbbLdaOXlmiF7ySUumF1it0BDTr6oXt8yjvlc9j0ea8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240909T221428Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRK663XKK%2F20240909%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cbc66d4e48056bc7153dd0fc8a0bc3c03fa16cfa5b6d0e3cc0480ef43bb6bb67&hash=003e910c8585c899e1352fe5fbc23962e254ef7610fe3d8c10d72cc7f864f13f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S074959781630125X&tid=spdf-fd0693cb-3bfb-485e-8f05-7a7712178f1d&sid=45edce937838a54cbc1b5f47c6c33c3edd23gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f165f0256035b5c500a0e&rr=8c0a880dafe38fea&cc=us>`_

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