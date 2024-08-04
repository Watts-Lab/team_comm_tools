.. _TEMPLATE:

ONLINE DISCUSSION TAGS
============

High-Level Intuition
*********************
This feature detects special metrics specific to communications in an online setting, such as capitalized words, hyperlinks and quotes, amongst others noted below.

Citation
*********
NA

Implementation Basics 
**********************

Calculates a number of metrics specific to communications in an online setting:

1. Num all caps: Number of words that are in all caps
2. Num links: Number of links to external resources
3. Num Reddit Users: Number of usernames referred to, in u/RedditUser format.
4. Num Emphasis: The number of times someone used **emphasis** in their message
5. Num Bullet Points: The number of bullet points used in a message.
6. Num Line Breaks: The number of line breaks in a message.
7. Num Quotes: The number of “quotes” in a message.
8. Num Block Quotes Responses: The number of times someone uses a block quote (”>”), indicating a longer quotation
9. Num Ellipses: The number of times someone uses ellipses (…) in their message
10. Num Parentheses: The number of sets of fully closed parenthetical statements in a message
11. Num Emoji: The number of emoticons in a message, e.g., “:)”

Implementation Notes/Caveats 
*****************************
1. This feature should be run on text that is not preprocesed to remove puncutations, hyperlinks and before conversion to lowercase
2. The "Reddit Users" features might not be informative in Non-Reddit contexts

Interpreting the Feature 
*************************
Note: These are a few examples for illustration. This is not a comprehensive list. 

1. Num all caps:
Example: This is a sentence with SEVERAL words in ALL CAPS.

Interpretation: This can be used to understand the number of emphasized words, often associated with high arousal

7. Num Quotes:
Example: Oh, yet another of the "amazing" meetings where we discuss the same thing for hours!

Interpretation: Can be interpreted as a sarcastic comment.


Related Features 
*****************
NA