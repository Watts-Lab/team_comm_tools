.. _politeness_strategies:

Politeness Strategies
======================

High-Level Intuition
*********************
A collection of conversational markers that indicates the use of politeness.

Citation
*********
`Danescu-Niculescu-Mizil et al. (2013) <https://www.cs.cornell.edu/~cristian/Politeness_files/politeness.pdf>`_

`Chang et al. (2020) <https://www.cs.cornell.edu/~cristian/ConvoKit_Demo_Paper_files/convokit-demo-paper.pdf>`_

Implementation Basics 
**********************

The PolitenessStrategies framework in Convokit identifies linguistic aspects of politeness using an annotated corpus of requests.
It evaluates and operationalizes politeness theory components like indirection and deference, with a classifier achieving near-human performance across domains. 

Each utterance (message) and a Spacy object (to do the parsing) is parsed through the transform_utterance() method of a PolitenessStrategies instance.
This method "Extract politeness strategies for raw string inputs (or individual utterances)." It calculates the following politeness strategies:

please
please_start
hashedge
indirect_btw
hedges
factuality
deference
gratitude
apologizing
1st_person_pl
1st_person
1st_person_start
2nd_person
2nd_person_start
indirect_greeting
direct_question
direct_start
haspositive
hasnegative
subjunctive
indicative

Implementation Notes/Caveats 
*****************************
NA

Interpreting the Feature 
*************************

List of politeness features returned by function (From cited papers):

====== ============================== ===================== ================== =====================================================
 No.   Strategy                        Politeness Score       In top quartile     Example
       (Positive = More Polite)
====== ============================== ===================== ================== =====================================================
 1.    Gratitude                       0.87***                78%***              I really appreciate that you’ve done them.
 2.    Deference                       0.78***                70%***              Nice work so far on your rewrite.
 3.    Greeting                        0.43***                45%***              Hey, I just tried to . . .
 4.    Positive lexicon                0.12***                32%***              Wow! / This is a great way to deal. . .
 5.    Negative lexicon                -0.13***               22%**               If you’re going to accuse me . . .
 6.    Apologizing                     0.36***                53%***              Sorry to bother you . . .
 7.    Please                          0.49***                57%***              Could you please say more. . .
 8.    Please start                    −0.30*                 22%                 Please do not remove warnings . . .
 9.    Indirect (btw)                  0.63***                58%**               By the way, where did you find . . .
 10.   Direct question                 −0.27***               15%***              What is your native language?
 11.   Direct start                    −0.43***               9%***               So can you retrieve it or not?
 12.   Counterfactual modal            0.47***                52%***              Could/Would you . . .
 13.   Indicative modal                0.09                   27%                 Can/Will you . . .
 14.   1st person start                0.12***                29%**               I have just put the article . . .
 15.   1st person pl.                  0.08*                  27%                 Could we find a less complex name . . .
 16.   1st person                      0.08***                28%***              It is my view that ...
 17.   2nd person                      0.05***                30%***              But what’s the good source you have in mind?
 18.   2nd person start                −0.30***               17%**               You’ve reverted yourself . . .
 19.   Hedges                          0.14***                28%                 I suggest we start with . . .
 20.   Factuality                      −0.38***               13%***              In fact you did link, . . .
====== ============================== ===================== ================== =====================================================

Related Features 
*****************
:ref:`politeness_receptiveness_markers` contains a similar list of markers related to politeness and receptiveness, computed by the SECR module (Yeomans et al., 2020); this can be though of as a more recent and upgraded version of the original politeness features.