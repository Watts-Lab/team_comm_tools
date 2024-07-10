.. _turn_taking_index:

Turn Taking Index
=================

High-Level Intuition
*********************
This feature is about the extent to which people take turns in a conversation. Turns are the number of distinct, broken up "chats" in a row during which a person has the "floor." 
The turn-taking index distinguishes between teams that have people who speak in big chunks versus teams that have people interrupting each other a great deal.

Citation
*********
`Almaatouq, Alsobay et al., 2023 <https://onlinelibrary.wiley.com/doi/pdf/10.1111/tops.12706>`_

Implementation Basics 
**********************
Read the code associated with this feature and give an intuitive explanation of what the code does. How does the technical implementation link to the high-level concept?

The turn-taking formula is: (Number of Turns - 1) / (Total Number of Chats - 1)

For example, in the following dataframe:
.. list-table:: Speaker Turn Counts
   :widths: 50 25
   :header-rows: 1

   * - speaker_nickname
     - turn_count
   * - emily
     - 5
   * - amy
     - 10
   * - nikhil
     - 2
   * - emily
     - 1
   * - amy
     - 2

Number of turns taken = 5 (Emily, Amy, Nikhil, Emily, Amy) Total Number of Chats = 5 + 10 + 2 + 1 + 2 = 20

Index = (5-1)/(20-1) = 0.21

The -1 at the top and bottom of the fraction is due to the fact that if one person talks for the entire time 
(e.g., the whole conversation is just a monologue), then we want the index to be 0, not 1 / (n_chats).

Implementation Notes/Caveats 
*****************************
In the original paper, turn-taking was actually measured in terms of actions in a game. 
Someone took a "turn" when they made a bunch of uninterrupted turns in a game; then, their turn-taking index divided the number of uninterrupted turns by the total number of turns taken:

> A group’s turn-taking index for a given round is measured by dividing the number of turns taken (a turn is an uninterrupted sequence of room assignments made by a single player, each defining an intermediate solution) by the total number of solutions generated on a particular task instance.

Why does this matter?

> This measure is intended to differentiate between groups that collaborate in blocks (e.g., Player 1 moves N times, then Player 2 moves N times, then Player 3 moves N times) and groups that collaborate more dynamically (e.g., Players 1, 2, and 3 alternate moves, for a total of 3N moves)—in the first example, the number of turns taken is 3, and in the second example, the number of turns taken is 3N, but the total number of solutions generated is the same in both cases.

In our case, the turns are be *turns within a conversation* --- the number of distinct, broken up "chats" in a row during which a person has the "floor." 
The turn-taking index will therefore distinguish between teams that have people speak in big chunks (you say your piece, then I say mine, debate-style), versus teams that have people interrupting each other a great deal.

Interpreting the Feature 
*************************
In the edge case where only a single person spoke the entire time, causing the denominator to be 0, the turn-taking index is set to 0.

Read the code associated with this feature and answer the following questions, if applicable:

1. What are the bounds of the score? What does a high versus low score mean? (How should you read this score?)
2. Give a concrete example (e.g., negative score versus positive score)
3. What DOESN’T the score measure? That is, what does the score take into account, and what are some ways that it might not capture the high-level social science concept?
4. Are there any edge cases that we should be aware of? (e.g., if the conversation contains only one chat?) To the best of your knowledge, how does the code handle it?

Related Features 
*****************
N/A