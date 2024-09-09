.. _team_burstiness:

Team Burstiness
================

High-Level Intuition
*********************
This conversation-level feature measures the extent to which successive messages in a conversation take place in a temporally "bursty" manner (that is, all happening within a short period of time), as opposed to being spread evenly throughout a conversation.

Citation
*********
`Riedl and Woolley (2018) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2384068>`_

Implementation Basics 
**********************
Burstiness requires as input a vector of wait times between successive messages (which we compute in the toolkit as `time_diff`; see :ref:`time_difference`). Then, the metric is given by :math:`B = \frac{\text{standard\_deviation} - \text{mean}}{\text{standard\_deviation} + \text{mean}}`.

Implementation Notes/Caveats 
*****************************
In the event that there are no valid time differences, we return 0. If the time stamp is not present in the data at all, we return None.

Interpreting the Feature 
*************************
The feature is a value from -1 (not at all bursty; completely periodic) to 1 (extremely bursty) indicating the "burstiness" level of team exchanges. From Riedl and Woolley (2018):

.. epigraph::

   To index the temporal coordination of communication and code submissions within the team, we constructed a measure of the burstiness of team activity. This measure captures the degree to which team members concentrated their communication and work effort during relatively contained time periods versus spreading them out over time more equally. Specifically, we constructed a measure that captures the bursty nature of team activity based on the wait times (in minutes) between each team activity (either sending a message to the team or making a code submission). Greater correlation in the timings of team activities indicates greater burstiness. That is, greater burstiness indicates higher responsiveness of activity among members of the team. Conversely, if team activities are not well coordinated, this is equivalent to team activities’ following a random Poisson process, resulting in a low degree of burstiness. Thus, low burstiness indicates that team activities are less temporally correlated.

   (...) This measure is meaningful when both the mean and the standard deviation of wait times P(τ) exist, which is always the case for real-world finite signals (Wooten & Ulrich, 2012). B has a value in the bounded range [-1, 1], and its magnitude indexes with the signal’s burstiness: B = 1 corresponds to the most bursty signal, B = 0 to neutral, and B = -1 to a completely regular (periodic) signal. Thus, higher values of B correspond to spiked patterns of high team activity (high correlation of activity), while lower values of B correspond to more regular team activity (low correlation of activity).


Related Features 
*****************
:ref:`time_difference` is a required input to this feature.