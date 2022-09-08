# Measuring Team Performance by Mapping Team Processes

## 1. Which interactions predict successful teams?
Social science traditionally focuses on studying just one variable at a time --- an approach known as the "one-at-a-time" experiment (Almaatouq et al., 2022, working paper). The literature on team performance is filled with papers that identify the role that a few specific team process variables play in improving team performance --- for example, having shared mental models (Kraiger and Wenzel, 2009); having discursive diversity (Lix et al., 2021); or being able to share workloads well (Campion, Medsker, and Higgs, 1993).

[**Figure 1:** examples of flowcharts and theories that each relate a small number of key variables to performance. While the individual flowcharts make sense on their own, how does everything connect together? Imagine that you meet a manager, who asks what she should do to improve her team's performance. There are so many competing models and theories that it's impossible to say for sure what she should focus on. For all the work that has taken place in pursuit of team performance, the field has not yet produced a unified or generalizable theory.]

<img width="843" alt="Screen Shot 2022-09-07 at 9 55 08 PM" src="https://user-images.githubusercontent.com/28793641/189016295-9b3df001-4016-43a2-983b-3353def7b764.png">

The literature on team performance thus contains a large number of *individual theories* about which team processes predict team outcomes, but no way to compare or unify them. To make matters worse, the studies are conducted on a diverse set of populations who performed different different tasks during different time periods. Some were measured in the lab, with university students or anonymous strangers recruited online, and others were performed on specific companies with employees who had a yearslong history of working with each other. It's hard to say which of these theories dominates the measure of a team's overall quality, becuase one simply cannot compare apples to oranges.

Making matters worse, many of these theories rely on vague concepts that can't be measured, much less turned into an actionable recommendation. Sure, it makes sense that teams will work better together if they have "shared mental models," but what does that mean? How do you measure that? How does it interact with all of the other variables that might be at play? Other concepts are equally vague --- what does it mean to have good "coordination," "low conflict," "good team spirit?" Answering the question of performance feels like having a conversation with a toddler: you are hopelessly lost in a nested list of "what's that?"

"Performance is a combination of team member inputs and the team process conditions."

"What are the relevant team process conditions?"

"Well, they need to have shared mental models ..."

"What's that?"

"..."

## 2. Our goal is to produce a quantitative model of team performance that will be dynamically trained on real team data.

Within the management literature, performance is (no pun intended) a work horse metric. Nearly every study of groups or teams includes a measure of "performance," or the idea that some groups are simply better or more productive than others. Performance even underlies the reason why we choose to have teams at all. It is because teams have *synergy* (Larson, 2010) --- a "performance gain" from when a group of people collaborate rather than work alone --- that teamwork is so appealing. After all, if teams do not perform well together, why use them at all?

But as much as performance has been discussed in the literature, and as much as managers demand new ways to improve it, performance is, in reality, a somewhat elusive concept. The study of teams suffers from a "unit of analysis problem:" that is, because teams are composed of individuals, as well as products of their context, it is not obvious what one is truly measuring when one measures performance.

This problem is by no means new. As Davis (1969) writes:

> It is commonly observed that “group behavior” is a function of three classes of variables: (a) person variables, such as abilities, personality traits, or motives; (b) environmental variables that reflect the effects of the immediate location and larger organization, community, or social context in which group action takes place; and (c) variables associated with the immediate task or goal that the group is pursuing.

Thus, performance is some unspecified mixture of the person, their environment, and their task. Other models echo this idea. A common framework for team performance posits that performance consists of some combination of *inputs* (the specific features of individuals on the team and the environment they belong to) and the *process* (the characteristics of their interactions together). A variant of this model has appeared in numerous papers across the decades (e.g., McGrath, 1964; Hackman and Morris, 1975; Dickinson and McIntyre, 1997; Cicek et al., 2005).

But what, exactly, is the right mixture? Even if we agree that performance is some combination of the people, their environment, and their process, we still don't have a "recipe" for which exact combination of elements constitutes strong performance. We don't even know which features within these broad categories matter.

This research project aims to fill this gap. Our goal is to systematically survey the literature on team processes and team performance, producing as comprehensive a list as possible of the features of teams that are theorized (based on prior research) to increase team performance. We will then "map out" how these features relate to each other, creating a comprehensive model that integrates prior work. Next, we will develop ways to computationally measure these features, using Natural Language Processing to translate vague concepts like "coorindation" into a suite of metrics that we can observe in real-life teams. In a final stage of the project, we will use our features as the basis of a model, which will then train on datasets (both existing ones and ones that we will collect through field experiments) of real teams in action. From those models, we will learn which features of team performance tend to matter most across different case studies and contexts. 

[**Figure 2**: A very early start to mapping out how different features relate to each other.]

<img width="1061" alt="Screen Shot 2022-09-07 at 10 16 22 PM" src="https://user-images.githubusercontent.com/28793641/189018669-dca75f76-b6bf-4d62-bcfa-e9b5d889f05f.png">

## 4. Next Steps
1. A first step, we need to collect as many individual concepts and theory papers about team performance as possible, creating a list of general features of a teams process and interactions that matter in measuring a team's performance. The idea is to have the map represent, to the extent possible, the current "state of the art" in what we know about peformance. Although it doesn't have to be perfectly comprehensive (a key advantage of the map is that it will be easy to add new rows or columns in the future), it should have enough coverage to be considered a useful and trustworthy starting point for this approach. Additionally, it will be important to document the original source of the theories --- were they developed based on artificial teams in the lab (e.g., students or online crowdworkers?) or were they developed based on observing teams at a specific company? Knowing these details will help us to "weight" which features are more important for which contexts, as well as compare which sources of performance insights are more generlizable to other contexts.

2. Next, we will need to *drill down as far as possible into how each concept can be measured*. For instance, team morale could be measured as having "X% message of positive sentiment," or it could be measured as "referring to the team by its team name Y% of the time," an so on. Some measures will be better than others, and the key is to try to document as many options and possibilities as precisely as possible. Notably, there will be quite a bit of NLP involved here, since most of our observations of team interactions are text chats.

[**Figure 3**: At this point in the project, our work would be summarized by the following figure. The inputs are the various independent theories; the output is a unified table of features and how to measure them.]

<img width="858" alt="Screen Shot 2022-09-07 at 10 23 36 PM" src="https://user-images.githubusercontent.com/28793641/189019499-38427555-2b27-4d5b-a4fd-ed62acdc75c7.png">

3. Then, we'll pit the measurements against real data, of both teams in the lab and in the field. We'll use datasets of recorded text conversations of people interacting in teams and see how well the features we designed in Steps 1-2 predict performance data. We can use this data to learn weights for the features --- identifying which features have the strongest weight in determining a team's level of success.

4. Future work can even build towards the creation of social computing systems. Can we build a bot that automatically measures peoples' projected team performance, and that makes suggestions and useful corrections as they work? Can that be deployed in a way that is helpful and non-annoying? Are more large-scale changes needed? That is, should we design new systems or ways of working altogether?

In short, our approach brings computer science and management science together. This project will develop an artifact that documents theories of team performance in a commensurable way, and which will then be combined with real-world data to learn develop cutting-edge about what makes a team "perform" well.

## Documents and Handy Links
- Our Team Email: csslab-team-process-map@wharton.upenn.edu (Ask Emily for the password!)

- This "master sheet" documents all of the team performance metrics that we are collecting so far:
https://docs.google.com/spreadsheets/d/1JnChOKFXkv944LvnYbzI1qrHLEPfCvEMN5XzP1AxvmA/edit?usp=sharing

Under "New Paper Collection - ADD HERE," we will collect papers and make an initial pass at the information. We are experimenting with the other tabs to group key themes together.

- Team Performance Metric Lit Review: https://docs.google.com/document/d/1RTvKzkLdporPWTxNDZ-kUYdT8yKGxMl7l1I_Q242_Ow/edit#

This is a slightly messy literature review done by Emily. Much of it has already been integrated into the master spreadsheet, but it is a good starting point for background reading/resources, even though it's not perfectly organized.

- Literature Review folder with a few starter PDF's:
https://drive.google.com/drive/u/0/folders/1NmZi4Z8ywP-1wzhsEDStMYQS4gj0kD7y
