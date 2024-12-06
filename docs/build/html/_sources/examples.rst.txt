.. _examples:

================
Worked Example
================

-------------------
Demo / Sample Code
-------------------

After following the "Getting Started" steps below, the Team Communication Toolkit can be imported at the top of any Python script. We have provided a simple example file, "featurize.py", and a demo notebook, "demo.ipynb," under our `examples folder <https://github.com/Watts-Lab/team_comm_tools/tree/main/examples>`_ on GitHub.

We also have demos available on Google Colab that you can copy and run on your own:

- `Demo 1: Overview of Team Communication Toolkit and 3 Levels of Features <https://colab.research.google.com/drive/1e8D5h_prRJsGs_N563EvpoQK0uZIAYsJ?usp=sharing>`_

- `Demo 2: Sample Analysis with the Group Affect and Performance Corpus <https://colab.research.google.com/drive/1wnuUC5yg6uQH0TYP1AXVPGRgfp2-npgJ?usp=sharing>`_

Finally, this page will walk you through a case study, highlighting top use cases and considerations when using the toolkit.

----------------
Getting Started
----------------

To use our tool, please ensure that you have Python >= 3.10 installed and a working version of `pip <https://pypi.org/project/pip/>`_, which is Python's package installer. Then, in your local environment, run the following:

.. code-block:: sh

   pip install team_comm_tools

This command will automatically install our package and all required dependencies.

Troubleshooting
================

In the event that some dependency installations fail (for example, you may get an error that ``en_core_web_sm`` from Spacy is not found, or that there is a missing NLTK resource), please run this simple one-line command in your terminal, which will force the installation of Spacy and NLTK dependencies:

.. code-block:: sh

   download_resources

If you encounter a further issue in which the 'wordnet' package from NLTK is not found, it may be related to a known bug in NLTK in which the wordnet package does not unzip automatically. If this is the case, please follow the instructions to manually unzip it, documented in `this thread <https://github.com/nltk/nltk/issues/3028>`_.

You can also find a full list of our requirements `here <https://github.com/Watts-Lab/team_comm_tools/blob/main/requirements.txt>`_.

Import Recommendations: Virtual Environment and Pip
=====================================================

**We strongly recommend using a virtual environment in Python to run the package.** We have several specific dependency requirements. One important one is that we are currently only compatible with numpy < 2.0.0 because `numpy 2.0.0 and above <https://numpy.org/devdocs/release/2.0.0-notes.html#changes>`_ made significant changes that are not compatible with other dependencies of our package. As those dependencies are updated, we will support later versions of numpy.

**We also strongly recommend that your version of pip is up-to-date (>=24.0).** There have been reports in which users have had trouble downloading dependencies (specifically, the Spacy package) with older versions of pip. If you get an error with downloading ``en_core_web_sm``, we recommend updating pip.

Importing the Package
======================

After you import the package and install dependencies, you can then use our tool in your Python script as follows:

.. code-block:: python
   
   from team_comm_tools import FeatureBuilder

Now you have access to the :ref:`feature_builder`. This is the main class that you'll need to interact with the Team Communication Toolkit.

*Note*: PyPI treats hyphens and underscores equally, so "pip install team_comm_tools" and "pip install team-comm-tools" are equivalent. However, Python does NOT treat them equally, and **you should use underscores when you import the package, like this: from team_comm_tools import FeatureBuilder**.

-------------------------------------------------------
Walkthrough: Running the FeatureBuilder on Your Data
-------------------------------------------------------

Next, we'll go through the details of running the FeatureBuilder on your data, discussing each of the specific options / parameters at your disposal.

Configuring the FeatureBuilder
================================

The FeatureBuilder accepts any Pandas DataFrame as the input, so you can read in data in whatever format you like. For the purposes of this walkthrough, we'll be using some jury deliberation data from `Hu et al. (2021) <https://dl.acm.org/doi/pdf/10.1145/3411764.3445433?casa_token=d-b5sCdwpNcAAAAA:-U-ePTSSE3rY1_BLXy1-0spFN_i4gOJqy8D0CeXHLAJna5bFRTee9HEnM0TnK_R-g0BOqOn35mU>`_. 

We first import Pandas and read in the dataframe:

.. code-block:: python
   
   import pandas as pd
   juries_df = pd.read_csv("./example_data/full_empirical_datasets/jury_conversations_with_outcome_var.csv", encoding='utf-8')


Now we are ready to call the FeatureBuilder on our data. All we need to do is declare a FeatureBuilder object and run the .featurize() function, like this:

.. code-block:: python

	jury_feature_builder = FeatureBuilder(
		input_df = juries_df,
		speaker_id_col = "speaker_nickname",
		message_col = "message",
		timestamp_col = "timestamp",
		grouping_keys = ["batch_num", "round_num"],
		vector_directory = "./vector_data/",
		output_file_base = "jury_output",
		turns = True
	)
	jury_feature_builder.featurize()

Basic Input Columns
---------------------

Conversation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

* The **input_df** parameter is where you pass in your dataframe. In this case, we want to run the FeatureBuilder on the juries data that we read in!

* The **speaker_id_col** refers to the name of the column containing a unique identifier for each speaker / participant in the conversation. Here, in the data, the name of our columns is called "speaker_nickname."

	* If you do not pass anything in, "speaker_nickname" is the default value for this parameter.

* The **message_col** refers to the name of the column containing the utterances/messages that you want to featurize. In our data, the name of this column is "message."

	* If you do not pass anything in, "message" is the default value for this parameter.

	* We assume that all messages are ordered chronologically.

* The **timestamp_col** refers to the name of the column containing when each utterance was said. In this case, we have exactly one timestamp for each message, stored in "timestamp." 

	* If you do not pass anything in, "timestamp" is the default value for this parameter.

	* Sometimes, you may have data on both the *start* and the *end* of a message; when people are speaking live, it's possible that they talk over each other! In this case, the parameter **timestamp_col** also accepts a tuple of two strings, assumed to be *(start, end)*. For example, if we had two columns insteac, we could use the following:

	.. code-block:: python

		timestamp_col = ("timestamp_start", "timestamp_end")

* **In the FeatureBuilder, we assume that every conversation has a unique identifying string, and that all the messages belonging to the same conversation have the same identifier.** Typically, we would use the column **conversation_id_col** to indicate the name of this identifier. However, we also support cases in which there is more than one identifer per conversation, and our example here illustrates this functionality. The **grouping_keys** parameter means that we want to group by more than one column, and allow the FeatureBuilder to treat unique combinations of the grouping keys as the "conversational identifier". This means that we treat each unique combination of "batch_num" and "round_num" as a different conversation, and we *override* the **conversation_id_col** if a list of **grouping_keys** is present.

	* In cases where you are using **conversation_id_col**, "conversation_num" is the default value for this parameter.

	* If we were to use just one of the columns as our conversation identifier instead --- for example, treat each instance of "batch_num" as a unique conversation, we would use this syntax: 

	.. code-block:: python

		conversation_id_col = "batch_num"

Vector Directory
*******************

* The **vector_directory** is the name of a directory in which we will store some pre-processed information. Some features require running inference from HuggingFace's `RoBERTa-based sentiment model <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment>`_, and others require generating `SBERT vectors <https://sbert.net/>`_. These processes take time, and we cache the outputs so that subsequent runs of the FeatureBuilder on the same dataset will not take as much time. Therefore, we require you to pass in a location where you'd like us to save these outputs.

	* By default, the directory is named "vector_data/."

	* **Note that we do not require the name of the vector directory to be a folder that already exists**; if it doesn't exist, we will create it for you.

	* Inside the folder, we will store the RoBERTa outputs in a subfolder called "sentiment", and the SBERT vectors in a subfolder called "sentence." We will create both of these subfolders for you.

	* The **turns** parameter, which we will discuss later, controls whether or not you'd like the FeatureBuilder to treat successive utterances by the same individual as a single "turn," or whether you'd like them to be treated separately. We will cache different versions of outputs based on this parameter; we use a subfolder called "chats" (when **turns=False**) or "turns" (when **turns=True**).

.. _output_file_details:

Output File Naming Details 
*****************************

* There are three output files for each run of the FeatureBuilder, which mirror the three levels of analysis: utterance-, speaker-, and conversation-level. (Please see the section on `Generating Features: Utterance-, Speaker-, and Conversation-Level <intro#generating_features>`_ for more details.) These are generated using the **output_file_base** parameter.

	* **All of the outputs will be generated in a folder called "output."**

	* Within the "output" folder, **we generate sub-folders such that the three files will be located in subfolders called "chat," "user," and "conv," respectively.**

	* Similar to the **vector_directory** parameter, the "chat" directory will be renamed to "turn" depending on the value of the **turns** parameter.

* It is possible to generate different names for each of the three output files, rather than using the same base file path by modifying **output_file_path_chat_level** (Utterance- or Chat-Level Features), **output_file_path_user_level** (Speaker- or User-Level Features), and **output_file_path_conv_level** (Conversation-Level Features). However, because outputs are organized in the specific locations described above, **we have specific requirements for inputting the output paths, and we will modify the path under the hood to match our file naming schema,** rather than saving the file directly to the specified location.

	* We expect that you pass in a **path**, not just a filename. For example, the path needs to be "./my_file.csv", and not just "my_file.csv"; you will get an error if you pass in only a name without the "/".

	* Regardless of your path location, we will automatically append the name "output" to the fornt of your file path.

	* Within the "output" folder, **we will also generate the chat/user/conv sub-folders.**

	* If you pass in a path that already contains the above automatically-generated elements (for example, "./output/chat/my_chat_features.csv"), we will skip these steps and directly save it in the relevant folder.

	* Similar to the **vector_directory** parameter, the "chat" directory will be renamed to "turn" depending on the value of the **turns** parameter.

	* This means that the following two ways of specifying an output path are equivalent, assuming that turns=False:

	.. code-block:: python

		output_file_path_chat_level = "./jury_output_chat_level.csv"

		output_file_path_chat_level = "./output/chat/jury_output_chat_level.csv"

	* And these two ways of specifying an output path are equivalent, assuming that turns=True:

	.. code-block:: python

		output_file_path_chat_level = "./jury_output_turn_level.csv"

		output_file_path_chat_level = "./output/turn/jury_output_turn_level.csv"


Turns
******

* The **turns** parameter controls whether we want to treat successive messages from the same person as a single turn. For example, in a text conversation, sometimes individuals will send many message in rapid succession, as follows:

	* **John**: Hey Michael

	* **John**: How are you?

	* **John**: I wanted to talk you real quick!

		* These messages by John can be thought of as a single turn, in which he says, "Hey Michael, how are you? I wanted to talk to you real quick!" Instead, however, John sent three messages in a row, suggesting that he took three "turns." When the **turns** parameter is set to True, the FeatureBuilder will automatically combine messages like this into a single "turn."

		* We note, however, that one of our features (:ref:`turn_taking_index`) will always give the value of "1" in the case when you set **turns=True**, since, by definition, people will never take multiple "turns" in a row.


Advanced Configuration Columns
-------------------------------

More advanced users of the FeatureBuilder should consider the following optional parameters, depending on their needs.

Regenerating Vector Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~

* The **regenerate_vectors** parameter controls whether you'd like the FeatureBuilder to re-generate the content in the **vector_directory**, even if we have already cached the output of a previous run. It is useful if the underlying data has changed, but you want to give the output file the same name as a previous run of the FeatureBuilder.

	* By default, **we assume that, if your output file is named the same, that the underlying vectors are the same**. If this isn't true, you should set **regenerate_vectors = True** in order to clear out the cache and re-generate the RoBERTa and SBERT outputs.

Custom Features
~~~~~~~~~~~~~~~~~

* The **custom_features** parameter allows you to specify features that do not exist within our default set. **We default to NOT generating four features that depend on SBERT vectors, as the process for generating the vectors tends to be slow.** However, these features can provide interesting insights into the extent to which individuals in a conversation speak "similarly" or not, based on a vector similarity metric. To access these features, simply use the **custom_features** parameter:

	.. code-block:: python

		custom_features = [
            "(BERT) Mimicry",
            "Moving Mimicry",
            "Forward Flow",
            "Discursive Diversity"]


    * You can chose to add any of these features depending on your preference.

Analyzing First Percentage (%)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The **analyze_first_pct** parameter allows you to "cut off" and separately analyze the first X% of a conversation, in case you wish to separately study different sections of a conversation as it progresses. For example, you may be interested in knowing how the attributes of the first 50% of a conversation differ from the attributes of the entire conversation. Then you can sepcify the following:

	.. code-block:: python

		analyze_first_pct: [0.5, 1.0]

	* This will first analyze the first 50% of each conversation, and then analyze the full conversation.

	* By default, we will simply analyze 100% of each conversation.

Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~

* The parameters **ner_training_df** and **ner_cutoff** are required if you would like the FeatureBuilder to identify named entities in your conversations. For example, the sentence, "John, did you talk to Michael this morning?" has two named entities: "John" and "Michael." The FeatureBuilder includes a tool that automatically detects these named entities, but it requires the user (you!) to specify some training data with examples of the types of named entities you'd like to recognize. This is because proper nouns can take many forms, from standard Western-style names (e.g., "John") to pseudonymous online nicknames (like "littleHorse"). More information about these parameters can be found in :ref:`named_entity_recognition`.

.. _custom_aggregation:

Custom Aggregation
~~~~~~~~~~~~~~~~~~~

Imagine that you, as a researcher, are interested in high-level characteristics of the entire conversation (for example, how much is said), but you only have measures at the (lower) level of each individual utterance (for example, the number of words in each message). How would you "aggregate" information from the lower level to the higher level?

A simple solution is to sum up to the total number of words per utterance, and group by the conversation identifier. Then, you would have the total number of words for the entire conversation. You can imagine doing similar aggregations for other types of statistics --- for example, the average number of words, the variance in the number of words, and so on.

The FeautureBuilder includes built-in functionality to perform aggregations across different levels of analysis. By default, all numeric attributes generated at the utterance (chat) level are aggregated using the functions ``mean``, ``max``, ``min``, and ``stdev``.

We perform three types of aggregations. Consider, for example, a conversation with messages containing 5, 10, and 15 words. Then we would have the following:

- **Conversation-Level Aggregates** transform statistics at the level of an utterance (chat) to the level of a conversation. An example is the mean number of words per utterance (10) and the maximum number of words in any utterance (15). 
- **Speaker(User)-Level Aggregates** transform statistics at the level of an utterance (chat) to the level of a given speaker (user; participant) in a conversation. An example is the mean number of words per message by a particular speaker. 
- **Conversation-Level Aggregates of Speaker-Level Information**: transform information about the speakers (users; participants) to the level of a conversation. An example is the average number of words for the most talkative speaker.

Given that there are multiple default aggregation functions and numerous utterance-level attributes, an (overwhelmingly) large number of aggregation statistics can be produced. As of **v.0.1.5**, aggregation behavior can be customized using the following parameters:

- ``convo_aggregation``: A boolean that defaults to ``True``; when turned to ``False``, aggregation at the conversation level is disabled **[NOTE 1]**.
- ``convo_methods``: A list specifying which aggregation methods to use at the conversation level. Options include ``mean``, ``max``, ``min``, ``stdev``, ``median``, and ``sum`` **[NOTE 2]; [NOTE 3]**. We default to using ``mean``, ``max``, ``min``, and ``stdev``.
- ``convo_columns``: A list specifying which utterance-level attributes to aggregate to the conversation level. These should be valid columns in the utterance (chat)-level data. This defaults to ``None``, which is configured to aggregate all available numeric outputs.

Equivalent parameters for the speaker (user) level are:

- ``user_aggregation``: A boolean that defaults to ``True``; when turned to ``False``, aggregation at the speaker (user) level is disabled **[NOTE 1]**.
- ``user_methods``: A list specifying which aggregation methods to use at the speaker/user level (with the same options as the conversation level).
- ``user_columns``: A list specifying which utterance-level attributes to aggregate at the speaker/user level.

The table below summarizes the different types of aggregation, and the ways in which they can be customized:

.. list-table:: Aggregation Overview
   :header-rows: 1
   :widths: 20 15 20 20 10 15 25

   * - Aggregation Type
     - Default Methods
     - Methods Available
     - Customization Parameters
     - Output DataFrame
     - Example Aggregation
     - Interpretation
   * - Utterance (Chat) -> Conversation
     - ``mean``, ``max``, ``min``, ``stdev``
     - ``mean``, ``max``, ``min``, ``stdev``, ``median``, ``sum``
     - ``convo_aggregation``, ``convo_methods``, ``convo_columns``
     - Conversation
     - ``mean_num_words``
     - Average number of words per utterance in the conversation
   * - Utterance (Chat) -> Speaker/User
     - ``mean``, ``max``, ``min``, ``stdev``
     - ``mean``, ``max``, ``min``, ``stdev``, ``median``, ``sum``
     - ``user_aggregation``, ``user_methods``, ``user_columns``
     - Speaker/User
     - ``mean_num_words``
     - Average number of words per utterance for a given individual
   * - Speaker (User) -> Conversation
     - ``mean``, ``max``, ``min``, ``stdev``
     - ``mean``, ``max``, ``min``, ``stdev``, ``median``, ``sum``
     - ``convo_aggregation``, ``convo_methods``, ``convo_columns``
     - Conversation
     - ``max_user_mean_num_words``
     - Average number of words per utterance for the person who talked the most


Example Usage of Custom Aggregation Parameters
************************************************

To customize aggregation behavior, simply add the following when constructing your FeatureBuilder:

.. code-block:: python

     convo_methods = ['max', 'median']  # This aggregates ONLY "positive_bert" at the conversation level using max and median.
     convo_columns = ['positive_bert'],
     user_methods = ['mean']            # This aggregates ONLY "negative_bert" at the speaker/user level using mean.
     user_columns = ['negative_bert']

To turn off aggregation, set the following parameters to ``False``. By default, both are ``True`` as aggregation is performed automatically:

.. code-block:: python

     convo_aggregation = False
     user_aggregation = False

Important Notes and Caveats
*****************************

- **[NOTE 1]** Even when aggregation is disabled, totals of words, messages, and characters are still summarized, as these are required for calculating the Gini Coefficient features.
- **[NOTE 2]** Be careful when choosing the "sum" aggregation method, as it is not always appropriate to use the "sum" as an aggregation function. While it is a sensible choice for utterance-level attributes that are *countable* (for example, the total number of words, or other lexical wordcounts), it is a less sensible choice for others (for example, it does not make sense to sum sentiment scores for each utterance in a conversation). Consequently, using the "sum" feature will come with an associated warning.
- **[NOTE 3]** In addition to aggregating from the utterance (chat) level to the conversation level, we also aggregate from the speaker (user) level to the conversation level, using the same methods specified in ``convo_methods`` to do so.

Cumulative Grouping
~~~~~~~~~~~~~~~~~~~~

* The parameters **cumulative_grouping** and **within_task** address a special case of having multiple conversational identifiers; **they assume that the same team has multiple sequential conversations, and that, in each conversation, they perform one or more separate activities**. This was originally created as a companion to a multi-stage Empirica game (see: `<https://github.com/Watts-Lab/multi-task-empirica>`_). For example, imagine that a team must complete 3 different tasks, each with 3 different subparts. Then we can model this event in terms of 1 team (High level), 3 tasks (Mid level), and 3 subparts per task (Low level).

	* In such an activity, we assume that there are three levels of identifiers: High, Mid, and Low.

	* The "High" level identifier can be thought of as the team's identifier, and the same team then completes multiple different activities (or has multiple different conversations), each with one or more subparts. 

	* The "Mid" level identifier is a sequence of separate conversations about different topics.

	* The "Low" level identifier assumes that, within each topic, there are one or more subparts/subtasks. For example, suppose that teams must discuss three different political issues (Gun Control, Death Penalty, and Abortion), and within each topic, they need to discuss it from two perspectives (Democrat, Republican). In this case, there would be an identifier for each of the 3 Mid-level activities (political issues), and for each Low-level subpart (Democrat/Republican).

	* If your activity does not have any subparts, set your Low-level identifier equal to the Mid-level identifier.

	* The **cumulative_grouping** parameter accounts for the case in which, in such a nested sequence of conversations, you may want to count a team's previous conversations as "part" of the current conversation. For example, suppose that the team first discussed the Gun Control issue, and then moves on to discuss the Death Penalty issue. You may imagine that a heated discussion about Gun Control might impact the later discussion about the Death Penalty, and you may want to incorporate the previous topic when analyzing the second conversation. **In effect, the cumulative_grouping paramter creates a duplicate of the "earlier" conversation and groups it with the later conversation, so that analyses of sequential conversations can incorporate information from what happened before.**

		* Thus, without **cumulative_grouping**, we would have 6 independent conversations:

			#. Gun Control, Democrat

			#. Gun Control, Republican

			#. Death Penalty, Democrat

			#. Death Penalty, Republican

			#. Abortion, Democrat

			#. Abortion, Republican

		* But with **cumulative_grouping = True**, we would have the following conversations, in which we treat each conversation as building on the last one:

			#. Gun Control, Democrat

			#. Gun Control, Democrat; Gun Control, Republican

			#. Gun Control, Democrat; Gun Control, Republican; Death Penalty, Democrat

			#. Gun Control, Democrat; Gun Control, Republican; Death Penalty, Democrat; Death Penalty, Republican

			#. Gun Control, Democrat; Gun Control, Republican; Death Penalty, Democrat; Death Penalty, Republican; Abortion, Democrat

			#. Gun Control, Democrat; Gun Control, Republican; Death Penalty, Democrat; Death Penalty, Republican; Abortion, Democrat; Abortion, Republican

	* A further consideration is that the user may only wish to make a conversation "cumulative" at the Mid level, but not across all Mid levels. For example, extending the political discussion case, you may think that discussing the Democratic perspective on the same issue might influence the discussion of the Republican perspective, but you may think the Gun Control, Death Penalty, and Abortion issues are separate topics that should not be treated as the same "conversation." In this case, setting **within_task = True** would combine conversations at the "Low" level, but would not combine conversations at the "Mid" level.

		* Thus, with **cumulative_grouping = True**, we would have the following conversations:
			
			#. Gun Control, Democrat

			#. Gun Control, Democrat; Gun Control, Republican

			#. Death Penalty, Democrat

			#. Death Penalty, Democrat; Death Penalty, Republican

			#. Abortion, Democrat

			#. Abortion, Democrat, Abortion, Republican

	* Finally, it is important to remember that, since cumulative groupings mean that we progressively consider more and more of the same conversation, **your conversation dataframe will substantially increase in size**, and this may affect the runtime of your FeatureBuilder.

Additional FeatureBuilder Considerations
------------------------------------------

Here are some additional design details of the FeatureBuilder that you may wish to keep in mind:

	* **Outside of the required columns (Conversation Identifier, Speaker Identifier, Message, and Timestamp), the FeatureBuilder will ignore any remaining columns in your conversation data.** The FeatureBuilder strictly *appends* new columns to the input dataset. We made this design decision so that researchers can run the FeatureBuilder and conduct additional analyses (e.g, regression) directly on the output; for example, you may have additional information (metadata, outcome variables) included in your input dataframe that you want to analyze alongside the conversation features. We will not touch them.

		* The only caveat to this rule is if you happen to have a column that is named exactly the same as one of the conversation features that we generate. In that case, your column will be overwritten. Please refer to `<https://teamcommtools.seas.upenn.edu/HowItWorks>`_ for a list of all the features we generate, along with their column names.

	* **When summarizing features from the utterance level to the conversation and speaker level, we only consider numeric features.** This is perhaps a simplifying assumption more than anything else; although we do extract non-numeric information (for example, a Dale-Chall label of whether an utterance is "Easy" to ready or not; a list of named entities identified), we cannot summarize these efficiently, so they are not considered.

Inspecting Generated Features
--------------------------------

Feature Information
~~~~~~~~~~~~~~~~~~~~

Every FeatureBuilder object has an underlying property called the **feature_dict**, which lists information and references about the features included in the toolkit. Assuming that **jury_feature_builder** is the name of your FeatureBuilder, you can access the feature dictionary as follows:

.. code-block:: python

   jury_feature_builder.feature_dict

The keys of this dictionary are the formal feature names, and the value is a JSON blob with information about the feature or collection of features. A more nicely-displayed version of this dictionary is also available on our `website <https://teamcommtools.seas.upenn.edu/HowItWorks>`_.

**New in v.0.1.4**: To access a list of the formal feature names that a FeatureBuilder will generate, you can use the **feature_names** property: 

.. code-block:: python
   
   jury_feature_builder.feature_names # a list of formal feature names included in featurization (e.g., "Team Burstiness")

You can also use the **feature_names** property in tandem with the **feature_dict** to learn more about a specific feature; for example, the following code will show the dictionary entry for the first feature in **feature_names**:

.. code-block:: python

	jury_feature_builder.feature_dict[jury_feature_builder.feature_names[0]]

Here is some example output (for the RoBERTa sentiment feature):

.. code-block:: text
	
	{'columns': ['positive_bert', 'negative_bert', 'neutral_bert'],
	 'file': './utils/check_embeddings.py',
	 'level': 'Chat',
	 'semantic_grouping': 'Emotion',
	 'description': 'The extent to which a statement is positive, negative, or neutral, as assigned by Cardiffnlp/twitter-roberta-base-sentiment-latest. The total scores (Positive, Negative, Neutral) sum to 1.',
	 'references': '(Hugging Face, 2023)',
	 'wiki_link': 'https://conversational-featurizer.readthedocs.io/en/latest/features_conceptual/positivity_bert.html',
	 'function': <function team_comm_tools.utils.calculate_chat_level_features.ChatLevelFeaturesCalculator.concat_bert_features(self) -> None>,
	 'dependencies': [],
	 'preprocess': [],
	 'vect_data': False,
	 'bert_sentiment_data': True}

Feature Column Names
~~~~~~~~~~~~~~~~~~~~~

Once you call **.featurize()**, you can also obtain a convenient list of the feature columns generated by the toolkit:

.. code-block:: python
   
   jury_feature_builder.chat_features # a list of the feature columns generated at the chat (utterance) level
   jury_feature_builder.conv_features_base # a list of the base (non-aggregated) feature columns at the conversation level
   jury_feature_builder.conv_features_all # a list of all feature columns at the conversation level, including aggregates

These lists may be useful to you if you'd like to inspect which features in the output dataframe come from the FeatureBuilder; for example:

.. code-block:: python

	jury_output_chat_level[jury_feature_builder.chat_features]