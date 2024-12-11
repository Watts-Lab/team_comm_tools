.. _index_main:

The Team Communication Toolkit
===============================
Welcome to the official documentation of the Team Communication Toolkit! If you're a social scientist looking to explore conversational data, or just someone interested in better understanding the dynamics of team conversations, you've come to the right place.

Getting Started
****************

To use our tool, please ensure that you have Python >= 3.10 installed and a working version of `pip <https://pypi.org/project/pip/>`_, which is Python's package installer. Then, in your local environment, run the following:

.. code-block:: sh

   pip install team_comm_tools

This command will automatically install our package and all required dependencies.

Troubleshooting
++++++++++++++++

In the event that some dependency installations fail (for example, you may get an error that ``en_core_web_sm`` from Spacy is not found, or that there is a missing NLTK resource), please run this simple one-line command in your terminal, which will force the installation of Spacy and NLTK dependencies:

.. code-block:: sh

   download_resources

If you encounter a further issue in which the 'wordnet' package from NLTK is not found, it may be related to a known bug in NLTK in which the wordnet package does not unzip automatically. If this is the case, please follow the instructions to manually unzip it, documented in `this thread <https://github.com/nltk/nltk/issues/3028>`_.

Import Recommendations: Virtual Environment and Pip
+++++++++++++++++++++++++++++++++++++++++++++++++++++

**We strongly recommend using a virtual environment in Python to run the package.** We have several specific dependency requirements. One important one is that we are currently only compatible with numpy < 2.0.0 because `numpy 2.0.0 and above <https://numpy.org/devdocs/release/2.0.0-notes.html#changes>`_ made significant changes that are not compatible with other dependencies of our package. As those dependencies are updated, we will support later versions of numpy.

**We also strongly recommend that your version of pip is up-to-date (>=24.0).** There have been reports in which users have had trouble downloading dependencies (specifically, the Spacy package) with older versions of pip. If you get an error with downloading ``en_core_web_sm``, we recommend updating pip.

After you import the package and install dependencies, you can then use our tool in your Python script as follows:

.. code-block:: python

   from team_comm_tools import FeatureBuilder

*Note*: PyPI treats hyphens and underscores equally, so "pip install team_comm_tools" and "pip install team-comm-tools" are equivalent. However, Python does NOT treat them equally, and **you should use underscores when you import the package, like this: from team_comm_tools import FeatureBuilder**.

Using the Package
******************

Declaring a FeatureBuilder
+++++++++++++++++++++++++++
Once you import the tool, you will be able to declare a FeatureBuilder object, which is the heart of our tool. Here is some sample syntax:

.. code-block:: python

   my_feature_builder = FeatureBuilder(
      input_df = my_pandas_dataframe,
      # this means there's a column in your data called 'conversation_id' that uniquely identifies a conversation
      conversation_id_col = "conversation_id",  
      # this means there's a column in your data called 'speaker_id' that uniquely identifies a speaker
      speaker_id_col = "speaker_id",
      # this means there's a column in your data called 'messagae' that contains the content you want to featurize
      message_col = "message",
      # this means there's a column in your data called 'timestamp' that conains the time associated with each message; we also accept a list of (timestamp_start, timestamp_end), in case your data is formatted in that way.
      timestamp_col= "timestamp",
      # this is where we'll cache things like sentence vectors; this directory doesn't have to exist; we'll create it for you!
      vector_directory = "./vector_data/",
      # this will be the base file path for which we generate the three outputs;
      # you will get your outputs in output/chat/my_output_chat_level.csv; output/conv/my_output_conv_level.csv; and output/user/my_output_user_level.
      output_file_base = "my_output"
      # it will also store the output into output/turns/my_output_chat_level.csv
      turns = False,
      # these features depend on sentence vectors, so they take longer to generate on larger datasets. Add them in manually if you are interested in adding them to your output!
      custom_features = [  
            "(BERT) Mimicry",
            "Moving Mimicry",
            "Forward Flow",
            "Discursive Diversity"
      ],
   )

   # this line of code runs the FeatureBuilder on your data
   my_feature_builder.featurize()

Inspecting Generated Features
++++++++++++++++++++++++++++++

Feature Information
^^^^^^^^^^^^^^^^^^^^^
Every FeatureBuilder object has an underlying property called the **feature_dict**, which lists information and references about the features included in the toolkit. Assuming that **my_feature_builder** is the name of your FeatureBuilder, you can access the feature dictionary as follows:

.. code-block:: python

   my_feature_builder.feature_dict

The keys of this dictionary are the formal feature names, and the value is a JSON blob with information about the feature or collection of features. A more nicely-displayed version of this dictionary is also available on our `website <https://teamcommtools.seas.upenn.edu/HowItWorks>`_.

**New in v.0.1.4**: To access a list of the formal feature names that a FeatureBuilder will generate, you can use the **feature_names** property: 

.. code-block:: python
   
   my_feature_builder.feature_names # a list of formal feature names included in featurization (e.g., "Team Burstiness")

You can also use the **feature_names** property in tandem with the **feature_dict** to learn more about a specific feature; for example, the following code will show the dictionary entry for the first feature in **feature_names**:

.. code-block:: python

   my_feature_builder.feature_dict[my_feature_builder.feature_names[0]]

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
^^^^^^^^^^^^^^^^^^^^^

Once you call **.featurize()**, you can also obtain a convenient list of the feature columns generated by the toolkit:

.. code-block:: python
   
   my_feature_builder.chat_features # a list of the feature columns generated at the chat (utterance) level
   my_feature_builder.conv_features_base # a list of the base (non-aggregated) feature columns at the conversation level
   my_feature_builder.conv_features_all # a list of all feature columns at the conversation level, including aggregates

These lists may be useful to you if you'd like to inspect which features in the output dataframe come from the FeatureBuilder; for example:

.. code-block:: python

   jury_output_chat_level[my_feature_builder.chat_features]


Table of Contents
******************

Use the Table of Contents below to learn more about our tool. We recommend that you begin in the "Introduction" section, then explore other sections of the documentation as they become relevant to you. We recommend reading :ref:`basics` for a high-level overview of the requirements and parameters, and then reading through the :ref:`examples` for a detailed walkthrough and discussion of considerations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   intro
   basics
   examples
   features/index
   features_conceptual/index
   feature_builder
   utils/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
