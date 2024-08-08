.. Team Communication Toolkit documentation master file, created by
   sphinx-quickstart on Fri Jun 14 12:54:37 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Team Communication Toolkit
===============================
Welcome to the official documentation of the Team Communication Toolkit! If you're a social scientist looking to explore conversational data, or just someone interested in better understanding the dynamics of team conversations, you've come to the right place.

Getting Started
****************

To use our tool, please ensure that you have Python >= 3.10 installed and a working version of `pip <https://pypi.org/project/pip/>`_, which is Python's package installer. Then, in your local environment, run the following:

.. code-block::

   pip install team_comm_tools

You will also need to ensure that Spacy and NLTK are installed in addition to the required dependencies. If you get an error that en_core_web_sm is not found, you should ensure the following:

.. code-block::

   spacy download en_core_web_sm

Additionally, we require the following NLTK dependencies, which need to be downloaded via Python. If you don't have them, run the following python script in your environment:

.. code-block::

   import nltk

   nltk.download('nps_chat')
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')

**We strongly recommend using a virtual environment in Python to run the package.** We have several specific dependency requirements. One important one is that we are currently only compatible with numpy < 2.0.0 because `numpy 2.0.0 and above <https://numpy.org/devdocs/release/2.0.0-notes.html#changes>`_ made significant changes that are not compatible with other dependencies of our package. As those dependencies are updated, we will support later versions of numpy.

After you import the package and install dependencies, you can then use our tool in your Python script as follows:

.. code-block:: python

   from team_comm_tools import FeatureBuilder

*Note*: PyPI treats hyphens and underscores equally, so "pip install team_comm_tools" and "pip install team-comm-tools" are equivalent. However, Python does NOT treat them equally, and **you should use underscores when you import the package, like this: from team_comm_tools import FeatureBuilder**.

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
      # give us names for the utterance (chat), speaker (user), and conversation-level outputs
      output_file_path_chat_level = "./my_output_chat_level.csv", 
      output_file_path_user_level = "./my_output_user_level.csv",
      output_file_path_conv_level = "./my_output_conversation_level.csv",
      # if true, this will combine successive turns by the same speaker.
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
   my_feature_builder.featurize(col="message")

Use the Table of Contents below to learn more about our tool. We recommend that you begin in the "Introduction" section, then explore other sections of the documentation as they become relevant to you. More information on using our tool can be found in :ref:`examples`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   intro
   feature_builder
   features/index
   features_conceptual/index
   examples
   utils/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
