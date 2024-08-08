[![Testing Features](https://github.com/Watts-Lab/team_comm_tools/workflows/Testing%20Features/badge.svg)](https://github.com/Watts-Lab/team_comm_tools/actions?query=workflow:"Testing+Features")
[![GitHub release](https://img.shields.io/github/release/Watts-Lab/team_comm_tools?include_prereleases=&sort=semver&color=blue)](https://github.com/Watts-Lab/team_comm_tools/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

# The Team Communication Toolkit
The Team Communication Toolkit is a Python package that makes it easy for social scientists to analyze and understand *text-based communication data*. Our aim is to facilitate seamless analyses of conversational data --- especially among groups and teams! --- by providing a single interface for researchers to generate and explore dozens of research-backed conversational features.

<div align="center">

[![View - Home Page](https://img.shields.io/badge/View_site-GH_Pages-2ea44f?style=for-the-badge)](https://teamcommtools.seas.upenn.edu/)

[![View - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://conversational-featurizer.readthedocs.io/en/latest/ "Go to project documentation")

</div>

# Getting Started

To use our tool, please ensure that you have Python >= 3.10 installed and a working version of [pip](https://pypi.org/project/pip/), which is Python's package installer. Then, in your local environment, run the following:

```sh
pip install team_comm_tools
```

You will also need to ensure that Spacy and NLTK are installed in addition to the required dependencies. If you get an error that en_core_web_sm is not found, you should ensure the following:

```sh
spacy download en_core_web_sm
```

Additionally, we require the following NLTK dependencies:

```sh
nltk.download('nps_chat')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**We strongly recommend using a virtual environment in Python to run the package.** We have several specific dependency requirements. One important one is that we are currently only compatible with numpy < 2.0.0 because [numpy 2.0.0 and above](https://numpy.org/devdocs/release/2.0.0-notes.html#changes) made significant changes that are not compatible with other dependencies of our package. As those dependencies are updated, we will support later versions of numpy.

After you import the package and install dependencies, you can then use our tool in your Python script as follows:

```python
from team_comm_tools import FeatureBuilder
```

This allows you to declare a FeatureBuilder object, which is the heart of our tool. Here is some sample syntax:

```python
# this section of code declares a FeatureBuilder object
my_feature_builder = FeatureBuilder(
   input_df = my_pandas_dataframe,
   conversation_id_col = "conversation_id",  # this means there's a column in your data called 'conversation_id' that uniquely identifies a conversation
   vector_directory = "./vector_data/",  # this is where we'll cache things like sentence vectors; this directory doesn't have to exist; we'll create it for you!
   output_file_path_chat_level = "./my_output_chat_level.csv", # give us names for the utterance (chat), speaker (user), and conversation-level outputs
   output_file_path_user_level = "./my_output_user_level.csv",
   output_file_path_conv_level = "./my_output_conversation_level.csv",
   turns = False,  # if true, this will combine successive turns by the same speaker.
   custom_features = [  # these features depend on sentence vectors, so they take longer to generate on larger datasets. Add them in manually if you are interested in adding them to your output!
         "(BERT) Mimicry",
         "Moving Mimicry",
         "Forward Flow",
         "Discursive Diversity"
   ],
)

# this line of code runs the FeatureBuilder on your data
my_feature_builder.featurize(col="message")
```

*Note*: PyPI treats hyphens and underscores equally, so `pip install team_comm_tools` and `pip install team-comm-tools` are equivalent. However, Python does NOT treat them equally, and **you should use underscores when you import the package, like this: `from team_comm_tools import FeatureBuilder`**.


# Learn More
Please visit our website, [https://teamcommtools.seas.upenn.edu/](https://teamcommtools.seas.upenn.edu/) for general information about our project and research. For more detailed documentation on our features and examples, please visit our [Read the Docs Page](https://conversational-featurizer.readthedocs.io/en/latest/)