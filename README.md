# A Conversational Analysis Framework for Predicting Team Success through Team Communication Processes
Our project seeks to answer the question, “which types of team communication processes matter in different team activities?” Depending on the type of task at hand (Straus, 1999) or the context of the research study (Levinthal & Rosenkopf, 2020), some strategies may be more useful than others. Thus, how might we synthesize the myriad studies of team behavior — which take place across many tasks and contexts — into actionable insights for managers? In this project, we will extract team communication processes from the organizational behavior literature (for example, “Positivity,” “Information Exchange,” “Equal Participation”), and then measure these features on real teams’ communication transcripts across a variety of tasks. We will then use these measured features to predict which types of communication are most associated with team success across different activities.

## Getting Started

If you are new to this repository, welcome! Please follow the steps below to get started.

### Step 1: Clone the Repo
First, clone this repository into your local development environment: 
```
git clone https://github.com/Watts-Lab/team-process-map.git
```

### Step 2: Download Dependencies
Second, we *strongly* recommend using a virtual environment to install the dependencies required for the project.=
The dependencies of the project are listed in `feature_engine/requirements.txt`: https://github.com/Watts-Lab/team-process-map/blob/main/feature_engine/requirements.txt

**Python Version**: We recommend `python3.11` when running this repository.

Later versions of Python (e.g., 3.12) are currently incompatible with the `sentence-transformers` library. There is an open issue here: https://github.com/google/sentencepiece/issues/968 [Updated as of January 30, 2024]

#### Run Initial Scripts for Dependencies
Before starting the featurizer, you need to run the following to obtain dependencies for the project:

```
python3 -m spacy download en_core_web_sm
```
```
import nltk
nltk.download('nps_chat')
nltk.download('punkt')
```

### Step 3: Add "Hidden" Files
Certain files associated with the project are "hidden" from the public GitHub repo at this time, for
reasons of copyright, research embargo, or other requests. 

These are instead saved on our private Google Drive. To access them, click this link: https://drive.google.com/drive/folders/1c-g4d-Pq6kT2el4oaCSiQGsQrtgd3VH3?usp=drive_link
This will take you to a folder called "GitHub Assets." Then do the following:

1. Unzip the folder "lexicons.zip"
2. Replace the folder under `feature_engine/features/lexicons` with the one in the ZIP file.

### Step 4: Initialize the data repository
Test data for the project lives in a separate repository. To populate the data repository when running for the first time, run the following command:

```
git submodule update --init --recursive
```
For more information on updating submodules, refer to [this documentation](https://stackoverflow.com/questions/1030169/pull-latest-changes-for-all-git-submodules).

### Step 5: Run the Featurizers
At this point, you should be ready to run the featurizer! Navigate to the `feature_engine` folder, and use the following command:

```
python3 featurize.py
```
This calls the `featurizer.py` file, which declares a FeatureBuilder object for different dataset of interest, and featurizes them using our framework.

### Automated Unit Testing
To formally conclude the development of a new feature, it's crucial to test it's performance! Outlined below are the steps to unit test features at the chat or conversation level.

The first step is to draft test inputs (`conversation_num`, `speaker`, `message`) and expected outputs for your feature. For example,  "This is a test message." should return 5 for `num_words` at the chat level (note that `conversation_num` and `speaker` have no effect on the ultimate result, so they can be chosen arbitrarily).

Testing a conversation level feature, say `discursive_diversity`, requires a series of chats. For example, "This is a test message." (speaker 1), "This is a test message." (speaker 1), "This is a test message." (speaker 2), "This is a test message." (speaker 2), within the same conversation, should return 0. Note that the `conversation_num` for each new test should be distinct from all previous `conversation_num`, even if the feature being tested is different.

Once putting together the test inputs, add each CHAT (and it's associated conversation_num and speaker) as a separate row in either `test_chat_level.csv` or `test_conv_level.csv`, within `./feature_engine/testing/data/cleaned_data`. The format of the CSV is as follows: `id, conversation_num, speaker_nickname, message, expected_column, expected_value`, where `expected_column` is the feature name (i.e. num_words).

Push all your changes to Github, including feature development and test dataset additions. Go under the "Actions" tab in the toolbar. Notice there's a new job running called "Testing-Features". A green checkmark at the conclusion of this job indicates all new tests have passed. A red cross means some test has failed. Navigate to the uploaded "Artifact" (near the bottom of the status page) for list of failed tests and their associated inputs/outputs.

Debug and iterate!

## Documents and Handy Links
- Our Team Email: csslab-team-process-map@wharton.upenn.edu (Ask Emily for the password!)

- Our "master sheet," where we track progress, literature, and new features to build:
https://docs.google.com/spreadsheets/d/1JnChOKFXkv944LvnYbzI1qrHLEPfCvEMN5XzP1AxvmA/edit?usp=sharing

## Database of Current Communication Features
For a list of our current documented communication features, please refer to [this database](https://glitter-runner-dfb.notion.site/e0fd0ceb6c6c47d9b8e3bec95d8af78f?v=3050cfbe883e4d9ea1954bc67bf12a46&pvs=4).
