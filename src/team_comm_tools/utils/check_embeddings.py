import pandas as pd
import numpy as np
import re
import os
import pickle

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import logging

logging.set_verbosity(40) # only log errors

model_vect = SentenceTransformer('all-MiniLM-L6-v2')
MODEL  = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if embeddings exist
def check_embeddings(chat_data, vect_path, bert_path, need_sentence, need_sentiment, regenerate_vectors, message_col = "message"):
    """
    Check if embeddings and required lexicons exist, and generate them if they don't.

    This function ensures the necessary vector and BERT embeddings are available. 
    It also checks for the presence of certainty and lexicon files, generating them if needed.

    :param chat_data: Dataframe containing chat data
    :type chat_data: pd.DataFrame
    :param vect_path: Path to the vector embeddings file (by default, we want SBERT vectors; embeddings for each utterance.)
    :type vect_path: str
    :param bert_path: Path to the RoBERTa sentiment inference output file
    :type bert_path: str
    :param need_sentence: Whether at least one feature will require SBERT vectors; we will not need to calculate them otherwise.
    :type need_sentence: bool
    :param need_sentiment: Whether at least one feature will require the RoBERTa sentiments; we will not need to calculate them otherwise.
    :type need_sentiment: bool
    :param regenerate_vectors: If true, will regenerate vector data even if it already exists
    :type regenerate_vectors: bool, optional
    :param message_col: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message_col: str, optional

    :return: None
    :rtype: None
    """
    if (regenerate_vectors or (not os.path.isfile(vect_path))) and need_sentence:
        generate_vect(chat_data, vect_path, message_col)
    if (regenerate_vectors or (not os.path.isfile(bert_path))) and need_sentiment:
        generate_bert(chat_data, bert_path, message_col)

    try:
        vector_df = pd.read_csv(vect_path)
        # check whether the given vector and bert data matches length of chat data 
        if len(vector_df) != len(chat_data):
            print("ERROR: The length of the vector data does not match the length of the chat data. Regenerating...")
            generate_vect(chat_data, vect_path, message_col)
    except FileNotFoundError: # It's OK if we don't have the path, if the sentence vectors are not necessary
        if need_sentence:
            generate_vect(chat_data, vect_path, message_col)

    try:
        bert_df = pd.read_csv(bert_path)
        if len(bert_df) != len(chat_data):
            print("ERROR: The length of the sentiment data does not match the length of the chat data. Regenerating...")
            generate_bert(chat_data, bert_path, message_col)
    except FileNotFoundError:
        if need_sentiment: # It's OK if we don't have the path, if the sentiment features are not necessary
            generate_bert(chat_data, bert_path, message_col)
    
    # Get the lexicon pickle(s) if they don't exist
    current_script_directory = Path(__file__).resolve().parent
    LEXICON_PATH_STATIC = current_script_directory.parent/"features/assets/lexicons_dict.pkl"
    if (not os.path.isfile(LEXICON_PATH_STATIC)):
        generate_lexicon_pkl()
    CERTAINTY_PATH_STATIC = current_script_directory.parent/"features/assets/certainty.pkl"
    if (not os.path.isfile(CERTAINTY_PATH_STATIC)):
        generate_certainty_pkl()

# Read in the lexicons (helper function for generating the pickle file)
def read_in_lexicons(directory, lexicons_dict):
    for filename in os.listdir(directory):
        with open(directory/filename, encoding = "mac_roman") as lexicons:
            if filename.startswith("."):
                continue
            lines = []
            for lexicon in lexicons:
                # get rid of parentheses
                lexicon = lexicon.strip()
                lexicon = lexicon.replace('(', '')
                lexicon = lexicon.replace(')', '')
                if '*' not in lexicon:
                    lines.append(r"\b" + lexicon.replace("\n", "") + r"\b")
                else:
                    # get rid of any cases of multiple repeat -- e.g., '**'
                    lexicon = lexicon.replace('\**', '\*')

                    # build the final lexicon
                    lines.append(r"\b" + lexicon.replace("\n", "").replace("*", "") + r"\S*\b")
        clean_name = re.sub('.txt', '', filename)
        lexicons_dict[clean_name] = "|".join(lines)

def generate_lexicon_pkl():
    """
    Helper function for generating the pickle file containing lexicons.

    This function reads in lexicon files from a specified directory, processes the content, 
    and appends the cleaned lexicon patterns to a dictionary.

    :param directory: The directory containing the lexicon files
    :type directory: Path
    :param lexicons_dict: Dictionary to store the processed lexicon patterns
    :type lexicons_dict: dict

    :return: None
    :rtype: None
    """
    print("Generating Lexicon pickle...")

    try:
        lexicons_dict = {}
        current_script_directory = Path(__file__).resolve().parent
        read_in_lexicons(current_script_directory.parent / "features/lexicons/liwc_lexicons/", lexicons_dict) # Reads in LIWC Lexicons
        read_in_lexicons(current_script_directory.parent / "features/lexicons/other_lexicons/", lexicons_dict) # Reads in Other Lexicons

        # Save as pickle
        with open(current_script_directory.parent/"features/assets/lexicons_dict.pkl", "wb") as lexicons_pickle_file:
            pickle.dump(lexicons_dict, lexicons_pickle_file)
    except:
        print("WARNING: Lexicons not found. Skipping pickle generation...")

def generate_certainty_pkl():
    """
    Helper function for generating the pickle file containing the certainty lexicon.

    This function reads in lexicon files from a specified directory, processes the content, 
    and appends the cleaned lexicon patterns to a dictionary.

    :param directory: The directory containing the lexicon files
    :type directory: Path
    :param lexicons_dict: Dictionary to store the processed lexicon patterns
    :type lexicons_dict: dict

    :return: None
    :rtype: None
    """
    print("Generating Certainty pickle...")

    try:
        current_script_directory = Path(__file__).resolve().parent
        with open(current_script_directory.parent/"features/lexicons/certainty.txt", "r") as file:
            text_content = file.read()

        # Pickle the text content
        with open(current_script_directory.parent/"features/assets/certainty.pkl", "wb") as file:
            pickle.dump(text_content, file)
    except:
        print("WARNING: Certainty lexicon not found. Skipping pickle generation...")


def generate_vect(chat_data, output_path, message_col):
    """
    Generates sentence vectors for the given chat data and saves them to a CSV file.

    :param chat_data: Contains message data to be vectorized.
    :type chat_data: pd.DataFrame
    :param output_path: Path to save the CSV file containing message embeddings.
    :type output_path: str
    :param message_col: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message_col: str, optional
    :raises FileNotFoundError: If the output path is invalid.
    :return: None
    :rtype: None
    """

    print(f"Generating SBERT sentence vectors...")

    embedding_arr = [row.tolist() for row in model_vect.encode(chat_data[message_col])]
    embedding_df = pd.DataFrame({'message': chat_data[message_col], 'message_embedding': embedding_arr})

    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_path, index=False)

def generate_bert(chat_data, output_path, message_col):
    """
    Generates RoBERTa sentiment scores for the given chat data and saves them to a CSV file.

    :param chat_data: Contains message data to be analyzed for sentiments.
    :type chat_data: pd.DataFrame
    :param output_path: Path to save the CSV file containing sentiment scores.
    :type output_path: str
    :param message_col: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message_col: str, optional
    :raises FileNotFoundError: If the output path is invalid.
    :return: None
    :rtype: None
    """
    print(f"Generating RoBERTa sentiments...")

    messages = chat_data[message_col]
    sentiments = messages.apply(get_sentiment)

    sent_arr = [list(dict.values()) for dict in sentiments]

    sent_df = pd.DataFrame(sent_arr, columns =['positive_bert', 'negative_bert', 'neutral_bert']) 
    
    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sent_df.to_csv(output_path, index=False)

def get_sentiment(text):
    """
    Analyzes the sentiment of the given text using a BERT model and returns the scores for positive, negative, and neutral sentiments.

    :param text: The input text to analyze.
    :type text: str or None
    :return: A dictionary with sentiment scores.
    :rtype: dict
    """

    if (pd.isnull(text)):
        return({'positive': np.nan, 'negative': np.nan, 'neutral': np.nan})
    
    text = ' '.join(text.split()[:512]) # handle cases when the text is too long: just take the first 512 chars (hacky, but BERT context window cannot be changed)
    encoded = tokenizer(text, return_tensors='pt')
    output = model_bert(**encoded)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # sample output format
    return({'positive': scores[2], 'negative': scores[0], 'neutral': scores[1]})