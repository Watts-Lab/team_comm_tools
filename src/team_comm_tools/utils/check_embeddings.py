import pandas as pd
import numpy as np
import re
import os
import pickle

from tqdm import tqdm
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

import ast

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import logging
from team_comm_tools.utils.preprocess import *

logging.set_verbosity(40) # only log errors

model_vect = SentenceTransformer('all-MiniLM-L6-v2')
MODEL  = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if embeddings exist
def check_embeddings(chat_data, vect_path, bert_path, original_vect_path, need_sentence, need_sentiment, regenerate_vectors, message_col = "message"):
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
    :param original_vect_path: Path to the DEFAULT vector embeddings file (SBERT vectors; embeddings for each utterance.)
    :type original_vect_path: str
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
            # reset vector path to default/original
            generate_vect(chat_data, original_vect_path, message_col)
        else:     
            # check that message in vector data matches chat data
            preprocessed_chat = chat_data[message_col].astype(str).apply(preprocess_text)
            
            # removed _original from message_col
            preprocessed_vector = vector_df[message_col[:-9]].astype(str).apply(preprocess_text)

            mismatches = chat_data[preprocessed_chat != preprocessed_vector]
            if len(mismatches) != 0:
                print("Messages in the vector data do not match the chat data. Regenerating...")
                generate_vect(chat_data, original_vect_path, message_col)
            
            if "message_embedding" in vector_df.columns:
                # check that message_embedding is numeric list
                if not vector_df["message_embedding"].apply(is_numeric_list).all():
                    print("message_embedding is not a numeric list. Regenerating ...")
                    generate_vect(chat_data, original_vect_path, message_col)
                else:    
                    # check if length of all vectors is the same
                    vect_lengths = vector_df["message_embedding"].apply(lambda x: ast.literal_eval(x)).apply(lambda x : len(x))                
                    
                    if (vect_lengths == 0).any():
                        print("One or more value in message_embedding are null. Regenerating ...")
                        generate_vect(chat_data, original_vect_path, message_col)
                            
                    if len(vect_lengths.unique()) > 1:
                        print("Not all vectors have the same length. Regenerating ...")
                        generate_vect(chat_data, original_vect_path, message_col)
                
                # check if vectors have a 1-1 mapping with the text
                embedding_message_map = {}
                for _, row in vector_df.iterrows():
                    embedding = row['message_embedding']
                    message = row['message']
                    
                    if embedding in embedding_message_map:
                        if message != embedding_message_map[embedding]:
                            print("Same embedding maps to multiple unique messages. Regenerating ...")
                            generate_vect(chat_data, original_vect_path, message_col)
                            break
                    else:
                        embedding_message_map[embedding] = message
                    
            else:
                print("no message_embedding column. Regenerating ...")
                generate_vect(chat_data, original_vect_path, message_col)
                
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

# checks if column is only numeric elements
def is_numeric_list(value):
    if isinstance(value, str):
        # try to convert string representations to actual lists
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return False 
                
    # ensure the value is a list
    if not isinstance(value, list):
        return False
                
    # ensure all elements in the list are numeric
    return all(isinstance(x, (int, float)) for x in value)
            
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

def str_to_vec(str_vec):
    """
    Takes a string representation of a vector and returns it as a 1D np array.
    """
    vector_list = [float(e) for e in str_vec[1:-1].split(',')]
    return np.array(vector_list)

def get_nan_vector():
    """
    Get a default value for an empty string (the "NaN vector") and returns it as a 1D np array.
    """
    current_dir = os.path.dirname(__file__)
    nan_vector_file_path = os.path.join(current_dir, '../features/assets/nan_vector.txt')
    nan_vector_file_path = os.path.abspath(nan_vector_file_path)

    with open(nan_vector_file_path, "r") as f:
        return str_to_vec(f.read())

def generate_vect(chat_data, output_path, message_col, batch_size = 64):
    """
    Generates sentence vectors for the given chat data and saves them to a CSV file.

    :param chat_data: Contains message data to be vectorized.
    :type chat_data: pd.DataFrame
    :param output_path: Path to save the CSV file containing message embeddings.
    :type output_path: str
    :param message_col: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message_col: str, optional
    :param batch_size: The size of each batch for processing sentiment analysis. Defaults to 64.
    :type batch_size: int
    :raises FileNotFoundError: If the output path is invalid.
    :return: None
    :rtype: None
    """
    print(f"Generating SBERT sentence vectors...")

    nan_vector = get_nan_vector()
    empty_to_nan = [text if text and text.strip() else None for text in chat_data[message_col].tolist()]
    non_empty_texts = [text for text in empty_to_nan if text is not None]
    all_embeddings = [emb for i in tqdm(range(0, len(non_empty_texts), batch_size)) for emb in model_vect.encode(non_empty_texts[i:i + batch_size])]
    embeddings = np.tile(nan_vector, (len(empty_to_nan), 1)) # default embeddings to the NAN vector
    non_empty_index = 0
    for idx, text in enumerate(empty_to_nan):
        if text is not None: # if it's a real text, fill it in with its actual embeding
            embeddings[idx] = all_embeddings[non_empty_index]
            non_empty_index += 1
    embedding_arr = [emb.tolist() for emb in embeddings]
    embedding_df = pd.DataFrame({'message': chat_data[message_col], 'message_embedding': embedding_arr})

    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_path, index=False)

def generate_bert(chat_data, output_path, message_col, batch_size=64):
    """
    Generates RoBERTa sentiment scores for the given chat data and saves them to a CSV file.

    :param chat_data: Contains message data to be analyzed for sentiments.
    :type chat_data: pd.DataFrame
    :param output_path: Path to save the CSV file containing sentiment scores.
    :type output_path: str
    :param message_col: A string representing the column name that should be selected as the message. Defaults to "message".
    :type message_col: str, optional
    :param batch_size: The size of each batch for processing sentiment analysis. Defaults to 64.
    :type batch_size: int
    :raises FileNotFoundError: If the output path is invalid.
    :return: None
    :rtype: None
    """
    print(f"Generating RoBERTa sentiments...")

    messages = chat_data[message_col].tolist()
    batch_sentiments_df = pd.DataFrame()

    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i + batch_size]
        batch_df = get_sentiment(batch)
        batch_sentiments_df = pd.concat([batch_sentiments_df, batch_df], ignore_index=True)

    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    batch_sentiments_df.to_csv(output_path, index=False)

def get_sentiment(texts):
    """
    Analyzes the sentiment of the given list of texts using a BERT model and returns a DataFrame with scores for positive, negative, and neutral sentiments.

    :param texts: The list of input texts to analyze.
    :type texts: list of str
    :return: A DataFrame with sentiment scores.
    :rtype: pd.DataFrame
    """

    # Handle and tokenize non-null and non-empty texts
    texts_series = pd.Series(texts)
    non_null_non_empty_texts = texts_series[texts_series.apply(lambda x: pd.notnull(x) and x.strip() != '')].tolist()

    if not non_null_non_empty_texts:
        # Return a DataFrame with NaN if there are no valid texts to process
        return pd.DataFrame(np.nan, index=texts_series.index, columns=['positive_bert', 'negative_bert', 'neutral_bert'])

    encoded = tokenizer(non_null_non_empty_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    output = model_bert(**encoded)

    scores = output[0].detach().numpy()
    scores = softmax(scores, axis=1)

    sent_dict = {
        'positive_bert': scores[:, 2],
        'negative_bert': scores[:, 0],
        'neutral_bert': scores[:, 1]
    }
    
    non_null_sent_df = pd.DataFrame(sent_dict)

    # Initialize the DataFrame such that null texts and empty texts get np.nan
    sent_df = pd.DataFrame(np.nan, index=texts_series.index, columns=['positive_bert', 'negative_bert', 'neutral_bert'])
    sent_df.loc[texts_series.apply(lambda x: pd.notnull(x) and x.strip() != ''), ['positive_bert', 'negative_bert', 'neutral_bert']] = non_null_sent_df.values

    return sent_df