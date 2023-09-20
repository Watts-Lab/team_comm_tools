import pandas as pd
import numpy as np
import re
import os

import torch
from sentence_transformers import SentenceTransformer, util

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


model_vect = SentenceTransformer('all-MiniLM-L6-v2')

MODEL  = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Check if embeddings exist
def check_embeddings(input_path, vect_path, bert_path):
    
    dataset = re.sub('../feature_engine/data/raw_data', '', input_path)
    
    if (not os.path.isfile(vect_path)):
        generate_vect(vect_path, dataset)
    if (not os.path.isfile(bert_path)):
        generate_bert(bert_path, dataset)


# Generate sentence vectors
def generate_vect(output_path, dataset):

    embedding_arr = [row.tolist() for row in model.encode(self.chat_data.messages)]
    embedding_df = pd.DataFrame({'message': self.chat_data.messages, 'message_embedding': embedding_arr})

    embedding_df.to_csv(output_path + dataset + '.csv')


# Generate BERT sentiments 
def generate_bert(output_path, dataset):
    
    # TODO: does this overwrite messages column?
    sentiments = self.chat_data.messages.apply(get_sentiment)

    sent_arr = [list(dict.values()) for dict in sentiments]

    sent_df = pd.DataFrame(sent_arr, columns =['positive_bert', 'negative_bert', 'neutral_bert']) 
    
    output_csv_folder = '../../sentiment_bert/'

    # TODO: validate the dataset name ipynb, parse from the input path?
    sent_df.to_csv(output_path + dataset + '.csv')

def get_sentiment(text):

    if (pd.isnull(text)):
        return({'positive': np.nan, 'negative': np.nan, 'neutral': np.nan})
    
    encoded = tokenizer(text, return_tensors='pt')
    output = model_bert(**encoded)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # sample output format
    return({'positive': scores[2], 'negative': scores[0], 'neutral': scores[1]})
