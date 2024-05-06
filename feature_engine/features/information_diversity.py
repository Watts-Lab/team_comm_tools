import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
nltk.download('stopwords')
nltk.download('wordnet')
stopword = list(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer  
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from pprint import pprint
from scipy.spatial.distance import cosine

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel

def get_info_diversity(df):
    """
    Computes information diversity (value between 0 and 1 inclusive) for all conversations
    """
    info_div_score = df.groupby("conversation_num").apply(lambda x : info_diversity(x)).reset_index().rename(columns={0: "info_diversity"})
    return info_div_score

def info_diversity(df):
    """
    Preprocess data and then create numeric mapping of words in dataset to pass into LDA model
    Uses n = 20 topics 
    """
    processed_data = df['message'].apply(preprocessing).tolist()

    if not processed_data:
         return 0

    mapping = corpora.Dictionary(processed_data)
    full_corpus = [mapping.doc2bow(text) for text in processed_data]
    lda = LdaModel(corpus=full_corpus, id2word=mapping, num_topics=20)

    topics = [lda.get_document_topics(bow) for bow in full_corpus]

    ID = calculate_ID_score(topics, 20)
    return ID

def preprocessing(data):
        """
        Preprocesses the data by lowercasing, lemmatizing, and removing words of size less than 4
        """
        le=WordNetLemmatizer()
        word_tokens=word_tokenize(data.lower())
        tokens=[le.lemmatize(w) for w in word_tokens if w not in stopword and len(w) > 3]
        return tokens

def calculate_ID_score(doc_topics, num_topics):
        """
        Computes info diversity score as suggested in Reidl & Woodley (2017); determine a topic vector 
        for every message, then compute a mean topic vector across all messages, and measure the average 
        cosine similarity between the message vectors and the mean vector
        """
        topic_matrix = []
        for doc in doc_topics:
            topic_dist = np.zeros(num_topics)
            for topic, prob in doc:
                topic_dist[topic] = prob
            topic_matrix.append(topic_dist)
        topic_matrix = np.array(topic_matrix)
        
        mean_topic_vector = np.mean(topic_matrix, axis=0)
        squared_cosine_distances = [(1 - cosine(doc, mean_topic_vector))**2 for doc in topic_matrix]
        score = np.sum(squared_cosine_distances) / len(squared_cosine_distances)
        return score
