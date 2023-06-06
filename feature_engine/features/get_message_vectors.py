import gensim
import pandas as pd
import numpy as np

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('../feature_engine/data/GoogleNews-vectors-negative300.bin', binary=True)


# get vectors for juries dataset 
def add_vector(arr):
    if not arr:
        return np.nan
    
    # 300 dimensional word embedding vector to represent each line/message
    currentVectorSum = [0.0] * 300
    count = 0

    # get message vector by adding averaging all the word vectors (from pretrained)
    for word in arr:
        if word in model:
            count += 1
            currentVectorSum = [x + y for x, y in zip(currentVectorSum, model[word])]

    if (count > 0):
        currentVectorSum = [x / count for x in currentVectorSum]

    return np.array(currentVectorSum, dtype="float64")
    # return str(currentVectorSum)[1:-1] + '\n'

def get_message_vector(line):
    
    arr = []

    if not pd.isnull(line):
        arr = gensim.utils.simple_preprocess(line)

    return add_vector(arr)
