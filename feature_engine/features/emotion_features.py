'''
Source for initial code: https://github.com/monologg/GoEmotions-pytorch
'''

# for importing submodules
import gitmodules

# for importing paths / specifying a parent directory
import sys

# setting path to the /modules folder
# note: this depends on which path you are accessing the file in
# currently, this is set to work if you call it from the feature_engine parent folder (because that's where all our main scaffolds lie)
sys.path.append("./modules")

# these are from the GoEmotions_pytorch submodule
from goemotionspytorch.model import BertForMultiLabelClassification
from goemotionspytorch.multilabel_pipeline import MultiLabelPipeline

# import the import nlp stuff
# note: had to import fixes from https://github.com/monologg/GoEmotions-pytorch/issues/7 (by 1sherhash)
from transformers import BertTokenizer
from pprint import pprint
import torch
import numpy as np


tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")


def get_emotions(text):
    inputs = tokenizer(text,return_tensors="pt")
    outputs = model(**inputs)
    scores =  1 / (1 + torch.exp(-outputs[0]))  # Sigmoid
    threshold = .3 # setting a specific threshold --- but this is a choice!
    for item in scores:
        # labels = []
        # scores = []
        result = []
        for idx, s in enumerate(item):
            if s > threshold:
                result.append((model.config.id2label[idx], s.detach().numpy()))
                # labels.append(model.config.id2label[idx])
                # scores.append(s)
        return result
        #return {"labels": labels, "scores": scores}

#pprint(get_emotions("It is very cold outside and I hate that so much"))

# # Output
#  [{'labels': ['neutral'], 'scores': [0.9750906]},
#  {'labels': ['curiosity', 'love'], 'scores': [0.9694574, 0.9227462]},
#  {'labels': ['love'], 'scores': [0.993483]},
#  {'labels': ['anger'], 'scores': [0.99225825]}]