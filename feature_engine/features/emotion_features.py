'''
Source for initial code: https://github.com/monologg/GoEmotions-pytorch
'''

# for importing submodules
import gitmodules

# for importing paths / specifying a parent directory
import sys

# setting path to the /modules folder
# note: this depends on which path you are accessing the file in
sys.path.append("../modules")

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

texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it’s happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

results = []
for txt in texts:
    inputs = tokenizer(txt,return_tensors="pt")
    outputs = model(**inputs)
    scores =  1 / (1 + torch.exp(-outputs[0]))  # Sigmoid
    threshold = .3
    for item in scores:
        labels = []
        scores = []
        for idx, s in enumerate(item):
            if s > threshold:
                labels.append(model.config.id2label[idx])
                scores.append(s)
        results.append({"labels": labels, "scores": scores})

pprint(results)

# # Output
#  [{'labels': ['neutral'], 'scores': [0.9750906]},
#  {'labels': ['curiosity', 'love'], 'scores': [0.9694574, 0.9227462]},
#  {'labels': ['love'], 'scores': [0.993483]},
#  {'labels': ['anger'], 'scores': [0.99225825]}]