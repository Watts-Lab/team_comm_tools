'''
Source for initial code: https://github.com/monologg/GoEmotions-pytorch
'''

from transformers import BertTokenizer
from pprint import pprint

# for importing paths / specifying a parent directory
import sys
import path

cur_directory = path.path(__file__).abspath() 
sys.path.append(cur_directory.parent.parent) # setting path

# these are from the GoEmotions_pytorch submodule
from GoEmotions_pytorch.model import BertForMultiLabelClassification
from GoEmotions_pytorch.multilabel_pipeline import MultiLabelPipeline

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it’s happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

pprint(goemotions(texts))


# # Output
#  [{'labels': ['neutral'], 'scores': [0.9750906]},
#  {'labels': ['curiosity', 'love'], 'scores': [0.9694574, 0.9227462]},
#  {'labels': ['love'], 'scores': [0.993483]},
#  {'labels': ['anger'], 'scores': [0.99225825]}]