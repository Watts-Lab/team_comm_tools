# Featurizer Documentation

The featurizer takes text conversations and transforms them into the following representations:

- A table of `chat-level features`, which generates a unique conversation feature for each chat (aka utterance, or message);
- A table of `conversation-level features`, which generates aggregations of features within each chat at the conversation level.

To set up and run the featurizer from scratch, you should do the following.

# Run separate iPython Scripts [Do this only ONCE]
Some features are computationally inefficient to run every time, so we do some of the pre-processing upfront. Before getting started, you should separately run the following:

- `features/preprocess_lexicons.ipynb` --> generates `features/lexicons_dict.pkl`
- `features/process_sent_vectors.ipynb` --> generates `embeddings/*`

The following do not have to be re-run (the outputs are already saved to this directory):
- `features/positivity_bert_analysis.ipynb` --> generates `sentiment_bert/*`

# Run the main featurizer [Do this every time you want to refresh/generate new features.]

In the terminal, run `python3 featurize.py`.
