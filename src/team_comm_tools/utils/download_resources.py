import nltk
import spacy
import subprocess

def download():
    # nltk
    for resource in [
        'corpora/nps_chat',
        'tokenizers/punkt',
        'corpora/stopwords',
        'corpora/wordnet']:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1])
    # spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            subprocess.check_call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        except subprocess.CalledProcessError as error:
            print(f"Error downloading spaCy model: {error}")
            raise

if __name__ == "__main__":
    download()