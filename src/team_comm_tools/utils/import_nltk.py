import nltk

def download_nltk_resources():
    nltk.download('nps_chat')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

if __name__ == "__main__":
    download_nltk_resources()
