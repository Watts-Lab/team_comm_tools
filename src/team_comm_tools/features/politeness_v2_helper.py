import os
import pandas as pd
import spacy
import en_core_web_sm
import re
import numpy as np
import regex
import pickle
import errno

from .keywords import kw

nlp = en_core_web_sm.load()
nlp.enable_pipe("senter")
# kw = keywords.kw

import nltk
from nltk.corpus import stopwords
from nltk import tokenize

def sentence_split(doc):
    """
    Splits a spaCy Doc object into a list of sentences, each with simple preprocessing.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be split into sentences.

    Returns:
        list: A list of preprocessed sentences from the input Doc object.
    """

    sentences = [str(sent) for sent in doc.sents]
    sentences = [' ' + prep_simple(str(s)) + ' ' for s in sentences]

    return sentences


def sentence_pad(doc):
    """
    Pads the sentences of a spaCy Doc object by concatenating them with simple preprocessing.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be padded.

    Returns:
        str: A single string with all sentences concatenated and preprocessed.
    """

    sentences = sentence_split(doc)

    return ''.join(sentences)


def count_matches(keywords, doc):
    """
    Counts the occurrences of prespecified keywords in a text.

    Args:
        keywords (dict): A dictionary where keys are feature names and values are lists of phrases to search for.
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame with the counts of keyword matches for each feature.
    """

    text = sentence_pad(doc)

    key_res = []
    phrase2_count = []

    for key in keywords:

        key_res.append(key)
        counter = 0

        check = any(item in text for item in keywords[key])

        if check == True:

            for phrase in keywords[key]:

                phrase_count = text.count(phrase)

                if phrase_count > 0:

                    counter = counter + phrase_count

        phrase2_count.append(counter)

    res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T

    return res


def get_dep_pairs(doc):
    """
    Extracts dependency pairs from a spaCy Doc object and handles negations.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        tuple: A tuple containing a list of dependency pairs and a list of negations.
    """

    dep_pairs = [[token.dep_, token.head.text, token.head.i, token.text, token.i] for token in doc]
    negations = [dep_pairs[i] for i in range(len(dep_pairs)) if dep_pairs[i][0] == 'neg']
    token_place = [dep_pairs[i][2] for i in range(len(dep_pairs)) if dep_pairs[i][0] == 'neg']

    dep_pairs2 = []

    if len(negations) > 0:

        for j in range(len(dep_pairs)):

            if dep_pairs[j][2] not in token_place and dep_pairs[j] not in dep_pairs2:
                dep_pairs2.append(dep_pairs[j])

    else:
        dep_pairs2 = dep_pairs.copy()

    dep_pairs2 = [[dep_pairs2[i][0], dep_pairs2[i][1], dep_pairs2[i][3]] for i in range(len(dep_pairs2))]

    return dep_pairs2, negations


def get_dep_pairs_noneg(doc):
    """
    Extracts dependency pairs from a spaCy Doc object without handling negations.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        list: A list of dependency pairs from the input text.
    """
    return [[token.dep_, token.head.text, token.text] for token in doc]


def count_spacy_matches(keywords, dep_pairs):
    """
    Counts occurrences of prespecified dependency pairs in a list of dependency pairs.

    Args:
        keywords (dict): A dictionary where keys are feature names and values are lists of dependency pairs to search for.
        dep_pairs (list): A list of dependency pairs extracted from the text.

    Returns:
        pd.DataFrame: A DataFrame with the counts of dependency pair matches for each feature.
    """

    key_res = []
    phrase2_count = []

    for key in keywords:
        key_res.append(key)
        counter = 0

        check = any(item in dep_pairs for item in keywords[key])

        if check == True:

            for phrase in keywords[key]:

                if phrase in dep_pairs:

                    for dep in dep_pairs:

                        if phrase == dep:

                            counter = counter + 1

        phrase2_count.append(counter)

    res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T

    return res


def token_count(doc):
    """
    Counts the number of tokens (words) in a spaCy Doc object.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        int: The number of tokens in the input text.
    """

    # Counts number of words in a text string
    return len([token for token in doc])


def bare_command(doc):
    """
    Checks if the first word of each sentence is a verb and not in a list of keywords.

    Args:
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        int: The count of sentences that start with a verb not in the keyword list.
    """

    keywords = set([' be ', ' do ', ' please ', ' have ', ' thank ', ' hang ', ' let '])

    first_words = [' ' + prep_simple(str(sent[0])) + ' ' for sent in doc.sents]

    POS_fw = [sent[0].tag_ for sent in doc.sents]

    # returns word if word is a verb and in list of keywords
    bc = [b for a, b in zip(POS_fw, first_words) if a == 'VB' and b not in keywords]

    return len(bc)


def is_in_subordinate_clause(tok, sent):
    """
    Check if a token is inside a subordinate clause rather than the main clause.
    """
    # Walk up from the token's head (not the token itself)
    current = tok
    while current.head != current and current != sent.root:
        # Check if the HEAD has a subordinate clause dependency
        if current.head.dep_ in {"advcl", "relcl", "acl", "ccomp", "xcomp"} and current.head != sent.root:
            # We're attached to something that's a subordinate clause
            return True
        current = current.head
    return False

def wh_is_real_question(tok, sent, auxiliaries, ends_with_question_mark=False):
    """
    Returns True if the WH-word token is part of a real main-clause question.
    """
    # For WH-determiners (both with and without ?), use special logic
    if tok.dep_ == "det":
        noun = tok.head
        
        # Check: is the noun inside a complement clause?
        current = noun
        while current.head != current and current != sent.root:
            if current.dep_ in {"ccomp", "xcomp"}:
                return False
            current = current.head
        
        # If the noun is a subject (nsubj) and has a relcl ancestor, it's likely a misparsed question
        if noun.dep_ in {"nsubj", "nsubjpass"}:
            # This looks like a question with the WH-noun as subject
            # Check for auxiliary
            for t in sent:
                if t.i > noun.i and t.text.lower() in auxiliaries:
                    return True
            
            # If sentence ends with ?, accept it
            if ends_with_question_mark:
                return True
        
        # For non-subject WH-determiners, check close ancestors for relcl
        if tok.dep_ == "det" and tok.head.dep_ != "relcl":
            # Check head and head's head
            if tok.head.head.dep_ == "relcl" and tok.head.head.i < tok.i:
                # relcl is before WH-word, likely a real relative clause
                return False
        
        # Check if there's an auxiliary after the WH-word/noun
        for t in sent:
            if t.i > noun.i and t.text.lower() in auxiliaries:
                return True
        
        return False
    
    # For other WH-words (not determiners)
    # First check for complement clauses (ccomp, xcomp) - these are embedded questions
    for anc in tok.ancestors:
        if anc.dep_ in {"ccomp", "xcomp"}:
            return False
    
    # Check if WH-word is attached to a verb that takes interrogative complements
    # Verbs like: tell, ask, know, wonder, understand, explain, show, see, remember, etc.
    complement_taking_verbs = {
        'tell', 'ask', 'know', 'wonder', 'understand', 'explain', 
        'show', 'see', 'remember', 'forget', 'realize', 'figure',
        'decide', 'consider', 'discover', 'find', 'learn', 'teach'
    }
    
    if tok.head.pos_ == "VERB" and tok.head.lemma_ in complement_taking_verbs:
        # Check if there are tokens before this verb (indicating it's not sentence-initial)
        tokens_before_verb = 0
        for t in sent:
            if t.i >= tok.head.i:
                break
            if t.pos_ not in {"PUNCT", "INTJ"}:
                tokens_before_verb += 1
        
        # If there are 2+ tokens before the verb, WH is likely embedded
        if tokens_before_verb >= 2:
            return False
    
    # Check if has relcl ancestor
    has_relcl_ancestor = False
    for anc in tok.ancestors:
        if anc.dep_ == "relcl":
            has_relcl_ancestor = True
            break
    
    if has_relcl_ancestor:
        # Check if this is a misparsed main question vs real relative clause
        # Count substantive tokens before the WH-word
        substantive_before = 0
        for t in sent:
            if t.i >= tok.i:
                break
            if t.pos_ not in {"INTJ", "PUNCT", "CCONJ", "DET"}:
                substantive_before += 1
        
        # If fewer than 3 substantive tokens before WH, likely a misparsed main question
        if substantive_before < 3:
            pass  # Don't exclude it
        else:
            # Likely a real relative clause
            return False
    
    # If the sentence ends with ?, be lenient for non-relcl WH-words
    if ends_with_question_mark:
        return True
    
    # For non-? sentences with non-determiner WH-words
    if is_in_subordinate_clause(tok, sent):
        return False
    
    if tok.dep_ not in {"nsubj", "nsubjpass", "csubj", "attr", "ROOT", "dobj", "pobj", "advmod"}:
        return False

    for t in sent:
        if not is_in_subordinate_clause(t, sent) and t.text.lower() in auxiliaries:
            return True

    return False

def Question(doc):
    """
    Counts the number of sentences containing question words and question marks.
    """
    search_tags = {'WRB', 'WP', 'WDT', 'WP$'}
    wh_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'}

    auxiliaries = {
        'do', 'does', 'did', 'have', 'has', 'had',
        'can', 'could', 'will', 'would', 
        'may', 'might', 'shall', 'should',
        'is', 'are', 'was', 'were', 'am'
    }
    pronoun_followers = {'i', 'you', 'we', 'he', 'she', 'they', 'it', 'these', 'those', 'this', 'that'}

    wh_count = 0
    yesno_count = 0
    counted_sentences = set()
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_tokens = list(sent)
        if not sent_tokens:
            continue
            
        # Method 1: Sentences ending with '?'
        if sent_text.endswith('?'):
            wh = False
            for tok1 in sent_tokens:
                t1_lower = tok1.text.lower()
                if t1_lower in wh_words and tok1.tag_ in search_tags:
                    if wh_is_real_question(tok1, sent, auxiliaries, ends_with_question_mark=True):
                        wh = True
                        break
            if wh:
                wh_count += 1
            else:
                yesno_count += 1
            counted_sentences.add(sent.start)
            continue
        
        # Method 2: Lexical rule-based detection for sentences without '?'
        found_question = False
        for tok1 in sent_tokens:
            t1_lower = tok1.text.lower()
            if t1_lower in wh_words and tok1.tag_ in search_tags:
                if wh_is_real_question(tok1, sent, auxiliaries, ends_with_question_mark=False):
                    wh_count += 1
                    counted_sentences.add(sent.start)
                    found_question = True
                    break
        
        if found_question:
            continue
            
        # Check for Yes/No questions
        for tok1, tok2 in zip(sent_tokens, sent_tokens[1:] + [None]):
            t1_lower = tok1.text.lower()
            t2_lower = tok2.text.lower() if tok2 else None
            
            if tok1.i - sent.start > 1:
                continue
                
            if t1_lower in auxiliaries and t2_lower in pronoun_followers:
                yesno_count += 1
                counted_sentences.add(sent.start)
                break

    return yesno_count, wh_count


def word_start(keywords, doc):
    """
    Finds the first words in text that match a list of keywords.

    Args:
        keywords (dict): A dictionary where keys are feature names and values are lists of first words to search for.
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame with the counts of first word matches for each feature.
    """

    key_res = []
    phrase2_count = []

    # doc = nlp(text)

    for key in keywords:

        first_words = [' ' + prep_simple(str(sent[0])) + ' ' for sent in doc.sents]
        cs = [w for w in first_words if w in keywords[key]]

        phrase2_count.append(len(cs))
        key_res.append(key)

    res = pd.DataFrame([key_res, phrase2_count], index=['Features', 'Counts']).T
    return res


def adverb_limiter(keywords, doc):
    """
    Searches for adverb modifiers in the text that match a list of keywords.

    Args:
        keywords (dict): A dictionary where the key 'Adverb_Limiter' contains a list of adverb modifiers to search for.
        doc (spacy.tokens.Doc): The spaCy Doc object containing the text to be analyzed.

    Returns:
        int: The count of adverb modifier matches in the text.
    """

    tags = [token.dep_ for token in doc if token.dep_ == 'advmod' and
            str(' ' + str(token) + ' ') in keywords['Adverb_Limiter']]

    return len(tags)


def feat_counts(text, kw):
    """
    Extracts various linguistic features from a text using predefined keywords and dependency pairs.

    Args:
        text (str): The text to be analyzed.
        kw (dict): A dictionary containing predefined keywords and dependency pairs.

    Returns:
        pd.DataFrame: A DataFrame with counts of various linguistic features.
    """

    # remove extraneous backslashes
    text = re.sub('\\\\', '', text)

    text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)
    text = text.lstrip()
    clean_text = prep_simple(text)
    doc_text = nlp(text)

    doc_clean_text = nlp(clean_text)

    kw_matches = count_matches(kw['word_matches'], doc_text)

    dep_pairs, negations = get_dep_pairs(doc_clean_text)
    dep_pair_matches = count_spacy_matches(kw['spacy_pos'], dep_pairs)

    dep_pairs_noneg = get_dep_pairs_noneg(doc_clean_text)
    disagreement = count_spacy_matches(kw['spacy_noneg'], dep_pairs_noneg)

    neg_dp = set([' ' + i[1] + ' ' for i in negations])
    neg_only = count_spacy_matches(kw['spacy_neg_only'], neg_dp)

    # count start word matches like conjunctions and affirmations
    start_matches = word_start(kw['word_start'], doc_text)

    scores = pd.concat([kw_matches, dep_pair_matches, disagreement, start_matches, neg_only])
    scores = scores.groupby('Features').sum()
    scores = scores.reset_index()

    bc = bare_command(doc_text)
    scores.loc[len(scores)] = ['Bare_Command', bc]

    ynq, whq = Question(doc_text)

    scores.loc[len(scores)] = ['YesNo_Questions', ynq]
    scores.loc[len(scores)] = ['WH_Questions', whq]

    adl = adverb_limiter(kw['spacy_tokentag'], doc_text)
    scores.loc[len(scores)] = ['Adverb_Limiter', adl]

    tokens = token_count(doc_text)
    scores.loc[len(scores)] = ['Token_count', tokens]

    return scores

def load_to_lists(path, words):
    """
    Loads keywords from text files in a specified directory into lists.

    Args:
        path (str): The directory path containing the text files.
        words (str): Specifies whether to load 'single' or 'multiple' words per line.

    Returns:
        tuple: A tuple containing a list of feature names and a list of keywords.
    """

    keywords = []

    all_files = os.listdir(path)

    all_files = [file for file in all_files if file.endswith(".txt")]
    all_filenames = [file.split('.', 1)[0] for file in all_files if file.endswith(".txt")]

    feature_names = []

    all_lines = []
    for i in range(len(all_files)):

        if all_files[i].endswith(".txt"):
            try:
                with open(os.path.join(path, all_files[i]), "r") as f:
                    for line in f:
                        splitLine = line.split()

                        if words == 'single':
                            splitLine = ' '.join(splitLine)
                            splitLine = [splitLine.center(len(splitLine) + 2)]
                            all_lines.extend(splitLine)

                        if words == 'multiple':
                            all_lines.append(splitLine)

                        feature_names.append(all_filenames[i])
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

    return feature_names, all_lines

def load_to_dict(path, words):
    """
    Loads keywords from text files in a specified directory into a dictionary.

    Args:
        path (str): The directory path containing the text files.
        words (str): Specifies whether to load 'single' or 'multiple' words per line.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of keywords.
    """

    keywords = {}

    all_files = os.listdir(path)

    all_files = [file for file in all_files if file.endswith(".txt")]
    all_filenames = [file.split('.', 1)[0] for file in all_files if file.endswith(".txt")]

    for i in range(len(all_files)):
        all_lines = []
        if all_files[i].endswith(".txt"):
            try:
                with open(os.path.join(path, all_files[i]), "r") as f:
                    for line in f:
                        splitLine = line.split()

                        if words == 'single':
                            splitLine = [' '.join(splitLine)]
                            all_lines.extend(splitLine)

                        if words == 'multiple':
                            #splitLine = multi_strings(line)
                            all_lines.append(splitLine)

                if words == 'single':
                    all_lines = [l.center(len(l) + 2) for l in all_lines]

                keywords[all_filenames[i]] = all_lines
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

    return keywords


def commit_data(path, path_in, folders, words_in_line):
    """
    Loads data from .txt files, creates one dictionary per folder
    and outputs each folder as a dictionary in a pickle file

    Args:
        path (str): The base directory path containing the folders with text files.
        path_in (str): The directory path to save the pickle files.
        folders (list): A list of folder names containing the text files.
        words_in_line (list): A list specifying whether each folder contains 'single' or 'multiple' words per line.

    Returns:
        None
    """

    for i in range(len(folders)):
        x = load_to_dict(path + folders[i], words_in_line[i])

        file = open(path_in + folders[i] + ".pkl", "wb")
        pickle.dump(x, file)
        file.close()


def load_saved_data(path_in, folders):
    """
    Loads predefined keywords and dependency pairs

    Args:
        path_in (str): The directory path containing the pickle files.
        folders (list): A list of folder names to load the pickle files from.

    Returns:
        dict: A dictionary where keys are folder names and values are dictionaries of keywords and dependency pairs.
    """

    dicts = {}

    for i in range(len(folders)):

        file = open(path_in + folders[i] + ".pkl", "rb")
        x = pickle.load(file)
        dicts[folders[i]] = x
        file.close()

    return dicts


def clean_text(text):
    """
    Cleans and normalizes text by replacing certain patterns and characters.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """

    orig = ["let's", "i'm", "won't", "can't", "shan't", "'d",
            "'ve", "'s", "'ll", "'re", "n't", "u.s.a.", "u.s.", "e.g.", "i.e.",
            "‘", "’", "“", "”", "100%", "  ", "mr.", "mrs.", "dont", "wont"]

    new = ["let us", "i am", "will not", "cannot", "shall not", " would",
           " have", " is", " will", " are", " not", "usa", "usa", "eg", "ie",
           "'", "'", '"', '"', "definitely", " ", "mr", "mrs", "do not", "would not"]

    for i in range(len(orig)):
        text = text.replace(orig[i], new[i])

    return text


def prep_simple(text):
    """
    Preprocesses text by cleaning and removing certain characters.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """

    # text cleaning

    t = text.lower()
    t = clean_text(t)
    t = re.sub(r"[.?!]+\ *", "", t) 
    t = re.sub('[^A-Za-z,]', ' ', t)  

    return t

def prep_whole(text):
    """
    Preprocesses text by cleaning, removing certain characters, and filtering out stopwords.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text with stopwords removed.
    """

    t = text.lower()
    t = clean_text(t)
    t = re.sub('[^A-Za-z]', ' ', t)

    words = nltk.word_tokenize(t)

    stopword = set(stopwords.words('english'))
    words = [w for w in words if not w in stopword]
    text = ' '.join(words)

    return text


def sentenciser(text):
    """
    Splits text into sentences using spaCy.

    Args:
        text (str): The input text to be split into sentences.

    Returns:
        list: A list of sentences from the input text.
    """

    nlp.enable_pipe("senter")

    doc = nlp(text)

    split_t = [sent.text for sent in doc.sents]

    return split_t


def punctuation_seperator(text):
    """
    Separates text into segments based on punctuation.

    Args:
        text (str): The input text to be separated by punctuation.

    Returns:
        list: A list of text segments with punctuation removed.
    """

    PUNCT_RE = regex.compile(r'(\p{Punctuation})')
    split_punct = PUNCT_RE.split(text)

    # Removing punctuation from the list
    no_punct = []
    for s in split_punct:
        s = re.sub(r'[^\w\s]', '', s)
        if s != '':
            no_punct.append(s)

    return no_punct


def conjection_seperator(text):
    """
    Separates text into segments based on conjunctions.

    Args:
        text (str): The input text to be separated by conjunctions.

    Returns:
        list: A list of text segments separated by conjunctions.
    """

    tags = nltk.pos_tag(nltk.word_tokenize(text))
    first_elements = [e[0] for e in tags]
    second_elements = [e[1] for e in tags]

    if 'CC' in second_elements:
        index = [i for i, e in enumerate(second_elements) if e == 'CC']
        index.insert(0, 0)
        parts = [first_elements[i:j] for i, j in zip(index, index[1:] + [None])]

        return [' '.join(p) for p in parts]
    else:
        return [' '.join(first_elements)]


def phrase_split(text):
    """
    Splits text into phrases based on punctuation and conjunctions.

    Args:
        text (str): The input text to be split into phrases.

    Returns:
        list: A list of phrases from the input text.
    """

    text = punctuation_seperator(text)
    phrases = []
    for t in text:
        t = conjection_seperator(t)

        phrases.extend(t)

    return phrases
