"""
file: calculate_chat_level_features.py
---
This file defines the ChatLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all chat level features here. 

The steps needed to add a feature would be to:
- First define any building blocks that the feature would need in the appropriate "features" module (like word counter).
- Define a function within the class that uses these building blocks to build the feature and appends it 
  to the chat level dataframe as columns.
- Call the feature defining function in the driver function.
"""

# Importing modules from features
from features.basic_features import *
from features.info_exchange_zscore import *
from features.lexical_features import *
from features.other_LIWC_features import *
from features.word_mimicry import *

##NEWLY ADDED BY PRIYA##
from features.readability import*
from features.cosine_similarity import*
from features.entropy import*
from features.hedge import*
from features.hedge_2 import*
from features.positivity_zscore import*
from features.sentiment_analysis import*
from features.temporal_features import*
from features.tf_idf import*

class ChatLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

        PARAMETERS:
            @param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
        """
        self.chat_data = chat_data
        
    def calculate_chat_level_features(self) -> pd.DataFrame:
        """
            This is the main driver function for this class.

        RETURNS:
            (pd.DataFrame): The chat level dataset given to this class during initialization along with 
                            new columns for each chat level feature.
        """
        # Text-Based Basic Features
        self.text_based_features()

        # Info Exchange Feature
        self.info_exchange_feature()
        
        # lexical features
        # self.lexical_features() # TODO - commenting this out to speed things up; also, these are not currently being summarized

        # Other lexical features
        self.other_lexical_features()

        # Word Mimicry
        self.calculate_word_mimicry()

        ##NEWLY ADDED BY PRIYA##
        self.calculate_readability()
        self.calculate_hedge1()
        self.calculate_tf_idf()

        '''
        self.calculate_cosine_similarity()
        self.calculate_entropy()
        self.calculate_hedge2()
        self.calculate_positivity_zscore()
        self.calculate_sentiment_analysis()
        self.calculate_temporal_features()
        '''

        # Return the input dataset with the chat level features appended (as columns)
        return self.chat_data
        
    def text_based_features(self) -> None:
        """
            This function is used to implement the common text based featuers.
        """
        # Count Words
        self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
        
        # Count Characters
        self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
        
        # Count Messages        
        self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
        
    def info_exchange_feature(self) -> None:
        """
            This function helps in extracting the different types of z-scores from the chats 
            (see features/info_exchange_zscore.py to learn more about how these features are calculated).
        """
        # Get Modified Wordcount: Total word count - first_singular pronouns
        self.chat_data["info_exchange_wordcount"] = self.chat_data["message"].apply(get_info_exchange_wordcount)
        
        # Get the z-score of each message across all chats
        self.chat_data = get_zscore_across_all_chats(self.chat_data, "info_exchange_wordcount")
        
        # Get the z-score within each conversation
        self.chat_data = get_zscore_across_all_conversations(self.chat_data, "info_exchange_wordcount")

    def lexical_features(self) -> None:
        """
            This is a driver function that calls relevant functions in features/lexical_features.py to implement the lexical features.
        """
        self.chat_data = pd.concat([self.chat_data, self.chat_data["message"].apply(lambda x: pd.Series(liwc_features(str(x))))], axis = 1)
        
    def other_lexical_features(self) -> None:
        """
            This function extract the number of questions, classify whether the message contains clarification questions,
            calculate the word type-to-token ratio, and the proportion of first person pronouns from the chats
            (see features/other_LIWC_features.py to learn more about how these features are calculated)
        """
        # Get the number of questions in each message
        self.chat_data["num_question_naive"] = self.chat_data["message_lower_with_punc"].apply(num_question_naive)
        
        # Classify whether the message contains clarification questions
        self.chat_data["NTRI"] = self.chat_data["message_lower_with_punc"].apply(classify_NTRI)
        
        # Calculate the word type-to-token ratio
        self.chat_data["word_TTR"] = self.chat_data["message"].apply(get_word_TTR)
        
        # Calculate the proportion of first person pronouns from the chats
        self.chat_data["first_pronouns_proportion"] = self.chat_data["message"].apply(get_proportion_first_pronouns)
        
    def calculate_word_mimicry(self) -> None:
        """
            This function calculate the number of function words that also used in other’s prior turn,
            and the sum of inverse frequency of each content word that also occurred in the other’s immediately prior turn.
            (see features/word_mimicry.py to learn more about how these features are calculated)
            
            Note: this function takes the dataset WITHOUT any punctuations as input
        """

        # Extract function words / content words from a message
        self.chat_data["function_words"] = self.chat_data["message"].apply(function_word)
        self.chat_data["content_words"] = self.chat_data["message"].apply(content_word)
        
        # Extract the function words / content words that also appears in the immediate previous turn
        self.chat_data["function_word_mimicry"] = mimic_words(self.chat_data, "function_words")
        self.chat_data["content_word_mimicry"] = mimic_words(self.chat_data, "content_words")
        
        # Compute the number of function words that also appears in the immediate previous turn
        self.chat_data["function_word_accommodation"] = self.chat_data["function_word_mimicry"].apply(Function_mimicry_score)
        
        # Compute the sum of inverse frequency of each content word that also occurred in the other’s immediately prior turn.
        self.chat_data["content_word_accommodation"] = Content_mimicry_score(self.chat_data, "content_words","content_word_mimicry")

        # Drop the function / content word columns -- we dont' need them in the output
        self.chat_data = self.chat_data.drop(columns=['function_words', 'content_words', 'function_word_mimicry', 'content_word_mimicry'])

    ##NEWLY ADDED BY PRIYA##
    def calculate_readability(self) -> None:
        self.chat_data['dale_chall_score'] = self.chat_data["message_lower_with_punc"].apply(dale_chall_helper)
        self.chat_data['dale_chall_classification'] = self.chat_data['dale_chall_score'].apply(classify_text)

    def calculate_cosine_similarity(self) -> None:
        self.chat_data['cosine_similarity'] = ngram_cosine_similarity(self.chat_data,"message",3)
    
    def calculate_entropy(self) -> None:
        #how to add the path for liwc_lexicons, which are not public?
        pos_words =  create_sets(filepath) 
        neg_words =  create_sets(filepath)

        self.chat_data['entropy_tag'] = ngram_dialog_act_entropy(self.chat_data,"message",3,pos_words,neg_words,"positive","negative")

    def calculate_hedge1(self) -> None:
        self.chat_data['is_hedged1'] = is_hedged_sentence_1(self.chat_data,"message")
        
    def calculate_hedge2(self) -> None:
        self.chat_data['is_hedged2'] = is_hedged_sentence2(self.chat_data,"message")
        
    def calculate_positivity_zscore(self) -> None:
        self.chat_data['positivity_zscore'] = chat_pos_zscore(self.chat_data,"message")

    def calculate_sentiment_analysis(self) -> None:
        self.chat_data['polarity_score'] = get_avg_polarity_score(self.chat_data,"message")
        self.chat_data['subjectivity_score'] = get_avg_subjectivity_score(self.chat_data,"message")
        
    def calculate_temporal_features(self) -> None:
        self.chat_data = mean_msg_duration(self.chat_data,"timestamp")
        self.chat_data = std_msg_duration(self.chat_data,"timestamp")

    def calculate_tf_idf(self) -> None:
        df = get_tfidf(self.chat_data,"message")
        self.chat_data = pd.concat([self.chat_data, df], axis=0)
