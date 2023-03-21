import pandas as pd
from textblob import TextBlob
import statistics as stat

'''
Polarity:Ranges between -1 to 1,  -1 is most negative, 0 is neutral, 1 is positive
Subjectivity: Ranges between 0 to 1, 0 - purely factual, 1 - pure opinion
'''

'''
@param text 
@return gets the average polarity score for the entire conversation
'''

def get_avg_polarity_score(df,on_column):
    return TextBlob(df[on_column]).sentiment.polarity

'''
@param text 
@return polarity score of the most polarizing speaker
'''

def get_highest_polarity_individual_score(df,on_column):
    speaker_polarity = TextBlob(df[on_column]).groupby(["speaker_nickname"]).sentiment.polarity
    speaker_polarity.sort()
    return  speaker_polarity[0]

'''
@param text 
@return polarity score of the least polarizing speaker
'''

def get_lowest_polarity_individual_score(df,on_column):
    speaker_polarity = TextBlob(df[on_column]).groupby(["speaker_nickname"]).sentiment.polarity
    speaker_polarity.sort()
    return  speaker_polarity[-1]

'''
@param text 
@return Standard Deviation in the polarity scores of all the speakers
'''

def get_polarity_variation(df,on_column):
    speaker_polarity = TextBlob(df[on_column]).groupby(["speaker_nickname"]).sentiment.polarity
    speaker_polarity.sort()
    return  stat.pstdev(speaker_polarity )

'''
@param text 
@return gets the average subjectivity score for the entire conversation
'''
def get_avg_subjectivity_score(df,on_column):
    return TextBlob(df[on_column]).sentiment.subjectivity

'''
@param text 
@return subjectivity score of the most subjective speaker
'''

def get_subjectivity_score(df,on_column):
    return TextBlob(df[on_column]).sentiment.subjectivity

'''
@param text 
@return subjectivity score of the least subjective speaker
'''

def get_highest_subjectivity_individual_score(df,on_column):
    speaker_subjectivity = TextBlob(df[on_column]).groupby(["speaker_nickname"]).sentiment.subjectivity
    speaker_subjectivity.sort()
    return  speaker_subjectivity[-1]

'''
@param text 
@return Standard Deviation in the subjectivity scores of all the speakers
'''

def get_highest_subjectivity_individual_score(df,on_column):
    speaker_subjectivity = TextBlob(df[on_column]).groupby(["speaker_nickname"]).sentiment.subjectivity
    speaker_subjectivity.sort()
    return  stat.pstdev(speaker_subjectivity)

def get_subjectivity_score2(string):
    return TextBlob(string).sentiment.subjectivity
