import pandas as pd



'''
@param df = the name of the dataframe on which the operation is being applied.
	- assumes that the df is a chat-by-chat setup in which each row is 1 chat.
	- assumes that the chat is stored in a column called 'message'
@param feature_name = the name of the column you want the feature to be named
@param function_name = the name of the function used to create the feature
'''
def create_chat_level_feature(df, feature_name, function_name):
	df[feature_name] = df['message'].apply(lambda x: function_name(str(x)))
	return(df)
