import pandas as pd
from features import feature_extraction

def get_politeness_v2(df,on_column):
    """
    @Args:
        The text dataframe
    @Returns:
        The dataframe after adding the politness v2 features
    """

    #extract the column headers by running the script for a random text. We sort the names of the features in alphabetical order. 

    '''
    TODO --- this code should be fixed stylistically; is there a cleaner way of doing this?
    '''
    # # This is done because the original package sorts the features by counts. It is not possible do so if we have a number of rows, as each row may have different counts for different features 
    column_headers = feature_extraction.feat_counts("hahaha",feature_extraction.kw).sort_values(by='Features')['Features'].tolist()
    
    # Apply the function to each row in 'text_column' and store the result in a new column 'output_column'. We sort the names of the features in alphabetical order
    df_output = df[on_column].apply(lambda x: feature_extraction.feat_counts(x,feature_extraction.kw).sort_values(by='Features')['Counts'])

    '''
        TODO -- this code breaks for me:
        df_output = df[on_column].apply(lambda x: feature_extraction.feat_counts(x,feature_extraction.kw).sort_values(by='Features')['Counts'])
                                                                               ^^^^^^^^^^^^^^^^^^^^^
        AttributeError: module 'features.feature_extraction' has no attribute 'kw'
    '''

    #add the column headers
    df_output.columns = column_headers

    return df_output
