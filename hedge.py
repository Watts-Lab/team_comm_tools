import pandas as pd

#returns the total number of hedge words present in the data set (We can add more hedge words)
def hedge1(df,on_column):
    hedge_words = ["sort of", "kind of", "I guess", "I think", "a little", "maybe", "possibly", "probably"]
    return  df[on_column].str.count("|".join(hedge_words)).sum()

#Test Data
data = {
    "name": ["I am sort of","I guess I am crazy"]
}
df = pd.DataFrame(data)
hedge1(df,"name")
