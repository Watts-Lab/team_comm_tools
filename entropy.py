#Get priors using the previous conversations. and try to check how 

def ngram_dialog_act_entropy(df,on_column,n,set1,set2,set1_label,set2_label):

    entropies = []
    # Initialize count vectorizer
    vectorizer = CountVectorizer(ngram_range=(n,n), vocabulary=set1+set2)

    #iterate through each row in the data frame and add the tag
    for i,row in df.iterrows():

        # Count occurrences of words in each bag
        X = vectorizer.fit_transform(df[on_column])

        X_dense = X.toarray()
        set1_counts = np.sum(X_dense[:, 0:len(set1)], axis=1)
        set2_counts = np.sum(X_dense[:, len(set2):], axis=1)    

        # Normalize counts to obtain probabilities
        set1_probs = set1_counts / np.sum(set1_counts)
        set2_probs = set2_counts / np.sum(set2_counts)

        # Calculate entropy for each bag
        set1_entropy = -np.sum(set1_probs.reshape(-1,) * np.log2(set1_probs.reshape(-1,)))
        set2_entropy = -np.sum(set2_probs.reshape(-1,) * np.log2(set2_probs.reshape(-1,)))


        #return the result
        if set1_entropy > set2_entropy:
            print(set1_entropy)
            print(set2_entropy)
            entropies.append(set1_label)
        elif set1_entropy < set2_entropy:
            print(set1_entropy)
            print(set2_entropy)
            entropies.append(set2_label)
        else:
            print(set1_entropy)
            print(set2_entropy)
            entropies.append("neutral")

    df['entropy'] = entropies

    #Test for entropy
data = {
    "name": ["you are a bad person","You are a happy, joyful, lovely person"]
}

df = pd.DataFrame(data)

set1 = ["happy", "joyful", "lovely"]
set2 = ["bad"]

ngram_dialog_act_entropy(df,'name',1,set1,set2,'positive','negative')
print(df['entropy'])