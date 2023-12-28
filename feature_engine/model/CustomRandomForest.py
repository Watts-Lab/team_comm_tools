from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

class CustomRandomForest(RandomForestRegressor):
    def __init__(self, n_estimators, max_depth, forced_features, n_additional_features, criterion = "squared_error"):
        self.n_estimators = n_estimators  # Number of Trees
        self.max_depth = max_depth # Max Depth
        self.forced_features = forced_features  # List of features forced to be part of it
        self.n_additional_features = n_additional_features
        self.estimators_ = []
        self.criterion = criterion

    def fit(self, X, y):
        n, p = X.shape
        print("Growing trees...")
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, criterion = self.criterion, random_state=np.random.randint(10000000)) # a new random decision tree every time
            # Select forced features
            X_forced = X[self.forced_features]
            X_forced.reset_index(drop=True, inplace=True)

            # Select additional non-forced features
            nonforced_cols = [col for col in X.columns if col not in self.forced_features]
            X_nonforced_pool = X[nonforced_cols]
            X_nonforced = X_nonforced_pool.sample(n=self.n_additional_features, axis='columns')
            X_nonforced.reset_index(drop=True, inplace=True)

            # Build up the X (forced + non-forced)
            X_selected = pd.concat([X_forced, X_nonforced], axis=1)

            # Fit the tree to the subset of features
            tree.fit(X_selected, y)

            # Save the trained tree
            self.estimators_.append(tree)
        print("Done!")

    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = []
        print("Making predictions...")
        for tree in self.estimators_:
            # Ensure columns in X match the columns used during training for this tree
            tree_columns = tree.feature_names_in_ 
            X_selected = X[tree_columns]
            tree_prediction = tree.predict(X_selected)
            predictions.append(tree_prediction)
        print("Done!")

        return np.mean(predictions, axis=0)