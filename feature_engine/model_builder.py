# 3rd Party Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LassoCV
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import optuna
from functools import partial

# Note -- running this class requires Python 3.9 or earlier (not compatible with more recent Python)

class ModelBuilder():
    def __init__(self, config_path="config.json", output_dir="output/", dataset_names=["juries"]):
        with open(config_path, "rb") as json_file:
            self.config = json.load(json_file)
            self.dataset_names = dataset_names
            self.conv = None
            self.convs_complete = [] # list version of conv_complete 

        '''
        [NEW] for the datasets, accept a *list* of datasets, so that it is possible to concatenate them and build
        a model for both of them
        '''
        if(not isinstance(dataset_names, list)):
            raise TypeError("Please provide the dataset names as a list!")

        if(len(dataset_names)==1): # only 1 dataset, so proceed as before
            try:
                self.conv_complete = pd.read_csv(output_dir + self.config[self.dataset_names[0]]["filename"])
            except KeyError:
                print("Are you sure that your dataset name is correct?")
            self.conv = self.conv_complete.drop(self.config[self.dataset_names[0]]["cols_to_ignore"], axis=1).dropna()
        else: # multiple datasets to concatenate together
            for dataset_name in dataset_names:
                try:
                    full_dataset = pd.read_csv(output_dir + self.config[dataset_name]["filename"]).dropna()
                    self.convs_complete.append(full_dataset)
                except KeyError:
                    print("Are you sure that your dataset name is correct?")
                df_extra_columns_dropped = full_dataset.drop(self.config[dataset_name]["cols_to_ignore"], axis=1).dropna()
                
                # add a column with the dataset name --- we can use this to regress on things, controlling for the task
                df_extra_columns_dropped = df_extra_columns_dropped.assign(dataset_name = dataset_name)

                # merge with self.conv
                if self.conv is None:
                    self.conv = df_extra_columns_dropped
                else:
                    self.conv = pd.concat([self.conv, df_extra_columns_dropped])

        self.X, self.y = None, None
        self.baseline_model = None
        self.optimized_model = None
    
    def select_target(self, target):
        if(len(self.dataset_names)==1):
            self.conv["target_raw"] = target
            scaler = StandardScaler()
            self.conv["target_std"] = scaler.fit_transform(self.conv[["target_raw"]])
        else: # concatenate all targets together
            target_raw = pd.Series([])
            target_std = pd.Series([])
            for i, t in enumerate(target):
                raw_dataset = self.convs_complete[i]
                target_raw = pd.concat([target_raw, raw_dataset[t]])
                raw_dataset[t] = (raw_dataset[t] - raw_dataset[t].mean()) / raw_dataset[t].std()
                target_std = pd.concat([target_std, raw_dataset[t]])
            self.conv["target_raw"] = target_raw
            self.conv["target_std"] = target_std

    def viz_target(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].hist(self.conv["target_raw"])
        ax[0].set(xlabel="Chosen Target (Raw)", ylabel="Frequency", title="Distribution of Chosen Target (Raw)")
        ax[0].grid()

        ax[1].hist(self.conv["target_std"])
        ax[1].set(xlabel="Chosen Target (Std)", ylabel="Frequency", title="Distribution of Chosen Target (Std)")
        ax[1].grid()
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        sns.heatmap(self.conv.drop(["target_std"], axis=1).corr(), cmap="RdBu", ax=ax[0])
        ax[0].set(title="Feature Correlations with Raw Target")
        ax[0].axis("off")

        sns.heatmap(self.conv.drop(["target_raw"], axis=1).corr(), cmap="RdBu", ax=ax[1])
        ax[1].set(title="Feature Correlations with Standardized Target")
        ax[1].axis("off")
    
    '''
    [NEW]/TODO let's make it possible to define different types of models, e.g., linear, RF, XGB
    I started initial work here for defining different kinds of models
    Model types = ['xgb', 'lasso']
    '''
    def define_model(self, model_type = "xgb"):
        self.model_type = model_type

        # TODO -- we should make it possible here to select whether you want to regress on the raw or the std target
        self.X, self.y = self.conv.drop(["target_raw", "target_std"], axis=1), self.conv["target_std"]

        # TODO -- this could be a good place to add in additional task features, beyond dummies
        
        # [NEW] Get dummies of categorical variable (task)
        self.X = pd.get_dummies(self.X)
        # [NEW] Apply Normalization to all columns --- otherwise, columns with largest values pop out
        self.X = pd.DataFrame(StandardScaler().fit_transform(self.X),columns = self.X.columns)
        
        if model_type == 'xgb':
            self.baseline_model = XGBRegressor(random_state=42)
        elif model_type == 'lasso':
            self.baseline_model = LassoCV(alphas = [0.01, 0.05, .1, 0.25, 0.5, 0.75, 1])
    
    def evaluate_model(self, model):
        self.model_metrics(model)
        self.model_diagnostics(model)

    def model_metrics(self, model):
        
        # TODO - can we specify how many folds to do CV on?
        # TODO - can we also do a version where we can calculate (1) define a train and test set, and (2) calculate out-of-sample prediction error on the test set?

        r2 = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring="r2").mean().round(4)
        mae = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_absolute_error, greater_is_better=False)).mean().round(4)
        mse = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_squared_error, greater_is_better=False)).mean().round(4)
        rmse = np.sqrt(mse).round(4)
        print("Model Metrics")
        print(f"R2: {r2}\tMAE: {mae}\tMSE: {mse}\tRMSE: {rmse}")

    def model_diagnostics(self, model):
        model = model.fit(self.X, self.y)

        if self.model_type == 'xgb':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X)
            shap.summary_plot(shap_values, self.X, feature_names=self.X.columns, plot_type="bar")
            shap.summary_plot(shap_values, self.X, feature_names=self.X.columns)

        # [NEW] Add different kinds of diagnostics for different types of models
        if self.model_type == 'lasso':
            self.print_lasso_coefs(model)
            self.plot_lasso_residuals(model)

    # [NEW] method for Lasso
    def print_lasso_coefs(self, model):

        coefficients = model.coef_
        
        # Get coefficients that are larger than minimal thresh = 0.001
        # This drops the worst coefficients in the graph and makes the image more readable
        MIN_THRESH = 0.01
        if(len(coefficients[abs(coefficients) > MIN_THRESH])>0):
            nonzero_coefficients,nonzero_indices = self.get_nonzero_lasso_coefs(coefficients, MIN_THRESH)
        else:
            MIN_THRESH = 0
            nonzero_coefficients,nonzero_indices = self.get_nonzero_lasso_coefs(coefficients, MIN_THRESH)

        nonzero_names = self.X.columns[nonzero_indices] 
        sorted_indices = np.argsort(np.abs(nonzero_coefficients))[::-1]
        sorted_coefficients = nonzero_coefficients[sorted_indices]
        sorted_names = nonzero_names[sorted_indices]

        fig, ax = plt.subplots(figsize=(20, 7))
        ax.barh(range(len(sorted_coefficients)), sorted_coefficients, color='steelblue')
        ax.invert_yaxis()

        coeff_range = np.abs(np.max(sorted_coefficients) - np.min(sorted_coefficients))
        offset = coeff_range * 0.03

        for i, (coefficient, name) in enumerate(zip(sorted_coefficients, sorted_names)):
            text_x = coefficient - offset if coefficient < 0 else coefficient + offset
            ax.text(text_x, i, f'{coefficient:.2f}', ha='left' if coefficient < 0 else 'right', va='center')

        ax.set_yticks(range(len(sorted_coefficients)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Coefficient Name')
        ax.set_title('Lasso Coefficients with CV-fitted alpha = ' + str(model.alpha_))

        plt.show()

    # [NEW] method for Lasso
    def get_nonzero_lasso_coefs(self, coefficients, MIN_THRESH):
        nonzero_coefficients = coefficients[abs(coefficients) > MIN_THRESH]
        nonzero_indices = np.where(abs(coefficients) > MIN_THRESH)[0]
        return(nonzero_coefficients,nonzero_indices)

    # [NEW] method for Lasso
    def plot_lasso_residuals(self, model):
   
        predicted_values = model.predict(self.X)
        residuals = self.y - predicted_values
        
        plt.figure(figsize=(8, 6))
        plt.scatter(predicted_values, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    def optimize_model(self):
        optimization_function = partial(self.optimize)
        study = optuna.create_study(direction="minimize")
        # TODO - curerntly optimization runs with 15 trials. Can we create a more intelligent stopping criterion?
        # Also TODO -- seems like for the most part, optimized models actually *underperform* baseline models
        # possible we need more iterations? 
        study.optimize(optimization_function, n_trials=15)
        self.optimized_model = XGBRegressor(**study.best_params, random_state=42, objective="reg:squarederror")
    
    def optimize(self, trial):
        """
        Function used to produce optimized (Bayesian Optimization) Random Forests regressor
        
        INPUTS:
            :trial: One Optimization trial.
            :X_train (np.ndarray): Training data
            :y_train (np.ndarray): Training labels
            :X_test (np.ndarray): Testing data
            :y_test (np.ndarray): Testing labels
        """
        # Define Parameter Space
        criterion = trial.suggest_categorical("criterion", ["absolute_error", "squared_error", "poisson"])
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
        max_samples = trial.suggest_uniform("max_samples", 0.01, 1.0)
        learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.01)
        gamma = trial.suggest_uniform("gamma", 0.001, 0.02)

        # Define Model
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            max_samples=max_samples,
            learning_rate=learning_rate,
            gamma=gamma,
            random_state=42, 
            objective="reg:squarederror"
        )
        
        mse = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_squared_error, greater_is_better=False)).mean().round(4)

        # Return MSE
        return mse