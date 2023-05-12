# 3rd Party Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import optuna
from functools import partial

class ModelBuilder():
    def __init__(self, config_path="config.json", output_dir="output/", dataset_name="juries"):
        with open(config_path, "rb") as json_file:
            self.config = json.load(json_file)
        self.dataset_name = dataset_name
        self.conv_complete = pd.read_csv(output_dir + self.config[self.dataset_name]["filename"])
        self.conv = self.conv_complete.drop(self.config[self.dataset_name]["cols_to_ignore"], axis=1)
        self.X, self.y = None, None
        self.baseline_model = None
        self.optimized_model = None
    
    def select_target(self, target):
        self.conv["target_raw"] = target
        scaler = StandardScaler()
        self.conv["target_std"] = scaler.fit_transform(self.conv[["target_raw"]])

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
    
    def define_model(self):
        self.X, self.y = self.conv.drop(["target_raw", "target_std"], axis=1), self.conv["target_std"]
        self.baseline_model = XGBRegressor(random_state=42)
    
    def evaluate_model(self, model):
        self.model_metrics(model)
        self.model_diagnostics(model)

    def model_metrics(self, model):
        r2 = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring="r2").mean().round(4)
        mae = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_absolute_error, greater_is_better=False)).mean().round(4)
        mse = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_squared_error, greater_is_better=False)).mean().round(4)
        rmse = np.sqrt(mse).round(4)
        print("Model Metrics")
        print(f"R2: {r2}\tMAE: {mae}\tMSE: {mse}\tRMSE: {rmse}")

    def model_diagnostics(self, model):
        model = model.fit(self.X, self.y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X, feature_names=self.X.columns, plot_type="bar")
        shap.summary_plot(shap_values, self.X, feature_names=self.X.columns)

    def optimize_model(self):
        optimization_function = partial(self.optimize)
        study = optuna.create_study(direction="minimize")
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