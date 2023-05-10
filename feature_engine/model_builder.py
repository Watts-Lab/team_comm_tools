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

class ModelBuilder():
    def __init__(self, config_path, output_dir):
        self.juries_conv_complete = pd.read_csv(output_dir + "jury_output_conversation_level.csv")
        with open(config_path, "rb") as json_file:
            self.config = json.load(json_file)
        self.juries_conv = self.juries_conv_complete.drop(self.config["juries_conv_cols_to_ignore"], axis=1)
        self.baseline_model = None
        self.X, self.y = None, None
    
    def select_target(self, target):
        self.juries_conv["target_raw"] = target
        scaler = StandardScaler()
        self.juries_conv["target_std"] = scaler.fit_transform(self.juries_conv[["target_raw"]])

    def viz_target(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].hist(self.juries_conv["target_raw"])
        ax[0].set(xlabel="Chosen Target (Raw)", ylabel="Frequency", title="Distribution of Chosen Target (Raw)")
        ax[0].grid()

        ax[1].hist(self.juries_conv["target_std"])
        ax[1].set(xlabel="Chosen Target (Std)", ylabel="Frequency", title="Distribution of Chosen Target (Std)")
        ax[1].grid()
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        sns.heatmap(self.juries_conv.drop(["target_std"], axis=1).corr(), cmap="RdBu", ax=ax[0])
        ax[0].set(title="Feature Correlations with Raw Target")
        ax[0].axis("off")

        sns.heatmap(self.juries_conv.drop(["target_raw"], axis=1).corr(), cmap="RdBu", ax=ax[1])
        ax[1].set(title="Feature Correlations with Standardized Target")
        ax[1].axis("off")
    
    def define_model(self):
        self.X, self.y = self.juries_conv.drop(["target_raw", "target_std"], axis=1), self.juries_conv["target_std"]
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
