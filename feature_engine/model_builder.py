# 3rd Party Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import optuna
from functools import partial

# Note -- running this class requires Python 3.9 or earlier (not compatible with more recent Python)

class ModelBuilder():
    def __init__(self, config_path="config.json", output_dir="output/", task_map_path = 'utils/task_map.csv', dataset_names=["csop"], test_dataset_names=None):
        with open(config_path, "rb") as json_file:
            self.config = json.load(json_file)
            self.dataset_names = dataset_names
            self.conv = None
            self.convs_complete = [] # list version of conv_complete 

            self.test_dataset_names = test_dataset_names
            self.test_conv = None
            self.test_convs_complete = [] # list version of conv_complete 

            self.output_dir = output_dir
            self.task_maps = pd.read_csv(task_map_path)
            self.task_maps = self.task_maps[self.task_maps['task'].isin(self.config['task_names'])]

        if self.test_dataset_names != None: 
            self.create_datasets(is_test_datasets=True)
        self.create_datasets(is_test_datasets=False)
        
        self.X, self.y = None, None
        self.baseline_model = None
        self.optimized_model = None

    def create_datasets(self, is_test_datasets=False):
        '''
        [NEW] for the datasets, accept a *list* of datasets, so that it is possible to concatenate them and build
        a model for both of them
        '''
        if is_test_datasets:
            dataset_names = self.test_dataset_names
        else:
            dataset_names = self.dataset_names
        
        convs_complete = []
        conv = None
        conv_complete = None
        if(not isinstance(dataset_names, list)):
            raise TypeError("Please provide the dataset names as a list!")

        if(len(dataset_names)==1): # only 1 dataset, so proceed as before
            try:
                conv_complete = pd.read_csv(self.output_dir + self.config[dataset_names[0]]["filename"])
            except KeyError:
                print("Are you sure that your dataset name is correct?")
                
            conv = conv_complete.drop(self.config[dataset_names[0]]["cols_to_ignore"], axis=1).dropna()
            conv['dataset_name'] = dataset_names[0]

        else: # multiple datasets to concatenate together
            for dataset_name in dataset_names:
                try:
                    full_dataset = pd.read_csv(self.output_dir + self.config[dataset_name]["filename"]).dropna()
                    convs_complete.append(full_dataset)
                except KeyError:
                    print("Are you sure that your dataset name is correct?")
                
                df_extra_columns_dropped = full_dataset.drop(self.config[dataset_name]["cols_to_ignore"], axis=1).dropna()
                
                # add a column with the dataset name --- we can use this to regress on things, controlling for the task
                df_extra_columns_dropped = df_extra_columns_dropped.assign(dataset_name = dataset_name)

                # merge with self.conv
                if conv is None:
                    conv = df_extra_columns_dropped
                else:
                    conv = pd.concat([conv, df_extra_columns_dropped])

        if is_test_datasets: # Handle / store test dataset
            self.test_conv = conv
            self.test_conv_complete = conv_complete
            self.test_convs_complete = convs_complete
        else:
            self.conv = conv
            # When we have multiple datasets, we have self.convs_complete
            self.convs_complete = convs_complete
            # When we have a single dataset, we have only self.conv_complete
            self.conv_complete = conv_complete
        
        self.integrate_task_level_features(is_test_datasets=is_test_datasets)

    def integrate_task_level_features(self, is_test_datasets=False):
        if is_test_datasets: 
            self.test_conv['task_name'] = self.test_conv['dataset_name'].map(self.config['task_mapping_keys'])
            self.test_conv = pd.merge(left=self.test_conv, right=self.task_maps, left_on=['task_name'], right_on=['task'], how='left')
            self.test_conv.drop(['dataset_name', 'task_name', 'task'], axis=1, inplace=True)
        else:
            self.conv['task_name'] = self.conv['dataset_name'].map(self.config['task_mapping_keys'])
            self.conv = pd.merge(left=self.conv, right=self.task_maps, left_on=['task_name'], right_on=['task'], how='left')
            self.conv.drop(['dataset_name', 'task_name', 'task'], axis=1, inplace=True)

    def set_target(self, target, is_test):
        if(not is_test):
            conversation_clean = self.conv
            conversation_complete = self.conv_complete
            conversation_complete_list = self.convs_complete
        else:
            conversation_clean = self.test_conv
            conversation_complete = self.test_conv_complete
            conversation_complete_list = self.test_convs_complete


        if(len(self.dataset_names)==1):
            if(isinstance(target, list)): # In case user provided target as list
                target = target[0]

            conversation_clean["target_raw"] = conversation_complete[target]
            conversation_clean["target_std"] = (conversation_clean["target_raw"] - conversation_clean["target_raw"].mean()) / conversation_clean["target_raw"].std()
        else: # concatenate all targets together
            target_raw_list, target_std_list = [], []
            for target_idx, target_name in enumerate(target):
                target_raw_list.extend(conversation_complete_list[target_idx][target_name].to_list())
                scaler = StandardScaler()
                target_std_list.extend(list(scaler.fit_transform(conversation_complete_list[target_idx][target_name].to_numpy().reshape(-1, 1))))
            conversation_clean["target_raw"] = target_raw_list
            conversation_clean["target_std"] = target_std_list
            # target_raw = pd.Series([])
            # target_std = pd.Series([])
            # for i, t in enumerate(target):
            #     raw_dataset = conversation_complete_list[i]
            #     target_raw = pd.concat([target_raw, raw_dataset[t]])
            #     raw_dataset[t] = (raw_dataset[t] - raw_dataset[t].mean()) / raw_dataset[t].std()
            #     target_std = pd.concat([target_std, raw_dataset[t]])
            # conversation_clean["target_raw"] = target_raw
            # conversation_clean["target_std"] = target_std

        # set everything
        if(not is_test):
            self.conv = conversation_clean
            self.conv_complete = conversation_complete
            self.convs_complete = conversation_complete_list
        else:
            self.test_conv = conversation_clean
            self.test_conv_complete = conversation_complete
            self.test_convs_complete = conversation_complete_list
            
    def select_target(self, target):
        self.set_target(target, is_test = False)

    def select_test_target(self, target):
        self.set_target(target, is_test = True)

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
    Model types = ['xgb', 'lasso', 'linear', 'rf']
    '''
    def define_model(self, model_type = "xgb"):
        self.model_type = model_type
        if model_type == 'xgb':
            self.baseline_model = XGBRegressor(random_state=42)
        elif model_type == 'lasso':
            self.baseline_model = LassoCV(alphas = [0.01, 0.05, .1, 0.25, 0.5, 0.75, 1])
        elif model_type == 'linear':
            self.baseline_model = LinearRegression()
        elif model_type == 'rf':
            self.baseline_model = RandomForestRegressor(random_state=42)

    def define_dataset_for_model(self, is_test=False):
        # TODO -- we should make it possible here to select whether you want to regress on the raw or the std target
        if is_test:
            X, y = self.test_conv.drop(["target_raw", "target_std"], axis=1), self.test_conv["target_std"]
        else:
            X, y = self.conv.drop(["target_raw", "target_std"], axis=1), self.conv["target_std"]
        X = pd.get_dummies(X)
        X = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)
        return X, y

    def evaluate_model(self, model, val_size=0.1, test_size=0.1):
        print('Checking Holdout Sets', end='...')
        if self.test_dataset_names == None:
            print('Creating Holdout Sets...')
            self.create_holdout_sets(val_size=val_size, test_size=test_size)
        else:
            self.create_holdout_sets(val_size=val_size)
        print('Done')
        print('Training Model', end='...')
        model = model.fit(self.X_train, self.y_train)
        print('Done')
        self.summarize_model_metrics(model)
        self.model_diagnostics(model)

    def create_holdout_sets(self, val_size=0.1, test_size=None):
        self.X, self.y = self.define_dataset_for_model()
        if test_size:
            X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, random_state=42, test_size=test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=42, test_size=val_size)
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = X_val, y_val
            self.X_test, self.y_test = X_test, y_test
        else:
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, random_state=42, test_size=val_size)
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = X_val, y_val
            self.X_test, self.y_test = self.define_dataset_for_model(is_test=True)

    def summarize_model_metrics(self, model):
        train_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_train, self.y_train))
        val_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_val, self.y_val))
        test_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_test, self.y_test))
        print("MODEL METRICS")
        print('Train Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(train_metrics['r2'], train_metrics['mae'], train_metrics['mse'], train_metrics['rmse']))
        
        print('Validation Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(val_metrics['r2'], val_metrics['mae'], val_metrics['mse'], val_metrics['rmse']))

        print('Test Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(test_metrics['r2'], test_metrics['mae'], test_metrics['mse'], test_metrics['rmse']))

    def calculate_model_metrics(self, model, dataset):
        X, y = dataset
        r2 = r2_score(y_true=y, y_pred=model.predict(X)).round(4)
        mae = mean_absolute_error(y_true=y, y_pred=model.predict(X)).round(4)
        mse = mean_squared_error(y_true=y, y_pred=model.predict(X)).round(4)
        rmse = np.sqrt(mse).round(4)
        return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}

    def model_diagnostics(self, model):
        if self.model_type in ['xgb', 'rf']:
            explainer = shap.TreeExplainer(model)
        # [NEW] Add different kinds of diagnostics for different types of models
        elif self.model_type in ['lasso', 'linear']:
            if self.model_type == 'lasso':
                self.print_lasso_coefs(model)
                self.plot_lasso_residuals(model)
            explainer = shap.LinearExplainer(model, self.X_val)
            
        shap_values = explainer.shap_values(self.X_val)
        shap.summary_plot(shap_values, self.X_val, feature_names=self.X.columns, plot_type="bar")
        shap.summary_plot(shap_values, self.X_val, feature_names=self.X.columns)

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

    def get_nonzero_lasso_coefs(self, coefficients, MIN_THRESH):
        nonzero_coefficients = coefficients[abs(coefficients) > MIN_THRESH]
        nonzero_indices = np.where(abs(coefficients) > MIN_THRESH)[0]
        return(nonzero_coefficients,nonzero_indices)

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

    def optimize_model(self, n_trials=15):
        optimization_function = partial(self.optimize)
        study = optuna.create_study(direction="minimize")
        # TODO - curerntly optimization runs with a fixed number of trials. Can we create a more intelligent stopping criterion?
        # Also TODO -- seems like for the most part, optimized models actually *underperform* baseline models
        # possible we need more iterations? 
        study.optimize(optimization_function, n_trials=n_trials)
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
        if self.model_type == 'xgb': gamma = trial.suggest_uniform("gamma", 0.001, 0.02)

        # Define Model
        if self.model_type == 'xgb':
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
        elif self.model_type == 'rf': # TODO -- despite 'rf', this fails with error
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                criterion=criterion,
                max_samples=max_samples,
                learning_rate=learning_rate,
                random_state=42
            )

        # TODO -- throw an error if user provides a model type (e.g., linear) that cannot be optimized
        
        mse = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_squared_error, greater_is_better=False, cv=self.folds)).mean().round(4)

        # Return MSE
        return mse