import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Define directories
base_dir = "/Users/evanrowbotham/Dev/data"
visualization_dir = os.path.join(base_dir, "level0_data/modeling/visualizations")
metrics_dir = os.path.join(base_dir, "level0_data/modeling/metrics")
os.makedirs(visualization_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Input Level 0 dataset
data = pd.read_csv(os.path.join(base_dir, "larger_level0_data_cleaned_chat_level.csv"))

# Drop the conversation number column
data = data.drop(columns=['conversation_num'])

# Separate features and labels
features = data.drop(columns=['oppositional_intensity', 'directness'])
labels_opp = data['oppositional_intensity']
labels_dir = data['directness']

# Keep only numeric features
numeric_features = features.select_dtypes(include=[np.number])

# Drop columns with no observed values
numeric_features = numeric_features.loc[:, numeric_features.isnull().mean() < 1.0]

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(kernel='linear')
}

# Initialize dictionaries to hold f1 scores and reports
f1_scores = {'Oppositional Intensity': {}, 'Directness': {}}
reports = []
feature_importances = {'Logistic Regression': {}, 'Random Forest': {}}
feature_names = numeric_features.columns

# Set up k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Function to process data
def process_data(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

# Function to evaluate models
def evaluate_models(X, y, task_name):
    for model_name, model in models.items():
        fold_f1_scores = []
        importances = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            X_train, X_test = process_data(X_train, X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

            if model_name == 'Logistic Regression':
                importances.append(model.coef_.flatten())
            elif model_name == 'Random Forest':
                importances.append(model.feature_importances_)

            report = classification_report(y_test, y_pred, output_dict=True)
            for avg_type in ['macro avg', 'weighted avg']:
                reports.append({
                    'Model': model_name,
                    'Task': task_name,
                    'Average Type': avg_type,
                    'Precision': report[avg_type]['precision'],
                    'Recall': report[avg_type]['recall'],
                    'F1-Score': report[avg_type]['f1-score'],
                    'Support': report[avg_type]['support']
                })
        
        f1_scores[task_name][model_name] = np.mean(fold_f1_scores)
        if model_name in ['Logistic Regression', 'Random Forest']:
            feature_importances[model_name][task_name] = np.mean(importances, axis=0)

# Evaluate models for oppositional intensity
evaluate_models(numeric_features.values, labels_opp.values, 'Oppositional Intensity')

# Evaluate models for directness
evaluate_models(numeric_features.values, labels_dir.values, 'Directness')

# Print formatted F1 scores 
def print_formatted_f1_scores():
    print("\n===================== F1 Scores ================================")
    for model_name in models.keys():
        opp_score = f1_scores['Oppositional Intensity'][model_name]
        dir_score = f1_scores['Directness'][model_name]
        print(f"{model_name}: Oppositional Intensity = {opp_score:.6f}, Directness = {dir_score:.6f}")
    print("================================================================\n")

print_formatted_f1_scores()

# Create a DataFrame from the reports
report_df = pd.DataFrame(reports)

# Reorder columns
report_df = report_df[['Model', 'Task', 'Average Type', 'Precision', 'Recall', 'F1-Score', 'Support']]

# Save the classification report to a CSV file
report_df.to_csv(os.path.join(metrics_dir, "classification_reports.csv"), index=False)

# Function to normalize feature importances
def normalize_importances(importances):
    norm_importances = importances / np.sum(np.abs(importances))
    return norm_importances

# Save the feature importances
np.save(os.path.join(base_dir, 'feature_importances_opp.npy'), feature_importances['Logistic Regression']['Oppositional Intensity'])
np.save(os.path.join(base_dir, 'feature_importances_dir.npy'), feature_importances['Logistic Regression']['Directness'])
np.save(os.path.join(base_dir, 'feature_names.npy'), np.array(feature_names, dtype=str))

# Function to plot logistic regression coefficients with adjusted y-axis spacing
def plot_logistic_regression_coefficients():
    if 'Logistic Regression' in feature_importances:
        for task_name, importances in feature_importances['Logistic Regression'].items():
            importances = normalize_importances(importances)
            sorted_idx = np.argsort(np.abs(importances))
            top_5_idx = sorted_idx[-5:]  # Get indices of the top 5 features
            
            top_5_features = [feature_names[i] for i in top_5_idx]
            top_5_importances = importances[top_5_idx]
            top_5_std = np.std(importances[top_5_idx]) / np.sqrt(len(top_5_importances))  # Standard error
            
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            fig, ax = plt.subplots(figsize=(5.05, 3.81))  # Set figure size to height = 3.81" by width = 5.05"
            for i, (feature, importance, std) in enumerate(zip(top_5_features, top_5_importances, [top_5_std]*5)):
                ax.plot([importance], [i], 'o', color=colors[i])
                ax.plot([importance - std, importance + std], [i, i], color=colors[i])
            
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_yticks(range(len(top_5_idx)))
            ax.set_yticklabels(top_5_features)  # Set y-axis labels to feature names
            ax.set_ylim(-1, len(top_5_idx))  # Adjust y-axis limits
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'Logistic Regression Coefficients: {task_name}')
            # Create a legend with dots
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=feature) for feature, color in zip(top_5_features, colors)]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, f'logistic_regression_coefficients_{task_name}.png'))
            plt.show()

# Call the function to plot logistic regression coefficients
plot_logistic_regression_coefficients()
