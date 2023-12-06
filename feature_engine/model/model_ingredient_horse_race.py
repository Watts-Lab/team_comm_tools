from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import colorsys
import json
import os
import scipy.stats as stats 
import numpy as np
import pandas as pd
import random
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
plt.rcParams["font.family"] = "Times New Roman"

"""
This Python script formalizes the process required to run the "Team Ingredient Horse Race."

It assumes the multi-task data as input.

For each task in the multi-task data, it fits models using 4 groups of features, progressively nesting:
- Team Composition
- Task Map Features
- Task Instance Complexity
- Conversation Features
"""

def drop_invariant_columns(df):
	"""
	Certain features are invariant throughout the training data (e.g., the entire column is 0 throughout).

	These feature obviously won't be very useful predictors, so we drop them.
	
	This function works by identifying columns that only have 1 unique value throughout the entire column,
	and then dropping them.

	@df: the dataframe containing the features (this should be X).
	"""
	nunique = df.nunique()
	cols_to_drop = nunique[nunique == 1].index
	return(df.drop(cols_to_drop, axis=1))

def read_and_preprocess_data(path, min_num_chats, num_conversation_components):
	conv_data  = pd.read_csv(path)

	# Fill NA with mean
	conv_data.fillna(conv_data.mean(numeric_only=True), inplace=True)

	# Filter this down to teams that have at least min_num of chats
	# Can also comment this out to re-run results on *all* conversations!
	conv_data = conv_data[conv_data["sum_num_messages"] >= min_num_chats]


	# Save the important information

	# DV
	dvs = conv_data[["score","speed","efficiency","raw_duration_min","default_duration_min"]]

	# Team Composition
	composition_colnames = ['birth_year', 'CRT', 'income_max', 'income_min', 'IRCS_GS', 'IRCS_GV', 'IRCS_IB', 'IRCS_IR',
				'IRCS_IV', 'IRCS_RS', 'political_fiscal', 'political_social', 'RME', 'country', 'education_level',
				'gender', 'marital_status', 'political_party', 'race', 'playerCount']
	
	# Select columns that contain the specified keywords
	composition = conv_data[[col for col in conv_data.columns if any(keyword in col for keyword in composition_colnames)]]

	# Task
	task = conv_data[['task', 'complexity']].copy()

	task_map_path = '../utils/task_map.csv' # get task map
	task_map = pd.read_csv(task_map_path)

	task_name_mapping = {
		"Moral Reasoning": "Moral Reasoning (Disciplinary Action Case)",
		"Wolf Goat Cabbage": "Wolf, goat and cabbage transfer",
		"Guess the Correlation": "Guessing the correlation",
		"Writing Story": "Writing story",
		"Room Assignment": "Room assignment task",
		"Allocating Resources": "Allocating resources to programs",
		"Divergent Association": "Divergent Association Task",
		"Word Construction": "Word construction from a subset of letters",
		"Whac a Mole": "Whac-A-Mole"
	}
	task.loc[:, 'task'] = task['task'].replace(task_name_mapping)
	task = pd.merge(left=task, right=task_map, on = "task", how='left')
	
	# Create dummy columns for 'complexity'
	complexity_dummies = pd.get_dummies(task['complexity'])
	task = pd.concat([task, complexity_dummies], axis=1)   
	task.drop(['complexity', 'task'], axis=1, inplace=True)

	# Conversation
	conversation = conv_data.drop(columns=list(dvs.columns) + list(composition.columns) + ['task', 'complexity', 'stageId', 'roundId', 'cumulative_stageId', 'gameId', 'message', 'message_lower_with_punc', 'speaker_nickname', 'conversation_num', 'timestamp'])
	conversation = drop_invariant_columns(conversation) # drop invariant conv features

	# additional preprocess --- get PC's of conversation to reduce dimensionality issues
	pca = PCA(n_components=num_conversation_components)
	pca_result = pca.fit_transform(conversation.transform(lambda x: (x - x.mean()) / x.std()))
	print("PCA explained variance:")
	print(np.sum(pca.explained_variance_ratio_))
	conversation = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

	return composition, task, conversation, dvs

# Note --- this uses k-fold cross-validation with k = 5 (the default)
# We are testing 10,000 different alphas, so I feel like this is an OK heuristic
def get_optimal_alpha(X, y, y_target, feature_columns_list, lasso):

	if(lasso == True):
		model = LassoCV(n_alphas = 10000)
		model.fit(X[feature_columns_list], y[y_target])
	else:
		model = RidgeCV(n_alphas = 10000)
		model.fit(X[feature_columns_list], y[y_target])
		
	return model.alpha_ # optimal alpha

def fit_regularized_linear_model(X, y, y_target, feature_columns_list, lasso=True, tune_alpha=False, prev_coefs = None, prev_alpha = None):

	if not tune_alpha:
		alpha = 1.0
	if (prev_alpha is not None):
		alpha = prev_alpha # use previous alpha
		print("Setting alpha to previous...")
		print(alpha)
	else:
		# Hyperparameter tune the alpha
		alpha = get_optimal_alpha(X, y, y_target, feature_columns_list, lasso=True)

	if lasso:
		model = Lasso(alpha=alpha)
	else:
		model = Ridge(alpha=alpha)

	if(prev_coefs is not None): # set weights to previous coefficients
		print("Setting coefficients ....")
		model.coef_ = prev_coefs

		print(model.coef_)

	# Calculation of Q^2 metric
	squared_model_prediction_errors = []
	squared_average_prediction_errors = []

	# Initialize a list to store coefficients
	coefficients_list = []

	# Leave one out -- iterate through the entire length of the dataset
	for i in range(len(y)):
		# Store the evaluation datapoint
		evaluation_X = X.iloc[[i]]
		evaluation_y = y.iloc[[i]][y_target]

		# Drop the ith datapoint (leave this one out)
		X_fold = X.drop(X.index[i])
		y_fold = y.drop(y.index[i])[y_target]

		# Fit the model
		model.fit(X_fold[feature_columns_list], y_fold)

		# Save the Prediction Error
		prediction = model.predict(evaluation_X[feature_columns_list])[0]
		squared_model_prediction_errors.append((evaluation_y - prediction) ** 2)

		# Save the Total Error for this fold
		squared_average_prediction_errors.append((evaluation_y - np.mean(y_fold)) ** 2)

		# Append the coefficients to the list
		coefficients_list.append(model.coef_)

	# Create a DataFrame with feature names as rows and iteration results as columns
	feature_coefficients = pd.DataFrame(coefficients_list, columns=feature_columns_list).T

	q_squared = 1 - (np.sum(squared_model_prediction_errors) / np.sum(squared_average_prediction_errors))
	print("Q^2: " + str(q_squared))

	return model, q_squared, feature_coefficients

def display_feature_coefficients(feature_coef_df):
	# Initialize a list to store DataFrames for each feature
	dfs = []

	# Iterate through the rows of the input DataFrame
	for feature_name, coefficients in feature_coef_df.iterrows():
		# Calculate the confidence interval without NaN values
		non_nan_coefficients = coefficients[~np.isnan(coefficients)]
		if len(non_nan_coefficients) == 0:
			# Handle the case where there are no valid coefficients
			continue

		mean_coef = non_nan_coefficients.mean()

		# Check if all coefficients in the row are the same
		if len(coefficients.unique()) == 1:
			# If all coefficients are the same, set the lower and upper CI to the mean
			confidence_interval = (mean_coef, mean_coef)
		else:
			std_error = non_nan_coefficients.sem()
			confidence_interval = stats.t.interval(0.95, len(non_nan_coefficients) - 1, loc=mean_coef, scale=std_error)

		# Create a DataFrame for the summary data
		temp_df = pd.DataFrame({
			"Feature": [feature_name],
			"Mean": [mean_coef],
			"Lower_CI": [confidence_interval[0]],
			"Upper_CI": [confidence_interval[1]]
		})

		# Append the temporary DataFrame to the list
		dfs.append(temp_df)

	# Concatenate all the DataFrames in the list into the final summary DataFrame
	summary_df = pd.concat(dfs, ignore_index=True)

	return summary_df

def sort_by_mean_abs(df):
	return df.reindex(df["Mean"].abs().sort_values(ascending=False).index)


"""
Function to run all experiments in 1 go
"""
def get_sample_with_replacement(df):
	num_rows = len(df)
	resampled_indices = pd.Series(range(num_rows)).sample(n=num_rows, replace=True).reset_index(drop=True)
	resampled_dataframe = df.iloc[resampled_indices]

	return resampled_dataframe

def resample(X, y):
	total_data = pd.concat([X, y], axis = 1)
	resampled_data = get_sample_with_replacement(total_data)
	resampled_X = resampled_data[list(X.columns)]
	resampled_y = resampled_data[list(y.columns)]

	return resampled_X, resampled_y

def train_and_evaluate_three_models(random_seed, X, y, composition_cols, task_map_cols, task_cols, conv_cols):
	random.seed(random_seed)

	# Set up the dataset by drawing 1,000 samples
	resampled_X, resampled_y = resample(X, y)

	# # Composition Features
	# print(".......composition only.......")
	# model_ridge_composition, mrc_q2, mrc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, composition_cols, lasso = False, tune_alpha = True)

	# # Composition + Task (Map Only)
	# print(".......composition + task map.......")
	# task_gen_comp_features = composition_cols+task_map_cols
	# model_ridge_taskgencomp, mrtgc_q2, mrtgc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, task_gen_comp_features, lasso = False, tune_alpha = True)

	# # Composition + Task (Map + Complexity)
	# print(".......composition + all task features.......")
	# task_comp_features = composition_cols+task_cols
	# model_ridge_taskcomp, mrtc_q2, mrtc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, task_comp_features, lasso = False, tune_alpha = True)

	# # Composition + Task + Conversation
	# print(".......composition + all task features + conversation.......")
	# all_features = composition_cols+task_cols+conv_cols
	# model_ridge_all, mrall_q2, mrall_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, all_features, lasso = False, tune_alpha = True)



	# for solo categories
	# Composition Features
	print(".......composition.......")
	model_ridge_composition, mrc_q2, mrc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, composition_cols, lasso = False, tune_alpha = True)

	# Composition + Task (Map Only)
	print(".......task map.......")
	model_ridge_taskgencomp, mrtgc_q2, mrtgc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, task_map_cols, lasso = False, tune_alpha = True)

	# Composition + Task (Map + Complexity)
	print(".......complexity.......")
	task_complexity_features = ["High", "Low", "Medium"]
	model_ridge_taskcomp, mrtc_q2, mrtc_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, task_complexity_features, lasso = False, tune_alpha = True)

	# Composition + Task + Conversation
	print(".......conversation.......")
	model_ridge_all, mrall_q2, mrall_feature_coefficients = fit_regularized_linear_model(resampled_X, resampled_y, desired_target, conv_cols, lasso = False, tune_alpha = True)


	return mrc_q2, mrtgc_q2, mrtc_q2, mrall_q2

def get_experimental_results_for_data(data_path, min_num_chats, num_conversation_components, N_ITERS):
	
	# Get lists of features
	team_composition_features, task_features, conv_features, targets = read_and_preprocess_data(data_path, min_num_chats=min_num_chats, num_conversation_components = num_conversation_components)

	# Set the full X and y dataframes
	X = pd.concat([team_composition_features, task_features, conv_features], axis = 1)
	y = targets

	# Column Names
	composition_cols = list(team_composition_features.columns)
	task_map_path = '../utils/task_map.csv' # get task map
	task_map_cols = list(pd.read_csv(task_map_path).drop(["task"], axis = 1).columns)
	task_cols = list(task_features.columns)
	conv_cols = list(conv_features.columns)

	# Bootstrap!
	# composition_only = []
	# composition_task_general = []
	# composition_task = []
	# all_features = []

	# for solo categories
	composition = []
	task_map = []
	task_complexity = []
	conversation = []


	random_seeds = [random.randint(1, 999999999) for i in range(N_ITERS)]

	for i in range(len(random_seeds)):
		if i % 10 == 0:
			print("Starting iteration #" + str(i) + " ...")
		seed = random_seeds[i]
		
		# comp, taskgencomp, taskcomp, taskcompconv = train_and_evaluate_three_models(seed, X, y, composition_cols, task_map_cols, task_cols, conv_cols)
		comp_res, taskmap_res, complexity_res, conv_res = train_and_evaluate_three_models(seed, X, y, composition_cols, task_map_cols, task_cols, conv_cols)

		#composition_only.append(comp)
		# composition_task_general.append(taskgencomp)
		# composition_task.append(taskcomp)
		# all_features.append(taskcompconv)
		
		# for solo categories
		composition.append(comp_res)
		task_map.append(taskmap_res)
		task_complexity.append(complexity_res)
		conversation.append(conv_res)

	return composition, task_map, task_complexity, conversation

	# return composition_only, composition_task_general, composition_task, all_features

"""
Plotting Utilities
"""
def plot_means_with_confidence_intervals_and_ttests(observation_lists, labels, save_path, title_appendix = "", confidence_level=0.95, alpha=0.05):
	# Calculate means and confidence intervals
	means = [np.mean(observation) for observation in observation_lists]
	errors = [(sms.DescrStatsW(observation).tconfint_mean()[1] - sms.DescrStatsW(observation).tconfint_mean()[0]) / 2. for observation in observation_lists]
	colors = plt.cm.tab20(np.arange(len(labels)))

	# Plot the bar graph with error bars
	plt.figure(figsize=(12, 8))
	plt.bar(range(len(means)), means, yerr=errors, align='center', alpha=0.7, ecolor='black', color=colors, capsize=10)

	# Add labels and title
	plt.xticks(range(len(means)), labels, rotation=45, ha="right")  # Rotate x-axis labels by 45 degrees
	plt.ylabel('Prediction Q^2', size = 16)
	plt.title('Predictive Power of Models; Pairwise t-test with B-H Correction' + title_appendix, size = 18)

	# Perform pairwise t-tests
	p_values = []
	line_height = np.mean(max(observation_lists)) * 0.1
	for i in range(len(observation_lists)):
		for j in range(i + 1, len(observation_lists)):
			t_stat, p_value = stats.ttest_ind(observation_lists[i], observation_lists[j])
			p_values.append(p_value)

			# Draw horizontal bar between compared groups only if p-value is significant
			if p_value < alpha:
				line_y = max(means) + max(errors) + np.mean(max(observation_lists)) * 0.03 + (i + j) * line_height * 1.5  # Adjust the multiplier for better spacing
				plt.plot([i, j], [line_y, line_y], color='black')

				# Display significance stars based on p-value
				if p_value < 0.001:
					significance_label = '***'
				elif p_value < 0.01:
					significance_label = '**'
				elif p_value < 0.05:
					significance_label = '*'
				else:
					significance_label = 'n.s.'

				# Display significance labels on the plot
				plt.text((i + j) / 2, line_y + np.mean(max(observation_lists)) * 0.025, significance_label, ha='center', va='center')

	# Correct p-values for multiple comparisons using Benjamini-Hochberg procedure
	_, p_values_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

	# Show the plot
	plt.savefig(save_path+".svg")
	plt.savefig(save_path+".png")

def save_cv_data(observation_lists, labels, file_path):
	data_dict = {label: column for label, column in zip(labels, observation_lists)}

	df = pd.DataFrame(data_dict)
	df.to_csv(file_path)

"""
The driver of the whole thing!
"""
if __name__ == "__main__":

	# Datasets with different aggregation methods
	multitask_cumulative_by_stage = '../output/conv/multi_task_output_conversation_level_stageId_cumulative.csv'
	multitask_cumulative_by_stage_and_task = '../output/conv/multi_task_output_conversation_level_stageId_cumulative_within_task.csv'

	# Key parameters
	num_conversation_components = 5
	min_num_chats = 0
	desired_target = "score"
	N_ITERS = 100
	
	#labels = ["Composition Only", "Composition + Task Map", "Composition + Task Map + Complexity", "Composition + Task Map + Complexity + Communication"]
	labels_solo = ["Composition", "Task Map", "Task Complexity", "Communication"]

	# Call the function for each type of grouping
	# print("Beginning Analysis for Multitask (Cumulative by StageID)....")
	
	# composition_stagecumu, composition_task_general_stagecumu, composition_task_stagecumu, all_stagecumu = get_experimental_results_for_data(multitask_cumulative_by_stage, min_num_chats, num_conversation_components, N_ITERS)
	# save_cv_data([composition_stagecumu, composition_task_general_stagecumu, composition_task_stagecumu, all_stagecumu], labels, './multi_task_results/multitask_cumulative_by_stage.csv')
	# plot_means_with_confidence_intervals_and_ttests([composition_stagecumu, composition_task_general_stagecumu, composition_task_stagecumu, all_stagecumu], labels, "./multi_task_results/multitask_cumulative_by_stage_ingredient_horserace", title_appendix = " (Chats Cumulative by StageId)", confidence_level=0.95, alpha=0.05)

	# plot the performance of each category alone
	# composition_stagecumu, task_general_stagecumu, complexity_stagecumu, conversation_stagecumu = get_experimental_results_for_data(multitask_cumulative_by_stage, min_num_chats, num_conversation_components, N_ITERS)
	# save_cv_data([composition_stagecumu, task_general_stagecumu, complexity_stagecumu, conversation_stagecumu], labels_solo, './multi_task_results/multitask_cumulative_by_stage_category_solo.csv')
	# plot_means_with_confidence_intervals_and_ttests([composition_stagecumu, task_general_stagecumu, complexity_stagecumu, conversation_stagecumu], labels_solo, "./multi_task_results/multitask_cumulative_by_stage_ingredient_category_solo", title_appendix = " (Chats Cumulative by StageId)", confidence_level=0.95, alpha=0.05)

	print("Beginning Analysis for Multitask (Cumulative by StageID and TASK)....")
	# composition_stagecumutask, composition_task_general_stagecumutask, composition_task_stagecumutask, all_stagecumutask = get_experimental_results_for_data(multitask_cumulative_by_stage_and_task, min_num_chats, num_conversation_components, N_ITERS)
	# save_cv_data([composition_stagecumutask, composition_task_general_stagecumutask, composition_task_stagecumutask, all_stagecumutask], labels, './multi_task_results/multitask_cumulative_by_stage_and_task.csv')
	# plot_means_with_confidence_intervals_and_ttests([composition_stagecumutask, composition_task_general_stagecumutask, composition_task_stagecumutask, all_stagecumutask], labels, "./multi_task_results/multitask_cumulative_by_stage_and_task_ingredient_horserace", title_appendix = " (Chats Cumulative by StageId and Task)", confidence_level=0.95, alpha=0.05)

	# plot the performance of each category alone
	composition_stagecumutask, task_general_stagecumutask, complexity_stagecumutask, conversastion_stagecumutask = get_experimental_results_for_data(multitask_cumulative_by_stage_and_task, min_num_chats, num_conversation_components, N_ITERS)
	save_cv_data([composition_stagecumutask, task_general_stagecumutask, complexity_stagecumutask, conversastion_stagecumutask], labels_solo, './multi_task_results/multitask_cumulative_by_stage_and_task_category_solo.csv')
	plot_means_with_confidence_intervals_and_ttests([composition_stagecumutask, task_general_stagecumutask, complexity_stagecumutask, conversastion_stagecumutask], labels_solo, "./multi_task_results/multitask_cumulative_by_stage_and_task_ingredient_category_solo", title_appendix = " (Chats Cumulative by StageId and Task)", confidence_level=0.95, alpha=0.05)

