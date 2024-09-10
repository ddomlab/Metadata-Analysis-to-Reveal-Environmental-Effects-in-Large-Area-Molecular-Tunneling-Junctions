import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import scipy.stats as stats

# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Load the data from the updated Excel file
file_path = r'C:\Users\Owner\OneDrive\Desktop\train_GBR_output_carbon_number_with_season_final.xlsx'
data = pd.read_excel(file_path)

# Select relevant columns for the model
model_df = data[['Humidity', 'Temperature', 'Current Density', 'Electrode', 'Written Carbon Number', 'Season', 'Time of Day', 'User Type']].dropna()

# Compute correlation with the target variable
def compute_correlation(df, target):
    correlations = df.corr()[target]
    return correlations

# Calculate correlations for numerical features
numeric_features = ['Humidity', 'Temperature']
correlations_numeric = compute_correlation(model_df[numeric_features + ['Current Density']], 'Current Density')

# Define numerical and categorical transformers
numeric_transformer = Pipeline([
    ('scaler', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
])

# Encode categorical features and apply polynomial feature expansion
categorical_features = ['Season', 'User Type', 'Time of Day', 'Written Carbon Number', 'Electrode']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Split the data into features and target variable
X = model_df[['Humidity', 'Temperature', 'Season', 'Time of Day', 'User Type', 'Written Carbon Number', 'Electrode']]
y = model_df['Current Density']

# Define the pipeline including preprocessing and regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=random_seed))
])

# Define search spaces for Bayesian Optimization
search_spaces = {
    'regressor__n_estimators': Integer(100, 300),
    'regressor__max_depth': Integer(4, 6),
    'regressor__learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'regressor__loss': Categorical(['squared_error', 'absolute_error', 'huber']),
    'regressor__min_samples_split': Integer(2, 20),
    'regressor__min_samples_leaf': Integer(1, 10),
    'regressor__max_features': Categorical([1.0, 'sqrt', 'log2']),
    'regressor__subsample': Real(0.6, 0.95, prior='uniform')
}

# Define outer cross-validation for model evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

# Define Bayesian Optimization
bayes_search = BayesSearchCV(
    pipeline,
    search_spaces,
    scoring='r2',
    cv=outer_cv,  # Use outer_cv for model evaluation
    n_iter=20,  # Number of parameter settings that are sampled
    random_state=random_seed,
    verbose=1
)

# Fit BayesSearchCV to find the best hyperparameters
bayes_search.fit(X, y)

# Predict and evaluate the model with BayesSearchCV
best_pipeline = bayes_search.best_estimator_

# Get cross-validated predictions
y_pred = cross_val_predict(best_pipeline, X, y, cv=outer_cv)

# Calculate MSE, RMSE, MAE, and R2 scores
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print('Best parameters found:')
print(bayes_search.best_params_)
print(f'Model MSE: {mse}')
print(f'Model RMSE: {rmse}')
print(f'Model MAE: {mae}')
print(f'Model R²: {r2}')

# Compute and print correlations for numerical features
print('Correlations with target for numerical features:')
print(correlations_numeric)

# Create a DataFrame with actual and predicted values
results_df = X.copy()
results_df['Actual'] = y
results_df['Predicted'] = y_pred
results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

# Save results to Excel
results_file_path = r'C:\Users\Owner\OneDrive\Desktop\New GBR\model_predictions.xlsx'
results_df.to_excel(results_file_path, index=False)

print(f"Results saved to {results_file_path}")

# Extract and plot feature importances
best_model = best_pipeline.named_steps['regressor']
feature_importances = best_model.feature_importances_

# Get the feature names from the preprocessor
preprocessor_transformers = best_pipeline.named_steps['preprocessor'].transformers_
numeric_feature_names = numeric_features
categorical_feature_names = best_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
all_feature_names = numeric_feature_names + list(categorical_feature_names)

# Combine the importances of one-hot encoded features back to their original categorical features
importances_dict = {feature: 0 for feature in numeric_features + categorical_features}
for feature, importance in zip(all_feature_names, feature_importances):
    # For numeric features, add the importance directly
    if feature in numeric_features:
        importances_dict[feature] += importance
    # For categorical features, find the original feature name
    else:
        original_feature = feature.split('_')[0]
        importances_dict[original_feature] += importance

# Convert the dictionary to lists for plotting
combined_feature_names = list(importances_dict.keys())
combined_feature_importances = list(importances_dict.values())

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    'Feature': combined_feature_names,
    'Importance': combined_feature_importances
})

# Save feature importances to Excel
importances_file_path = r'C:\Users\Owner\OneDrive\Desktop\New GBR\feature_importances.xlsx'
feature_importances_df.to_excel(importances_file_path, index=False)

print(f"Feature importances saved to {importances_file_path}")

# Plotting the combined feature importances
plt.figure(figsize=(10, 8))
plt.barh(combined_feature_names, combined_feature_importances)
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Combined Feature Importance in Gradient Boosting Regressor', fontsize=16, fontweight='bold')
plt.show()

# Boxplot for each feature as a function of current density
for feature in numeric_features + categorical_features:
    plt.figure(figsize=(12, 8))
    plt.boxplot([model_df[model_df[feature] == val]['Current Density'] for val in model_df[feature].unique()],
                labels=model_df[feature].unique())
    plt.xlabel(feature, fontsize=14, fontweight='bold')
    plt.ylabel('Average Log Current Density', fontsize=14, fontweight='bold')
    plt.title(f'Boxplot of Current Density by {feature}', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.show()

# Error and Residual Analysis
# Plot residuals
plt.figure(figsize=(12, 8))
plt.scatter(results_df['Predicted'], results_df['Residual'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values', fontsize=14, fontweight='bold')
plt.ylabel('Residuals', fontsize=14, fontweight='bold')
plt.title('Residuals vs Predicted Values', fontsize=16, fontweight='bold')
plt.show()

# Plot histogram of residuals
plt.figure(figsize=(12, 8))
plt.hist(results_df['Residual'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Histogram of Residuals', fontsize=16, fontweight='bold')
plt.show()

# Q-Q plot for residuals
plt.figure(figsize=(12, 8))
stats.probplot(results_df['Residual'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals', fontsize=16, fontweight='bold')

plt.show()

# Assuming `results_df` is already defined
# Filter based on Current Density > 0
filtered_results_df = model_df[model_df['Current Density'] < 0]

# Split the filtered data into features and target variable
X_filtered = filtered_results_df[['Humidity', 'Temperature', 'Season', 'Time of Day', 'User Type', 'Written Carbon Number', 'Electrode']]
y_filtered = filtered_results_df['Current Density']

# Re-fit the model with filtered data
pipeline.fit(X_filtered, y_filtered)

# Predict with the filtered model
y_filtered_pred = pipeline.predict(X_filtered)

# Re-evaluate the model
mse_filtered = mean_squared_error(y_filtered, y_filtered_pred)
rmse_filtered = np.sqrt(mse_filtered)
mae_filtered = mean_absolute_error(y_filtered, y_filtered_pred)
r2_filtered = r2_score(y_filtered, y_filtered_pred)

print('Filtered Model Performance:')
print(f'Model MSE: {mse_filtered}')
print(f'Model RMSE: {rmse_filtered}')
print(f'Model MAE: {mae_filtered}')
print(f'Model R²: {r2_filtered}')

# Create a DataFrame with actual and predicted values for filtered data
filtered_results_df = X_filtered.copy()
filtered_results_df['Actual'] = y_filtered
filtered_results_df['Predicted'] = y_filtered_pred
filtered_results_df['Residual'] = filtered_results_df['Actual'] - filtered_results_df['Predicted']

# Save results to Excel
filtered_results_file_path = r'C:\Users\Owner\OneDrive\Desktop\New GBR\filtered_model_predictions.xlsx'
filtered_results_df.to_excel(filtered_results_file_path, index=False)

print(f"Filtered results saved to {filtered_results_file_path}")


# Feature importance extraction
best_model = pipeline.named_steps['regressor']
feature_importances = best_model.feature_importances_

# Extract feature names from the preprocessor
preprocessor_transformers = pipeline.named_steps['preprocessor'].transformers_
numeric_feature_names = ['Humidity', 'Temperature']
categorical_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
all_feature_names = numeric_feature_names + list(categorical_feature_names)

# Combine the importances of one-hot encoded features back to their original categorical features
importances_dict = {feature: 0 for feature in numeric_feature_names + categorical_features}
for feature, importance in zip(all_feature_names, feature_importances):
    # For numeric features, add the importance directly
    if feature in numeric_feature_names:
        importances_dict[feature] += importance
    # For categorical features, find the original feature name
    else:
        original_feature = feature.split('_')[0]
        importances_dict[original_feature] += importance

# Convert the dictionary to lists for plotting
combined_feature_names = list(importances_dict.keys())
combined_feature_importances = list(importances_dict.values())

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    'Feature': combined_feature_names,
    'Importance': combined_feature_importances
})

# Save feature importances to Excel
importances_file_path = r'C:\Users\Owner\OneDrive\Desktop\New GBR\feature_importances.xlsx'
feature_importances_df.to_excel(importances_file_path, index=False)

print(f"Feature importances saved to {importances_file_path}")

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Feature Importance in Gradient Boosting Regressor', fontsize=16, fontweight='bold')
plt.show()



