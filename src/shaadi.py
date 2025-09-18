#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np


# In[21]:


file_path = r'C:\Users\Admin\OneDrive\Documents\GitHub\shaadi\data\Dataset.txt'
df = pd.read_csv(file_path, sep='\t', index_col=0)
print(df.head())


# In[22]:


df.shape


# In[23]:


df.info()


# In[24]:


# view object columns

print(df.F15)


# In[25]:


print(df.F16)


# In[26]:


pd.set_option('display.max_columns', None)


# In[27]:


df.head()


# In[28]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F22.value_counts().index, df.F22.value_counts().values)


# In[29]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F21.value_counts().index, df.F21.value_counts().values)


# In[30]:


df.F21.value_counts().values


# In[31]:


df.F22.value_counts().values


# In[32]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F17.value_counts().index, df.F17.value_counts().values)


# In[33]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F18.value_counts().index, df.F18.value_counts().values)


# In[34]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F20.value_counts().index, df.F20.value_counts().values)


# In[35]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.F19.value_counts().index, df.F19.value_counts().values)


# In[36]:


import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_histograms(dataframe):
    """
    Selects features from F1 to F16, excludes object types,
    and plots a histogram for each.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
    """
    # Create a list of column names from 'F1' to 'F16'
    feature_columns = [f'F{i}' for i in range(1, 24)]
    
    # Select only the columns that exist in the dataframe and are not of object type
    numeric_features = [
        col for col in feature_columns 
        if col in dataframe.columns and dataframe[col].dtype != 'object'
    ]
    
    # Determine the grid size for the subplots
    num_features = len(numeric_features)
    num_cols = 3  # Set a fixed number of columns for the grid
    num_rows = (num_features + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten() # Flatten the grid to make it easy to iterate over

    print("Generating histograms for numeric features from F1 to F16...")
    
    # Loop through each numeric feature and create a histogram
    for i, col in enumerate(numeric_features):
        ax = axes[i]
        dataframe[col].hist(ax=ax, bins=50)
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()

# --- Call the function with your DataFrame ---
# Assuming 'df' is your loaded DataFrame from the notebook
plot_feature_histograms(df)


# In[37]:


plt.scatter(df.F17, df.F18)


# In[38]:


plt.scatter(df.F19, df.F20)


# In[39]:


# plot histogram for all numerical features

import matplotlib.pyplot as plt
 
plt.bar(df.C.value_counts().index, df.C.value_counts().values)


# In[40]:


pip install sklearn


# In[41]:


from sklearn.svm import SVC


# In[42]:


model = SVC()


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X = df.drop(columns=['C', 'F15', 'F16'])
y = df[['C']]


# In[45]:


X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)


# In[46]:


X_train


# In[47]:


y_train


# In[48]:


model.fit(X_train, y_train)


# In[49]:


X_test


# In[50]:


y_test


# In[51]:


y_pred = model.predict(X_test)


# In[52]:


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[53]:


f1_score(y_test, y_pred)


# In[54]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[55]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Drop the date columns and the target variable
X = df.drop(columns=['C', 'F15', 'F16'])
y = df['C']

# Split the data first to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Create a scaler and fit it ONLY on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the SAME scaler
X_test_scaled = scaler.transform(X_test)

# Train the model on the SCALED training data
model = SVC()
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
print(f"F1 Score after scaling: {f1_score(y_test, y_pred)}")


# In[56]:


model.classes_


# In[57]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[67]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# 5. Evaluate the Model
y_pred = rf_model.predict(X_test_scaled)

print(" --- Random Forest Classification Report --- ")
print(classification_report(y_test, y_pred))

# 6. Display the confusion matrix
print("\n--- Confusion Matrix ---")
cm    = confusion_matrix(y_test, y_pred)
disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.show()


# In[68]:


# Add the class_weight='balanced' parameter
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # This tells the model to weigh classes inversely to their frequency
    random_state=42,
    n_jobs=-1
)

# ... then fit and evaluate as before ...
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))


# In[69]:


# Display the confusion matrix
print("\n --- Confusion Matrix --- ")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.show()


# In[71]:


pip install imblearn


# In[62]:


from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assume X_train_scaled, X_test_scaled, y_train, and y_test are already created

# 1. Apply SMOTE to the training data
# This creates new synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# You can check the new class distribution
print(f"Original training set distribution: {Counter(y_train)}")
print(f"Resampled training set distribution: {Counter(y_train_resampled)}")

# 2. Train a new model on the BALANCED (resampled) data
rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_smote.fit(X_train_resampled, y_train_resampled)

# 3. Evaluate on the ORIGINAL, untouched test data
y_pred_smote = rf_model_smote.predict(X_test_scaled)

print("\n--- Classification Report after SMOTE ---")
print(classification_report(y_test, y_pred_smote))


# 
# 
# ## Next Steps for Further Improvement
# 
# An F1-score of 0.27 for the minority class is a good start, but there's still room to improve. Here are the best next steps.
# 
# 1. Hyperparameter Tuning (Most Recommended Next Step)
# 
# Now that you have a balanced training set, you can fine-tune the Random Forest model's parameters (hyperparameters) to find the best combination for your data. The default settings are rarely optimal.

# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


from sklearn.model_selection import RandomizedSearchCV

# Define a grid of parameters to test
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a base model
rf = RandomForestClassifier(random_state=42)

# # Instantiate the randomized search model
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
#                                n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1,
#                                scoring='f1') # Optimize for f1-score

from sklearn.metrics import make_scorer, recall_score

# Define a custom scoring function for minority class recall
minority_class = 1  # Replace with the label of your minority class
recall_minority = make_scorer(recall_score, pos_label=minority_class)

# Instantiate the randomized search model
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1,
                               scoring=recall_minority)  # Optimize for minority class recall

# Fit the random search model (using your resampled data)
rf_random.fit(X_train_resampled, y_train_resampled)

# Print the best parameters found
print(f"Best parameters found: {rf_random.best_params_}")

# Evaluate the best model
best_rf_model = rf_random.best_estimator_
y_pred_tuned = best_rf_model.predict(X_test_scaled)
print("\n--- Classification Report after Tuning ---")
print(classification_report(y_test, y_pred_tuned))


# In[75]:


pip install xgboost


# In[29]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Use the same resampled training data from SMOTE
# X_train_resampled, y_train_resampled

# Initialize and train the XGBoost model
# You can also tune XGBoost's hyperparameters later
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the XGBoost model on the untouched test set
y_pred_xgb = xgb_model.predict(X_test_scaled)

print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb))


# In[29]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# 1. Define the parameter grid for XGBoost
# These are some of the most important parameters to tune
param_grid_xgb = {
    
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# 2. Create a base XGBoost model
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# 3. Instantiate the randomized search model
# n_iter controls how many different parameter combinations are tried.
# Scoring='f1' is crucial for imbalanced datasets.
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=50,  # You can increase this for a more thorough search
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

# 4. Fit the model on your resampled training data
# This step might take some time
xgb_random.fit(X_train_resampled, y_train_resampled)

# 5. Print the best parameters found
print(f"Best parameters found for XGBoost: {xgb_random.best_params_}")

# 6. Evaluate the best model on the test set
best_xgb_model = xgb_random.best_estimator_
y_pred_tuned_xgb = best_xgb_model.predict(X_test_scaled)

print("\n--- XGBoost Classification Report after Tuning ---")
print(classification_report(y_test, y_pred_tuned_xgb))


# In[30]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get the predicted probabilities for the positive class (1)
y_probs = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()


# In[31]:


from sklearn.metrics import classification_report

# Get the predicted probabilities for the positive class (1)
y_probs = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

# --- SET YOUR NEW THRESHOLD HERE ---
new_threshold = 0.4

# Get new predictions based on the new threshold
y_pred_new_threshold = (y_probs >= new_threshold).astype(int)

# Evaluate the model with the new threshold
print(f"--- Classification Report with Threshold = {new_threshold} ---")
print(classification_report(y_test, y_pred_new_threshold))


# In[32]:


import joblib

# Save the trained model
joblib.dump(best_xgb_model, 'xgb_model.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved!")


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_pairs(dataframe, feature_pairs):
    """
    Generates scatter plots for specified pairs of features in a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the training data (e.g., X_train).
        feature_pairs (list of tuples): A list where each tuple contains two feature names
                                        to be plotted against each other.
    """
    # Determine the number of plots to create a grid
    num_plots = len(feature_pairs)
    # Arrange plots in a grid, with a maximum of 2 columns
    num_cols = min(num_plots, 2)
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 6 * num_rows))
    # Flatten the axes array for easy iteration, handling the single plot case
    axes = axes.flatten() if num_plots > 1 else [axes]

    print("Generating scatter plots for feature pairs...")

    # Loop through each pair and create a scatter plot
    for i, (feat1, feat2) in enumerate(feature_pairs):
        ax = axes[i]
        # Using a sample of the data to avoid overplotting if the dataset is large
        sample_df = dataframe.sample(n=min(1000, len(dataframe)), random_state=42)
        ax.scatter(sample_df[feat1], sample_df[feat2], alpha=0.5)
        ax.set_title(f'Scatter Plot of {feat1} vs. {feat2}', fontsize=12)
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# --- Example of how to use the function ---

# First, ensure X_train is a DataFrame (it should be if you followed previous steps)
# If it's a NumPy array, convert it back:
# X_train_df = pd.DataFrame(X_train, columns=X.columns)

# Define the pairs of features you suspect are correlated
suspected_pairs = [
    ('F19', 'F20'),
    ('F17', 'F18'),
    ('F21', 'F22')
]

# Call the function with your training data and the list of pairs
# Make sure X_train is a pandas DataFrame
plot_feature_pairs(X_train, suspected_pairs)


# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume 'df' is your original loaded DataFrame

# 1. Select only the numerical features from the original DataFrame
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# 2. Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# 3. Create the heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=.5
)
plt.title('Correlation Heatmap of Original Numerical Features', fontsize=16)
plt.show()


# hat's an excellent insight from the visualizations. Your observation is a key step in effective feature engineering.
# 
# When two features have similar distributions and a negative correlation, it often means they are capturing similar underlying information but in opposite directions. For example, they could be measuring the same concept from two different perspectives, where an increase in one is associated with a decrease in the other.
# 
# Keeping both features in the model can introduce redundancy (multicollinearity), which can sometimes make the model less stable or harder to interpret. A great way to handle this is to combine each pair of correlated features into a single, more powerful feature using Principal Component Analysis (PCA).

# In[37]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assume 'X_train', 'X_test', 'y_train', 'y_test' are from your original train_test_split

# --- 1. PCA Feature Engineering (on original splits) ---
pairs = [('F17', 'F18'), ('F21', 'F22')]
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

# Scale only the columns intended for PCA to start
pca_cols = [col for pair in pairs for col in pair]
scaler_pca = StandardScaler()
X_train_pca[pca_cols] = scaler_pca.fit_transform(X_train[pca_cols])
X_test_pca[pca_cols] = scaler_pca.transform(X_test[pca_cols])

# Apply PCA and create new dataframes
for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_train_pca[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_train_pca[[feat1, feat2]])
    X_test_pca[f'PCA_{feat1}_{feat2}'] = pca.transform(X_test_pca[[feat1, feat2]])
    X_train_pca = X_train_pca.drop(columns=[feat1, feat2])
    X_test_pca = X_test_pca.drop(columns=[feat1, feat2])

print("PCA feature engineering complete.")

# --- 2. Use Your Pre-existing Scaled and Resampled Data ---
# **This is where the error occurred.** You must re-run scaling and SMOTE
# on the new dataframes that include the PCA features.

# Re-run Scaling on the data with PCA features
final_scaler = StandardScaler()
X_train_scaled_new = final_scaler.fit_transform(X_train_pca)
X_test_scaled_new = final_scaler.transform(X_test_pca)

# Re-run SMOTE on the new scaled training data
smote = SMOTE(random_state=42)
X_train_resampled_new, y_train_resampled_new = smote.fit_resample(X_train_scaled_new, y_train)
print("SMOTE resampling complete on new feature set.")


# --- 3. Train and Evaluate XGBoost Model ---
xgb_model_pca = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# Fit on the newly created resampled data
xgb_model_pca.fit(X_train_resampled_new, y_train_resampled_new)

# Predict on the new scaled test data
y_pred_pca = xgb_model_pca.predict(X_test_scaled_new)

# Evaluate the results
print("\n--- Classification Report after PCA ---")
print(classification_report(y_test, y_pred_pca))

print("\n--- Confusion Matrix ---")
cm_pca = confusion_matrix(y_test, y_pred_pca)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=xgb_model_pca.classes_)
disp.plot()
plt.show()


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc

# --- 1. Define the Hyperparameter Grid for XGBoost ---
# This grid covers the most important parameters to tune
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# --- 2. Perform Randomized Search ---
# We optimize for the 'f1' score, which is ideal for imbalanced datasets
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,       # Number of cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

# Fit the model on your final resampled training data
# Assumes X_train_resampled_new and y_train_resampled_new are available
xgb_random.fit(X_train_resampled_new, y_train_resampled_new)

# Print the best parameters found
print(f"Best parameters found for XGBoost: {xgb_random.best_params_}")

# Use the best model found by the search
best_xgb_model = xgb_random.best_estimator_

# --- 3. Evaluate the Best Model ---
# Predict on the final scaled test data (with PCA features)
y_pred_tuned = best_xgb_model.predict(X_test_scaled_new)
print("\n--- Tuned XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_tuned))


# --- 4. Plot the Precision-Recall Curve ---
# Get the predicted probabilities for the positive class (1)
y_probs = best_xgb_model.predict_proba(X_test_scaled_new)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calculate the area under the curve (AUC-PR)
pr_auc = auc(recall, precision)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Tuned XGBoost Model')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# In[42]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assume 'X_train', 'X_test', 'y_train', 'y_test' are from your original train_test_split

# --- 1. PCA Feature Engineering (on original splits) ---
pairs = [('F17', 'F19'), ('F20', 'F18')]
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

# Scale only the columns intended for PCA to start
pca_cols = [col for pair in pairs for col in pair]
scaler_pca = StandardScaler()
X_train_pca[pca_cols] = scaler_pca.fit_transform(X_train[pca_cols])
X_test_pca[pca_cols] = scaler_pca.transform(X_test[pca_cols])

# Apply PCA and create new dataframes
for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_train_pca[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_train_pca[[feat1, feat2]])
    X_test_pca[f'PCA_{feat1}_{feat2}'] = pca.transform(X_test_pca[[feat1, feat2]])
    X_train_pca = X_train_pca.drop(columns=[feat1, feat2])
    X_test_pca = X_test_pca.drop(columns=[feat1, feat2])

print("PCA feature engineering complete.")

# --- 2. Use Your Pre-existing Scaled and Resampled Data ---
# **This is where the error occurred.** You must re-run scaling and SMOTE
# on the new dataframes that include the PCA features.

# Re-run Scaling on the data with PCA features
final_scaler = StandardScaler()
X_train_scaled_new = final_scaler.fit_transform(X_train_pca)
X_test_scaled_new = final_scaler.transform(X_test_pca)

# Re-run SMOTE on the new scaled training data
smote = SMOTE(random_state=42)
X_train_resampled_new, y_train_resampled_new = smote.fit_resample(X_train_scaled_new, y_train)
print("SMOTE resampling complete on new feature set.")


# --- 3. Train and Evaluate XGBoost Model ---
xgb_model_pca = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# Fit on the newly created resampled data
xgb_model_pca.fit(X_train_resampled_new, y_train_resampled_new)

# Predict on the new scaled test data
y_pred_pca = xgb_model_pca.predict(X_test_scaled_new)

# Evaluate the results
print("\n--- Classification Report after PCA ---")
print(classification_report(y_test, y_pred_pca))

print("\n--- Confusion Matrix ---")
cm_pca = confusion_matrix(y_test, y_pred_pca)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=xgb_model_pca.classes_)
disp.plot()
plt.show()


# In[43]:


import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc

# --- 1. Define the Hyperparameter Grid for XGBoost ---
# This grid covers the most important parameters to tune
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# --- 2. Perform Randomized Search ---
# We optimize for the 'f1' score, which is ideal for imbalanced datasets
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,       # Number of cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

# Fit the model on your final resampled training data
# Assumes X_train_resampled_new and y_train_resampled_new are available
xgb_random.fit(X_train_resampled_new, y_train_resampled_new)

# Print the best parameters found
print(f"Best parameters found for XGBoost: {xgb_random.best_params_}")

# Use the best model found by the search
best_xgb_model = xgb_random.best_estimator_

# --- 3. Evaluate the Best Model ---
# Predict on the final scaled test data (with PCA features)
y_pred_tuned = best_xgb_model.predict(X_test_scaled_new)
print("\n--- Tuned XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_tuned))


# --- 4. Plot the Precision-Recall Curve ---
# Get the predicted probabilities for the positive class (1)
y_probs = best_xgb_model.predict_proba(X_test_scaled_new)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calculate the area under the curve (AUC-PR)
pr_auc = auc(recall, precision)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Tuned XGBoost Model')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# In[44]:


# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Assume 'df' is your original loaded DataFrame

# --- 1. Datetime Feature Engineering ---
df_featured = df.copy()
# Convert date columns to datetime objects
date_cols = ['F15', 'F16']

for col in date_cols:
    df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
    df_featured[f'{col}_year'] = df_featured[col].dt.year
    df_featured[f'{col}_month'] = df_featured[col].dt.month
    df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek

df_featured['F15_dayofweek'] = df_featured['F15'].dt.dayofweek
df_featured['date_diff_days'] = (df_featured['F16'] - df_featured['F15']).dt.days

print("Datetime feature engineering complete.")

# --- 2. Initial Data Preparation ---
X = df_featured.drop(columns=['C', 'F15', 'F16']) # Drop original dates and target
y = df_featured['C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# --- 3. PCA Feature Engineering ---
pairs = [('F17', 'F18'), ('F21', 'F22')]
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

pca_cols = [col for pair in pairs for col in pair]
scaler_pca = StandardScaler()
X_train_pca[pca_cols] = scaler_pca.fit_transform(X_train[pca_cols])
X_test_pca[pca_cols] = scaler_pca.transform(X_test[pca_cols])

for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_train_pca[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_train_pca[[feat1, feat2]])
    X_test_pca[f'PCA_{feat1}_{feat2}'] = pca.transform(X_test_pca[[feat1, feat2]])
    X_train_pca = X_train_pca.drop(columns=[feat1, feat2])
    X_test_pca = X_test_pca.drop(columns=[feat1, feat2])

print("PCA feature engineering complete.")

# --- 4. Final Scaling ---
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_pca)
X_test_scaled = final_scaler.transform(X_test_pca)

# --- 5. SMOTE Resampling ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("SMOTE resampling complete.")

# --- 6. XGBoost Hyperparameter Tuning ---
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=25, # Using a smaller number of iterations for speed
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

xgb_random.fit(X_train_resampled, y_train_resampled)
best_xgb_model = xgb_random.best_estimator_
print(f"Best parameters found: {xgb_random.best_params_}")

# --- 7. Final Evaluation ---
y_pred_final = best_xgb_model.predict(X_test_scaled)
print("\n--- Final Classification Report ---")
print(classification_report(y_test, y_pred_final))


# --- 8. Feature Importance ---
# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({
    'feature': X_train_pca.columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('XGBoost Feature Importance', fontsize=16)
plt.show()


# In[47]:


print(feature_importances)


# In[53]:


df.head()


# In[61]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Assume 'df' is your original loaded DataFrame

# --- 1. Datetime Feature Engineering ---
df_featured = df.copy()
# Convert date columns to datetime objects
date_cols = ['F15', 'F16']

for col in date_cols:
    df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
    df_featured[f'{col}_year'] = df_featured[col].dt.year
    df_featured[f'{col}_month'] = df_featured[col].dt.month
    df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek

df_featured['F15_dayofweek'] = df_featured['F15'].dt.dayofweek
df_featured['date_diff_days'] = (df_featured['F16'] - df_featured['F15']).dt.days

print("Datetime feature engineering complete.")

# --- 2. Initial Data Preparation ---
X = df_featured.drop(columns=['C', 'F15', 'F16']) # Drop original dates and target
y = df_featured['C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# --- 3. PCA Feature Engineering ---
pairs = [('F17', 'F19'), ('F18', 'F20')]
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

pca_cols = [col for pair in pairs for col in pair]
scaler_pca = StandardScaler()
X_train_pca[pca_cols] = scaler_pca.fit_transform(X_train[pca_cols])
X_test_pca[pca_cols] = scaler_pca.transform(X_test[pca_cols])

for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_train_pca[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_train_pca[[feat1, feat2]])
    X_test_pca[f'PCA_{feat1}_{feat2}'] = pca.transform(X_test_pca[[feat1, feat2]])
    X_train_pca = X_train_pca.drop(columns=[feat1, feat2])
    X_test_pca = X_test_pca.drop(columns=[feat1, feat2])

print("PCA feature engineering complete.")

# --- 4. Final Scaling ---
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_pca)
X_test_scaled = final_scaler.transform(X_test_pca)

# --- 5. SMOTE Resampling ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("SMOTE resampling complete.")

# --- 6. XGBoost Hyperparameter Tuning ---
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid_xgb,
    n_iter=25, # Using a smaller number of iterations for speed
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

xgb_random.fit(X_train_resampled, y_train_resampled)
best_xgb_model = xgb_random.best_estimator_
print(f"Best parameters found: {xgb_random.best_params_}")

# --- 7. Final Evaluation ---
y_pred_final = best_xgb_model.predict(X_test_scaled)
print("\n--- Final Classification Report ---")
print(classification_report(y_test, y_pred_final))


# --- 8. Feature Importance ---
# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({
    'feature': X_train_pca.columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('XGBoost Feature Importance', fontsize=16)
plt.show()


# In[62]:


print(feature_importances)


# In[64]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Assume 'df' is your original loaded DataFrame

# --- 1. Perform ALL Feature Engineering Steps First ---
df_featured = df.copy()
date_cols = ['F15', 'F16']
for col in date_cols:
    df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
    df_featured[f'{col}_year'] = df_featured[col].dt.year
    df_featured[f'{col}_month'] = df_featured[col].dt.month
    df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek
df_featured['date_diff_days'] = (df_featured['F16'] - df_featured['F15']).dt.days

X_full = df_featured.drop(columns=['C', 'F15', 'F16'])
y = df_featured['C']

pairs = [('F18', 'F20'), ('F17', 'F19')]
X_pca_full = X_full.copy()
for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_pca_full[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_full[[feat1, feat2]])
    X_pca_full = X_pca_full.drop(columns=[feat1, feat2])

# --- 2. Select Only the Important Features ---
important_features = [
    'PCA_F18_F20', 'PCA_F17_F19', 'F22', 'F15_dayofweek', 'F16_dayofweek',
    'F15_month', 'F16_month', 'F21', 'F4', 'F15_year', 'F14', 'F2', 'F9', 'F16_year'
]
X_important = X_pca_full[important_features]
print(f"Model will be trained on these {len(X_important.columns)} important features.")

# --- 3. Run Preprocessing on the Reduced Feature Set ---
X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# --- 4. Basic XGBoost Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=15, # Lower n_iter for a "basic" but still effective search
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

xgb_random.fit(X_train_resampled, y_train_resampled)
best_model = xgb_random.best_estimator_
print(f"Best parameters found: {xgb_random.best_params_}")

# --- 5. Evaluate the Final, Simplified, and Tuned Model ---
y_pred_final = best_model.predict(X_test_scaled)
print("\n--- Classification Report for Final Model ---")
print(classification_report(y_test, y_pred_final))


# In[78]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Assume 'df' is your original loaded DataFrame

# --- 1. Perform ALL Feature Engineering on the Entire Dataset ---
df_featured = df.copy()
# Datetime features
date_cols = ['F15', 'F16']
for col in date_cols:
    df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
    df_featured[f'{col}_year'] = df_featured[col].dt.year
    df_featured[f'{col}_month'] = df_featured[col].dt.month
    df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek
df_featured['date_diff_days'] = (df_featured['F16'] - df_featured['F15']).dt.days

X_full = df_featured.drop(columns=['C', 'F15', 'F16'])

# PCA features
pairs = [('F18', 'F20'), ('F17', 'F19')]
X_pca_full = X_full.copy()
for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_pca_full[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_full[[feat1, feat2]])
    X_pca_full = X_pca_full.drop(columns=[feat1, feat2])

# --- 2. Select the Important Features ---
important_features = [
    'PCA_F18_F20', 'PCA_F17_F19', 'F22', 'F15_dayofweek', 'F16_dayofweek',
    'F15_month', 'F16_month', 'F21', 'F4', 'F15_year', 'F14', 'F2', 'F9', 'F16_year'
]
X_important = X_pca_full[important_features]
print("Preprocessing complete. Final features selected.")

# --- 3. Scale the Final Feature Set ---
# We fit a new scaler on the entire dataset for the final step.
scaler = StandardScaler()
X_scaled_final = scaler.fit_transform(X_important)

# Note: SMOTE is not used here. It's a technique for the training/evaluation phase.
# The final model should be trained on the true data distribution.
y_full = df['C']

# --- 4. Train the Best Model on ALL Data ---
# Using the best hyperparameters you found previously
final_model =  xgb_random.best_estimator_

# print("Training final model on the entire dataset...")
# final_model.fit(X_scaled_final, y_full)
# print("Training complete.")

# --- 5. Generate Predictions for the Entire Dataset ---
predictions = final_model.predict(X_scaled_final)

# Add the predictions back to your original DataFrame for easy review
df['predictions'] = predictions
print("\nPredictions have been generated and added to the DataFrame.")

# Display the counts of predicted classes
print("\nPrediction Counts:")
print(df['predictions'].value_counts())

# Display the head of the DataFrame with the new predictions column
print("\nDataFrame with Predictions:")
print(df.head())


# In[79]:


print(classification_report(df.C, df.predictions))


# In[80]:


# Assume 'df' is the DataFrame with the 'predictions' column already added

# Select only the Index and the new 'predictions' column
prediction_output = df[['predictions']]

# Reset the index so that the original 'Index' column becomes a regular column
prediction_output = prediction_output.reset_index()

# Rename the columns to 'Index' and 'Class' as per your requirement
prediction_output.columns = ['Index', 'Class']

# Define the output file path
output_file_path = r'C:\Users\Admin\OneDrive\Documents\GitHub\shaadi\predictions\training_predictions.txt'

# Save the DataFrame to a tab-delimited text file, without the new pandas index
prediction_output.to_csv(output_file_path, sep='\t', index=False)

print(f"Predictions saved to {output_file_path}")


# In[81]:


file_path = r'C:\Users\Admin\OneDrive\Documents\GitHub\shaadi\data\Dataset_test.txt'
df = pd.read_csv(file_path, sep='\t', index_col=0)
print(df.head())


# In[82]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Assume 'df' is your original loaded DataFrame

# --- 1. Perform ALL Feature Engineering on the Entire Dataset ---
df_featured = df.copy()
# Datetime features
date_cols = ['F15', 'F16']
for col in date_cols:
    df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
    df_featured[f'{col}_year'] = df_featured[col].dt.year
    df_featured[f'{col}_month'] = df_featured[col].dt.month
    df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek
df_featured['date_diff_days'] = (df_featured['F16'] - df_featured['F15']).dt.days

X_full = df_featured.drop(columns=['F15', 'F16'])

# PCA features
pairs = [('F18', 'F20'), ('F17', 'F19')]
X_pca_full = X_full.copy()
for feat1, feat2 in pairs:
    pca = PCA(n_components=1)
    X_pca_full[f'PCA_{feat1}_{feat2}'] = pca.fit_transform(X_full[[feat1, feat2]])
    X_pca_full = X_pca_full.drop(columns=[feat1, feat2])

# --- 2. Select the Important Features ---
important_features = [
    'PCA_F18_F20', 'PCA_F17_F19', 'F22', 'F15_dayofweek', 'F16_dayofweek',
    'F15_month', 'F16_month', 'F21', 'F4', 'F15_year', 'F14', 'F2', 'F9', 'F16_year'
]
X_important = X_pca_full[important_features]
print("Preprocessing complete. Final features selected.")

# --- 3. Scale the Final Feature Set ---
# We fit a new scaler on the entire dataset for the final step.
scaler = StandardScaler()
X_scaled_final = scaler.fit_transform(X_important)

# --- 5. Generate Predictions for the Entire Dataset ---
predictions = final_model.predict(X_scaled_final)

# Add the predictions back to your original DataFrame for easy review
df['predictions'] = predictions
print("\nPredictions have been generated and added to the DataFrame.")

# Display the counts of predicted classes
print("\nPrediction Counts:")
print(df['predictions'].value_counts())

# Display the head of the DataFrame with the new predictions column
print("\nDataFrame with Predictions:")
print(df.head())


# In[83]:


# Assume 'df' is the DataFrame with the 'predictions' column already added

# Select only the Index and the new 'predictions' column
prediction_output = df[['predictions']]

# Reset the index so that the original 'Index' column becomes a regular column
prediction_output = prediction_output.reset_index()

# Rename the columns to 'Index' and 'Class' as per your requirement
prediction_output.columns = ['Index', 'Class']

# Define the output file path
output_file_path = r'C:\Users\Admin\OneDrive\Documents\GitHub\shaadi\predictions\test_predictions.txt'

# Save the DataFrame to a tab-delimited text file, without the new pandas index
prediction_output.to_csv(output_file_path, sep='\t', index=False)

print(f"Predictions saved to {output_file_path}")

