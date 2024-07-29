from tkinter import Y
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import itertools
import shap
# Load and prepare the data
csv_file_path = '../data/Raw Data ALL.csv'
raw_df = pd.read_csv(csv_file_path)


# Extract features and target
X = raw_df.drop(columns=['TYPE', 'SUBJECT_ID'])
y = raw_df['TYPE']
col_clean = ['AFP', 'CA19-9', 'CA125']

for col in col_clean:
    X[col] = X[col].apply(lambda x: re.sub(r'><', '', x) if isinstance(x, str) else x)
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Initialize the imputer and scaler as used during training
imp = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Fit the imputer and scaler on the original training data
imp.fit(X)
scaler.fit(X)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


#Train data on only selected biomarkers. In this instance, we are training the data on a combination of HE4 and CEA
X_train_selected = X_train[[ 'CEA', 'HE4']]
X_test_selected_rf = X_test[[ 'CEA', 'HE4']]


# Define parameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'random_state': [42]
}
#GridSearchCV
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X_train_selected, y_train)

best_rf = rf_grid_search.best_estimator_
print("Best parameters for Random Forest:", rf_grid_search.best_params_)


# Predict on test set
y_test_pred_rf = best_rf.predict(X_test_selected_rf)
y_test_prob_rf = best_rf.predict_proba(X_test_selected_rf)[:, 1]

# Print classification report
print("Biomarker Classification:")
print(classification_report(y_test, y_test_pred_rf))

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_rf).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
roc_auc = roc_auc_score(y_test, y_test_prob_rf).ravel()
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")