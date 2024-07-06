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

# Read the CSV files into DataFrames
csv_file_path = '../data/Raw Data ALL.csv'
raw_df = pd.read_csv(csv_file_path)


train_df = pd.read_csv('../data/raw train data.csv')
test_df = pd.read_csv('../data/raw test data.csv')

# Print info about data types & null values for all columns
raw_df.info()
raw_df.describe()
train_df.info()
test_df.info()

# Check for null values in training data
null_rows = train_df[train_df.isna().any(axis=1)]
#print(null_rows)

# Define target column
target_column = 'TYPE'

# Prepare training data
X_train_full = train_df.drop(columns=[target_column, 'SUBJECT_ID'])
y_train_full = train_df[target_column]
y_test_full = test_df[target_column]
X_test_full= test_df.drop(columns=[target_column, 'SUBJECT_ID'])
# Encode categorical columns
columns_to_clean = ['AFP', 'CA19-9', 'CA125']

# Loop through each column and clean/convert both training and test data
for col in columns_to_clean:
    # Clean and convert training data
    X_train_full[col] = pd.to_numeric(X_train_full[col].apply(lambda x: re.sub(r'\\t', '', x) if isinstance(x, str) else x), errors='coerce')
    
    # Clean and convert test data
    X_test_full[col] = pd.to_numeric(X_test_full[col].apply(lambda x: re.sub(r'\\t', '', x) if isinstance(x, str) else x), errors='coerce')

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)


# Imputer to fill missing values with mean value (fit on training data only)
imp = SimpleImputer(strategy='mean')
imp.fit(X_train)


# Transform training and validation data
X_train = imp.transform(X_train)
X_val = imp.transform(X_val)

# Ensure test data is also imputed with the same imputer
X_test_full = imp.transform(X_test_full)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val= scaler.transform(X_val)
X_test_full= scaler.transform(X_test_full)
# Model 1: Logistic Regression with L1 regularization
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter values to test
    'penalty': ['l1'],  # L1 regularization (lasso)
    'solver': ['saga'],  # Algorithm for optimization
    'max_iter': [10000],  # Maximum number of iterations
    'random_state': [42]  # Random state for reproducibility
}

logreg_l1 = LogisticRegression()

# Train Logistic Regression on training data
grid_search = GridSearchCV(estimator=logreg_l1, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_logreg = grid_search.best_estimator_
print("Best parameters for Logistic Regression:", grid_search.best_params_)
best_logreg.fit(X_train, y_train)
# Predict on validation set
y_val_pred = best_logreg.predict(X_val)  # Predicted labels
y_val_prob = best_logreg.predict_proba(X_val)  # Predicted probabilities

print("Validation Set - Logistic Regression:")
print(classification_report(y_val, y_val_pred))

#Model 2: Graident Boosting Machine

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

            
            
xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42)
xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
print("Best parameters for XGBoost:", xgb_grid_search.best_params_)
#feature selection
selector = SelectFromModel(best_xgb, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
best_xgb.fit(X_train_selected, y_train)

y_val_pred_xgb = best_xgb.predict(X_val_selected)  
y_val_prob_xgb = best_xgb.predict_proba(X_val_selected)[:, 1]

print("Validation Set - XGBoost:")
print(classification_report(y_val, y_val_pred_xgb))

## Model 3: Artifical Neural Network

from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
# Neural Network with Grid Search and Feature Selection
mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50), (100,), (100, 100), (100, 50), (50, 50, 50), (200, 100), (200, 200), (300,200),(300,300), (300,100)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [500, 1000, 2000],
    'early_stopping': [True],
    'validation_fraction': [0.1, 0.15, 0.2]
}

mlp_clf = MLPClassifier(random_state=42)
mlp_grid_search = GridSearchCV(estimator=mlp_clf, param_grid=mlp_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)

mlp_grid_search.fit(X_train, y_train)

best_mlp = mlp_grid_search.best_estimator_
perm_importance = permutation_importance(best_mlp, X_val, y_val, n_repeats=10, random_state=42, scoring='roc_auc')

print("Best parameters for MLPClassifier:", mlp_grid_search.best_params_)


# Set a threshold for selecting features based on permutation importance scores
threshold = 0.001 

# Select features based on the threshold
selected_mlp =np.where(perm_importance.importances_mean > threshold)[0]
# Filter data to include only selected features
X_train_mlp = X_train[:,selected_mlp]
X_val_mlp= X_val[:,selected_mlp]
X_test_mlp = X_test_full[:,selected_mlp]

# Train MLPClassifier on selected features
best_mlp.fit(X_train_mlp, y_train)

# Predict on validation set
y_val_pred_mlp = best_mlp.predict(X_val_mlp)
y_val_prob_mlp = best_mlp.predict_proba(X_val_mlp)[:, 1]


# Feature Importance Plot
# Get feature importance using coefficients (magnitude)
feature_importance = np.abs(best_logreg.coef_[0])
sorted_idx = np.argsort(feature_importance)[::-1]  # Sort feature indices by importance (descending)

# Select top 20 features
top20_feature_indices = sorted_idx[:20]
top20_feature_names = X_train_full.columns[top20_feature_indices]
top20_feature_importance = feature_importance[top20_feature_indices]

# Plotting top 20 features based on importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(top20_feature_indices)), top20_feature_importance, align='center')
plt.yticks(range(len(top20_feature_indices)), top20_feature_names, fontsize=12)
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()



# ROC Curve and AUC for logistic regression
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-Logistic Regression')
plt.legend(loc="lower right")
plt.show()
#Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_val, y_val_prob[:, 1])
print("AUC-ROC Score:", auc_roc)
#Print 10-fold cross validation AUC-ROC Scores
cv_scores = cross_val_score(logreg_l1, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score-Logistic Regression:", np.mean(cv_scores))

# ROC Curve and AUC for XGBoost 
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob_xgb)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-XGBoost')
plt.legend(loc="lower right")
plt.show()
#Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_val, y_val_prob_xgb)
print("AUC-ROC Score-XGBoost:", auc_roc)
#Print 10-fold cross validation AUC-ROC Scores
cv_scores_xgb = cross_val_score(xgb_clf, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score-XGBoost:", np.mean(cv_scores_xgb))

# ROC Curve and AUC for MLP 
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob_mlp)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-MLP')
plt.legend(loc="lower right")
plt.show()
# Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_val, y_val_prob_mlp)
print("AUC-ROC Score-MLP:", auc_roc)
# Print 10-fold cross validation AUC-ROC Scores
cv_scores_mlp = cross_val_score(best_mlp, X_train_mlp, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score-MLP:", np.mean(cv_scores_mlp))





# Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Confusion Matrix for Logistic Regression
cnf_matrix_logreg = confusion_matrix(y_val, y_val_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix_logreg, classes=['OC', 'BOT'], title='Confusion matrix - Logistic Regression')
plt.figure()
plot_confusion_matrix(cnf_matrix_logreg, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - Logistic Regression')

# Confusion Matrix for XGBoost
cnf_matrix_gb = confusion_matrix(y_val, y_val_pred_xgb)
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], title='Confusion matrix - XGBoost')
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - XGBoost')
# Confusion Matrix for MLP
cnf_matrix_gb = confusion_matrix(y_val, y_val_pred_mlp)
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], title='Confusion matrix - MLP')
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - MLP')


plt.show()


# SHAP Analysis for Logistic Regression
shap.initjs()

explainer = shap.Explainer(best_logreg, X_train)
shap_values = explainer(X_val)

# Summary Plot
plt.title("SHAP Summary Plot - Logistic Regression")
shap.summary_plot(shap_values, X_val, feature_names=X_train_full.columns)

# SHAP Analysis for Gradient Boosting
shap.initjs()
explainer_xgb = shap.TreeExplainer(best_xgb)

# Summary Plot
shap_values_xgb = explainer_xgb(X_val_selected)  
plt.title("SHAP Summary Plot - XGBoost")

shap.summary_plot(shap_values_xgb, X_val_selected, feature_names=X_train_full.columns)

# SHAP Analysis for MLP
shap.initjs()
explainer_mlp = shap.DeepExplainer(best_mlp)

# Summary Plot
shap_values_mlp = explainer_mlp(X_val_mlp)  
plt.title("SHAP Summary Plot - MLP")

shap.summary_plot(shap_values_mlp, X_val_mlp, feature_names=X_train_full.columns)
plt.show()