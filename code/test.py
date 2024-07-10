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

# Read the CSV files into DataFrames
csv_file_path = '../data/Raw Data ALL.csv'
raw_df = pd.read_csv(csv_file_path)

# Print info about data types & null values for all columns
raw_df.info()
raw_df.describe()


# Check for null values in raw data
null_rows = raw_df[raw_df.isna().any(axis=1)]
#print(null_rows)

# Prepare training data
X = raw_df.drop(columns=['TYPE', 'SUBJECT_ID'])
y = raw_df['TYPE']

# Encode categorical columns
columns_to_clean = ['AFP', 'CA19-9', 'CA125']

for col in columns_to_clean:
    X[col] = pd.to_numeric(X[col].apply(lambda x: re.sub(r'\\t', '', x) if isinstance(x, str) else x), errors='coerce')
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

imp = SimpleImputer(strategy='mean')
imp.fit(X_train)


# Transform training and testing data
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# Ensure test data is also imputed with the same imputer
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)
# Model 1: Logistic Regression with L1 regularization
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1'],  # L1 regularization (lasso)
    'solver': ['saga'],  
    'max_iter': [10000], 
    'random_state': [42]  
}

logreg_l1 = LogisticRegression()

# Train Logistic Regression on training data
grid_search = GridSearchCV(estimator=logreg_l1, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_logreg = grid_search.best_estimator_
print("Best parameters for Logistic Regression:", grid_search.best_params_)
best_logreg.fit(X_train, y_train)
# Predict on test set
y_test_pred = best_logreg.predict(X_test) 
y_test_prob = best_logreg.predict_proba(X_test)  

print("Test Set - Logistic Regression:")
print(classification_report(y_test, y_test_pred))

#Model 2: Graident Boosting Machine

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

                    
xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42)
xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
print("Best parameters for XGBoost:", xgb_grid_search.best_params_)
#feature selection
selector = SelectFromModel(best_xgb, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_test)
best_xgb.fit(X_train_selected, y_train)

y_test_pred_xgb = best_xgb.predict(X_val_selected)  
y_test_prob_xgb = best_xgb.predict_proba(X_val_selected)[:, 1]

print("Test Set - XGBoost:")
print(classification_report(y_test, y_test_pred_xgb))

## Model 3: Artifical Neural Network
from tensorflow.keras.models import load_model

# Load the saved model
best_mlp = load_model('bestmodel.keras')

# Predict on test set using the loaded MLP model
y_test_pred_mlp = best_mlp.predict(X_test)
y_test_pred_mlp = (y_test_pred_mlp > 0.5).astype(int) 
print("Test Set - MLP:")
print(classification_report(y_test, y_test_pred_mlp))

#Model 4: Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

# Perform feature selection based on importance
selector = SelectFromModel(estimator=rf)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected_rf = selector.transform(X_test)


# Define parameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'random_state': [42]
}

# Perform GridSearchCV 
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X_train_selected, y_train)

# Get the best model
best_rf = rf_grid_search.best_estimator_
print("Best parameters for Random Forest:", rf_grid_search.best_params_)


# Predict on test set
y_test_pred_rf = best_rf.predict(X_test_selected_rf)
y_test_prob_rf = best_rf.predict_proba(X_test_selected_rf)[:, 1]

# Print classification report
print("Test Set - Random Forest:")
print(classification_report(y_test, y_test_pred_rf))

# Feature Importance Plot
feature_importance = np.abs(best_logreg.coef_[0])
sorted_idx = np.argsort(feature_importance)[::-1] 

# Select top 20 features
top20_feature_indices = sorted_idx[:20]
top20_feature_names = X.columns[top20_feature_indices]
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
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
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
auc_roc = roc_auc_score(y_test, y_test_prob[:, 1])
print("AUC-ROC Score:", auc_roc)
#Print 10-fold cross validation AUC-ROC Scores
cv_scores = cross_val_score(logreg_l1, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score-Logistic Regression:", np.mean(cv_scores))

# ROC Curve and AUC for XGBoost 
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob_xgb)
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
auc_roc = roc_auc_score(y_test, y_test_prob_xgb)
print("AUC-ROC Score-XGBoost:", auc_roc)
#Print 10-fold cross validation AUC-ROC Scores
cv_scores_xgb = cross_val_score(xgb_clf, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score-XGBoost:", np.mean(cv_scores_xgb))

# ROC Curve and AUC for MLP 
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_mlp)
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
auc_roc = roc_auc_score(y_test, y_test_pred_mlp)
print("AUC-ROC Score-MLP:", auc_roc)


# ROC Curve and AUC for RF 
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-RF')
plt.legend(loc="lower right")
plt.show()
# Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_test_prob_rf)
print("AUC-ROC Score-RF:", auc_roc)





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
cnf_matrix_logreg = confusion_matrix(y_test, y_test_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix_logreg, classes=['OC', 'BOT'], title='Confusion matrix - Logistic Regression')
plt.figure()
plot_confusion_matrix(cnf_matrix_logreg, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - Logistic Regression')

# Confusion Matrix for XGBoost
cnf_matrix_gb = confusion_matrix(y_test, y_test_pred_xgb)
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], title='Confusion matrix - XGBoost')
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - XGBoost')
# Confusion Matrix for MLP
cnf_matrix_gb = confusion_matrix(y_test, y_test_pred_mlp)
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], title='Confusion matrix - MLP')
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - MLP')
#Confusion Matrix for RF
cnf_matrix_gb = confusion_matrix(y_test, y_test_pred_rf)
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], title='Confusion matrix - RF')
plt.figure()
plot_confusion_matrix(cnf_matrix_gb, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix - RF')

plt.show()


# SHAP Analysis for Logistic Regression
shap.initjs()

explainer = shap.Explainer(best_logreg, X_train)
shap_values = explainer(X_test)

# Summary Plot
plt.title("SHAP Summary Plot - Logistic Regression")
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# SHAP Analysis for Gradient Boosting
explainer_xgb = shap.TreeExplainer(best_xgb)

# Summary Plot
shap_values_xgb = explainer_xgb(X_val_selected)  
plt.title("SHAP Summary Plot - XGBoost")

shap.summary_plot(shap_values_xgb, X_val_selected, feature_names=X.columns)

# SHAP Analysis for MLP
explainer_mlp = shap.Explainer(best_mlp, X_train)

# Summary Plot
shap_values_mlp = explainer_mlp(X_test)  
plt.title("SHAP Summary Plot - MLP")

shap.summary_plot(shap_values_mlp, X_test, feature_names=X.columns)

# SHAP Analysis for RF
explainer_rf = shap.TreeExplainer(best_rf)

# Compute SHAP values
shap_values_rf = explainer_rf.shap_values(X_test_selected_rf)

# Summary Plot
plt.title("SHAP Summary Plot - Random Forest")
shap.summary_plot(shap_values_rf, X_test_selected_rf, feature_names=X.columns)
plt.show()
