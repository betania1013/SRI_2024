import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sns
import itertools

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

# Logistic Regression with L1 regularization
logreg_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42)

# Train Logistic Regression on training data
logreg_l1.fit(X_train, y_train)


# Predict on validation set
y_val_pred = logreg_l1.predict(X_val)  # Predicted labels
y_val_prob = logreg_l1.predict_proba(X_val)  # Predicted probabilities

# Evaluate performance on validation set
print("Validation Set:")
print(classification_report(y_val, y_val_pred))

# Feature Importance Plot
# Get feature importance using coefficients (magnitude)
feature_importance = np.abs(logreg_l1.coef_[0])
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


# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
#Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_val, y_val_prob[:, 1])
print("AUC-ROC Score:", auc_roc)
#Print 10-fold cross validation AUC-ROC Scores
cv_scores = cross_val_score(logreg_l1, X_train, y_train, cv=10, scoring='roc_auc')
print("Mean AUC-ROC score:", np.mean(cv_scores))
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

cnf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['OC', 'BOT'], title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix')