import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
csv_file_path = r"C:\Users\betan\Documents\Raw Data ALL.csv"
# Read the CSV file into a DataFrame
raw_df = pd.read_csv(csv_file_path)
#returns num of cols and rows in raw data
raw_df.shape
#display first few rows of raw data
raw_df.head()
#summary stats
raw_df.describe()
# info abt data types & null values for all columns
raw_df.info()

#training data
train_df = pd.read_csv(r"C:\Users\betan\Documents\raw train data.csv")
#print(train_df.info())
#check location of null values and see the whole row
null_rows=train_df[train_df.isna().any(axis=1)]
print(null_rows)
test_df=pd.read_csv(r"C:\Users\betan\Documents\raw test data.csv")


#import necessary libaries for handling missing data and creating log regression model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#prep train and test data
#TARGET COLUMN = TYPE (1 - BOT --- Benign Ovarian Tumor and 0 - OC --- Ovarian Cancer)
target_column = 'TYPE'

# Prepare training and test data
X_train = train_df.drop(columns=[target_column])

y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]
#If there is a column in the training or test set with an object data type, encode it. 
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col], _ = X_train[col].factorize()
        X_test[col] = pd.Categorical(X_test[col], categories=_).codes


#check to see if object data types converted correctly
(X_train.dtypes)
#imputer to fill missing values with mean value
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#create simple logistic regression model
logreg = LogisticRegression()

#create pipeline to ensure that same preprocessing steps are applied to both training and test data
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipeline = Pipeline(steps)
#fit pipeline on training data
pipeline.fit(X_train, y_train)
#predict on test data
y_pred = pipeline.predict(X_test)
#each row corresponds to sample in x_test
yhat_prob = pipeline.predict_proba(X_test)
#prints out probability of sample belonging to benign tumor or ovarian cancer
#higher probabilites = more confidence in prediction
print(yhat_prob)
#Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')



#confusion matrix creation
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, y_pred, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
#OC= Ovarian Cancer BOT= Benign Ovarian Tumor
plot_confusion_matrix(cnf_matrix, classes=['OC', 'BOT'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['OC', 'BOT'], normalize=True, title='Normalized confusion matrix')

plt.show()



