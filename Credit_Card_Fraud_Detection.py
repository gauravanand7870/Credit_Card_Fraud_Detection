# # Credit Card Fraud Detection
# ## Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# #### Loading the dataset

dataset = pd.read_csv('creditcard.csv')

pd.options.display.max_columns=None #To Display each column of the dataset

#Displaying the top 10 rows of the dataset
dataset.head(10)

#Displaying the last 10 rows of the dataset
dataset.tail(10)

#Displaying the number of rows and columns of the dataset
dataset.shape

dataset.info()
# The above info shows that there are no null values

dataset=dataset.drop(['Time'],axis=1)
# ## Feature Scaling

# All the values of features V1,V2,V3,V4,....,V26,V27,V28 are in the same range
# So we have to apply fetaure scaling only on Amount feature
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset['Amount']=scaler.fit_transform(pd.DataFrame(dataset['Amount']))

dataset.head()

# Checking for Duplicate values
dataset.duplicated().any()

# Dropping the Duplicate values
dataset = dataset.drop_duplicates()

dataset.shape

dataset['Class'].value_counts()

sns.countplot(data=dataset, x='Class')
plt.show()

# declaring the independent variable X and dependent variable y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# ### Splitting the datset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# #### Without handling imbalanced data
# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred1=logreg.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred1)

# Accuracy and scores
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("Accuracy",accuracy_score(y_test,y_pred1))
print("Precision",precision_score(y_test,y_pred1))
print("Recall",recall_score(y_test,y_pred1))
print("F1_score",f1_score(y_test,y_pred1))
# #### Handling imbalanced dataset
# ##### Undersampling

regular = dataset[dataset['Class']==0]
fraud = dataset[dataset['Class']==1]

regular.shape

fraud.shape

# Selecting 473 (=no. of fraudulent transactions) random samples of regular transactions
regular_sample=regular.sample(n=473)

undersampled_dataset=pd.concat([regular_sample,fraud],ignore_index=True)

undersampled_dataset['Class'].value_counts()

undersampled_dataset.head()

XU=undersampled_dataset.iloc[:,:-1].values
yU=undersampled_dataset.iloc[:,-1].values

X_trainU,X_testU,y_trainU,y_testU=train_test_split(XU,yU,test_size=0.2,random_state=42)
# LOGISTIC REGRESSION

logregU=LogisticRegression()
logregU.fit(X_trainU,y_trainU)

y_pred1U=logreg.predict(X_testU)

# Confusion Matrix
confusion_matrix(y_testU,y_pred1U)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testU,y_pred1U))
print("Precision",precision_score(y_testU,y_pred1U))
print("Recall",recall_score(y_testU,y_pred1U))
print("F1_score",f1_score(y_testU,y_pred1U))
# DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
dtU = DecisionTreeClassifier()
dtU.fit(X_trainU,y_trainU)

y_pred2U=dtU.predict(X_testU)

# Confusion Matrix
confusion_matrix(y_testU,y_pred2U)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testU,y_pred2U))
print("Precision",precision_score(y_testU,y_pred2U))
print("Recall",recall_score(y_testU,y_pred2U))
print("F1_score",f1_score(y_testU,y_pred2U))
# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
rcU=RandomForestClassifier()
rcU.fit(X_trainU,y_trainU)

y_pred3U=rcU.predict(X_testU)

# Confusion Matrix
confusion_matrix(y_testU,y_pred3U)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testU,y_pred3U))
print("Precision",precision_score(y_testU,y_pred3U))
print("Recall",recall_score(y_testU,y_pred3U))
print("F1_score",f1_score(y_testU,y_pred3U))
# K-NEAREST NEIGHBORS

from sklearn.neighbors import KNeighborsClassifier
knnU=KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)

knnU.fit(X_trainU,y_trainU)

y_pred4U=rcU.predict(X_testU)

# Confusion Matrix
confusion_matrix(y_testU,y_pred4U)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testU,y_pred4U))
print("Precision",precision_score(y_testU,y_pred4U))
print("Recall",recall_score(y_testU,y_pred4U))
print("F1_score",f1_score(y_testU,y_pred4U))
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
nbU=GaussianNB()
nbU.fit(X_trainU,y_trainU)

y_pred5U=nbU.predict(X_testU)

# Confusion Matrix
confusion_matrix(y_testU,y_pred5U)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testU,y_pred5U))
print("Precision",precision_score(y_testU,y_pred5U))
print("Recall",recall_score(y_testU,y_pred5U))
print("F1_score",f1_score(y_testU,y_pred5U))
# Visualizing the Models

Model_comparisionU=pd.DataFrame({'Models':['Logistic Regression','Decision Tree','Random Forest','KNN','Naive Bayes'],'Accuracy':[accuracy_score(y_testU,y_pred1U)*100,accuracy_score(y_testU,y_pred2U)*100,accuracy_score(y_testU,y_pred3U)*100,accuracy_score(y_testU,y_pred4U)*100,accuracy_score(y_testU,y_pred5U)*100]})

Model_comparisionU

sns.barplot(x=Model_comparisionU['Models'],y=Model_comparisionU['Accuracy'])
# ##### Oversampling

from imblearn.over_sampling import SMOTE

XO,yO=SMOTE().fit_resample(X,y)

pd.Series(yO).value_counts()

X_trainO,X_testO,y_trainO,y_testO=train_test_split(XO,yO,test_size=0.2,random_state=42)
# LOGISTIC REGRESSION

logregO=LogisticRegression()
logregO.fit(X_trainO,y_trainO)

y_pred1O=logregO.predict(X_testO)

# Confusion Matrix
confusion_matrix(y_testO,y_pred1O)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testO,y_pred1O))
print("Precision",precision_score(y_testO,y_pred1O))
print("Recall",recall_score(y_testO,y_pred1O))
print("F1_score",f1_score(y_testO,y_pred1O))
# DECISION TREE CLASSIFIER

dtO=DecisionTreeClassifier()
dtO.fit(X_trainO,y_trainO)

y_pred2O=dtO.predict(X_testO)

# Confusion Matrix
confusion_matrix(y_testO,y_pred2O)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testO,y_pred2O))
print("Precision",precision_score(y_testO,y_pred2O))
print("Recall",recall_score(y_testO,y_pred2O))
print("F1_score",f1_score(y_testO,y_pred2O))
# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
rcO=RandomForestClassifier()
rcO.fit(X_trainO,y_trainO)

y_pred3O=rcO.predict(X_testO)

# Confusion Matrix
confusion_matrix(y_testO,y_pred3O)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testO,y_pred3O))
print("Precision",precision_score(y_testO,y_pred3O))
print("Recall",recall_score(y_testO,y_pred3O))
print("F1_score",f1_score(y_testO,y_pred3O))
# K-NEAREST NEIGHBOR

from sklearn.neighbors import KNeighborsClassifier
knnO=KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
knnO.fit(X_trainO,y_trainO)

y_pred4O=knnO.predict(X_testO)

# Confusion Matrix
confusion_matrix(y_testO,y_pred4O)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testO,y_pred4O))
print("Precision",precision_score(y_testO,y_pred4O))
print("Recall",recall_score(y_testO,y_pred4O))
print("F1_score",f1_score(y_testO,y_pred4O))
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
nbO=GaussianNB()
nbO.fit(X_trainO,y_trainO)

y_pred5O=nbO.predict(X_testO)

# Confusion Matrix
confusion_matrix(y_testO,y_pred5O)

# Accuracy and scores
print("Accuracy",accuracy_score(y_testO,y_pred5O))
print("Precision",precision_score(y_testO,y_pred5O))
print("Recall",recall_score(y_testO,y_pred5O))
print("F1_score",f1_score(y_testO,y_pred5O))
# Visualizing the Models

Model_comparisionO=pd.DataFrame({'Models':['Logistic Regression','Decision Tree','Random Forest','KNN','Naive Bayes'],'Accuracy':[accuracy_score(y_testO,y_pred1O)*100,accuracy_score(y_testO,y_pred2O)*100,accuracy_score(y_testO,y_pred3O)*100,accuracy_score(y_testO,y_pred4O)*100,accuracy_score(y_testO,y_pred5O)*100]})

Model_comparisionO

sns.barplot(x=Model_comparisionO['Models'],y=Model_comparisionU['Accuracy'])
# Here, Random Forest Classification with Oversampling gives best results
# ### Saving The Model

rfc=RandomForestClassifier()
rfc.fit(XO,yO)

import joblib
joblib.dump(rfc,"credit_card_model")

model=joblib.load("credit_card_model")

pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])


if pred == 0:
 print('Regular Transaction')
else:
 print('Fraudulent Transaction')