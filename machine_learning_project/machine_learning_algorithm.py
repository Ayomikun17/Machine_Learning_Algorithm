# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing Packages

import warnings
warnings.filterwarnings('ignore')
import pandas as pd  
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt


#Importing the file
data=pd.read_csv('android.csv')
data
data.shape

data = data.sample(frac=1).reset_index(drop=True)


#Gettig the number of malware infected or not
target_count = data.malware.value_counts()

#Assigning class 0 to infected and class 1 to unifected
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])

count_class_0, count_class_1 = data.malware.value_counts()

#Getting the corresponding data(rest of the columns) that corresponds 
#with malware or not
df_class_0 = data[data['malware'] == 0]
df_class_1 = data[data['malware'] == 1]


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)


#Independent and Dependent variable
X=df_test_over.iloc[:,df_test_over.columns !='malware']
Y=df_test_over.iloc[:,df_test_over.columns =="malware"]



from sklearn.utils import shuffle
X, Y=shuffle(X, Y)

#Dropping the name column because it contains strings and we only want numbers
X=X.drop(columns='name')


#Creating the model 
#Split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,
                                                    random_state=0)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# DecisionTreeClassifier algorithm
from sklearn.tree import DecisionTreeClassifier 

tree = DecisionTreeClassifier() 
tree.fit(X_train,y_train)


prediction_1 = tree.predict(X_test)
prediction_1

model_accuracy=metrics.accuracy_score(y_test,prediction_1)



#Random forest algortihm

clf = DecisionTreeClassifier(max_depth=2, random_state=0)
 
randomModel=clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
prediction_2 = randomModel.predict(X_test)
model_accuracy2 = accuracy_score(y_test,prediction_2)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
abc = LogisticRegression(random_state=0)

logModel = abc.fit(X_train, y_train)
prediction_3 = logModel.predict(X_test)
prediction_3

model_accuracy3=metrics.accuracy_score(y_test,prediction_3)