#!/usr/bin/env python
#coding:utf-8
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import os
import csv as csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
dataset_path = '../data'
train_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_datafile_salaries = os.path.join(dataset_path, 'train_salaries_2013-03-07.csv')
test_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_data_features = pd.read_csv(train_datafile_features)
train_data_salaries = pd.read_csv(train_datafile_salaries)
test_data_features = pd.read_csv(test_datafile_features)
train_data = pd.merge(train_data_features, train_data_salaries, on=['jobId'], how='left')
train_data.to_csv('train_data.csv')


feat_names = train_data.columns[1:-1].tolist()
i = 1
label_enc = OneHotEncoder()


X = train_data[feat_names].values
Y = train_data_salaries[train_data.columns[-1]].values
for i in range(len(X[1])-2):
	train_d = label_enc.fit_transform(X[:,i])
	X[:,i] = train_d
print (X[:,1])
'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/4, random_state=0)
parameters = {'C': [0.01, 1, 100]}
#clf = GridSearchCV(LogisticRegression(), parameters, cv=3, scoring='accuracy')

clf =LinearRegression()
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)


print (np.mean((y_train-pred_train) ** 2))
print (np.mean((y_test-pred_test) ** 2))



#X_train[:][1] = train_d 
'''


#print (train_data.head())