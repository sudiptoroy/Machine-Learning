# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:58:14 2018

@author: Sudipto
"""
#import dataset
from sklearn import datasets
iris = datasets.load_iris()

#definte x and y for train and test
X = iris.data
y = iris.target


#spliting the dataset for train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))