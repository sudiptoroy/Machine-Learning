# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 23:31:45 2018

@author: Sudipto
"""

import pandas as pd

dataset = pd.read_csv('wine.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)


from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, predictions))

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
