import pandas as pd #to import .csv file

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)# test size is a percent of whole dataset here test size is 20%
prediction=X_test
X_test = X_test[:,1:4]
X_train = X_train[:,1:4]

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
lebelencoder_X_2=LabelEncoder()
X_test[:,0]=labelencoder_X_1.fit_transform(X_test[:,0])
X_train[:,0]=lebelencoder_X_2.fit_transform(X_train[:,0])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=3))
model.add(Dense(output_dim=6,init='uniform',activation='relu'))
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
model.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 100)
y_pred=model.predict(X_test)
for i in range(len(y_pred)):
    if(y_pred[i]>0.5):    
        prediction[i][4]=1
    else:
        prediction[i][4]=0
        

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

import csv
myFile = open('final_predict.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(prediction)
final_predict=pd.read_csv('Heart_output.csv')
