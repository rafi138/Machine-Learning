import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('ML.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 12].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
input_test=X_test
X_test=X_test[:,3:12]
X_train=X_train[:,3:12]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])
labelencoder_X_2 = LabelEncoder()
X_test[:, 2] = labelencoder_X_2.fit_transform(X_test[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_11 = LabelEncoder()
X_train[:, 1] = labelencoder_X_11.fit_transform(X_train[:, 1])
labelencoder_X_12 = LabelEncoder()
X_train[:, 2] = labelencoder_X_12.fit_transform(X_train[:, 2])
onehotencoder1 = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder1.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

leen=len(X_test)
for i in range(leen):
    if(y_pred[i]>0.5):    
        input_test[i][12]=1
    else:
        input_test[i][12]=2    
import csv
myFile = open('ML-output.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(input_test)
result=pd.read_csv('ML-output.csv')
