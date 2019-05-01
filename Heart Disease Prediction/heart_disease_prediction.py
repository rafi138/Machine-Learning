import pandas as pd
import numpy as np
dataset=pd.read_csv('heart_disease_dataset.csv')
X=dataset.iloc[:,0:11].values
Y=dataset.iloc[:,10].values

#training and testing split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)# test size is a percent of whole dataset here test size is 20%
prediction=X_test
X_test = X_test[:,1:10]
X_train = X_train[:,1:10]

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
lebelencoder_X_2=LabelEncoder()
X_test[:,4]=labelencoder_X_1.fit_transform(X_test[:,4])
X_train[:,4]=lebelencoder_X_2.fit_transform(X_train[:,4])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(output_dim=6,init='uniform',activation='tanh',input_dim=9))
model.add(Dense(output_dim=6,init='uniform',activation='relu'))
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 100)
y_pred=model.predict(X_test)
for i in range(len(y_pred)):
    s=prediction[i][4]
    if(y_pred[i]>0.5):    
        prediction[i][10]=1
    else:
        prediction[i][10]=0

import csv
myFile = open('Heart_output.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(prediction)
result=pd.read_csv('Heart_output.csv')
