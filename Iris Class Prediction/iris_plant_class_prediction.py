import pandas as pd
import numpy as np
dataset=pd.read_csv('iris.csv')
X=dataset.iloc[:,0:4].values
Y=dataset.iloc[:,4].values

#output lebel dummy create
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
labelencoder.fit(Y)
Y = labelencoder.transform(Y)
Y = np_utils.to_categorical(Y)

#training and testing split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)# test size is a percent of whole dataset here test size is 20%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=4,activation='relu',input_dim=4))
model.add(Dense(output_dim=4,activation='relu'))
model.add(Dense(output_dim=3,activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 100)
y_pred= model.predict(X_test)
y_pred= (y_pred>=0.5)
prediction = [[str(0) for i in range(1)] for j in range(len(y_pred))]#2D array 
for i in range(0,len(y_pred)):
    s=y_pred[i]
    if(str(s[0])=="True"):
        prediction[i][0]="Iris-Setosa"
    elif(str(s[1])=="True"):
        prediction[i][0]="Iris-Versicolor"
    elif(str(s[2])=="True"):
        prediction[i][0]="Iris-Virginica"
    else:
        prediction[i][0]="Not Matched"
