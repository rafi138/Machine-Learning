import pandas as pd

dataset=pd.read_csv('red_wine_data.csv')
X=dataset.iloc[:,0:12].values
Y=dataset.iloc[:,11].values
    
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
label=LabelEncoder()
label.fit(Y)
Y = label.transform(Y)
Y = np_utils.to_categorical(Y)

output_test=[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide',
             'density','pH','sulphates','alcohol','quality']]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.4,random_state=0)# test size is a percent of whole dataset here test size is 20%

input_test=X_test
X_test=X_test[:,0:11]
X_train=X_train[:,0:11]



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=8,activation='tanh',input_dim=11))
model.add(Dense(output_dim=8,activation='tanh'))
model.add(Dense(output_dim=6,activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 100)
y_pred=model.predict(X_test)
prediction = [[str(0) for i in range(12)] for j in range(len(y_pred))]

for i in range(len(y_pred)):
    s=y_pred[i]
    if(str(s[0])=="True"):
        input_test[i][11]='3'
    elif(str(s[1])=="True"):
        input_test[i][11]='4'
    elif(str(s[2])=="True"):
        input_test[i][11]='5'
    elif(str(s[3])=="True"):
        input_test[i][11]='6'
    elif(str(s[4])=="True"):
        input_test[i][11]='7'
    elif(str(s[5])=="True"):
        input_test[i][11]='8'

import csv
myFile = open('prediction_quality.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(output_test)
myFile = open('prediction_quality.csv', 'a')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(input_test)
result=pd.read_csv('prediction_quality.csv')