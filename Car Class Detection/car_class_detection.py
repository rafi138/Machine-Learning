import pandas as pd

dataset=pd.read_csv('car_evaluation.csv')
X=dataset.iloc[:,0:6].values
Y=dataset.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils

#output dummy
labelencoder=LabelEncoder()
labelencoder.fit(Y)
Y = labelencoder.transform(Y)
Y = np_utils.to_categorical(Y)
#input dummy
X0=LabelEncoder()
X[:,0]=X0.fit_transform(X[:,0])
X1=LabelEncoder()
X[:,1]=X1.fit_transform(X[:,1])
X2=LabelEncoder()
X[:,2]=X2.fit_transform(X[:,2])
X3=LabelEncoder()
X[:,3]=X3.fit_transform(X[:,3])
X4=LabelEncoder()
X[:,4]=X4.fit_transform(X[:,4])
X5=LabelEncoder()
X[:,5]=X5.fit_transform(X[:,5])

onehotencoder= OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#training and testing split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#modeling
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=10,activation='relu',input_dim=8))
model.add(Dense(output_dim=8,activation='relu'))
model.add(Dense(output_dim=4,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 30)
y_pred=model.predict(X_test)
y_pred = (y_pred>=0.5)

names=['acc','good','unacc','vgood']

prediction = [[str(0) for i in range(1)] for j in range(len(y_pred))]
for i in range(0,len(y_pred)):
    s=y_pred[i]
    paisi=0
    for j in range(len(s)):
        if(str(s[j])=="True"):
            paisi=1
            prediction[i][0]=names[j]
            break
    if(not paisi):
        prediction[i][0]='Not Matched'