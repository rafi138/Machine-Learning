import pandas as pd

dataset = pd.read_csv('bank_marketing.csv')

X = dataset.iloc[:,0:16].values
Y = dataset.iloc[:,16].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

X1=LabelEncoder()
X[:,1]=X1.fit_transform(X[:,1])
X2=LabelEncoder()
X[:,2]=X2.fit_transform(X[:,2])
X3=LabelEncoder()
X[:,3]=X3.fit_transform(X[:,3])
X4=LabelEncoder()
X[:,4]=X4.fit_transform(X[:,4])
X6=LabelEncoder()
X[:,6]=X6.fit_transform(X[:,6])
X7=LabelEncoder()
X[:,7]=X7.fit_transform(X[:,7])
X8=LabelEncoder()
X[:,8]=X8.fit_transform(X[:,8])
X10=LabelEncoder()
X[:,10]=X10.fit_transform(X[:,10])
X15=LabelEncoder()
X[:,15]=X15.fit_transform(X[:,15])

YY=LabelEncoder()
Y = YY.fit_transform(Y);

onehotencoder= OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#training and testing split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.5,random_state=0) # test size is a percent of whole dataset here test size is 20%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=26))
model.add(Dense(output_dim=6,init='uniform',activation='relu'))
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 20)
y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred);