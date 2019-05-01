7import pandas as pd

dataset=pd.read_csv('red_wine_data.csv')
X=dataset.iloc[:,0:12].values
Y=dataset.iloc[:,11].values
    
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
label=LabelEncoder()
label.fit(Y)
Y = label.transform(Y)
Y = np_utils.to_categorical(Y)

output_test=[['fixed_acidity ','volatile_acidity ','citric_acid ','residual_sugar ','chlorides ','free_sulfur_dioxide ','total_sulfur_dioxide ',
             'density ','pH ','sulphates ','alcohol ','quality']]

input_test=X[0:1,0:12]
X_test=X[0:1,0:11]
X_train=X[:,0:11]

#prediction with input
for i in range(len(output_test[0])-1):
    X_test[0][i]=input(output_test[0][i])
    input_test[0][i]=X_test[0][i]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=8,activation='tanh',input_dim=11))
model.add(Dense(output_dim=8,activation='tanh'))
model.add(Dense(output_dim=6,activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y,batch_size=10, nb_epoch = 10)
y_pred=model.predict(X_test)

y_pred=(y_pred>=0.5)

for i in range(1):
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
