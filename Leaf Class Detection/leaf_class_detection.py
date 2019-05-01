import pandas as pd
dataset=pd.read_csv('leaf.csv')
X=dataset.iloc[:,1:17].values
Y=dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
labelencoder=LabelEncoder()
labelencoder.fit(Y)
Y = labelencoder.transform(Y)
Y = np_utils.to_categorical(Y)

#training and testing split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3,random_state=0)# test size is a percent of whole dataset here test size is 20%


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#modeling
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(output_dim=12,activation='relu',input_dim=15))
model.add(Dense(output_dim=8,activation='relu'))
model.add(Dense(output_dim=30,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10, nb_epoch = 100)
y_pred=model.predict(X_test)
y_pred = (y_pred>=0.5)
names=['Quercus suber','Salix atrocinerea','Populus nigra','Alnus sp',
       'Quercus robur','Crataegus monogyna','Ilex aquifolium','Nerium oleander','Betula pubescens',
       'Tilia tomentosa','Acer palmaturu','Celtis sp','Corylus avellana','Castanea sativa',
       'Populus alba','Primula vulgaris','Erodium sp','Bougainvillea sp','Arisarum vulgare',
       'Euonymus japonicus','Ilex perado ssp azorica','Magnolia soulangeana','Buxus sempervirens',
       'Urtica dioica','Podocarpus sp','Acca sellowiana','Hydrangea sp','Pseudosasa japonica',
       'Magnolia grandiflora','Geranium sp']
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