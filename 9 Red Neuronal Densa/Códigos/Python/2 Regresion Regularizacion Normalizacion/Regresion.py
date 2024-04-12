import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

# Dense layer: Just your regular densely-connected NN layer.

# Dropout layer: Dropout is a simple and powerful regularization 
# technique. Randomly selected neurons are ignored during training.

# BatchNormalization layer: Batch normalization applies a transformation 
# that maintains the mean output close to 0 and the output standard 
# deviation close to 1.

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('regresion_nolineal.csv') # 1, 2, 3

x = np.asanyarray(df[['time']])
y = np.asanyarray(df[['temp']])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


model = Sequential()

model.add(Dense(100,activation='relu',input_shape=[x.shape[1]]))
#model.add(BatchNormalization())
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error',optimizer='adam')

# loss <- funcion de perdida
# categorical_crossentropy <- clasificacion multiclase
# binary_crossentropy <- clasificacion binaria
# mean_squared_error <- regresion

hist = model.fit(xtrain,ytrain,batch_size=1000,epochs=50,verbose=1,validation_data=(xtest,ytest))

# loss <- para entrenamiento
# val_loss <- para validacion

# summarize history for loss
plt.figure()
plt.grid()
plt.plot(hist.history['loss'],lw=2)
plt.plot(hist.history['val_loss'],lw=2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Métricas de regresión
e_train = r2_score(ytrain,model.predict(xtrain))
e_test = r2_score(ytest,model.predict(xtest))

print('Train score: ', e_train)
print('Test score: ', e_test)


# Gráficas
yp = model.predict(x)
plt.figure()
plt.grid()
plt.title('Regresion no lineal')
plt.xlabel('time')
plt.ylabel('temp')

plt.plot(x,y,'bo')
plt.plot(x,yp,'r-')
plt.legend(['Entrenamiento','Generalizacion'])
plt.show()