import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

# Dense layer: Just your regular densely-connected NN layer.

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('df_regresion_nolineal_3.csv') # 1, 2, 3

x = np.asanyarray(df[['x']])
y = np.asanyarray(df[['y']])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


model = Sequential()
model.add(Dense(100,activation='relu',input_shape=[x.shape[1]]))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error',optimizer='adam')

# loss <- funcion de perdida
# categorical_crossentropy <- clasificacion multiclase
# binary_crossentropy <- clasificacion binaria
# mean_squared_error <- regresion

hist = model.fit(xtrain,ytrain,batch_size=2,epochs=100,verbose=1,validation_data=(xtest,ytest))

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
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,'bo')
plt.plot(x,yp,'r-')
plt.legend(['Entrenamiento','Generalizacion'])
plt.show()