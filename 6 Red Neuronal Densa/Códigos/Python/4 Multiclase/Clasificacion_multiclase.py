import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics
from sklearn.model_selection import train_test_split


# Clasificación multiclase
df = pd.read_csv('Sensor.csv')

x = np.asanyarray(df.drop(columns=['D']))
y = np.asanyarray(df['D'])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

ytrain = keras.utils.to_categorical(ytrain, 9)
ytest = keras.utils.to_categorical(ytest, 9)

model = Sequential()

model.add(Dense(100,activation='relu',input_shape=[x.shape[1]]))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(9,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# loss <- funcion de perdida
# categorical_crossentropy <- clasificacion multiclase
# binary_crossentropy <- clasificacion binaria
# mean_squared_error <- regresion

hist = model.fit(xtrain,ytrain,batch_size=100,epochs=100,verbose=1,validation_data=(xtest,ytest))

# loss, accuracy <- para entrenamiento
# val_loss, val_accuracy <- para validacion

# summarize history for accuracy
plt.figure()
plt.grid()
plt.plot(hist.history['accuracy'],lw=2)
plt.plot(hist.history['val_accuracy'],lw=2)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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


yp = model.predict(x)
yp = np.argmax(yp,axis=1)

# Métricas
print('Metricas: \n', metrics.classification_report(y,yp))

# Matriz de Confusión
print('Confusion matrix: \n', metrics.confusion_matrix(y,yp))
