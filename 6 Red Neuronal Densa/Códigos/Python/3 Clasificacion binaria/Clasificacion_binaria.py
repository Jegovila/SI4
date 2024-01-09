import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv('df_clasificacion_3.csv') # 1, 2, 3

x = np.asanyarray(df.drop(columns=['y']))
y = np.asanyarray(df['y'])

plt.figure
plt.grid()
plt.title('Clasificacion')

plt.plot(x[y==0,0],x[y==0,1],'ro',fillstyle='none',markersize=10)
plt.plot(x[y==1,0],x[y==1,1],'bo',fillstyle='none',markersize=10)

plt.legend(['y==0','y==1'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


model = Sequential()

model.add(Dense(100,activation='relu',input_shape=[x.shape[1]]))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# loss <- funcion de perdida
# categorical_crossentropy <- clasificacion multiclase
# binary_crossentropy <- clasificacion binaria
# mean_squared_error <- regresion

hist = model.fit(xtrain,ytrain,batch_size=4,epochs=100,verbose=1,validation_data=(xtest,ytest))

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
yp = (yp.ravel()>0.5)*1


# Decision Boundary Display
plt.figure()
plt.title('Clasificacion')

x_min, x_max = x[:, 0].min() - 0.1, x[:,0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.linspace(x_min,x_max, 100),np.linspace(y_min, y_max, 100))

x_in = np.c_[xx.ravel(), yy.ravel()]

y_pred = model.predict(x_in)
y_pred = np.round(y_pred).reshape(xx.shape)

plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.5)
plt.scatter(x[:,0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# Métricas
print('Metricas: \n', metrics.classification_report(y,yp))

# Matriz de Confusión
print('Confusion matrix: \n', metrics.confusion_matrix(y,yp))

