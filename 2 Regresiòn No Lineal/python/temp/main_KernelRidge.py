import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.kernel_ridge import KernelRidge

#
df = pd.read_csv('temp.csv') # 1, 2, 3

x = np.asanyarray(df[['time']])
y = np.asanyarray(df[['temp']])

x_train, x_test, y_train, y_test = train_test_split(x, y)

#
model = Pipeline([('scaler',StandardScaler()),
                  ('reg',KernelRidge(alpha=0.001, kernel='rbf'))])

model.fit(x_train,y_train)

#
print('Train score: ',model.score(x_train,y_train))
print('Test score: ',model.score(x_test,y_test))

#
x_plot = np.linspace(x.min(),x.max(),50).reshape(-1,1)
y_plot = model.predict(x_plot)

plt.figure()
plt.grid()
plt.title('Regresion no lineal - Kernel Ridge')
plt.xlabel('time')
plt.ylabel('temp')

plt.plot(x_train,y_train,'bo')
plt.plot(x_test,y_test,'ro')
plt.plot(x_plot,y_plot,'k-',lw=2)

plt.legend(['entrenamiento','generalizacion','prediccion'])
plt.show()
