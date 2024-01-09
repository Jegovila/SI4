import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

#
df = pd.read_csv('df_regresion_nolineal_1.csv') # 1, 2, 3

x = np.asanyarray(df[['x']])
y = np.asanyarray(df[['y']])

x_train, x_test, y_train, y_test = train_test_split(x,y) # %75 datos para entrenar, %25 datos para probar

#
model = Pipeline([('poly',PolynomialFeatures(degree=25)),('scaler',StandardScaler()),('reg',Ridge(0.1))])
model.fit(x_train,y_train)

#
print('Train score: ', model.score(x_train,y_train))
print('Test score: ', model.score(x_test,y_test))

#
x_plot = np.linspace(x.min(),x.max(),50).reshape(-1,1)
y_plot = model.predict(x_plot)

plt.figure()
plt.grid()
plt.title('Regresion polinomial')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x_train,y_train,'bo')
plt.plot(x_test,y_test,'ro')
plt.plot(x_plot,y_plot,'k-',lw=2)

plt.legend(['entrenamiento','generalizacion','prediccion'])

plt.show()