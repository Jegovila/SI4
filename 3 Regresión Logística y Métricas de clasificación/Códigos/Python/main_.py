import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay


# 
df = pd.read_csv('df_clasificacion_3.csv') # 1, 2, 3

x = np.asanyarray(df.drop(columns=['y']))
y = np.asanyarray(df['y'])

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.9)


#
model = Pipeline([('scaler',StandardScaler()),('cla',LogisticRegression(max_iter=2000,C=200))])
model.fit(x_train,y_train)

print('Train score:',model.score(x_train,y_train))
print('Test score:',model.score(x_test,y_test))

#
disp = DecisionBoundaryDisplay.from_estimator(model,x,response_method="predict",alpha=0.5,eps=0.2,cmap=plt.cm.RdBu,xlabel='x1',ylabel='x2')
disp.ax_.scatter(x[y==0,0],x[y==0,1],color='r',edgecolor='k')
disp.ax_.scatter(x[y==1,0],x[y==1,1],color='b',edgecolor='k')


yp = model.predict(x)

# Métricas
print('Metricas: \n', metrics.classification_report(y,yp))

# Matriz de Confusión
print('Confusion matrix: \n', metrics.confusion_matrix(y,yp))

# Curvas ROC
g = model.predict_proba(x)  
fpr, tpr, _ = metrics.roc_curve(y,g[:,1])
display = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,estimator_name="ROC")
display.plot()
plt.show()
