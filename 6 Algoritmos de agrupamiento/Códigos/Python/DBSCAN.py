import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


df = pd.read_csv('df_agrupacion_1.csv') # 1-6
print(df)

x = np.asanyarray(df)


model = DBSCAN(eps=0.2,min_samples=5)
model.fit(x)

y = model.labels_
n = np.max(model.labels_)+1


plt.figure()
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('DBSCAN')

for i in range(n):
  plt.plot(x[y==i,0],x[y==i,1],'o')

plt.plot(x[y==-1,0],x[y==-1,1],'rx')

plt.show()
