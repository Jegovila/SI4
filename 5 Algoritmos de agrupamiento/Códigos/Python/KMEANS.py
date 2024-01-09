import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


df = pd.read_csv('df_agrupacion_5.csv') # 1-6
print(df)

x = np.asanyarray(df)


n = 3;
model = KMeans(n_clusters=n)
model.fit(x)

y = model.predict(x)


plt.figure()
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('K-Means')

for i in range(n):
    plt.plot(x[y==i,0],x[y==i,1],'o')

plt.show()
