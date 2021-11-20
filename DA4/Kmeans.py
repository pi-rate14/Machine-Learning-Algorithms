import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv('Kmeans.csv')
print("data: ",data)
plt.scatter(data['X'],data['Y'])
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()
x = data.iloc[:,1:3] # 1t for rows and second for columns
print("x: ", x)
kmeans = KMeans(3)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
print("identified clusters: ",identified_clusters)
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Y'],data_with_clusters['X'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()