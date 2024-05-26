# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:45:48 2024

@author: aksha
"""
# Hierarchical Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Mall_Customers.csv")


X = df.iloc[:, [3, 4]].values

# Using the dendogram to find the optimal number of clusters

from scipy.cluster import hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")

plt.show()



# Fitting hierarchical Clustering to the mall datasets
from sklearn.cluster import AgglomerativeClustering
aghc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
yhat = aghc.fit_predict(X)


# Visualizing the cluster
plt.scatter(X[yhat==0, 0], X[yhat==0, 1], c='red', s=100, label='Cluster1')
plt.scatter(X[yhat==1, 0], X[yhat==1, 1], c = 'magenta', s =100, label='Clusters2')
plt.scatter(X[yhat==2, 0], X[yhat==2, 1], c='green', s=100, label='Cluster3')
plt.scatter(X[yhat==3, 0], X[yhat==3, 1], c = 'cyan', s =100, label='Clusters4')
plt.scatter(X[yhat==4, 0], X[yhat==4, 1], c='blue', s=100, label='Clusters5')
plt.title("Clusters of customers")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


plt.scatter(X[yhat==0, 0], X[yhat==0, 1], c='red', s=100, label='Careful')
plt.scatter(X[yhat==1, 0], X[yhat==1, 1], c = 'magenta', s =100, label='Standard')
plt.scatter(X[yhat==2, 0], X[yhat==2, 1], c='green', s=100, label='Target')
plt.scatter(X[yhat==3, 0], X[yhat==3, 1], c = 'cyan', s =100, label='Careless')
plt.scatter(X[yhat==4, 0], X[yhat==4, 1], c='blue', s=100, label='Sensible')
plt.title("Clusters of customers in aglomerative")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.savefig("Section 25 - Hierarchical Clustering/hc_clustering.png")
plt.show()

