# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:14:51 2024

@author: aksha
"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("./Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', max_iter=300, n_init= 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Within Cluster Sum of Squares - (WCSS)")
plt.show()


# Applying KMeans to the mall datasets
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)


# Visualizing the clusters
plt.scatter(x =X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], s = 100, c='red', label='Cluster1')
plt.scatter(x =X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], s = 100, c='blue', label='Cluster2')
plt.scatter(x =X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], s = 100, c='green', label='Cluster3')
plt.scatter(x =X[y_kmeans == 3, 0], y=X[y_kmeans == 3, 1], s = 100, c='magenta', label='Cluster4')
plt.scatter(x =X[y_kmeans == 4, 0], y=X[y_kmeans == 4, 1], s = 100, c='cyan', label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c='yellow', label='Centroids')
plt.title("Cluster of clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()




plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c='red', s=100, label='Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c = 'magenta', s =100, label='Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c='green', s=100, label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c = 'cyan', s =100, label='Careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], c='blue', s=100, label='Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c='yellow', label='Centroids')
plt.title("Cluster of clients in kmeans")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.savefig("Section 24 - K-Means Clustering/kmeans_clustering.png")
plt.show()

df = dataset.copy()

df['ykmeans'] = y_kmeans

labels = {0: 'Careful', 1: 'Standard', 2 : 'Target', 3: 'Careless', 4: 'Sensible'}
# df['labels'] = df[[] = labels[0], df['ykmeans'] == 1] = labels[1], df['ykmeans'] == 2] = labels[2], df['ykmeans'] == 3] = labels[3], df['ykmeans'] == 4] == labels[4]]

    
    

    