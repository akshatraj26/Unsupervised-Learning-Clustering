# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:07:45 2024

@author: aksha
"""

# importing a required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


# Loading and cleaning the dataset

X = pd.read_csv("CC GENERAL.csv")

# Droping the CUST_ID column from the data
X = X.drop('CUST_ID', axis =1)

# Checking and Handling missing value 

null_value = X.isna().sum()

X.fillna(method='ffill', inplace=True)

# Preprocessing the data

# Scaling the data so that all the features becomes comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


plt.plot(X_scaled)

# Normalize the data so data approximately follow gaussian distribution
X_normalized = normalize(X_scaled)

plt.plot(X_normalized)


# Converting the numpy array to dataframe
X_normalized = pd.DataFrame(X_normalized)


# Reducing the dimensionality of the data

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)

X_principal.columns = ['P1', 'P2']

# Visualizing the working of the dendograms

plt.figure(figsize=(8, 8))
plt.title("Visualizing the data")
dendogram = shc.dendrogram((shc.linkage(X_principal, method='ward')))

# this shows optimal number of cluster is 2
ag = AgglomerativeClustering(n_clusters=2)
yhat = ag.fit_predict(X_principal)


# Visualize th clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat, cmap='rainbow')
plt.title("Number of clusters ")
plt.show()



# when take k =3
ag3 = AgglomerativeClustering(n_clusters=3)
yhat3 = ag3.fit_predict(X_principal)

# Visualize th clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat3, cmap='rainbow')
plt.title("Number of clusters ")
plt.show()



# k =4
ag4 = AgglomerativeClustering(n_clusters=4)
yhat4= ag4.fit_predict(X_principal)

# Visualize th clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat4, cmap='rainbow')
plt.title("Number of clusters ")
plt.show()


# k =5
ag5 = AgglomerativeClustering(n_clusters=5)
yhat5= ag5.fit_predict(X_principal)

# Visualize th clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat5, cmap='rainbow')
plt.title("Number of clusters ")
plt.show()



# K = 6
ag6 = AgglomerativeClustering(n_clusters=5)
yhat6= ag6.fit_predict(X_principal)

# Visualize th clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat6, cmap='rainbow')
plt.title("Number of clusters ")
plt.show()


## Evaluateing the different models and visualizing the  different models and visualizing the results

k = [2, 3, 4, 5, 6]
# Silhouette Scores
silhouette_scores = []

silhouette_scores.append(silhouette_score(X_principal, yhat))
silhouette_scores.append(silhouette_score(X_principal, yhat3))
silhouette_scores.append(silhouette_score(X_principal, yhat4))
silhouette_scores.append(silhouette_score(X_principal, yhat5))
silhouette_scores.append(silhouette_score(X_principal, yhat6))

# ploting the bargraph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel("Number of clusters", fontsize=20)
plt.ylabel("S(i)", fontsize=20)
plt.show()



