# Clustering Algorithms

## What is Clustering?

Clustering is an unsupervised machine learning technique used to group similar data points into clusters. The goal is to ensure that data points within a cluster are more similar to each other than to those in other clusters. Clustering is widely used in exploratory data analysis, pattern recognition, image segmentation, and other areas.

## Types of Clustering Algorithms

### 1. K-Means Clustering
![K-Means Clustering](https://github.com/akshatraj26/Unsupervised-Learning-Clustering/blob/main/K-Means%20Clustering/kmeans_clustering.png)

K-Means is one of the most popular and simplest clustering algorithms. It partitions the dataset into K clusters, where each data point belongs to the cluster with the nearest mean.

#### How Does K-Means Work?

1. **Initialization**: Choose K initial centroids randomly from the dataset.
2. **Assignment**: Assign each data point to the nearest centroid, forming K clusters.
3. **Update**: Calculate the new centroids as the mean of all data points in each cluster.
4. **Repeat**: Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

#### Example

Consider a dataset with the following points: {1, 2, 3, 8, 9, 10} and K=2.

1. **Initialization**: Choose initial centroids, e.g., 2 and 9.
2. **Assignment**:
    - Cluster 1: {1, 2, 3}
    - Cluster 2: {8, 9, 10}
3. **Update**:
    - New centroid for Cluster 1: (1+2+3)/3 = 2
    - New centroid for Cluster 2: (8+9+10)/3 = 9
4. **Repeat**: No change in centroids, so the algorithm terminates.

#### Advantages and Disadvantages

- **Advantages**:
  - Simple to understand and implement.
  - Efficient for large datasets.

- **Disadvantages**:
  - Requires the number of clusters (K) to be specified.
  - Sensitive to initial centroid selection.
  - Assumes clusters are spherical and equally sized.

### 2. Agglomerative Clustering

Agglomerative clustering is a type of hierarchical clustering that builds nested clusters in a bottom-up manner. It starts with each data point as a single cluster and iteratively merges the closest pairs of clusters until only one cluster remains or a specified number of clusters is reached.

![Hierarchical Clustering](https://github.com/akshatraj26/Unsupervised-Learning-Clustering/blob/main/Hierarchical%20Clustering/hc_clustering.png)
![Dendogram Clustering](https://github.com/akshatraj26/Unsupervised-Learning-Clustering/blob/main/Hierarchical%20Clustering/Figure%202024-05-07%20130420.png)


#### How Does Agglomerative Clustering Work?

1. **Initialization**: Start with each data point as its own cluster.
2. **Merge**: Find the pair of clusters with the smallest distance between them and merge them into a single cluster.
3. **Repeat**: Repeat the merge step until the desired number of clusters is achieved.

#### Example

Consider a dataset with the following points: {1, 2, 3, 8, 9, 10}.

1. **Initialization**: Start with individual clusters: {1}, {2}, {3}, {8}, {9}, {10}.
2. **Merge**:
    - Merge {1} and {2} to form {1, 2}.
    - Merge {8} and {9} to form {8, 9}.
    - Merge {1, 2} and {3} to form {1, 2, 3}.
    - Merge {8, 9} and {10} to form {8, 9, 10}.
3. **Repeat**: Continue merging until the desired clustering structure is achieved.

#### Advantages and Disadvantages

- **Advantages**:
  - Does not require the number of clusters to be specified.
  - Produces a dendrogram, which can be used to choose the appropriate number of clusters.

- **Disadvantages**:
  - Computationally expensive for large datasets.
  - Sensitive to noise and outliers.

## Conclusion

Clustering is a powerful tool for discovering structure in data. K-Means and Agglomerative clustering are two widely used algorithms, each with its strengths and weaknesses. K-Means is efficient and straightforward but requires the number of clusters to be predefined. Agglomerative clustering provides a flexible hierarchical structure but can be computationally intensive. Choosing the right algorithm depends on the specific characteristics of the dataset and the goals of the analysis.

---




## Setting Up the Environment

1. Create a virtual environment:
    ```bash
    python -m venv .env
    ```

2. Activate the virtual environment:
    - On Windows:
        ```bash
        .env\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source .env/bin/activate
        ```

## Installing Dependencies

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

