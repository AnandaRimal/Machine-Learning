# K-Means Clustering

## Table of Contents
- [Introduction](#introduction)
- [What is K-Means?](#what-is-kmeans)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use K-Means?](#why-use-kmeans)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Algorithm Details](#algorithm-details)
- [Choosing Optimal K](#choosing-optimal-k)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)

## Introduction

K-Means is one of the most popular **unsupervised learning** algorithms used for clustering. It partitions data into K distinct, non-overlapping clusters by grouping similar data points together.

**Key Idea**: Find K cluster centers (centroids) such that the sum of squared distances between data points and their nearest centroid is minimized.

K-Means is widely used in:
- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection
- Feature learning

## What is K-Means?

K-Means clustering aims to partition $n$ observations into $K$ clusters where each observation belongs to the cluster with the nearest mean (centroid).

### Visual Intuition

```
Initial Random Centroids:
   X    X    X
  ●  ●  ●  ●
 ●  ●  ●  ●  ●
  ●  ●  ●  ●

After Iterations:
Cluster 1   Cluster 2   Cluster 3
   X           X           X
  ●●●        ●●●         ●●●
 ●●●●       ●●●●        ●●●●
```

### Key Components

1. **Centroids**: Center points of clusters
2. **Assignment**: Each point belongs to nearest centroid
3. **Update**: Recompute centroids as mean of assigned points
4. **Iteration**: Repeat until convergence

## Mathematical Foundation

### Objective Function

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)** or **Inertia**:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where:
- $K$ = number of clusters
- $C_k$ = set of points in cluster $k$
- $\boldsymbol{\mu}_k$ = centroid of cluster $k$
- $\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ = squared Euclidean distance

### Centroid Calculation

The centroid of cluster $C_k$ is the mean of all points in that cluster:

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$

where $|C_k|$ is the number of points in cluster $k$.

### Distance Metrics

**Euclidean Distance** (most common):
$$d(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{\sum_{j=1}^{d} (x_j - \mu_j)^2}$$

**Squared Euclidean Distance**:
$$d^2(\mathbf{x}, \boldsymbol{\mu}) = \sum_{j=1}^{d} (x_j - \mu_j)^2$$

**Manhattan Distance**:
$$d(\mathbf{x}, \boldsymbol{\mu}) = \sum_{j=1}^{d} |x_j - \mu_j|$$

### Cluster Assignment

Each point $\mathbf{x}_i$ is assigned to the nearest centroid:

$$c_i = \arg\min_{k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

This creates a **Voronoi tessellation** of the feature space.

### Convergence

K-Means converges when one of these occurs:

1. **Centroids don't change**: $\boldsymbol{\mu}_k^{(t+1)} = \boldsymbol{\mu}_k^{(t)}$ for all $k$
2. **Assignments don't change**: $C_k^{(t+1)} = C_k^{(t)}$ for all $k$
3. **Inertia change below threshold**: $|J^{(t+1)} - J^{(t)}| < \epsilon$
4. **Maximum iterations reached**

**Guaranteed**: K-Means always converges (inertia decreases monotonically).

## Why Use K-Means?

### 1. **Simplicity**
Easy to understand and implement.

### 2. **Efficiency**
Fast even for large datasets: $O(nKd \cdot \text{iterations})$.

### 3. **Scalability**
Works well with millions of samples.

### 4. **Versatility**
Applicable to many domains.

### 5. **Interpretability**
Cluster centers provide clear interpretation.

### 6. **Feature Learning**
Used for vector quantization and dimensionality reduction.

## Advantages and Disadvantages

### Advantages

1. **Fast and Efficient**: Computationally inexpensive
2. **Simple**: Easy to implement and understand
3. **Scalable**: Handles large datasets well
4. **Guaranteed Convergence**: Always converges (though possibly to local optimum)
5. **Interpretable**: Clear cluster centers
6. **Parallelizable**: Can distribute computations
7. **Works Well on Spherical Clusters**: Natural fit for round, evenly-sized clusters

### Disadvantages

1. **Must Choose K**: Need to specify number of clusters beforehand
2. **Sensitive to Initialization**: Different starting points → different results
3. **Local Optima**: May not find global optimum
4. **Assumes Spherical Clusters**: Struggles with elongated or irregular shapes
5. **Sensitive to Outliers**: Outliers can distort centroids
6. **Equal Cluster Sizes Assumption**: Performs poorly with imbalanced clusters
7. **Euclidean Distance Limitation**: Not suitable for categorical data
8. **Scale Sensitivity**: Features must be normalized

## Algorithm Details

### Standard K-Means Algorithm

**Input**:
- Data: $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n\}$
- Number of clusters: $K$

**Algorithm**:

```
1. Initialization:
   Randomly select K data points as initial centroids:
   μ₁, μ₂, ..., μₖ

2. Repeat until convergence:
   
   a) Assignment Step:
      For each data point xᵢ:
          Find nearest centroid:
          cᵢ = argmin_k ||xᵢ - μₖ||²
          Assign xᵢ to cluster Cₖ
   
   b) Update Step:
      For each cluster k:
          Recompute centroid:
          μₖ = (1/|Cₖ|) Σ(xᵢ ∈ Cₖ) xᵢ
   
   c) Check Convergence:
      If centroids don't change or max iterations reached:
          STOP
      
3. Output:
   Cluster assignments {c₁, c₂, ..., cₙ}
   Final centroids {μ₁, μ₂, ..., μₖ}
```

### K-Means++ Initialization

Better initialization to avoid poor local optima:

```
1. Choose first centroid μ₁ uniformly at random from data

2. For k = 2 to K:
   a) For each point xᵢ:
      Compute distance to nearest existing centroid:
      D(xᵢ) = min_j ||xᵢ - μⱼ||²
   
   b) Choose next centroid μₖ with probability:
      P(xᵢ) = D(xᵢ)² / Σⱼ D(xⱼ)²
   
   (Points farther from existing centroids more likely to be chosen)

3. Proceed with standard K-Means
```

**Advantage**: K-Means++ provides theoretical guarantees on solution quality.

### Mini-Batch K-Means

For very large datasets:

```
1. Initialize centroids using K-Means++

2. For each iteration:
   a) Sample random mini-batch of data (e.g., 1000 points)
   b) Assign mini-batch points to nearest centroids
   c) Update centroids using only mini-batch:
      μₖ^(new) = (1-η)μₖ^(old) + η·(mean of assigned points)
      where η is learning rate
   
3. Repeat until convergence
```

**Advantage**: Much faster on large datasets with slight accuracy trade-off.

## Choosing Optimal K

### 1. Elbow Method

Plot inertia vs. K and find the "elbow":

$$\text{Inertia}(K) = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Elbow**: Point where adding more clusters doesn't significantly reduce inertia.

```
Inertia
   |
   |\
   | \
   |  \___
   |      --------
   +-----------------> K
   1  2  3  4  5  6
      ↑
    Elbow (optimal K ≈ 3)
```

### 2. Silhouette Score

Measures how similar a point is to its own cluster compared to other clusters:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

where:
- $a_i$ = average distance to points in same cluster
- $b_i$ = average distance to points in nearest other cluster

**Range**: $s_i \in [-1, 1]$
- $s_i \approx 1$: Well-clustered
- $s_i \approx 0$: On cluster boundary
- $s_i \approx -1$: Misclassified

**Average Silhouette Score**:
$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s_i$$

Choose K that maximizes $\bar{s}$.

### 3. Gap Statistic

Compares inertia with expected inertia under null reference distribution:

$$\text{Gap}(K) = \mathbb{E}[\log(\text{Inertia}_{ref})] - \log(\text{Inertia}_{data})$$

Choose smallest K where:
$$\text{Gap}(K) \geq \text{Gap}(K+1) - s_{K+1}$$

### 4. Davies-Bouldin Index

Measures average similarity between each cluster and its most similar cluster:

$$DB = \frac{1}{K} \sum_{k=1}^{K} \max_{k' \neq k} \frac{\sigma_k + \sigma_{k'}}{d(\boldsymbol{\mu}_k, \boldsymbol{\mu}_{k'})}$$

**Lower is better** (well-separated clusters).

## Mathematical Examples

### Example 1: 2D K-Means (K=2)

**Data**: 6 points in 2D
$$\begin{array}{|c|c|c|}
\hline
\text{Point} & x_1 & x_2 \\
\hline
A & 1 & 1 \\
B & 2 & 1 \\
C & 4 & 3 \\
D & 5 & 4 \\
E & 1 & 2 \\
F & 4 & 4 \\
\hline
\end{array}$$

**Iteration 0** (Initialization):
- Random centroids: $\boldsymbol{\mu}_1 = (1, 1)$, $\boldsymbol{\mu}_2 = (5, 4)$

**Iteration 1**:

*Assignment Step*:
- $d(A, \boldsymbol{\mu}_1) = \sqrt{(1-1)^2 + (1-1)^2} = 0$ → Cluster 1
- $d(A, \boldsymbol{\mu}_2) = \sqrt{(1-5)^2 + (1-4)^2} = 5$ → Cluster 1

- $d(B, \boldsymbol{\mu}_1) = \sqrt{1^2 + 0^2} = 1$ → Cluster 1
- $d(B, \boldsymbol{\mu}_2) = \sqrt{9 + 9} = 4.24$ → Cluster 1

- $d(C, \boldsymbol{\mu}_1) = \sqrt{9 + 4} = 3.61$ → Cluster 2
- $d(C, \boldsymbol{\mu}_2) = \sqrt{1 + 1} = 1.41$ → Cluster 2

- $d(D, \boldsymbol{\mu}_1) = \sqrt{16 + 9} = 5$ → Cluster 2
- $d(D, \boldsymbol{\mu}_2) = 0$ → Cluster 2

- $d(E, \boldsymbol{\mu}_1) = \sqrt{0 + 1} = 1$ → Cluster 1
- $d(E, \boldsymbol{\mu}_2) = \sqrt{16 + 4} = 4.47$ → Cluster 1

- $d(F, \boldsymbol{\mu}_1) = \sqrt{9 + 9} = 4.24$ → Cluster 2
- $d(F, \boldsymbol{\mu}_2) = \sqrt{1 + 0} = 1$ → Cluster 2

**Clusters**:
- $C_1 = \{A, B, E\}$
- $C_2 = \{C, D, F\}$

*Update Step*:
$$\boldsymbol{\mu}_1 = \frac{1}{3}[(1,1) + (2,1) + (1,2)] = \frac{1}{3}(4, 4) = (1.33, 1.33)$$

$$\boldsymbol{\mu}_2 = \frac{1}{3}[(4,3) + (5,4) + (4,4)] = \frac{1}{3}(13, 11) = (4.33, 3.67)$$

**Iteration 2**:
Reassign with new centroids... (continues until convergence)

### Example 2: Inertia Calculation

Given final clusters from Example 1:
- $C_1 = \{A, B, E\}$, $\boldsymbol{\mu}_1 = (1.33, 1.33)$
- $C_2 = \{C, D, F\}$, $\boldsymbol{\mu}_2 = (4.33, 3.67)$

**Inertia**:
$$J = \sum_{k=1}^{2} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

For Cluster 1:
$$J_1 = (1-1.33)^2 + (1-1.33)^2 + (2-1.33)^2 + (1-1.33)^2 + (1-1.33)^2 + (2-1.33)^2$$
$$= 0.11 + 0.11 + 0.45 + 0.11 + 0.11 + 0.45 = 1.34$$

For Cluster 2:
$$J_2 = (4-4.33)^2 + (3-3.67)^2 + (5-4.33)^2 + (4-3.67)^2 + (4-4.33)^2 + (4-3.67)^2$$
$$= 0.11 + 0.45 + 0.45 + 0.11 + 0.11 + 0.11 = 1.34$$

**Total Inertia**:
$$J = J_1 + J_2 = 1.34 + 1.34 = 2.68$$

## Implementation in Python

### Basic K-Means with Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, 
                       cluster_std=0.60, random_state=42)

# Create and fit K-Means model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("="*60)
print("K-MEANS CLUSTERING RESULTS")
print("="*60)
print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"\nCentroids:\n{centroids}")

# Visualize
plt.figure(figsize=(12, 5))

# Original data with true labels
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
            s=50, alpha=0.6, edgecolors='k')
plt.title('True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-Means clustering
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
            s=50, alpha=0.6, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', 
            linewidths=2, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
```

### K-Means from Scratch

```python
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        
        # Randomly initialize centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, 
                                          replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # Assignment step
            labels = self._assign_clusters(X)
            
            # Update step
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration+1}")
                break
                
            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels)
        
        return self
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Recompute centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def _calculate_inertia(self, X, labels):
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += ((cluster_points - self.centroids[k])**2).sum()
        return inertia
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X)

# Test custom implementation
custom_kmeans = KMeansFromScratch(n_clusters=4, random_state=42)
custom_kmeans.fit(X)

print(f"\nCustom K-Means Inertia: {custom_kmeans.inertia_:.2f}")
print(f"Sklearn K-Means Inertia: {kmeans.inertia_:.2f}")
```

### Elbow Method for Optimal K

```python
from sklearn.metrics import silhouette_score

# Test different values of K
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow method
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
ax1.set_title('Elbow Method for Optimal K', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=4, color='r', linestyle='--', label='Optimal K=4')
ax1.legend()

# Silhouette score
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score vs K', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=4, color='r', linestyle='--', label='Optimal K=4')
ax2.legend()

plt.tight_layout()
plt.show()

# Print results
print("\nK\tInertia\t\tSilhouette")
print("-" * 40)
for k, inertia, score in zip(K_range, inertias, silhouette_scores):
    print(f"{k}\t{inertia:.2f}\t\t{score:.4f}")
```

### K-Means++ Initialization

```python
def kmeans_plus_plus(X, n_clusters, random_state=None):
    """K-Means++ initialization algorithm"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Step 1: Choose first centroid randomly
    centroids = [X[np.random.choice(n_samples)]]
    
    # Step 2-K: Choose remaining centroids
    for _ in range(1, n_clusters):
        # Compute distances to nearest existing centroid
        distances = np.array([min([np.linalg.norm(x - c)**2 
                                   for c in centroids]) 
                             for x in X])
        
        # Choose next centroid with probability proportional to D(x)²
        probabilities = distances / distances.sum()
        cumulative_probs = probabilities.cumsum()
        r = np.random.rand()
        
        for idx, cum_prob in enumerate(cumulative_probs):
            if r < cum_prob:
                centroids.append(X[idx])
                break
    
    return np.array(centroids)

# Compare random init vs K-Means++
results = []

for method in ['random', 'k-means++']:
    inertias_method = []
    for _ in range(10):  # 10 random trials
        kmeans = KMeans(n_clusters=4, init=method, n_init=1, random_state=None)
        kmeans.fit(X)
        inertias_method.append(kmeans.inertia_)
    results.append(inertias_method)

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=['Random Init', 'K-Means++'])
plt.ylabel('Inertia', fontsize=12)
plt.title('Initialization Method Comparison (10 trials)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print("\nInitialization Comparison:")
print(f"Random Init - Mean: {np.mean(results[0]):.2f}, Std: {np.std(results[0]):.2f}")
print(f"K-Means++   - Mean: {np.mean(results[1]):.2f}, Std: {np.std(results[1]):.2f}")
```

### Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Large dataset
X_large, _ = make_blobs(n_samples=100000, centers=5, 
                        cluster_std=0.8, random_state=42)

# Standard K-Means
start = time.time()
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_large)
time_kmeans = time.time() - start

# Mini-Batch K-Means
start = time.time()
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=1000, 
                            random_state=42)
mb_kmeans.fit(X_large)
time_mb_kmeans = time.time() - start

print("\n" + "="*60)
print("STANDARD vs MINI-BATCH K-MEANS")
print("="*60)
print(f"Dataset size: {X_large.shape[0]:,} samples\n")
print(f"Standard K-Means:")
print(f"  Time: {time_kmeans:.2f}s")
print(f"  Inertia: {kmeans.inertia_:.2f}\n")
print(f"Mini-Batch K-Means:")
print(f"  Time: {time_mb_kmeans:.2f}s")
print(f"  Inertia: {mb_kmeans.inertia_:.2f}\n")
print(f"Speedup: {time_kmeans/time_mb_kmeans:.1f}x")
```

## Practical Applications

### 1. **Customer Segmentation**
Group customers by purchasing behavior for targeted marketing.

```python
# Example: Customer features
features = ['Age', 'Income', 'Purchase_Frequency', 'Avg_Basket_Size']
# K-Means to find customer segments
```

### 2. **Image Compression**
Reduce colors in image by clustering pixel values.

```python
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

# Load image
image = mpimg.imread('image.jpg')
pixels = image.reshape(-1, 3)

# Cluster to K colors
kmeans = KMeans(n_clusters=16, random_state=42)
labels = kmeans.fit_predict(pixels)
compressed = kmeans.cluster_centers_[labels].reshape(image.shape)
```

### 3. **Document Clustering**
Group similar documents using TF-IDF features.

### 4. **Anomaly Detection**
Points far from all centroids are potential anomalies.

### 5. **Feature Engineering**
Cluster membership as categorical feature for supervised learning.

## Summary

K-Means is a foundational clustering algorithm that partitions data into K clusters.

**Key Mathematical Concepts**:
1. **Objective**: Minimize $J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
2. **Centroid**: $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$
3. **Assignment**: $c_i = \arg\min_{k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
4. **Guaranteed Convergence**: Inertia decreases monotonically

**Choosing K**:
- **Elbow Method**: Look for bend in inertia curve
- **Silhouette Score**: Maximize average silhouette
- **Gap Statistic**: Compare to null distribution
- **Domain Knowledge**: Use business context

**Best Practices**:
✅ Normalize features before clustering
✅ Use K-Means++ initialization
✅ Run multiple times with different random seeds
✅ Validate with silhouette score
✅ Consider Mini-Batch K-Means for large data
✅ Visualize clusters when possible

❌ Don't forget to scale features
❌ Don't use with categorical data
❌ Don't assume K-Means works for all cluster shapes
❌ Don't ignore outliers
❌ Don't use without validating K

K-Means is fast, simple, and effective for spherical clusters!
