# Principal Component Analysis (PCA)

## Table of Contents
- [Introduction](#introduction)
- [What is PCA?](#what-is-pca)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use PCA?](#why-use-pca)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Step-by-Step Algorithm](#step-by-step-algorithm)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Interpreting PCA Results](#interpreting-pca-results)
- [Practical Applications](#practical-applications)
- [Comparison with Other Methods](#comparison-with-other-methods)

## Introduction

Principal Component Analysis (PCA) is one of the most fundamental and widely used dimensionality reduction techniques in machine learning and data science. It transforms a dataset with many correlated variables into a smaller set of uncorrelated variables called **principal components**, while retaining as much of the original information (variance) as possible.

PCA is both an art and a science:
- **Science**: Rigorous mathematical foundation based on linear algebra
- **Art**: Choosing the right number of components requires judgment and domain knowledge

## What is PCA?

PCA is an **unsupervised** linear transformation technique that:

1. **Identifies patterns** in high-dimensional data
2. **Reduces dimensionality** while preserving variance
3. **Removes correlation** between variables
4. **Creates new features** (principal components) as linear combinations of original features

### Key Concepts

**Principal Components (PCs)**:
- Orthogonal (perpendicular) vectors in feature space
- Ordered by variance explained (PC1 > PC2 > PC3 > ...)
- Linear combinations of original features

**Dimensionality Reduction**:
- Transform from $n$ dimensions to $k$ dimensions (where $k < n$)
- Keep the $k$ components that explain most variance
- Discard components with low variance

**Variance Preservation**:
- First PC captures maximum variance
- Second PC captures maximum remaining variance (orthogonal to PC1)
- Continue until all variance is explained

## Mathematical Foundation

### Problem Formulation

Given a dataset $\mathbf{X} \in \mathbb{R}^{n \times p}$ with:
- $n$ samples (observations)
- $p$ features (dimensions)

**Goal**: Find a lower-dimensional representation $\mathbf{Z} \in \mathbb{R}^{n \times k}$ where $k < p$

### Covariance Matrix

The covariance matrix $\mathbf{\Sigma}$ captures the relationships between features:

$$\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$$

For mean-centered data, the covariance between features $i$ and $j$ is:

$$\text{Cov}(X_i, X_j) = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$

The covariance matrix is:
- **Symmetric**: $\mathbf{\Sigma} = \mathbf{\Sigma}^T$
- **Positive semi-definite**: All eigenvalues $\geq 0$
- **Size**: $p \times p$ matrix

### Eigenvalue Decomposition

PCA finds eigenvectors and eigenvalues of the covariance matrix:

$$\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v}$$

where:
- $\mathbf{v}$ is an eigenvector (principal component direction)
- $\lambda$ is an eigenvalue (variance along that direction)

**Spectral Decomposition**:

$$\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$

where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_p]$ is the matrix of eigenvectors
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_p)$ is the diagonal matrix of eigenvalues
- $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p \geq 0$

### Principal Components

The $i$-th principal component is:

$$\mathbf{z}_i = \mathbf{X} \mathbf{v}_i$$

where $\mathbf{v}_i$ is the $i$-th eigenvector.

In matrix form, the projection onto $k$ components:

$$\mathbf{Z} = \mathbf{X} \mathbf{V}_k$$

where $\mathbf{V}_k = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k]$ contains the first $k$ eigenvectors.

### Variance Explained

The variance explained by the $i$-th principal component:

$$\text{Var}(PC_i) = \lambda_i$$

**Proportion of variance** explained by $PC_i$:

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

**Cumulative variance** explained by first $k$ components:

$$\text{Cumulative Variance}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

### Reconstruction

To reconstruct data from reduced dimensions:

$$\mathbf{\hat{X}} = \mathbf{Z} \mathbf{V}_k^T$$

**Reconstruction error**:

$$\text{Error} = ||\mathbf{X} - \mathbf{\hat{X}}||^2 = \sum_{i=k+1}^{p} \lambda_i$$

## Why Use PCA?

### 1. **Curse of Dimensionality**

High-dimensional data suffers from:
- Sparse data points (distance between points increases)
- Computational complexity ($O(p^2)$ or worse)
- Overfitting in machine learning models
- Difficulty in visualization

PCA reduces dimensions while retaining information.

### 2. **Feature Correlation**

When features are highly correlated:
- Redundant information
- Multicollinearity in regression
- Inefficient models

PCA creates uncorrelated components.

### 3. **Visualization**

Humans can visualize up to 3 dimensions. PCA enables:
- 2D scatter plots (PC1 vs PC2)
- 3D visualizations (PC1, PC2, PC3)
- Pattern identification
- Cluster visualization

### 4. **Noise Reduction**

Components with low variance often represent noise:
- Keep high-variance components (signal)
- Discard low-variance components (noise)
- Improves model robustness

### 5. **Computational Efficiency**

Fewer features mean:
- Faster model training
- Lower memory requirements
- Easier hyperparameter tuning
- Simpler model interpretation

## Advantages and Disadvantages

### Advantages

1. **Reduces Dimensionality**: Decreases from $p$ to $k$ features
2. **Removes Correlation**: Creates orthogonal components
3. **Preserves Variance**: Retains most important information
4. **Unsupervised**: Doesn't require labels
5. **Interpretable**: Components show feature importance
6. **Noise Filtering**: Removes low-variance noise
7. **Visualization**: Enables 2D/3D plotting
8. **No Parameters**: Deterministic algorithm
9. **Computational**: Efficient for large datasets

### Disadvantages

1. **Linear Assumption**: Only captures linear relationships
2. **Interpretability Loss**: Components are combinations of features
3. **Sensitive to Scale**: Requires standardization
4. **Variance ≠ Importance**: High variance may not mean high predictive power
5. **Information Loss**: Discarding components loses some information
6. **Outlier Sensitivity**: Extreme values affect covariance
7. **Choosing $k$**: No automatic way to select number of components
8. **Not Suitable for**: Categorical data, sparse data, non-linear patterns

## Step-by-Step Algorithm

### PCA Algorithm

**Input**: Dataset $\mathbf{X} \in \mathbb{R}^{n \times p}$, number of components $k$

**Output**: Transformed data $\mathbf{Z} \in \mathbb{R}^{n \times k}$

**Steps**:

1. **Standardize the data** (mean = 0, std = 1):
   $$\mathbf{X}_{std} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
   where $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\sigma}$ is the standard deviation vector

2. **Compute the covariance matrix**:
   $$\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}_{std}^T \mathbf{X}_{std}$$

3. **Calculate eigenvalues and eigenvectors**:
   $$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

4. **Sort eigenvectors** by eigenvalues in descending order:
   $$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p$$

5. **Select top $k$ eigenvectors**:
   $$\mathbf{V}_k = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k]$$

6. **Transform the data**:
   $$\mathbf{Z} = \mathbf{X}_{std} \mathbf{V}_k$$

7. **Return** $\mathbf{Z}$ and variance explained

### Pseudocode

```
function PCA(X, k):
    // Step 1: Standardize
    μ = mean(X, axis=0)
    σ = std(X, axis=0)
    X_std = (X - μ) / σ
    
    // Step 2: Covariance matrix
    Σ = (1/(n-1)) * X_std^T * X_std
    
    // Step 3: Eigendecomposition
    (eigenvalues, eigenvectors) = eig(Σ)
    
    // Step 4: Sort
    indices = argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    
    // Step 5: Select k components
    V_k = eigenvectors[:, 0:k]
    
    // Step 6: Transform
    Z = X_std * V_k
    
    // Calculate variance explained
    variance_explained = eigenvalues[0:k] / sum(eigenvalues)
    
    return Z, variance_explained, V_k
```

## Mathematical Examples

### Example 1: Simple 2D to 1D Reduction

**Dataset**: 5 samples, 2 features

$$\mathbf{X} = \begin{bmatrix} 
2 & 3 \\
3 & 4 \\
4 & 5 \\
5 & 6 \\
6 & 7
\end{bmatrix}$$

**Step 1: Standardize**

Mean: $\boldsymbol{\mu} = [4, 5]$

Std: $\boldsymbol{\sigma} = [1.58, 1.58]$

$$\mathbf{X}_{std} = \begin{bmatrix} 
-1.26 & -1.26 \\
-0.63 & -0.63 \\
0 & 0 \\
0.63 & 0.63 \\
1.26 & 1.26
\end{bmatrix}$$

**Step 2: Covariance Matrix**

$$\mathbf{\Sigma} = \frac{1}{4} \mathbf{X}_{std}^T \mathbf{X}_{std} = \begin{bmatrix} 
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}$$

**Step 3: Eigenvalues and Eigenvectors**

Solving $|\mathbf{\Sigma} - \lambda \mathbf{I}| = 0$:

$$\begin{vmatrix} 
1-\lambda & 1 \\
1 & 1-\lambda
\end{vmatrix} = 0$$

$$(1-\lambda)^2 - 1 = 0$$

$$\lambda^2 - 2\lambda = 0$$

$$\lambda(\lambda - 2) = 0$$

**Eigenvalues**: $\lambda_1 = 2, \lambda_2 = 0$

For $\lambda_1 = 2$:
$$\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

For $\lambda_2 = 0$:
$$\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

**Step 4: Project onto PC1**

$$\mathbf{Z} = \mathbf{X}_{std} \mathbf{v}_1 = \begin{bmatrix} 
-1.26 & -1.26 \\
-0.63 & -0.63 \\
0 & 0 \\
0.63 & 0.63 \\
1.26 & 1.26
\end{bmatrix} \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix} = \begin{bmatrix} 
-1.78 \\
-0.89 \\
0 \\
0.89 \\
1.78
\end{bmatrix}$$

**Variance Explained**:
$$\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{2}{2+0} = 100\%$$

All variance is captured by PC1!

### Example 2: Choosing Number of Components

**Eigenvalues**: $\lambda = [25, 15, 8, 5, 2]$

**Total Variance**: $\sum \lambda = 55$

**Variance Explained**:

| Component | Eigenvalue | Individual | Cumulative |
|-----------|-----------|------------|------------|
| PC1 | 25 | 45.5% | 45.5% |
| PC2 | 15 | 27.3% | 72.7% |
| PC3 | 8 | 14.5% | 87.3% |
| PC4 | 5 | 9.1% | 96.4% |
| PC5 | 2 | 3.6% | 100.0% |

**Decision**: Choose $k=3$ to retain ~87% of variance.

## Implementation in Python

### Basic PCA Implementation from Scratch

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA_FromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        
    def fit(self, X):
        """
        Fit PCA on dataset X
        """
        # Step 1: Center the data (mean = 0)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Step 4: Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues
        
        return self
    
    def transform(self, X):
        """
        Transform X using fitted principal components
        """
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        """
        self.fit(X)
        return self.transform(X)
    
    def explained_variance_ratio(self):
        """
        Return proportion of variance explained by each component
        """
        return self.eigenvalues[:self.n_components] / np.sum(self.eigenvalues)
    
    def inverse_transform(self, Z):
        """
        Reconstruct original data from principal components
        """
        return Z @ self.components.T + self.mean

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)

pca = PCA_FromScratch(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("\nExplained variance ratio:", pca.explained_variance_ratio())
print("Cumulative variance:", np.sum(pca.explained_variance_ratio()))
```

### Using Scikit-Learn PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 10)

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print("="*60)
print("PCA Results")
print("="*60)
print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nCumulative explained variance:")
cumsum = np.cumsum(pca.explained_variance_ratio_)
for i, var in enumerate(cumsum):
    print(f"  PC1-PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nEigenvalues (variance):")
for i, val in enumerate(pca.explained_variance_):
    print(f"  PC{i+1}: {val:.4f}")

# Principal component loadings
print(f"\nPrincipal Components (loadings):")
components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components)],
    index=[f'Feature{i+1}' for i in range(X.shape[1])]
)
print(components_df)
```

### Visualizing PCA

```python
def plot_pca_analysis(X, y=None, n_components=2):
    """
    Complete PCA analysis with visualizations
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA with all components first
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Then reduce to n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scree Plot (Variance Explained)
    ax1 = plt.subplot(2, 3, 1)
    components = range(1, len(pca_full.explained_variance_ratio_) + 1)
    ax1.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, 
            color='skyblue', edgecolor='black')
    ax1.plot(components, pca_full.explained_variance_ratio(), 'ro-', linewidth=2)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.grid(alpha=0.3)
    
    # 2. Cumulative Variance
    ax2 = plt.subplot(2, 3, 2)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax2.plot(components, cumsum, 'bo-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.axhline(y=0.90, color='g', linestyle='--', label='90% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. PC1 vs PC2 Scatter
    ax3 = plt.subplot(2, 3, 3)
    if y is not None:
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                             alpha=0.6, edgecolor='black', s=50)
        plt.colorbar(scatter, ax=ax3)
    else:
        ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, 
                   edgecolor='black', s=50, color='skyblue')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('First Two Principal Components')
    ax3.grid(alpha=0.3)
    
    # 4. Component Loadings Heatmap
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(pca.components_, cmap='RdBu_r', aspect='auto')
    ax4.set_xlabel('Original Features')
    ax4.set_ylabel('Principal Components')
    ax4.set_title('Component Loadings')
    ax4.set_yticks(range(n_components))
    ax4.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    plt.colorbar(im, ax=ax4)
    
    # 5. Biplot (PC1 vs PC2 with feature vectors)
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=30, color='gray')
    
    # Plot feature vectors
    for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
        ax5.arrow(0, 0, comp1*3, comp2*3, head_width=0.1, head_length=0.1,
                 fc='red', ec='red', alpha=0.7)
        ax5.text(comp1*3.2, comp2*3.2, f'F{i+1}', color='red', fontsize=9)
    
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax5.set_title('Biplot: Samples and Features')
    ax5.grid(alpha=0.3)
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.axvline(0, color='black', linewidth=0.5)
    
    # 6. Variance explained table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    variance_data = []
    for i in range(min(10, len(pca_full.explained_variance_ratio_))):
        variance_data.append([
            f'PC{i+1}',
            f'{pca_full.explained_variance_ratio_[i]:.4f}',
            f'{cumsum[i]:.4f}',
            f'{pca_full.explained_variance_[i]:.4f}'
        ])
    
    table = ax6.table(cellText=variance_data,
                     colLabels=['Component', 'Variance', 'Cumulative', 'Eigenvalue'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax6.set_title('Variance Explained Summary', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return X_pca, pca

# Example usage
np.random.seed(42)
X_sample = np.random.randn(150, 8)
y_sample = np.random.randint(0, 3, 150)

X_transformed, pca_model = plot_pca_analysis(X_sample, y_sample, n_components=3)
```

### Complete Real-World Example

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assume we have a dataset with many features
# Example: Load your data
# df = pd.read_csv('your_data.csv')

# For demonstration, create synthetic data
np.random.seed(42)
n_samples = 500
n_features = 20

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(n_samples)*0.5 > 0).astype(int)

print("="*70)
print("PCA FOR DIMENSIONALITY REDUCTION - Complete Pipeline")
print("="*70)
print(f"\nOriginal dataset: {n_samples} samples × {n_features} features")

# Step 1: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 2: Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Apply PCA to determine optimal components
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Find number of components for 95% variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1

print(f"\nComponents needed for 95% variance: {n_components_95}")
print(f"Components needed for 90% variance: {np.argmax(cumsum >= 0.90) + 1}")

# Step 4: Apply PCA with chosen components
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nReduced dataset: {n_samples} samples × {n_components_95} features")
print(f"Dimensionality reduction: {n_features} → {n_components_95} "
      f"({n_components_95/n_features*100:.1f}%)")

# Step 5: Train models - Compare with and without PCA
print(f"\n{'='*70}")
print("MODEL COMPARISON: Original vs PCA-reduced features")
print(f"{'='*70}")

# Model on original features
model_original = LogisticRegression(random_state=42, max_iter=1000)
model_original.fit(X_train_scaled, y_train)
y_pred_original = model_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)

# Model on PCA features
model_pca = LogisticRegression(random_state=42, max_iter=1000)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"\n{'Model':<30} {'Accuracy':<15} {'Features':<15}")
print("-"*60)
print(f"{'Original Features':<30} {acc_original:<15.4f} {n_features:<15}")
print(f"{'PCA Features':<30} {acc_pca:<15.4f} {n_components_95:<15}")
print(f"\nAccuracy difference: {abs(acc_original - acc_pca):.4f}")

# Step 6: Visualize
print(f"\n{'='*70}")
print("PCA ANALYSIS")
print(f"{'='*70}")

print(f"\nTop 5 Principal Components:")
for i in range(min(5, len(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
          f"({pca.explained_variance_ratio_[i]*100:.2f}%) "
          f"- Eigenvalue: {pca.explained_variance_[i]:.4f}")

print(f"\nCumulative variance: {np.sum(pca.explained_variance_ratio_):.4f} "
      f"({np.sum(pca.explained_variance_ratio_)*100:.2f}%)")
```

## Interpreting PCA Results

### 1. **Explained Variance Ratio**

Shows how much information each component captures:
- PC1 typically explains 20-50% of variance
- First 2-3 components often explain 60-80%
- Use cumulative variance to choose $k$

### 2. **Component Loadings**

Each principal component is a linear combination:

$$PC_1 = w_{11}x_1 + w_{12}x_2 + ... + w_{1p}x_p$$

High absolute values of $w_{ij}$ indicate feature $j$ is important for component $i$.

### 3. **Scree Plot**

Plot eigenvalues vs component number:
- Look for "elbow" where variance drops sharply
- Components after elbow contribute little
- Choose $k$ at or before elbow

### 4. **Biplot**

Combines:
- Sample scores (dots): How samples project onto PCs
- Variable loadings (arrows): How original features relate to PCs
- Long arrows = important features
- Arrow direction = correlation with PCs

## Practical Applications

### 1. **Image Compression**

Reduce image dimensions while preserving visual quality:
- Original: 1000×1000 pixels = 1M dimensions
- PCA: Keep 100 components
- Compression ratio: 99.99%
- Reconstruct with minimal loss

### 2. **Face Recognition (Eigenfaces)**

Represent faces as linear combinations of "eigenfaces":
- Each face = weights on principal components
- Fast face matching
- Dimensionality: 10,000 pixels → 50-100 components

### 3. **Gene Expression Analysis**

Analyze thousands of genes:
- Original: 20,000 genes (features)
- PCA: Reduce to 10-20 components
- Identify gene expression patterns
- Cluster similar samples

### 4. **Anomaly Detection**

Detect outliers using reconstruction error:
- Normal samples reconstruct well
- Anomalies have high reconstruction error
- Threshold on $||\mathbf{x} - \hat{\mathbf{x}}||^2$

### 5. **Data Visualization**

Visualize high-dimensional clusters:
- Customer segmentation
- Document clustering
- Species classification
- 2D plot of complex data

## Comparison with Other Methods

### PCA vs t-SNE

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| **Type** | Linear | Non-linear |
| **Purpose** | Dimensionality reduction | Visualization |
| **Speed** | Fast | Slow |
| **Preservation** | Global structure | Local structure |
| **Deterministic** | Yes | No (random initialization) |
| **Interpretation** | Easy (linear combinations) | Difficult |

### PCA vs Autoencoders

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| **Method** | Linear algebra | Neural network |
| **Complexity** | Simple | Complex |
| **Non-linearity** | No | Yes |
| **Training** | Closed-form solution | Iterative optimization |
| **Overfitting** | No | Possible |

### PCA vs Feature Selection

| Aspect | PCA | Feature Selection |
|--------|-----|-------------------|
| **Creates Features** | Yes (combinations) | No (selects subset) |
| **Interpretability** | Lower | Higher |
| **Information** | Preserves max variance | May lose information |
| **Correlation** | Removes correlation | Retains correlation |

## Summary

PCA is a powerful technique for dimensionality reduction that transforms correlated variables into uncorrelated principal components while preserving maximum variance.

**Key Mathematical Concepts**:

1. **Covariance Matrix**: $\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$
2. **Eigendecomposition**: $\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v}$
3. **Transformation**: $\mathbf{Z} = \mathbf{X} \mathbf{V}_k$
4. **Variance Explained**: $\frac{\lambda_i}{\sum_j \lambda_j}$
5. **Reconstruction**: $\mathbf{\hat{X}} = \mathbf{Z} \mathbf{V}_k^T$

**Best Practices**:

✅ Always standardize data before PCA
✅ Examine scree plot and cumulative variance
✅ Choose $k$ to retain 90-95% variance
✅ Check component loadings for interpretation
✅ Validate model performance with cross-validation

❌ Don't use PCA on categorical data
❌ Don't assume variance equals importance
❌ Don't forget to transform test data the same way
❌ Don't ignore outliers (they affect PCA)
❌ Don't use PCA if interpretability is critical

PCA remains one of the most fundamental and widely-used techniques in machine learning, serving as both a preprocessing step and a powerful analytical tool!
