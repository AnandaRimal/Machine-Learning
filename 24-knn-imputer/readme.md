# KNN Imputer - Distance-Based Imputation Using Nearest Neighbors

## General Idea

KNN Imputer (K-Nearest Neighbors Imputer) fills missing values by finding the K most similar samples (neighbors) based on other features, then using the average (for numerical) or mode (for categorical) of those neighbors' values. Unlike simple imputation that uses global statistics, KNN imputation leverages local patterns and feature relationships, providing context-aware estimates that reflect the similarity structure in the data.

## Why Use KNN Imputation?

1. **Uses Feature Relationships**: Considers correlations between features
2. **Locally Adaptive**: Different imputation for different contexts
3. **More Accurate**: Often better than mean/median for MAR data
4. **Preserves Variance**: Maintains data variability better
5. **Handles Non-Linear Relationships**: Doesn't assume linearity
6. **Multivariate**: Uses all available features
7. **Intuitive**: "Similar samples have similar values"
8. **No Model Training**: Non-parametric approach

## Role in Machine Learning

### Comparison with Simple Imputation

**Mean Imputation**:
```
Age  Income  Education  →  Imputed_Income
25   50k     Bachelor       (Global mean)
30   NaN     Master    →    60k
45   NaN     PhD       →    60k
50   70k     Bachelor       (Global mean)
```

**All missing values** get same global mean (60k)

**Ignores**: Education level, Age

**KNN Imputation (K=2)**:
```
Age  Income  Education  →  Imputed_Income
25   50k     Bachelor       
30   NaN     Master    →    Find 2 nearest neighbors based on Age, Education
45   NaN     PhD       →    Different neighbors!
50   70k     Bachelor       
```

**For row 2** (Age=30, Master):
- Find K=2 most similar rows
- Similar in Age and Education
- Average their Income
- Result: ~60k (younger, master level)

**For row 3** (Age=45, PhD):
- Different nearest neighbors (older, PhD)
- Result: ~70k (higher education, older)

**KNN adapts** imputation to context!

### Mathematical Foundation

**Goal**: Impute missing value for sample $i$ in feature $j$

**Step 1**: Compute distance to all other samples using **complete features**

**Distance metric** (Euclidean):
$$d(i, i') = \sqrt{\sum_{k \in \text{complete}} (x_{ik} - x_{i'k})^2}$$

Where "complete" = features observed for both $i$ and $i'$

**Step 2**: Find K nearest neighbors
$$\mathcal{N}_K(i) = \{i'_1, i'_2, ..., i'_K\}$$

The K samples with smallest distance to $i$

**Step 3**: Impute as average of neighbors
$$\hat{x}_{ij} = \frac{1}{K} \sum_{i' \in \mathcal{N}_K(i)} x_{i'j}$$

**For categorical**: Use mode instead of mean

### Example Calculation

**Data**:
```
     Age  Income  Score
 0   25   50      80
 1   30   NaN     85
 2   28   55      82
 3   45   70      90
 4   27   52      81
```

**Impute Income for sample 1** (K=2)

**Step 1**: Compute distance using Age and Score

$$d(1, 0) = \sqrt{(30-25)^2 + (85-80)^2} = \sqrt{25 + 25} = 7.07$$

$$d(1, 2) = \sqrt{(30-28)^2 + (85-82)^2} = \sqrt{4 + 9} = 3.61$$

$$d(1, 3) = \sqrt{(30-45)^2 + (85-90)^2} = \sqrt{225 + 25} = 15.81$$

$$d(1, 4) = \sqrt{(30-27)^2 + (85-81)^2} = \sqrt{9 + 16} = 5.00$$

**Step 2**: Find 2 nearest neighbors

Smallest distances: $d(1,2) = 3.61$, $d(1,4) = 5.00$

Nearest neighbors: samples 2 and 4

**Step 3**: Average their Income
$$\hat{x}_{1,\text{Income}} = \frac{55 + 52}{2} = 53.5$$

**Result**: Impute Income[1] = 53.5

**Interpretation**: Sample 1 (Age 30, Score 85) is similar to samples 2 and 4 (younger, mid-80s score), so gets their average income (~53.5) rather than global mean (56.75).

## Scikit-Learn KNNImputer

### Basic Usage

```python
from sklearn.impute import KNNImputer
import numpy as np

X = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [7, 8, 9],
    [10, 11, 12]
])

imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)

print(X_imputed)
# [[1.  2.  3. ]
#  [4.  5.  6. ]  ← Imputed with neighbors
#  [7.  8.  9. ]
#  [10. 11. 12.]]
```

### Parameters

**n_neighbors**: int, default=5
- Number of neighbors to use
- Higher K: More smoothing (global)
- Lower K: More local (variance)
- Typical: 3-10

**weights**: 'uniform' or 'distance', default='uniform'
- `'uniform'`: All neighbors equal weight
  $$\hat{x}_{ij} = \frac{1}{K} \sum_{i' \in \mathcal{N}_K} x_{i'j}$$
  
- `'distance'`: Weight by inverse distance
  $$\hat{x}_{ij} = \frac{\sum_{i' \in \mathcal{N}_K} w_{i'} x_{i'j}}{\sum_{i' \in \mathcal{N}_K} w_{i'}}$$
  
  Where $w_{i'} = \frac{1}{d(i, i')}$ (closer neighbors have more influence)

**metric**: str or callable, default='nan_euclidean'
- Distance metric
- `'nan_euclidean'`: Euclidean ignoring NaNs (default, recommended)
- `'manhattan'`: $L_1$ distance $\sum |x_i - x_{i'}|$
- `'cosine'`: Cosine similarity
- Custom: Any function(X, Y) → distance matrix

**add_indicator**: bool, default=False
- If True, add missing indicator features
- Same as SimpleImputer's add_indicator

**keep_empty_features**: bool, default=False
- If True, keep features that are all NaN (filled with 0)
- If False, drop them

### Example with Parameters

```python
from sklearn.impute import KNNImputer

# Distance-weighted, 3 neighbors, Manhattan metric
imputer = KNNImputer(
    n_neighbors=3,
    weights='distance',
    metric='manhattan',
    add_indicator=True
)

X_imputed = imputer.fit_transform(X_train)
# Returns: [imputed_features, missing_indicators]
```

## Choosing K (Number of Neighbors)

### Effect of K

**Small K** (e.g., K=1):
- Very local
- High variance
- Sensitive to noise
- Overfitting risk

**Large K** (e.g., K=50):
- More global
- Low variance
- Smoothing effect
- Approaches mean imputation

**Optimal K**: Balance between local accuracy and stability

### Selection Strategy

**Rule of thumb**: $K = \sqrt{n}$ where $n$ = number of samples

**Cross-validation**:
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

best_k = None
best_score = 0

for k in [3, 5, 7, 10, 15, 20]:
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=k)),
        ('classifier', RandomForestClassifier())
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_score = scores.mean()
    
    print(f"K={k}: {mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nBest K: {best_k}")
```

**Dataset size guidance**:
- Small (n < 100): K = 3-5
- Medium (100 < n < 1000): K = 5-10
- Large (n > 1000): K = 10-20

### Empirical Example

**Dataset**: 500 samples, 10% missing

**Results**:
```
K=1:  Accuracy = 0.82  (too local, overfits)
K=3:  Accuracy = 0.86
K=5:  Accuracy = 0.88  ← Best
K=10: Accuracy = 0.87
K=20: Accuracy = 0.85  (too smooth)
K=50: Accuracy = 0.83  (approaching global mean)
```

**Optimal**: K=5 for this dataset

## Distance Weighting

### Uniform vs Distance Weights

**Uniform** (`weights='uniform'`):

**All K neighbors equal**:
$$\hat{x} = \frac{x_1 + x_2 + x_3}{3}$$

**Example**: K=3 neighbors with distances [2, 5, 8]

Values: [100, 110, 120]

Imputed: $(100 + 110 + 120) / 3 = 110$

**Distance-weighted** (`weights='distance'`):

**Closer neighbors have more influence**:
$$w_i = \frac{1}{d_i}$$

$$\hat{x} = \frac{\sum w_i x_i}{\sum w_i}$$

**Same example**:

Distances: [2, 5, 8]

Weights: $[1/2, 1/5, 1/8] = [0.5, 0.2, 0.125]$

Normalized: $[0.606, 0.242, 0.152]$

Values: [100, 110, 120]

Imputed: $0.606 \times 100 + 0.242 \times 110 + 0.152 \times 120 = 104.9$

**Result**: Closer to nearest neighbor (100) than uniform (110)

### When to Use Distance Weighting

**Use `weights='distance'`**:
- When closer neighbors more reliable
- Continuous features
- Non-uniform density data
- Want smoother transitions

**Use `weights='uniform'`**:
- Simpler, faster
- All neighbors equally reliable
- Small K (differences minimal)
- Categorical features

## Distance Metrics

### Euclidean (default: 'nan_euclidean')

**Formula**:
$$d(x, x') = \sqrt{\sum_{k=1}^p (x_k - x'_k)^2}$$

**Properties**:
- Standard distance
- Sensitive to scale (standardize first!)
- Works well for continuous features

**nan_euclidean**: Ignores missing values in distance calculation

$$d(x, x') = \sqrt{\frac{p}{p_{\text{common}}} \sum_{k \in \text{common}} (x_k - x'_k)^2}$$

Where $p_{\text{common}}$ = features observed in both samples

### Manhattan

**Formula**:
$$d(x, x') = \sum_{k=1}^p |x_k - x'_k|$$

**When to use**:
- Less sensitive to outliers
- High-dimensional data
- Features on different scales (if not standardized)

### Cosine

**Formula**:
$$d(x, x') = 1 - \frac{x \cdot x'}{||x|| ||x'||}$$

**When to use**:
- Direction matters more than magnitude
- Text data (TF-IDF vectors)
- Sparse features

## Feature Scaling for KNN Imputation

### Why Scaling Matters

**Problem**: Features on different scales dominate distance

**Example**:
```
Age    Income   Score
30     50000    85
35     55000    88
```

**Distance without scaling**:
$$d = \sqrt{(35-30)^2 + (55000-50000)^2 + (88-85)^2}$$
$$d = \sqrt{25 + 25000000 + 9} \approx 5000$$

**Income dominates** (5000 vs 5 vs 3)

**Distance with scaling** (standardized):
```
Age_scaled    Income_scaled   Score_scaled
-0.5          -0.5            -0.5
 0.5           0.5             0.5
```

$$d = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3} \approx 1.73$$

**Equal contribution** from all features

### Recommended Preprocessing

**Option 1: Scale Before KNN**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale first
    ('imputer', KNNImputer(n_neighbors=5))
])

X_imputed = pipeline.fit_transform(X)
```

**Problem**: StandardScaler can't handle NaN!

**Solution**: Temporary imputation for scaling

**Option 2: RobustScaler (handles outliers)**

```python
from sklearn.preprocessing import RobustScaler

# RobustScaler also requires no NaN
# Use SimpleImputer → RobustScaler → KNNImputer pattern
```

**Option 3: Manual Scaling with NaN Handling**

```python
import numpy as np

def scale_with_nan(X):
    """Standardize ignoring NaN"""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    return (X - mean) / std

X_scaled = scale_with_nan(X)
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_scaled)
```

**Best Practice**: 
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('simple_impute', SimpleImputer(strategy='median')),  # Fill NaN for scaling
    ('scale', StandardScaler()),                           # Scale
    ('knn_impute', KNNImputer(n_neighbors=5))             # Refine with KNN
])
```

Or use KNN directly (metric='nan_euclidean' handles scaling differences reasonably)

## Advantages and Limitations

### Advantages

1. **Uses feature relationships**: Imputes based on correlations
2. **Locally adaptive**: Context-aware imputation
3. **Preserves variance**: Better than mean imputation
4. **No model assumptions**: Non-parametric
5. **Handles non-linearity**: Doesn't assume linear relationships
6. **Multivariate**: Uses all features simultaneously
7. **Often more accurate**: Especially for MAR data
8. **Interpretable**: "Similar samples have similar values"

### Limitations

1. **Computationally expensive**: $O(n^2 p)$ for distance calculations
   - Slow for large datasets (n > 10,000)
   
2. **Sensitive to scaling**: Requires feature scaling

3. **Curse of dimensionality**: Performance degrades with many features
   - Distances become meaningless in high dimensions
   
4. **Sensitive to K**: Requires tuning n_neighbors

5. **Doesn't work well for MNAR**: Assumes MAR mechanism

6. **Requires sufficient data**: Needs enough samples for neighbors

7. **Can't extrapolate**: Imputed values within observed range

8. **Memory intensive**: Stores entire training set

## When to Use KNN Imputation

### Good Fit

**Use KNN Imputation when**:

1. **MAR data**: Missingness depends on observed features
2. **Strong feature correlations**: Features are related
3. **Moderate dataset size**: 100 < n < 10,000
4. **Low to moderate dimensionality**: p < 50
5. **Continuous features**: Numerical data
6. **Want higher accuracy**: Than simple imputation
7. **Can afford computation**: Have time/resources

### Poor Fit

**Avoid KNN Imputation when**:

1. **Large datasets**: n > 100,000 (too slow)
2. **High dimensionality**: p > 100 (curse of dimensionality)
3. **MCAR or MNAR**: KNN assumes MAR
4. **Mostly categorical**: KNN works best for numerical
5. **Real-time constraints**: Need fast imputation
6. **Simple baseline needed**: Mean/median sufficient

### Alternatives Comparison

**Simple Imputation** (Mean/Median):
- Pros: Fast, simple
- Cons: Ignores correlations
- Use when: Quick baseline, MCAR data

**KNN Imputation**:
- Pros: Uses correlations, more accurate
- Cons: Slow, needs tuning
- Use when: MAR data, accuracy important

**Iterative Imputation** (MICE):
- Pros: Models each feature, very accurate
- Cons: Even slower, complex
- Use when: High accuracy needed, complex relationships

**Model-based** (e.g., predict missing with RF):
- Pros: Very flexible, accurate
- Cons: Slow, risk of overfitting
- Use when: Critical feature, worth effort

## Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data with missing values
df = pd.read_csv('data_with_missing.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: Simple Imputation
baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

baseline_pipeline.fit(X_train, y_train)
y_pred_baseline = baseline_pipeline.predict(X_test)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"Baseline (Simple Imputation): {accuracy_baseline:.4f}")

# KNN Imputation
knn_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"KNN Imputation: {accuracy_knn:.4f}")
print(f"Improvement: {(accuracy_knn - accuracy_baseline)*100:.2f}%")

# Tune K
print("\nTuning K:")
for k in [3, 5, 7, 10, 15]:
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=k)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"K={k}: {score:.4f}")
```

**Typical output**:
```
Baseline (Simple Imputation): 0.8245
KNN Imputation: 0.8567
Improvement: 3.22%

Tuning K:
K=3: 0.8423
K=5: 0.8567  ← Best
K=7: 0.8512
K=10: 0.8489
K=15: 0.8401
```

## Best Practices

### 1. Scale Features Before KNN

**Always standardize** (or normalize) features:
```python
from sklearn.preprocessing import StandardScaler

# Option 1: Manual
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Handle NaN separately
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X_scaled)

# Option 2: Pipeline (recommended)
pipeline = Pipeline([
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler())  # Scale after imputation
])
```

### 2. Start with K=5

**Default**: K=5 is reasonable for most cases

**Then tune** if needed via cross-validation

### 3. Use weights='distance' for Better Accuracy

**Distance weighting** often improves results with minimal cost:
```python
KNNImputer(n_neighbors=5, weights='distance')
```

### 4. Consider Computational Cost

**For large datasets** (n > 10,000):
- Use SimpleImputer instead
- Or sample data for KNN, then impute rest with simple method
- Or use approximate nearest neighbors (not in sklearn)

### 5. Check for Convergence Issues

**If many features missing**: KNN may struggle

**Check**:
```python
missing_pct = X.isnull().mean()
print(missing_pct[missing_pct > 0.3])  # Features with >30% missing

# Consider dropping very sparse features
X_filtered = X.loc[:, missing_pct < 0.3]
```

### 6. Validate Imputation Quality

**Cross-validation with artificial missingness**:
```python
# Remove values, impute, compare to original
X_complete = X[X.notnull().all(axis=1)]  # Complete cases
X_test = X_complete.copy()

# Artificially remove 10%
mask = np.random.rand(*X_test.shape) < 0.1
X_missing = X_test.copy()
X_missing[mask] = np.nan

# Impute
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_missing)

# Compare
error = np.abs(X_imputed[mask] - X_test[mask]).mean()
print(f"Mean absolute error: {error:.4f}")
```

### 7. Combine with Missing Indicators

**Preserve missingness information**:
```python
KNNImputer(n_neighbors=5, add_indicator=True)
```

## Summary

KNN Imputation fills missing values using the average (or mode) of K nearest neighbors, providing context-aware imputation that leverages feature correlations and local data patterns.

**Key Concepts**:

**Algorithm**:
1. Compute distance to all samples using complete features
2. Find K nearest neighbors
3. Impute as average (numerical) or mode (categorical) of neighbors

**Mathematical Formula**:
$$\hat{x}_{ij} = \frac{1}{K} \sum_{i' \in \mathcal{N}_K(i)} x_{i'j}$$

**Distance-weighted**:
$$\hat{x}_{ij} = \frac{\sum_{i' \in \mathcal{N}_K} w_{i'} x_{i'j}}{\sum_{i' \in \mathcal{N}_K} w_{i'}}$$

Where $w_{i'} = 1/d(i, i')$

**sklearn Implementation**:
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
X_imputed = imputer.fit_transform(X)
```

**Parameters**:
- **n_neighbors**: K (typically 3-10)
- **weights**: 'uniform' or 'distance'
- **metric**: 'nan_euclidean' (default), 'manhattan', 'cosine'
- **add_indicator**: Add missing flags

**Advantages**:
- Uses feature relationships
- Locally adaptive (context-aware)
- Preserves variance better than simple imputation
- No parametric assumptions
- Handles non-linear relationships
- Often more accurate for MAR data

**Limitations**:
- Computationally expensive ($O(n^2)$)
- Requires feature scaling
- Curse of dimensionality
- Needs tuning (K)
- Slower than simple imputation
- Assumes MAR mechanism

**Best Practices**:
- Scale features before KNN
- Start with K=5, then tune
- Use weights='distance'
- Check computational cost (large n)
- Validate imputation quality
- Consider missing indicators
- Compare to simple imputation baseline

**When to Use**:
- MAR data with feature correlations
- Moderate dataset size (100 < n < 10,000)
- Low-moderate dimensions (p < 50)
- Accuracy matters
- Have computational resources

**Alternatives**:
- Simple imputation: Faster, simpler
- Iterative imputation: More accurate, slower
- Model-based: Most flexible, most complex

KNN Imputation provides a practical middle ground between simple statistical methods and complex model-based approaches, leveraging local similarity to produce more accurate and context-aware imputations.
