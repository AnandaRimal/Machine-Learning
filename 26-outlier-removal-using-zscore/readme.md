# Outlier Removal Using Z-Score - Statistical Distance Method

## General Idea

The Z-Score method detects outliers by measuring how many standard deviations a data point is from the mean. A z-score quantifies the distance of each value from the mean in units of standard deviation, with extreme z-scores (typically |z| > 3) indicating outliers that are statistically unusual. This parametric method assumes normally distributed data and identifies values that are unlikely to occur by chance under this distribution.

## Why Use Z-Score for Outlier Detection?

1. **Statistical Foundation**: Based on probability theory and normal distribution
2. **Interpretable**: Z-score directly shows "how unusual" a value is
3. **Simple Calculation**: Mean and standard deviation only
4. **Standard Thresholds**: Well-established cutoffs (2, 2.5, 3)
5. **Symmetric**: Detects both high and low outliers
6. **Widely Used**: Standard practice in many fields
7. **Works with Scaling**: Automatically accounts for feature scale
8. **Fast**: $O(n)$ complexity

## Role in Machine Learning

### Impact of Outliers

**Outliers harm many ML algorithms**:

**Linear Regression**: Coefficients pulled toward outliers
```
Without outlier: y = 2x + 1 (good fit)
With outlier (100, 500): y = 5x + 50 (distorted)
```

**Mean-based Imputation**: Outliers skew mean
```
Data: [1, 2, 3, 4, 100]
Mean: 22 (not representative!)
Without outlier: Mean = 2.5 (representative)
```

**Distance-based (KNN, K-Means)**: Outliers distort distances

**Gradient Descent**: Outliers cause unstable gradients

**Solution**: Remove or cap outliers before training

### Z-Score Intuition

**Normal distribution**: 68-95-99.7 rule

- **68%** of data within 1 standard deviation ($|z| < 1$)
- **95%** within 2 standard deviations ($|z| < 2$)
- **99.7%** within 3 standard deviations ($|z| < 3$)

**Implication**: If $|z| > 3$, only 0.3% chance (very rare!)

**Example**:
```
Ages: [25, 30, 28, 150, 32, 27]
Mean: 48.67
Std: 48.99

Z-scores:
25:   (25-48.67)/48.99 = -0.48  ✔ Normal
30:   (30-48.67)/48.99 = -0.38  ✔ Normal
28:   (28-48.67)/48.99 = -0.42  ✔ Normal
150:  (150-48.67)/48.99 = 2.07  ⚠ Check (borderline)
32:   (32-48.67)/48.99 = -0.34  ✔ Normal
27:   (27-48.67)/48.99 = -0.44  ✔ Normal
```

**After removing 150**:
```
Ages: [25, 30, 28, 32, 27]
Mean: 28.4
Std: 2.61
```

**More stable statistics**!

## Mathematical Foundation

### Z-Score Formula

**Definition**: Number of standard deviations from mean

$$z_i = \frac{x_i - \mu}{\sigma}$$

Where:
- $x_i$: Individual data point
- $\mu$: Population mean (or sample mean $\bar{x}$)
- $\sigma$: Population standard deviation (or sample $s$)

**Sample version** (used in practice):
$$z_i = \frac{x_i - \bar{x}}{s}$$

Where:
$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}$$

### Interpretation

**Z-score magnitude**:

- $z = 0$: Exactly at mean
- $z = 1$: One std above mean
- $z = -1$: One std below mean
- $z = 3$: Three std above mean (rare!)
- $|z| > 3$: Outlier (occurs <0.3% of time)

**Probability under normal distribution**:

$$P(|Z| > z) = 2 \times \Phi(-|z|)$$

Where $\Phi$ is standard normal CDF

**Common thresholds**:

$$
\begin{align*}
|z| > 2 &\implies P < 0.046 \text{ (4.6% of data)} \\
|z| > 2.5 &\implies P < 0.012 \text{ (1.2%)} \\
|z| > 3 &\implies P < 0.003 \text{ (0.3%)}
\end{align*}
$$

### Example Calculation

**Data**: Income = [40k, 45k, 50k, 55k, 200k]

**Step 1**: Calculate mean
$$\bar{x} = \frac{40 + 45 + 50 + 55 + 200}{5} = \frac{390}{5} = 78 \text{k}$$

**Step 2**: Calculate standard deviation
$$s = \sqrt{\frac{(40-78)^2 + (45-78)^2 + (50-78)^2 + (55-78)^2 + (200-78)^2}{5-1}}$$

$$s = \sqrt{\frac{1444 + 1089 + 784 + 529 + 14884}{4}} = \sqrt{\frac{18730}{4}} = \sqrt{4682.5} \approx 68.4 \text{k}$$

**Step 3**: Calculate z-scores

$$
\begin{align*}
z_{40} &= \frac{40 - 78}{68.4} = \frac{-38}{68.4} \approx -0.56 \\
z_{45} &= \frac{45 - 78}{68.4} = \frac{-33}{68.4} \approx -0.48 \\
z_{50} &= \frac{50 - 78}{68.4} = \frac{-28}{68.4} \approx -0.41 \\
z_{55} &= \frac{55 - 78}{68.4} = \frac{-23}{68.4} \approx -0.34 \\
z_{200} &= \frac{200 - 78}{68.4} = \frac{122}{68.4} \approx 1.78
\end{align*}
$$

**Step 4**: Apply threshold ($|z| > 3$)

All $|z| < 3$, so technically **no outliers** by strict rule

**However**: 200k clearly different. Using $|z| > 1.5$ threshold:
- $z_{200} = 1.78 > 1.5$ → **Outlier**

**Lesson**: Threshold choice matters!

## Choosing the Threshold

### Standard Thresholds

**Conservative** ($|z| > 3$):
- Keep 99.7% of data (if normal)
- Only remove extreme outliers
- Low risk of removing valid data
- May miss some outliers

**Moderate** ($|z| > 2.5$ or $|z| > 2$):
- Remove ~1-5% of data
- Balance between precision and recall
- Common in practice

**Aggressive** ($|z| > 1.5$):
- Remove ~13% of data
- Catches more outliers
- Higher risk of false positives

### Domain-Specific Guidelines

**Finance**: Often $|z| > 3$ (careful with rare events)

**Medical**: Often $|z| > 2.5$ (safety critical)

**Manufacturing**: Often $|z| > 2$ (quality control)

**Research**: Often $|z| > 3$ (conservative)

### Data-Driven Selection

**Visual inspection**:
```python
import matplotlib.pyplot as plt
import numpy as np

z_scores = (data - data.mean()) / data.std()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(z_scores, bins=50)
plt.axvline(-3, color='r', linestyle='--', label='z=-3')
plt.axvline(3, color='r', linestyle='--', label='z=3')
plt.axvline(-2, color='orange', linestyle='--', label='z=-2')
plt.axvline(2, color='orange', linestyle='--', label='z=2')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Z-Scores')

plt.subplot(1, 2, 2)
plt.scatter(range(len(data)), data)
plt.axhline(data.mean() + 3*data.std(), color='r', linestyle='--')
plt.axhline(data.mean() - 3*data.std(), color='r', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data with 3-sigma bounds')
plt.show()
```

**Cross-validation**: Choose threshold that maximizes model performance

## Implementation

### NumPy

```python
import numpy as np

# Data
data = np.array([40, 45, 50, 55, 200, 42, 48, 52])

# Calculate z-scores
mean = np.mean(data)
std = np.std(data, ddof=1)  # ddof=1 for sample std
z_scores = (data - mean) / std

print("Z-Scores:", z_scores)

# Detect outliers (|z| > 3)
threshold = 3
outliers_mask = np.abs(z_scores) > threshold
outliers = data[outliers_mask]
normal = data[~outliers_mask]

print(f"Outliers: {outliers}")
print(f"Normal data: {normal}")
```

### Pandas

```python
import pandas as pd

df = pd.DataFrame({
    'Age': [25, 30, 28, 150, 32, 27],
    'Income': [40, 45, 50, 200, 55, 42]
})

# Calculate z-scores for each column
from scipy import stats
z_scores = np.abs(stats.zscore(df))

print("Z-Scores:\n", z_scores)

# Filter: keep rows where all z-scores < 3
filtered_df = df[(z_scores < 3).all(axis=1)]

print("\nFiltered Data:\n", filtered_df)
```

### Scikit-Learn Style (Custom)

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=1)
        return self
    
    def transform(self, X):
        z_scores = np.abs((X - self.mean_) / self.std_)
        mask = (z_scores < self.threshold).all(axis=1)
        return X[mask]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Usage
remover = ZScoreOutlierRemover(threshold=3)
X_clean = remover.fit_transform(X)
```

### SciPy

```python
from scipy import stats
import numpy as np

# Data
data = np.array([40, 45, 50, 55, 200, 42, 48, 52])

# Calculate z-scores (convenience function)
z_scores = np.abs(stats.zscore(data))

# Remove outliers
threshold = 3
filtered_data = data[z_scores < threshold]

print(f"Original: {len(data)} points")
print(f"Filtered: {len(filtered_data)} points")
print(f"Removed: {len(data) - len(filtered_data)} outliers")
```

## Handling Multivariate Data

### Column-wise Z-Score

**Approach**: Calculate z-score per feature, remove row if ANY feature is outlier

```python
import numpy as np
from scipy import stats

X = np.array([
    [25, 50, 80],   # Age, Income, Score
    [30, 55, 85],
    [150, 60, 90],  # Age is outlier
    [35, 200, 88],  # Income is outlier
    [28, 58, 82]
])

# Z-score per column
z_scores = np.abs(stats.zscore(X, axis=0))

print("Z-Scores per feature:\n", z_scores)

# Remove if ANY feature has |z| > 3
threshold = 3
mask = (z_scores < threshold).all(axis=1)  # All features must be < 3
X_clean = X[mask]

print(f"\nOriginal: {X.shape[0]} samples")
print(f"Cleaned: {X_clean.shape[0]} samples")
print(f"Removed: {X.shape[0] - X_clean.shape[0]} outliers")
```

### Mahalanobis Distance (Alternative)

**Problem with column-wise**: Doesn't account for correlations

**Example**: Height and weight are correlated
- Tall person with high weight: Normal
- Short person with high weight: Outlier

**Column-wise z-score**: Might miss this

**Mahalanobis distance**: Accounts for correlations

$$D^2 = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

Where:
- $\mathbf{x}$: Data point (vector)
- $\boldsymbol{\mu}$: Mean vector
- $\mathbf{\Sigma}$: Covariance matrix

**Usage**:
```python
from scipy.spatial.distance import mahalanobis
import numpy as np

mean = np.mean(X, axis=0)
cov = np.cov(X.T)
cov_inv = np.linalg.inv(cov)

mahal_dist = [mahalanobis(x, mean, cov_inv) for x in X]
threshold = 3  # Chi-squared threshold (p=3 for 3 features)
outliers = np.array(mahal_dist) > threshold

X_clean = X[~outliers]
```

**Benefit**: Considers feature interactions

## Assumptions and Limitations

### Assumption: Normal Distribution

**Z-score assumes** data is normally distributed

**Problem**: If data is skewed, z-score misleading

**Example**: Right-skewed income data
```
Income: [20k, 25k, 30k, 35k, 40k, 100k]
Mean: 41.67k
Median: 32.5k  (better center for skewed data)
```

**Z-score for 100k**: 
$$z = \frac{100 - 41.67}{26.9} \approx 2.17$$

Not flagged as outlier ($|z| < 3$), but clearly different!

**Solution**: 
- **Transform data** (log, Box-Cox) to normalize
- **Use robust methods** (IQR, percentiles)
- **Check distribution** first (histogram, Q-Q plot)

### Limitation: Sensitivity to Mean/Std

**Mean and std are sensitive to outliers**!

**Example**:
```
Data: [1, 2, 3, 4, 100]

Mean: 22 (pulled by 100)
Std: 43.6 (inflated by 100)

Z-score for 100:
z = (100 - 22) / 43.6 = 1.79  ← Not flagged!
```

**Problem**: Extreme outlier inflates std, making itself appear normal!

**Solution**: 
- **Modified Z-Score** (uses median and MAD)
- **IQR method** (robust to outliers)
- **Iterative removal** (remove, recalculate, repeat)

### Limitation: Works for Univariate or Column-wise Only

**Standard z-score**: Per feature

**Doesn't capture**: Multivariate outliers (unusual combinations)

**Example**:
```
Age  Income
25   50k     ← Normal individually
30   55k     ← Normal individually
80   60k     ← Age high, Income normal = unusual combo!
```

**Solution**: Mahalanobis distance, Isolation Forest, Local Outlier Factor

## Modified Z-Score (Robust Alternative)

### Motivation

**Standard z-score**: Uses mean and std (sensitive to outliers)

**Modified z-score**: Uses **median** and **MAD** (robust statistics)

### Formula

$$M_i = \frac{0.6745 \times (x_i - \text{median})}{\text{MAD}}$$

Where:

**MAD** (Median Absolute Deviation):
$$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$

**Constant 0.6745**: Scaling factor so modified z-score ≈ standard z-score for normal data

### Implementation

```python
import numpy as np

def modified_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

# Data with extreme outlier
data = np.array([1, 2, 3, 4, 100])

# Standard z-score
z_standard = (data - np.mean(data)) / np.std(data)
print("Standard Z-Scores:", z_standard)
# [-0.48, -0.46, -0.43, -0.41,  1.79]  ← 100 not flagged!

# Modified z-score
z_modified = modified_z_score(data)
print("Modified Z-Scores:", z_modified)
# [-1.01, -0.67, -0.34,  0.00, 32.26]  ← 100 clearly outlier!

# Apply threshold
threshold = 3.5  # Common for modified z-score
outliers = np.abs(z_modified) > threshold
print("Outliers:", data[outliers])  # [100]
```

**Advantage**: Robust to outliers themselves!

## Best Practices

### 1. Check Distribution First

**Visualize**:
```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# Histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Distribution')
plt.show()

# Q-Q Plot (normal distribution check)
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

**If skewed**: Transform or use robust method

### 2. Use Modified Z-Score for Robustness

**Especially** when:
- Suspect extreme outliers
- Small datasets (n < 100)
- Skewed distributions

### 3. Document Threshold Choice

```python
threshold = 3
print(f"Using z-score threshold: {threshold}")
print(f"Expected false positive rate: {2*(1-stats.norm.cdf(threshold))*100:.2f}%")
```

### 4. Track Removed Outliers

```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df[['Age', 'Income']]))
outlier_mask = (z_scores > 3).any(axis=1)

print(f"Total outliers: {outlier_mask.sum()}")
print(f"Outlier percentage: {outlier_mask.mean()*100:.2f}%")
print("\nOutlier rows:")
print(df[outlier_mask])

# Save for audit
df_outliers = df[outlier_mask]
df_outliers.to_csv('removed_outliers.csv', index=False)
```

### 5. Consider Domain Context

**Don't blindly remove**!

**Example**: CEO salary in employee data
- Statistically outlier
- But **valid** and **important**
- May want to keep or handle separately

**Ask**: Is this a data error or real phenomenon?

### 6. Iterative Removal (Optional)

**Problem**: First outlier removal changes mean/std

**Solution**: Iterate

```python
import numpy as np
from scipy import stats

def iterative_zscore_removal(data, threshold=3, max_iter=10):
    for i in range(max_iter):
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        
        if not outliers.any():
            print(f"Converged after {i} iterations")
            break
        
        print(f"Iteration {i+1}: Removing {outliers.sum()} outliers")
        data = data[~outliers]
    
    return data

clean_data = iterative_zscore_removal(original_data)
```

### 7. Validate Impact on Model

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Before outlier removal
model = LinearRegression()
score_before = cross_val_score(model, X, y, cv=5).mean()

# After outlier removal
z_scores = np.abs(stats.zscore(X, axis=0))
mask = (z_scores < 3).all(axis=1)
X_clean, y_clean = X[mask], y[mask]

score_after = cross_val_score(model, X_clean, y_clean, cv=5).mean()

print(f"Score before: {score_before:.4f}")
print(f"Score after: {score_after:.4f}")
print(f"Improvement: {(score_after - score_before)*100:.2f}%")
```

## Complete Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Load data
df = pd.read_csv('house_prices.csv')

# Select numerical features
numerical_features = ['LotArea', 'GrLivArea', 'SalePrice']
df_num = df[numerical_features].copy()

print(f"Original data shape: {df_num.shape}")
print(f"\nOriginal statistics:\n{df_num.describe()}")

# Visualize before
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numerical_features):
    axes[i].boxplot(df_num[col])
    axes[i].set_title(f'{col} (Before)')
    axes[i].set_ylabel('Value')
plt.tight_layout()
plt.show()

# Calculate z-scores
z_scores = np.abs(stats.zscore(df_num))

print(f"\nZ-Score statistics:")
print(f"Max z-score per feature:\n{pd.DataFrame(z_scores, columns=numerical_features).max()}")

# Identify outliers (any feature |z| > 3)
threshold = 3
outlier_mask = (z_scores > threshold).any(axis=1)

print(f"\nOutliers detected: {outlier_mask.sum()} ({outlier_mask.mean()*100:.2f}%)")
print(f"\nOutlier examples:")
print(df_num[outlier_mask].head())

# Remove outliers
df_clean = df_num[~outlier_mask]

print(f"\nCleaned data shape: {df_clean.shape}")
print(f"\nCleaned statistics:\n{df_clean.describe()}")

# Visualize after
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numerical_features):
    axes[i].boxplot(df_clean[col])
    axes[i].set_title(f'{col} (After)')
    axes[i].set_ylabel('Value')
plt.tight_layout()
plt.show()

# Compare distributions
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, 10))
for i, col in enumerate(numerical_features):
    # Before
    axes[i, 0].hist(df_num[col], bins=50, edgecolor='black')
    axes[i, 0].set_title(f'{col} - Before')
    axes[i, 0].set_ylabel('Frequency')
    
    # After
    axes[i, 1].hist(df_clean[col], bins=50, edgecolor='black', color='green')
    axes[i, 1].set_title(f'{col} - After')
    axes[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Save cleaned data
df_clean.to_csv('house_prices_cleaned.csv', index=False)
print("\nCleaned data saved to 'house_prices_cleaned.csv'")
```

## Summary

The Z-Score method detects outliers by measuring statistical distance from the mean in units of standard deviation, identifying values that are unlikely under a normal distribution assumption.

**Key Concepts**:

**Formula**:
$$z_i = \frac{x_i - \bar{x}}{s}$$

**Threshold**:
- $|z| > 3$: Conservative (0.3% of normal data flagged)
- $|z| > 2.5$: Moderate (1.2% flagged)
- $|z| > 2$: Aggressive (4.6% flagged)

**Implementation**:
```python
from scipy import stats
z_scores = np.abs(stats.zscore(data))
clean_data = data[z_scores < 3]
```

**68-95-99.7 Rule**:
- 68% within 1σ
- 95% within 2σ
- 99.7% within 3σ

**Advantages**:
- Statistically principled
- Simple and fast
- Interpretable ("N standard deviations away")
- Standard thresholds available
- Detects both high and low outliers

**Limitations**:
- **Assumes normality** (fails for skewed data)
- **Sensitive to outliers** (mean/std affected by outliers themselves)
- **Univariate** (doesn't capture multivariate outliers)
- **Fixed threshold** (domain-dependent)

**Robust Alternative: Modified Z-Score**:
$$M_i = \frac{0.6745 \times (x_i - \text{median})}{\text{MAD}}$$

Uses median and MAD instead of mean/std (robust)

**Best Practices**:
- Check normality first (histogram, Q-Q plot)
- Use modified z-score for robustness
- Document threshold choice
- Track removed outliers
- Consider domain context (valid vs. error)
- Validate impact on model performance
- Use Mahalanobis distance for multivariate

**When to Use**:
- Normally distributed data
- Quick outlier screening
- Linear models sensitive to outliers
- Standard statistical analysis

**Alternatives**:
- **IQR Method**: Robust to skewness
- **Percentiles**: Non-parametric
- **Isolation Forest**: Machine learning approach
- **DBSCAN**: Density-based clustering

Z-Score outlier detection provides a fast and interpretable statistical method for identifying unusual values, most effective when data follows a normal distribution.
