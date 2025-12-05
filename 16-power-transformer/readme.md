# Power Transformer

## Introduction

The Power Transformer is a preprocessing technique that applies power transformations to make data more Gaussian-like (normally distributed). It automatically determines the optimal transformation from the Box-Cox or Yeo-Johnson family to stabilize variance and minimize skewness in the data. This is particularly useful for algorithms that assume normally distributed features.

## Why Power Transformation?

Many machine learning algorithms perform better when features follow a normal (Gaussian) distribution:
- **Linear Models**: Assumptions of normality improve inference
- **Gaussian Processes**: Explicitly assume Gaussian distributions
- **Neural Networks**: Faster convergence with normalized inputs
- **Distance-Based Methods**: Reduce impact of scale differences

Real-world data is often skewed (salary distributions, housing prices, count data), and power transformations can make it symmetric and bell-shaped.

## Mathematical Foundation

### Box-Cox Transformation

For **strictly positive** data ($x > 0$), the Box-Cox transformation is:

$$x_{\lambda} = \begin{cases}
\frac{x^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

Where:
- $x$: Original value (must be positive)
- $\lambda$ (lambda): Transformation parameter
- $x_{\lambda}$: Transformed value

**Special cases**:
- $\lambda = 1$: Linear transformation (no change except shifting)
- $\lambda = 0.5$: Square root transformation
- $\lambda = 0$: Log transformation
- $\lambda = -1$: Reciprocal transformation

**Finding optimal $\lambda$**: Maximum likelihood estimation maximizes:

$$L(\lambda) = -\frac{n}{2}\log(\sigma^2(\lambda)) + (\lambda - 1)\sum_{i=1}^{n}\log(x_i)$$

where $\sigma^2(\lambda)$ is the variance of transformed data.

### Yeo-Johnson Transformation

For data that can be **positive or negative**, the Yeo-Johnson transformation is:

$$x_{\lambda} = \begin{cases}
\frac{(x + 1)^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0, x \geq 0 \\
\log(x + 1) & \text{if } \lambda = 0, x \geq 0 \\
-\frac{(-x + 1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, x < 0 \\
-\log(-x + 1) & \text{if } \lambda = 2, x < 0
\end{cases}$$

**Advantage**: Works with zero and negative values, making it more versatile than Box-Cox.

### Standardization After Transformation

After applying the power transformation, data is typically standardized:

$$z = \frac{x_{\lambda} - \mu}{\sigma}$$

where $\mu$ and $\sigma$ are the mean and standard deviation of transformed values.

## Detailed Example

**Dataset**: Income data (highly right-skewed)

$$X = [30000, 35000, 40000, 45000, 50000, 100000, 200000]$$

This data is right-skewed due to the high outliers.

### Step 1: Visualize Original Distribution

```
Original Statistics:
Mean = 71,428.57
Median = 45,000
Skewness = 1.96 (highly right-skewed)
```

### Step 2: Apply Box-Cox Transformation

Using maximum likelihood, suppose optimal $\lambda = 0.15$ is found.

For $x_1 = 30000$:

$$x_{\lambda} = \frac{30000^{0.15} - 1}{0.15} = \frac{5.172 - 1}{0.15} = \frac{4.172}{0.15} = 27.81$$

For $x_2 = 35000$:

$$x_{\lambda} = \frac{35000^{0.15} - 1}{0.15} = \frac{5.301 - 1}{0.15} = 28.67$$

Continuing for all values:

| Original ($x$) | Transformed ($x_{\lambda}$) |
|----------------|------------------------------|
| 30,000         | 27.81                        |
| 35,000         | 28.67                        |
| 40,000         | 29.41                        |
| 45,000         | 30.06                        |
| 50,000         | 30.65                        |
| 100,000        | 35.15                        |
| 200,000        | 40.48                        |

### Step 3: Check New Distribution

```
Transformed Statistics:
Mean = 31.75
Median = 30.06
Skewness = 0.42 (much more symmetric!)
```

The transformation reduced skewness from 1.96 to 0.42, making the data more normally distributed.

### Step 4: Standardize

$$z_i = \frac{x_{\lambda,i} - 31.75}{\sigma}$$

If $\sigma = 4.5$:

For $x_1$: $z_1 = \frac{27.81 - 31.75}{4.5} = -0.88$

For $x_7$: $z_7 = \frac{40.48 - 31.75}{4.5} = 1.94$

## Comparison: Box-Cox vs Yeo-Johnson

| Aspect | Box-Cox | Yeo-Johnson |
|--------|---------|-------------|
| **Data Requirement** | $x > 0$ only | Any real number |
| **Use Case** | Positive features (prices, counts) | Features with negatives (profits, temperatures) |
| **Formula Complexity** | Simpler | More complex (4 cases) |
| **Common Choice** | When all values positive | When zeros or negatives present |

## Advantages

1. **Automatic Optimization**: Finds optimal $\lambda$ automatically using MLE
2. **Reduces Skewness**: Transforms skewed distributions toward normal
3. **Stabilizes Variance**: Makes variance more constant across the range
4. **Improves Model Performance**: Helps algorithms assuming normality
5. **Handles Outliers**: Reduces impact of extreme values
6. **Invertible**: Can transform back to original scale
7. **Versatile**: Yeo-Johnson works with any real numbers

## Disadvantages

1. **Interpretability Loss**: Transformed values hard to interpret in original units
2. **Not Always Effective**: Some distributions can't be normalized
3. **Overfitting Risk**: Optimizing $\lambda$ on small datasets may overfit
4. **Computational Cost**: Finding optimal $\lambda$ requires optimization
5. **Tree-Based Models**: Not needed for Random Forest, Gradient Boosting
6. **Requires Careful Application**: Need to apply same $\lambda$ to test data
7. **Assumptions**: Assumes transformation from power family sufficient

## When to Use Power Transformer

### Use When:
- Features are **highly skewed** (skewness > 1 or < -1)
- Using **linear models**, **SVMs**, or **neural networks**
- Features have **exponential-like** distributions
- Data contains **outliers** that should be de-emphasized
- Algorithm **assumes normality** (Gaussian Naive Bayes, LDA)
- Need **variance stabilization**

### Don't Use When:
- Features already **normally distributed**
- Using **tree-based models** (Random Forest, XGBoost)
- **Interpretability** is critical (use simple log transform instead)
- Data is **already bounded** in [0,1] (use other normalization)
- Very **small datasets** (risk of overfitting $\lambda$)
- **Categorical** or **binary** features

## Code Example

### Basic Usage

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Create skewed data
np.random.seed(42)
data = np.random.exponential(scale=2, size=1000).reshape(-1, 1)

# Initialize Power Transformer
# method='box-cox' for positive data
# method='yeo-johnson' for any data (default)
pt = PowerTransformer(method='yeo-johnson', standardize=True)

# Fit and transform
data_transformed = pt.fit_transform(data)

# Check fitted lambda
print(f"Optimal lambda: {pt.lambdas_[0]:.4f}")

# Inverse transform
data_original = pt.inverse_transform(data_transformed)
```

### Comparing Before and After

```python
import pandas as pd
from scipy import stats

# Original statistics
print("Original Data:")
print(f"Mean: {np.mean(data):.2f}")
print(f"Std: {np.std(data):.2f}")
print(f"Skewness: {stats.skew(data):.2f}")

# Transformed statistics
print("\nTransformed Data:")
print(f"Mean: {np.mean(data_transformed):.2f}")
print(f"Std: {np.std(data_transformed):.2f}")
print(f"Skewness: {stats.skew(data_transformed):.2f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(data, bins=30, edgecolor='black')
axes[0].set_title('Original Data (Skewed)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(data_transformed, bins=30, edgecolor='black')
axes[1].set_title('Transformed Data (More Gaussian)')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Multiple Features

```python
# Multiple features with different distributions
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=3, noise=10)

# Add skewness to features
X[:, 0] = np.exp(X[:, 0] / 5)  # Exponential
X[:, 1] = X[:, 1] ** 2         # Squared
# X[:, 2] remains relatively normal

# Transform all features
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

print("Fitted lambdas for each feature:")
for i, lam in enumerate(pt.lambdas_):
    print(f"Feature {i}: λ = {lam:.4f}")
```

### In a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create pipeline
pipeline = Pipeline([
    ('power', PowerTransformer(method='yeo-johnson')),
    ('regressor', LinearRegression())
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"Average R² score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Box-Cox for Positive Data Only

```python
# For strictly positive data
positive_data = np.random.gamma(shape=2, scale=2, size=1000).reshape(-1, 1)

pt_boxcox = PowerTransformer(method='box-cox', standardize=True)
data_boxcox = pt_boxcox.fit_transform(positive_data)

print(f"Box-Cox lambda: {pt_boxcox.lambdas_[0]:.4f}")
```

## What's in the Notebook

The `day31.ipynb` notebook demonstrates:

### 1. **Understanding Skewness**
- Loading and visualizing skewed datasets
- Calculating skewness and kurtosis
- Identifying features needing transformation

### 2. **Box-Cox Transformation**
- Applying Box-Cox to positive features
- Finding optimal lambda values
- Comparing distributions before/after
- Q-Q plots to check normality

### 3. **Yeo-Johnson Transformation**
- Handling features with negative values
- Comparing with Box-Cox
- Use cases for each method

### 4. **Impact on Model Performance**
- Training models with/without transformation
- Comparing accuracy, R², MSE
- Cross-validation results
- Feature importance changes

### 5. **Practical Examples**
- Real dataset: Housing prices (right-skewed)
- Temperature data (can be negative)
- Count data with zeros
- Financial data with outliers

### 6. **Pitfalls and Best Practices**
- When transformation doesn't help
- Checking assumptions
- Handling test data correctly
- Combining with other preprocessing

### 7. **Inverse Transformation**
- Converting predictions back to original scale
- Interpreting transformed coefficients
- Error propagation

## Mathematical Intuition

### Why Does Power Transformation Work?

**Problem**: Skewed data compresses most values in a narrow range, with few extreme values stretching the distribution.

**Solution**: Power transformation stretches the compressed region and compresses the stretched region.

**For right-skewed data** (long right tail):
- $\lambda < 1$: Compresses large values, stretches small values
- Example: Log transformation ($\lambda = 0$) makes $1000$ much closer to $100$ in transformed space

**For left-skewed data** (long left tail):
- $\lambda > 1$: Compresses small values, stretches large values

### Visual Intuition

```
Original right-skewed:
|*********************|---------|---|
0                    50        100  200

After transformation (λ=0.15):
|*********|*********|*********|
2.0      2.5       3.0       3.5
(More evenly distributed)
```

## Relationship to Other Transformations

### Simple Transformations as Special Cases

1. **Log**: $\lambda = 0$ (Box-Cox limit)
   $$\log(x) = \lim_{\lambda \to 0} \frac{x^{\lambda} - 1}{\lambda}$$

2. **Square Root**: $\lambda = 0.5$
   $$\sqrt{x} \approx \frac{x^{0.5} - 1}{0.5}$$

3. **Reciprocal**: $\lambda = -1$
   $$\frac{1}{x} \approx \frac{x^{-1} - 1}{-1}$$

### Advantages Over Manual Selection

- **Data-driven**: Optimal $\lambda$ from data, not guesswork
- **Flexible**: Continuous range, not just discrete choices
- **Consistent**: Same procedure for all features

## Common Pitfalls

### 1. Forgetting to Save Lambda
```python
# WRONG: Fitting on train, test separately
pt_train = PowerTransformer().fit_transform(X_train)
pt_test = PowerTransformer().fit_transform(X_test)  # Different λ!

# CORRECT: Fit on train, apply to test
pt = PowerTransformer()
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)  # Same λ
```

### 2. Applying to Already Normal Data
- May make distribution worse
- Always check skewness first

### 3. Using Box-Cox with Non-Positive Data
```python
# If data has zeros or negatives, use Yeo-Johnson
data_with_zeros = np.array([0, 1, 2, 3]).reshape(-1, 1)

# This will fail:
# pt = PowerTransformer(method='box-cox')
# pt.fit_transform(data_with_zeros)  # ValueError!

# Use Yeo-Johnson instead:
pt = PowerTransformer(method='yeo-johnson')
pt.fit_transform(data_with_zeros)  # Works!
```

### 4. Not Checking Transformation Effectiveness
```python
from scipy.stats import skew

# Always verify improvement
before_skew = skew(data)
after_skew = skew(data_transformed)

print(f"Skewness before: {before_skew:.2f}")
print(f"Skewness after: {after_skew:.2f}")

if abs(after_skew) > abs(before_skew):
    print("Warning: Transformation made skewness worse!")
```

## Real-World Applications

### 1. Housing Prices
```python
# Prices often exponentially distributed
house_prices = pd.read_csv('houses.csv')['price'].values.reshape(-1, 1)

pt = PowerTransformer(method='box-cox')
prices_transformed = pt.fit_transform(house_prices)

# Use in regression model
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, prices_transformed)

# Predictions need inverse transform
predictions_transformed = model.predict(X_test)
predictions_original = pt.inverse_transform(predictions_transformed)
```

### 2. Income Data
```python
# Income highly right-skewed
income_data = data[['income']].values
pt = PowerTransformer(method='yeo-johnson')
income_normalized = pt.fit_transform(income_data)
```

### 3. Website Traffic
```python
# Page views per day (often long-tailed)
traffic = data[['daily_visits']].values
pt = PowerTransformer(method='box-cox')  # All positive
traffic_transformed = pt.fit_transform(traffic)
```

## Summary

Power Transformer is a sophisticated preprocessing technique that:
- **Automatically finds** the best transformation to make data Gaussian
- **Reduces skewness** and stabilizes variance
- **Improves performance** of algorithms assuming normality
- **Handles both** positive-only (Box-Cox) and any real numbers (Yeo-Johnson)

**Key takeaway**: Use PowerTransformer when you have skewed features and are using algorithms that benefit from normally distributed data. Always apply the same transformation parameters (lambda) to training and test data.

The transformation is particularly powerful for:
- Linear regression and logistic regression
- Support Vector Machines
- Neural networks
- Gaussian-based algorithms

But remember: tree-based models (Random Forest, XGBoost) don't need this transformation and may even perform worse with it!

---

*"The power transformation is like finding the right lens to view your data through—it doesn't change the information, just makes patterns easier to see."*
