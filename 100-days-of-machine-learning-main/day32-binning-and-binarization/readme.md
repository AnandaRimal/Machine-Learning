# Binning and Binarization - Discretizing Continuous Features

## General Idea

Binning (also called discretization) is the process of transforming continuous numerical variables into discrete categorical bins or intervals. Binarization is a special case where continuous values are converted into binary (0/1) based on a threshold. These techniques reduce the complexity of continuous data, make models more robust to outliers, and can capture non-linear relationships more easily in linear models.

## Why Use Binning and Binarization?

1. **Non-Linearity Capture**: Linear models can learn non-linear patterns
2. **Outlier Robustness**: Extreme values get grouped into bins
3. **Interpretability**: Easier to understand "young/middle-aged/senior" than exact ages
4. **Missing Value Handling**: Can create "missing" bin
5. **Reduce Overfitting**: Less granular data prevents memorization
6. **Match Domain Knowledge**: Age groups, income brackets align with business logic
7. **Improve Tree Performance**: Can speed up tree-based models
8. **Reduce Storage**: Fewer unique values

## Role in Machine Learning

### For Linear Models

**Problem**: Linear models assume linear relationships
$$y = \beta_0 + \beta_1 x$$

**Reality**: Many relationships are non-linear
- Age vs health risk: U-shaped (young and old have higher risk)
- Temperature vs energy usage: piecewise (heat/cool at extremes)

**Solution**: Binning creates piecewise constant regions
$$y = \begin{cases}
\beta_1 & \text{if } x \in \text{bin}_1 \\
\beta_2 & \text{if } x \in \text{bin}_2 \\
\beta_3 & \text{if } x \in \text{bin}_3
\end{cases}$$

**After one-hot encoding bins**: Linear model learns different coefficient per bin

### For Tree-Based Models

**Note**: Trees already discretize features via splits

**Binning can still help**:
- Pre-binning reduces search space (faster training)
- Domain-aligned bins improve interpretability
- Can prevent overly granular splits

**But**: Often unnecessary, trees handle continuous values well

### For Neural Networks

**Binning + Embedding**: Treat bins as categories
- Bin continuous feature
- Use embedding layer for bins
- Network learns representation

**Benefit**: Can be more efficient than raw continuous values

## Types of Binning

### 1. Equal-Width Binning (Uniform Binning)

**Definition**: Divide range into bins of equal width

**Formula**:
$$\text{bin width} = \frac{\max(X) - \min(X)}{n_{bins}}$$

**Bin edges**:
$$\text{edges} = [\min(X), \min(X) + w, \min(X) + 2w, ..., \max(X)]$$

Where $w$ is bin width

**Example**:
- Data: $[0, 100]$, $n_{bins} = 5$
- Width: $(100 - 0) / 5 = 20$
- Bins: $[0, 20), [20, 40), [40, 60), [60, 80), [80, 100]$

**Pros**:
- Simple to understand
- Easy to interpret
- Fast to compute

**Cons**:
- Sensitive to outliers (can create empty bins)
- Uneven distribution of data across bins
- May have many points in one bin, few in others

**When to use**:
- Data is uniformly distributed
- Need easy interpretability
- Range is well-defined without outliers

**sklearn**: `KBinsDiscretizer(strategy='uniform')`

### 2. Equal-Frequency Binning (Quantile Binning)

**Definition**: Divide data so each bin has approximately equal number of samples

**Formula**:
$$\text{bin edges} = \text{quantiles}(X, [0, 1/n, 2/n, ..., 1])$$

**Example**:
- Data: $[1, 2, 3, 5, 7, 10, 15, 20, 30, 50]$ (10 points)
- $n_{bins} = 5$
- Points per bin: $10 / 5 = 2$
- Bins:
  - Bin 1: $[1, 2]$ (2 points)
  - Bin 2: $(2, 5]$ (2 points)
  - Bin 3: $(5, 10]$ (2 points)
  - Bin 4: $(10, 20]$ (2 points)
  - Bin 5: $(20, 50]$ (2 points)

**Pros**:
- Balanced bin populations
- Robust to outliers
- Better statistical properties

**Cons**:
- Bin edges less intuitive (not round numbers)
- Variable bin widths
- Two equal values might be in different bins (edge cases)

**When to use**:
- Skewed distributions
- Presence of outliers
- Want balanced representation in each bin

**sklearn**: `KBinsDiscretizer(strategy='quantile')`

### 3. K-Means Binning (Clustering-Based)

**Definition**: Use k-means clustering to find optimal bin centers, assign points to nearest center

**Process**:
1. Run 1D k-means on feature values
2. Cluster centers become bin "centers"
3. Assign each value to closest center's bin

**Mathematical formulation**:
$$\min_{C} \sum_{i=1}^{n} \min_{j \in \{1,...,k\}} (x_i - c_j)^2$$

Where $C = \{c_1, ..., c_k\}$ are cluster centers

**Pros**:
- Adapts to data distribution
- Minimizes within-bin variance
- Can capture natural groupings

**Cons**:
- More complex
- Requires iterative optimization
- Results depend on initialization
- Bin edges less interpretable

**When to use**:
- Multi-modal distributions
- Want data-driven bin boundaries
- Prioritize minimizing variance within bins

**sklearn**: `KBinsDiscretizer(strategy='kmeans')`

### 4. Custom Binning (Domain-Driven)

**Definition**: Manually specify bin edges based on domain knowledge

**Example - Age Groups**:
- Bins: $[0, 18), [18, 35), [35, 50), [50, 65), [65, \infty)$
- Labels: ['Minor', 'Young Adult', 'Middle-Aged', 'Senior', 'Elderly']

**Example - Income Brackets**:
- Bins: $[0, 30k), [30k, 70k), [70k, 150k), [150k, \infty)$
- Labels: ['Low', 'Middle', 'Upper-Middle', 'High']

**Pros**:
- Aligned with business/domain understanding
- Interpretable
- Consistent with regulations or standards

**Cons**:
- Requires domain expertise
- May not be optimal for model performance
- Needs manual specification

**sklearn**: Use `pd.cut()` with custom bins or write custom transformer

## Binarization

### Definition

Convert continuous values to binary (0 or 1) based on threshold:

$$x' = \begin{cases}
1 & \text{if } x > \text{threshold} \\
0 & \text{if } x \leq \text{threshold}
\end{cases}$$

### Common Use Cases

**1. Presence/Absence**:
- Feature: Transaction amount
- Binarized: Has made purchase (yes/no)
- Threshold: 0

**2. Above/Below Average**:
- Feature: Test score
- Binarized: Above average (yes/no)
- Threshold: mean(scores)

**3. Health Indicators**:
- Feature: BMI
- Binarized: Overweight (yes/no)
- Threshold: 25

**4. Time-Based**:
- Feature: Account age (days)
- Binarized: Established customer (yes/no)
- Threshold: 180 days

### Mathematical Properties

**Loss of Information**:
- Continuous: $x \in \mathbb{R}$ (infinite values)
- Binarized: $x' \in \{0, 1\}$ (2 values)

**Result**: Significant information loss, only "above/below" preserved

**When acceptable**:
- Only threshold crossing matters (e.g., passed/failed)
- Want simple interpretability
- Feature is noisy (binarization acts as regularization)

**sklearn**: `Binarizer(threshold=value)`

## KBinsDiscretizer - sklearn Implementation

### Syntax

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(
    n_bins=5,              # Number of bins per feature
    encode='ordinal',      # Encoding method
    strategy='quantile'    # Binning strategy
)

kbd.fit(X_train)
X_binned = kbd.transform(X_train)
```

### Parameters

**n_bins**: int or array, default=5
- Number of bins
- Can specify different per feature: `n_bins=[3, 5, 10]`

**encode**: {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
- `'onehot'`: Sparse one-hot encoding (each bin → binary vector)
- `'onehot-dense'`: Dense one-hot encoding
- `'ordinal'`: Integer encoding (0, 1, 2, ..., n_bins-1)

**strategy**: {'uniform', 'quantile', 'kmeans'}, default='quantile'
- `'uniform'`: Equal-width bins
- `'quantile'`: Equal-frequency bins
- `'kmeans'`: Clustering-based bins

### Fitted Attributes

**bin_edges_**: list of arrays
- Bin edges for each feature
- Access: `kbd.bin_edges_[feature_idx]`

**n_bins_**: array
- Actual number of bins per feature (may differ from requested if too many bins)

### Example Usage

**Ordinal Encoding** (0, 1, 2, ...):
```python
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
kbd.fit(X)
X_binned = kbd.transform(X)
# X_binned: [[0], [0], [1], [2], [2], ...]
```

**One-Hot Encoding**:
```python
kbd = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='quantile')
kbd.fit(X)
X_binned = kbd.transform(X)
# X_binned: [[1,0,0], [1,0,0], [0,1,0], [0,0,1], ...]
```

## Binarizer - sklearn Implementation

### Syntax

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0)
X_binary = binarizer.transform(X)
```

### Parameters

**threshold**: float, default=0.0
- Threshold for binarization
- Values $\leq$ threshold become 0
- Values $>$ threshold become 1

**copy**: bool, default=True
- Whether to copy X or modify in-place

### No fit() Required

**Note**: Binarizer is stateless (no parameters learned)
```python
binarizer = Binarizer(threshold=5)
X_binary = binarizer.transform(X)  # No fit needed
```

**But**: Still has `fit()` for pipeline compatibility
```python
binarizer.fit(X)  # Does nothing, returns self
```

### Example

```python
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

binarizer = Binarizer(threshold=5)
X_binary = binarizer.transform(X)

print(X_binary)
# [[0, 0, 0],
#  [0, 0, 1],
#  [1, 1, 1]]
```

## Using in Pipelines

### Binning Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('binning', KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Flow**:
$$X \xrightarrow{\text{bin}} X_{binned} \xrightarrow{\text{one-hot}} X_{encoded} \xrightarrow{\text{model}} \hat{y}$$

### Selective Binning with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer([
    ('binning', KBinsDiscretizer(n_bins=5, encode='onehot'), ['age', 'income']),
    ('scaling', StandardScaler(), ['height', 'weight']),
    ('binarize', Binarizer(threshold=0), ['balance'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

**Benefit**: Different strategies for different feature types

### Binning + Interaction Terms

```python
from sklearn.preprocessing import PolynomialFeatures

pipeline = Pipeline([
    ('binning', KBinsDiscretizer(n_bins=3, encode='ordinal')),
    ('interactions', PolynomialFeatures(degree=2, interaction_only=True)),
    ('model', Ridge())
])
```

**Result**: Captures interactions between binned features

## Mathematical Examples

### Example 1: Equal-Width Binning

**Data**: Ages $[18, 22, 25, 35, 45, 50, 60, 70]$

**Range**: $70 - 18 = 52$

**n_bins = 4**:
$$w = 52 / 4 = 13$$

**Bins**:
1. $[18, 31)$: Contains 18, 22, 25 → 3 samples
2. $[31, 44)$: Contains 35 → 1 sample
3. $[44, 57)$: Contains 45, 50 → 2 samples
4. $[57, 70]$: Contains 60, 70 → 2 samples

**Ordinal encoding**: $[0, 0, 0, 1, 2, 2, 3, 3]$

**Issue**: Unbalanced (bin 2 has only 1 sample)

### Example 2: Equal-Frequency Binning

**Same data**: $[18, 22, 25, 35, 45, 50, 60, 70]$ (8 samples)

**n_bins = 4**: Each bin should have $8/4 = 2$ samples

**Bins** (from quantiles):
1. $[18, 23.5]$: 18, 22 → 2 samples
2. $(23.5, 40]$: 25, 35 → 2 samples
3. $(40, 55]$: 45, 50 → 2 samples
4. $(55, 70]$: 60, 70 → 2 samples

**Ordinal encoding**: $[0, 0, 1, 1, 2, 2, 3, 3]$

**Balanced**: Each bin has exactly 2 samples

### Example 3: Binarization

**Data**: Test scores $[45, 62, 73, 88, 91]$

**Threshold**: 70 (passing grade)

**Binarization**:
$$x' = \mathbb{1}_{x > 70}$$

**Result**: $[0, 0, 1, 1, 1]$ (fail, fail, pass, pass, pass)

**Interpretation**: First two failed, last three passed

### Example 4: K-Means Binning

**Data**: $[1, 2, 3, 10, 11, 12, 20, 21, 22]$ (3 clusters obvious)

**n_bins = 3**, **strategy='kmeans'**:

**K-means finds centers**:
- $c_1 = 2$ (cluster: 1, 2, 3)
- $c_2 = 11$ (cluster: 10, 11, 12)
- $c_3 = 21$ (cluster: 20, 21, 22)

**Bin assignment** (to nearest center):
$$\text{bin}(x) = \arg\min_j |x - c_j|$$

**Result**: $[0, 0, 0, 1, 1, 1, 2, 2, 2]$

**Bin edges** (midpoints between centers):
- Bin 0: $(-\infty, 6.5]$
- Bin 1: $(6.5, 16]$
- Bin 2: $(16, \infty)$

## Encoding Binned Features

### Ordinal Encoding

**Result**: Integer codes $0, 1, 2, ..., n_{bins} - 1$

**Implies ordering**: Bin 0 < Bin 1 < Bin 2

**When to use**:
- Bins have natural order (age groups, income brackets)
- Using tree-based models (can handle ordinal naturally)

**Issue with linear models**: Assumes equal spacing
- Difference between bins 0 and 1 = difference between bins 1 and 2
- May not be true (e.g., unequal bin widths)

### One-Hot Encoding

**Result**: Binary vectors

**Example**: 3 bins → $[1,0,0], [0,1,0], [0,0,1]$

**No ordering assumed**: Each bin is independent

**When to use**:
- Bins don't have meaningful order
- Using linear models or neural networks
- Want separate coefficient per bin

**Dimension increase**: 1 feature → n_bins features

**sklearn**:
```python
KBinsDiscretizer(encode='onehot')  # Sparse
KBinsDiscretizer(encode='onehot-dense')  # Dense array
```

## Effect on Model Performance

### Linear Models - Before Binning

**Model**: $y = \beta_0 + \beta_1 x_{age}$

**Limitation**: Single slope for all ages
- Can't capture: Young and old both have higher health costs (U-shape)

### Linear Models - After Binning

**Bins**: Child (0-18), Adult (18-65), Senior (65+)

**One-hot encoded**: $x_{child}, x_{adult}, x_{senior}$

**Model**: $y = \beta_0 + \beta_1 x_{child} + \beta_2 x_{adult} + \beta_3 x_{senior}$

**Result**: Different coefficient per age group (piecewise constant)

**Can now capture**: Higher costs for children and seniors, lower for adults

### Tree Models

**Effect**: Minimal benefit, sometimes harmful

**Reason**: Trees already split on thresholds
- Pre-binning restricts split points
- Tree can find better splits in continuous space

**Exception**: Very deep trees might benefit from pre-binning (regularization)

## Advantages and Disadvantages

### Advantages

**1. Non-linearity in Linear Models**:
- Piecewise constant approximation
- Different coefficient per bin

**2. Outlier Robustness**:
- Extreme values grouped with near-neighbors
- Reduces outlier influence

**3. Interpretability**:
- "Senior citizens" vs "age 67.3"
- Aligns with domain concepts

**4. Missing Value Handling**:
- Can create "missing" bin
- Treat as separate category

**5. Regularization**:
- Reduces model complexity
- Can prevent overfitting

**6. Computational Efficiency**:
- Fewer unique values
- Faster processing for some algorithms

### Disadvantages

**1. Information Loss**:
- Lose within-bin variation
- Age 18 and 25 treated identically in same bin

**2. Arbitrary Boundaries**:
- Why 65 and not 64.5 for "senior"?
- Discontinuities at bin edges

**3. Curse of Dimensionality** (with one-hot):
- Many bins → many features
- Sparse data in high dimensions

**4. Overfitting Risk** (too many bins):
- Memorize training data
- Poor generalization

**5. Underfitting Risk** (too few bins):
- Miss important patterns
- Overly simplified

**6. Requires Tuning**:
- Number of bins hyperparameter
- Strategy choice
- Threshold selection

## Best Practices

### 1. Choose Appropriate Number of Bins

**Too few bins** ($n < 3$): Oversimplification, information loss

**Too many bins** ($n > 20$): Overfitting, high dimensionality

**Rule of thumb**: 
$$n_{bins} \approx \sqrt{n_{samples}} \quad \text{or} \quad n_{bins} \approx \log_2(n_{samples})$$

**Example**: 1000 samples → $\sqrt{1000} \approx 32$ or $\log_2(1000) \approx 10$ bins

**Best**: Cross-validation to select optimal number

### 2. Use Quantile Strategy for Skewed Data

**Skewed distribution**: Equal-width creates many empty bins

**Solution**: `strategy='quantile'` ensures balanced bin populations

### 3. Use Custom Bins for Domain Knowledge

**Example**: Age groups defined by life stages, not statistics

```python
import pandas as pd
pd.cut(age, bins=[0, 18, 25, 40, 60, 100], labels=['Child', 'Youth', 'Adult', 'Middle', 'Senior'])
```

### 4. One-Hot Encode for Linear Models

**Unless**: Bins have true ordinal relationship AND equal importance

**Usually**: One-hot safer for linear models

```python
KBinsDiscretizer(encode='onehot')
```

### 5. Don't Bin Tree-Based Model Features (Usually)

**Trees**: Already discretize optimally

**Exception**: 
- Interpretability requirements
- Regularization (limit tree depth implicitly)

### 6. Visualize Bin Distributions

```python
import matplotlib.pyplot as plt

kbd = KBinsDiscretizer(n_bins=5, strategy='quantile')
X_binned = kbd.fit_transform(X)

plt.hist(X, bins=kbd.bin_edges_[0], edgecolor='black')
plt.title('Bin Distribution')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()
```

**Check**: Balanced populations, sensible edges

### 7. Apply Binning After Imputation

**Order**: Impute → Bin

**Reason**: Can't bin NaN values

```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('binning', KBinsDiscretizer()),
    ('model', LogisticRegression())
])
```

### 8. Consider Impact on Interpretability

**Binning**: Easier interpretation ("high income" vs "$73,492")

**But**: Coefficients apply to bins, not original scale

**Trade-off**: Simplicity vs precision

## Common Patterns

### Pattern 1: Binning Age for Linear Model

```python
preprocessor = ColumnTransformer([
    ('age_bins', KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile'), ['age']),
    ('scale', StandardScaler(), other_numeric_features)
])
```

### Pattern 2: Binarizing Transaction Flag

```python
preprocessor = ColumnTransformer([
    ('transaction', Binarizer(threshold=0), ['transaction_amount']),
    # 0: No transaction, 1: Any transaction
    ('scale', StandardScaler(), other_features)
])
```

### Pattern 3: Custom Domain Bins + One-Hot

```python
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def custom_age_bins(X):
    return pd.cut(X.ravel(), bins=[0, 18, 30, 50, 70, 100], labels=False).reshape(-1, 1)

pipeline = Pipeline([
    ('bins', FunctionTransformer(custom_age_bins)),
    ('onehot', OneHotEncoder()),
    ('model', LogisticRegression())
])
```

## Summary

Binning and binarization are discretization techniques that convert continuous features into categorical bins or binary values, enabling linear models to capture non-linear relationships and improving robustness to outliers.

**Key Concepts**:

**Binning**: Continuous → Discrete bins
- **Equal-Width**: $w = (\max - \min) / n_{bins}$
- **Equal-Frequency**: Quantile-based, balanced populations
- **K-Means**: Clustering-based, minimize within-bin variance
- **Custom**: Domain-driven edges

**Binarization**: Continuous → Binary {0, 1}
$$x' = \mathbb{1}_{x > threshold}$$

**sklearn Tools**:
- `KBinsDiscretizer`: Binning with multiple strategies
- `Binarizer`: Threshold-based binarization

**Encoding**:
- **Ordinal**: Integer codes (0, 1, 2, ...)
- **One-Hot**: Binary vectors per bin

**Benefits**:
- Non-linearity for linear models
- Outlier robustness
- Interpretability
- Regularization

**Drawbacks**:
- Information loss
- Arbitrary boundaries
- Dimension increase (one-hot)
- Requires hyperparameter tuning

**Best Practices**:
- Choose $n_{bins}$ via cross-validation
- Use quantile strategy for skewed data
- One-hot encode for linear models
- Don't bin for tree models (usually)
- Visualize bin distributions
- Apply after imputation

**When to Use**:
- Linear models with non-linear relationships
- Need interpretability (age groups, income brackets)
- Skewed/outlier-heavy data
- Want regularization

**When NOT to Use**:
- Tree-based models (redundant)
- Data already normally distributed
- Need precise values (not ranges)
- Too few samples (unstable bins)

Binning and binarization are powerful preprocessing techniques that bridge the gap between continuous and categorical data, enabling simpler models to capture complex patterns while improving interpretability and robustness.

---

**Video Link**: https://youtu.be/kKWsJGKcMvo
