# Imputing Numerical Data - Filling Missing Values

## General Idea

Imputation is the process of replacing missing values with estimated values based on observed data. For numerical features (continuous or discrete), common strategies include replacing missing values with the mean, median, or a constant value. Unlike complete case analysis which discards incomplete rows, imputation preserves all observations by filling gaps with plausible values, maintaining sample size and statistical power.

## Why Impute Numerical Data?

1. **Preserve Sample Size**: Keep all observations, no data deletion
2. **Maintain Statistical Power**: Larger $n$ → better inference
3. **Reduce Bias**: Better than deletion under MAR
4. **Enable ML Algorithms**: Most require complete numeric input
5. **Utilize Partial Information**: Don't discard complete features in incomplete rows
6. **Improve Model Performance**: More training data → better generalization
7. **Standard Practice**: Expected in production pipelines
8. **Multiple Strategies Available**: Choose based on data distribution

## Role in Machine Learning

### The Imputation Trade-off

**Complete Case Analysis**:
- Pros: No artificial values, simple
- Cons: Data loss, reduced power

**Imputation**:
- Pros: Preserve sample size, more information
- Cons: Introduces uncertainty, may distort distributions

**Mathematical perspective**:

**Without imputation**: Model trained on $n_{complete} < n$ samples

**With imputation**: Model trained on $n$ samples, but some $X$ values estimated

**Trade-off**: More data (good) vs estimation error (bad)

**Typically**: Benefits outweigh costs, especially when:
- Missingness $< 40\%$
- MAR assumption reasonable
- Appropriate imputation method chosen

### Impact on Model Performance

**Variance reduction**:
$$\text{Var}(\hat{\theta}) \propto \frac{1}{n}$$

Larger $n$ (via imputation) → lower variance → more stable estimates

**Bias introduction**:
$$\text{Bias} = E[\hat{\theta}] - \theta_{true}$$

Imputation can introduce bias if:
- Wrong imputation method
- MNAR data
- Severe missingness

**Mean Squared Error**:
$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

**Goal**: Choose imputation method that minimizes MSE

## Imputation Strategies for Numerical Data

### 1. Mean Imputation

**Method**: Replace missing with mean of observed values

**Formula**:
$$x_{\text{imputed}} = \bar{x}_{\text{observed}} = \frac{1}{n_{obs}}\sum_{i: x_i \text{ observed}} x_i$$

**Example**:
```
Original: [10, 20, NaN, 40, NaN, 60]
Mean of observed: (10 + 20 + 40 + 60) / 4 = 32.5
Imputed: [10, 20, 32.5, 40, 32.5, 60]
```

**Properties**:
- **Preserves mean**: $\bar{x}_{imputed} = \bar{x}_{observed}$
- **Reduces variance**: $\text{Var}(x_{imputed}) < \text{Var}(x_{observed})$
- **Distorts distribution**: Creates artificial peak at mean
- **Weakens correlations**: Imputed values not correlated with other features

**When to use**:
- Data approximately normally distributed
- Low missingness (<10%)
- Quick baseline
- Mean is meaningful measure

**When NOT to use**:
- Skewed distributions (mean not representative)
- Outliers present (mean sensitive)
- High missingness (severe variance reduction)

### 2. Median Imputation

**Method**: Replace missing with median of observed values

**Formula**:
$$x_{\text{imputed}} = \text{median}(x_{\text{observed}})$$

**Example**:
```
Original: [10, 20, NaN, 40, NaN, 100]
Median of observed: median([10, 20, 40, 100]) = 30
Imputed: [10, 20, 30, 40, 30, 100]
```

**Properties**:
- **Robust to outliers**: Median unaffected by extreme values
- **Preserves median**: $\text{median}(x_{imputed}) = \text{median}(x_{observed})$
- **Doesn't preserve mean**: $\bar{x}_{imputed} \neq \bar{x}_{observed}$ (generally)
- **Better for skewed data**: More representative than mean

**Comparison** (with outlier):
```
Data: [10, 20, 30, 40, 1000]  # 1000 is outlier
Mean: 220 (not representative!)
Median: 30 (representative)
```

**When to use**:
- Skewed distributions
- Outliers present
- Robust estimate needed
- Ordinal-like numerical data

**When NOT to use**:
- Normal distribution (mean equally good, more efficient)
- Need to preserve mean exactly

### 3. Mode Imputation (Numerical)

**Method**: Replace with most frequent value

**Formula**:
$$x_{\text{imputed}} = \text{mode}(x_{\text{observed}})$$

**Example**:
```
Original: [1, 2, 2, NaN, 2, 3, NaN, 3]
Mode: 2 (appears 3 times)
Imputed: [1, 2, 2, 2, 2, 3, 2, 3]
```

**When to use**:
- Discrete numerical data with few unique values
- Count data with dominant value
- Rating scales (1-5 stars)

**Rarely used** for truly continuous numerical data

### 4. Constant Imputation

**Method**: Replace with arbitrary constant

**Formula**:
$$x_{\text{imputed}} = c \quad \text{(constant)}$$

**Common constants**:
- **0**: For counts, frequencies ("no occurrences")
- **-1**: Out-of-range flag
- **9999**: Obviously fake value (for debugging)
- **Domain-specific**: e.g., -999 for temperature (impossible value)

**Example**:
```
Original: [5, 10, NaN, 20, NaN]
Constant: 0
Imputed: [5, 10, 0, 20, 0]
```

**When to use**:
- Missing means "zero" (e.g., transaction count)
- Want to preserve missingness information
- Placeholder for further processing
- Domain knowledge suggests specific value

**When NOT to use**:
- Constant not meaningful
- Would distort distribution severely
- Tree models might overfit to constant

### 5. Random Imputation

**Method**: Sample from observed distribution

**Formula**:
$$x_{\text{imputed}} \sim \text{Empirical Distribution}(x_{\text{observed}})$$

**Example**:
```
Original: [10, 20, NaN, 40, NaN, 60]
Random sample from {10, 20, 40, 60}: e.g., 20, 60
Imputed: [10, 20, 20, 40, 60, 60]
```

**Properties**:
- **Preserves distribution**: Mean, variance approximately maintained
- **Adds randomness**: Different imputations each run
- **Maintains variability**: Unlike mean/median

**When to use**:
- Want to preserve distribution shape
- Variance important
- Multiple imputation framework

**Limitation**: Doesn't use relationships with other features

## Scikit-Learn SimpleImputer

### Syntax

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Create imputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'

# Fit on training data
imputer.fit(X_train)

# Transform train and test
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

### Parameters

**strategy**: str, default='mean'
- `'mean'`: Replace with mean
- `'median'`: Replace with median
- `'most_frequent'`: Replace with mode
- `'constant'`: Replace with fill_value

**fill_value**: str or numerical, default=None
- Used when strategy='constant'
- Value to replace missing

**missing_values**: int, float, str, np.nan, default=np.nan
- Placeholder for missing values
- Default: np.nan
- Can specify: 0, -1, 'missing', etc.

**add_indicator**: bool, default=False
- If True, add binary indicator for missingness
- Creates additional columns: was_missing_feature1, etc.

**copy**: bool, default=True
- If True, create copy; if False, modify in-place

**keep_empty_features**: bool, default=False
- If True, keep all-NaN columns (filled)
- If False, remove all-NaN columns

### Fitted Attributes

**statistics_**: array
- Imputation values for each feature
- Access: `imputer.statistics_`

**Example**:
```python
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)
print(imputer.statistics_)  # [25.3, 60000.5, 3.2, ...]  # Mean of each column
```

**indicator_**: MissingIndicator object (if add_indicator=True)
- Access: `imputer.indicator_`

### Examples

**Mean imputation**:
```python
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Median imputation**:
```python
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
```

**Constant imputation**:
```python
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer.fit_transform(X)
```

**With missing indicator**:
```python
imputer = SimpleImputer(strategy='mean', add_indicator=True)
X_imputed = imputer.fit_transform(X)
# X_imputed includes original features (imputed) + binary indicators
```

## Mathematical Considerations

### Variance Reduction with Mean Imputation

**Original variance** (observed data only):
$$\sigma^2_{obs} = \frac{1}{n_{obs}}\sum_{i: x_i \text{ obs}} (x_i - \bar{x}_{obs})^2$$

**After mean imputation**:
$$\sigma^2_{imputed} = \frac{1}{n}\left[\sum_{i: x_i \text{ obs}} (x_i - \bar{x})^2 + \sum_{i: x_i \text{ miss}} (\bar{x} - \bar{x})^2\right]$$

Since imputed values = mean, second term = 0:
$$\sigma^2_{imputed} = \frac{n_{obs}}{n} \sigma^2_{obs} < \sigma^2_{obs}$$

**Result**: Variance artificially reduced by factor $\frac{n_{obs}}{n}$

**Example**: 30% missing → variance reduced by 30%

### Correlation Attenuation

**Original correlation** between $X$ and $Y$:
$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**After mean imputation** in $X$:
- Imputed $X$ values uncorrelated with $Y$
- Correlation diluted

$$\rho_{X_{imputed},Y} < \rho_{X_{observed},Y}$$

**Impact**: Weakened relationships between features

### Standard Error Underestimation

**True standard error**:
$$SE = \frac{\sigma}{\sqrt{n}}$$

**With imputation**: $\sigma$ underestimated → $SE$ underestimated

**Result**: 
- Confidence intervals too narrow
- p-values too small
- False confidence in precision

**Solution**: Multiple imputation or bootstrap

## Using Imputation in Pipelines

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Order matters**: Impute → Scale → Model

### Different Strategies per Feature Type

```python
from sklearn.compose import ColumnTransformer

# Identify feature types
normal_features = ['age', 'height', 'weight']
skewed_features = ['income', 'house_price']
count_features = ['num_purchases']

preprocessor = ColumnTransformer([
    ('normal_impute', SimpleImputer(strategy='mean'), normal_features),
    ('skewed_impute', SimpleImputer(strategy='median'), skewed_features),
    ('count_impute', SimpleImputer(strategy='constant', fill_value=0), count_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

### Imputation with Missing Indicator

```python
# Add binary flags for missingness
imputer = SimpleImputer(strategy='median', add_indicator=True)

pipeline = Pipeline([
    ('impute', imputer),
    ('model', GradientBoostingClassifier())
])
```

**Result**: Model can learn if missingness itself is predictive

## Best Practices

### 1. Fit on Training Data Only

**Correct** (no leakage):
```python
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)  # Uses train statistics
```

**Wrong** (leakage):
```python
imputer.fit(X_all)  # Test data leaks!
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

**Why**: Test data shouldn't influence imputation values

### 2. Choose Strategy Based on Distribution

**Check distribution**:
```python
import matplotlib.pyplot as plt

plt.hist(df['feature'], bins=30)
plt.title('Feature Distribution')
plt.show()

# Check skewness
skewness = df['feature'].skew()
if abs(skewness) < 0.5:
    strategy = 'mean'  # Approximately normal
else:
    strategy = 'median'  # Skewed
```

### 3. Document Imputation Choices

```python
missing_before = X.isnull().sum()
X_imputed = imputer.fit_transform(X)
missing_after = pd.DataFrame(X_imputed).isnull().sum()

print(f"Missing before imputation:\n{missing_before}")
print(f"\nMissing after imputation:\n{missing_after}")
print(f"\nImputation values: {imputer.statistics_}")
```

### 4. Validate Imputation Reasonableness

```python
# Check if imputed values make sense
for i, col in enumerate(X.columns):
    imputed_value = imputer.statistics_[i]
    obs_min = X[col].min()
    obs_max = X[col].max()
    
    if obs_min <= imputed_value <= obs_max:
        print(f"{col}: Imputed {imputed_value:.2f} ✓ (within range)")
    else:
        print(f"{col}: Imputed {imputed_value:.2f} ✗ (outside range!)")
```

### 5. Consider Adding Missing Indicators

```python
# If missingness might be informative
imputer = SimpleImputer(strategy='median', add_indicator=True)
```

**Example**: Missing income might indicate:
- Unemployed (predictive for loan default)
- Privacy concern (predictive for behavior)

### 6. Compare Imputation Strategies

```python
from sklearn.model_selection import cross_val_score

strategies = ['mean', 'median', 'most_frequent']
scores = {}

for strategy in strategies:
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy=strategy)),
        ('model', RandomForestClassifier())
    ])
    score = cross_val_score(pipeline, X, y, cv=5).mean()
    scores[strategy] = score
    print(f"{strategy}: {score:.3f}")

best_strategy = max(scores, key=scores.get)
print(f"\nBest: {best_strategy}")
```

### 7. Handle Edge Cases

**All values missing in a column**:
```python
# Option 1: Remove column before imputation
X = X.dropna(axis=1, how='all')

# Option 2: Use keep_empty_features
imputer = SimpleImputer(keep_empty_features=True, strategy='constant', fill_value=0)
```

**Different missing placeholders**:
```python
# If missing encoded as -999
imputer = SimpleImputer(missing_values=-999, strategy='mean')
```

### 8. Visualize Before and After

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before
axes[0].hist(X_train[:, 0], bins=30, alpha=0.7, label='Observed')
axes[0].set_title('Before Imputation')
axes[0].legend()

# After
X_imputed = imputer.fit_transform(X_train)
axes[1].hist(X_imputed[:, 0], bins=30, alpha=0.7, label='With Imputed', color='orange')
axes[1].axvline(imputer.statistics_[0], color='red', linestyle='--', label=f'Imputed Value')
axes[1].set_title('After Imputation')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Limitations of Simple Imputation

### 1. Ignores Feature Relationships

**Problem**: Imputes each feature independently

**Example**:
- Age and Income correlated
- Simple imputation: Replace missing Income with overall median
- Better: Use Age to predict missing Income (see KNN, Iterative Imputation)

### 2. Underestimates Uncertainty

**Problem**: Treats imputed values as if truly observed

**Reality**: Imputed values are estimates with uncertainty

**Solution**: Multiple imputation (create several imputed datasets, combine results)

### 3. Distorts Distributions

**Mean imputation**: Creates spike at mean

**Impact**: 
- Reduces variance
- Changes distribution shape
- Affects downstream analyses

### 4. Weakens Correlations

**Imputed values**: No correlation with other features

**Result**: Attenuated correlation coefficients

**Impact**: Missed relationships, weaker predictive models

### 5. Not Suitable for MNAR

**MNAR**: Missingness depends on unobserved values

**Example**: High earners hide income

**Simple imputation**: Uses observed values (low/medium earners)

**Result**: Biased estimates (underestimate true mean)

## When to Use Each Strategy

### Mean Imputation

**Use**:
- Normal distribution
- Low missingness (<10%)
- Outliers absent
- Linear models

**Avoid**:
- Skewed data
- Outliers present
- High missingness

### Median Imputation

**Use**:
- Skewed distribution
- Outliers present
- Robust estimate needed
- Default choice (safe)

**Avoid**:
- Need exact mean preservation
- Perfectly normal data (mean equally good)

### Constant Imputation

**Use**:
- Missing = meaningful (e.g., zero)
- Want to flag missingness
- Domain-specific constant

**Avoid**:
- Constant not meaningful
- Would distort severely

### Most Frequent (Mode)

**Use**:
- Discrete numerical data
- Few unique values
- Dominant value exists

**Avoid**:
- Continuous data
- Uniform distribution

## Summary

Simple imputation replaces missing numerical values with summary statistics (mean, median, mode) or constants, preserving sample size while introducing estimated values.

**Key Concepts**:

**Imputation Strategies**:
1. **Mean**: $\bar{x}_{obs}$ — for normal distributions
2. **Median**: $\text{median}(x_{obs})$ — robust, for skewed data
3. **Mode**: $\text{mode}(x_{obs})$ — for discrete numerical
4. **Constant**: $c$ — domain-specific value

**sklearn Implementation**:
```python
SimpleImputer(strategy='mean'|'median'|'most_frequent'|'constant')
```

**Advantages**:
- Preserves sample size
- Simple, fast
- Maintains statistical power
- Works with any algorithm
- Better than deletion (usually)

**Disadvantages**:
- Reduces variance (mean/median)
- Weakens correlations
- Ignores feature relationships
- Underestimates uncertainty
- Creates distribution artifacts

**Best Practices**:
- Fit on training data only
- Choose strategy by distribution
- Validate imputed values
- Consider missing indicators
- Compare strategies via CV
- Visualize impact
- Document choices

**When to Use**:
- Missingness < 40%
- MAR assumption reasonable
- Need complete data
- Baseline approach

**Alternatives**:
- KNN Imputer: Uses feature relationships
- Iterative Imputer: Model-based (MICE)
- Missing indicators: Preserve information
- Algorithms with native support: XGBoost, LightGBM

**Mathematical Impact**:
- Variance: $\sigma^2_{imp} = \frac{n_{obs}}{n}\sigma^2_{obs}$ (reduced)
- Correlation: $\rho_{imp} < \rho_{obs}$ (attenuated)
- Standard error: Underestimated (over-confident)

Simple imputation is a practical, widely-used approach that provides a reasonable balance between simplicity and effectiveness for handling missing numerical data in machine learning pipelines.

---

**Video Link**: https://youtu.be/mCL2xLBDw8M
