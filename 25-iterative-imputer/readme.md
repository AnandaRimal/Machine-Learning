# Iterative Imputer - MICE (Multivariate Imputation by Chained Equations)

## General Idea

Iterative Imputer, also known as MICE (Multivariate Imputation by Chained Equations) or FCS (Fully Conditional Specification), fills missing values by modeling each feature with missing values as a function of other features in a round-robin fashion. It iteratively predicts missing values using a regression model trained on observed values, cycling through features until convergence. This sophisticated approach captures complex feature relationships and provides highly accurate imputations.

## Why Use Iterative Imputation?

1. **Models Feature Dependencies**: Captures complex relationships between features
2. **Multivariate Approach**: Uses all available information
3. **High Accuracy**: Often most accurate imputation method
4. **Flexible Models**: Can use linear, tree-based, or any regressor
5. **Handles MAR Well**: Leverages observed data patterns
6. **Iterative Refinement**: Improves estimates over iterations
7. **Statistically Principled**: Based on conditional distributions
8. **Uncertainty Quantification**: Multiple imputation variant available

## Role in Machine Learning

### Comparison with Other Methods

**Simple Imputation** (Mean):
```
Age   Income   Score
25    50k      NaN   →  Fill Score with global mean (85)
```

**Ignores**: Age, Income

**KNN Imputation**:
```
Age   Income   Score
25    50k      NaN   →  Find similar samples, average their Score
```

**Uses**: Similarity (distance)

**Iterative Imputation**:
```
Age   Income   Score
25    50k      NaN   →  Train model: Score = f(Age, Income)
                      Predict: Score_pred = f(25, 50k)
```

**Uses**: Predictive model trained on complete cases

**Advantage**: Learns functional relationships (e.g., "higher income → higher score")

### The MICE Algorithm

**Goal**: Impute missing values in features $X_1, X_2, ..., X_p$

**Process**:

**Step 0: Initial Imputation**
- Fill all missing with simple method (e.g., mean)
- Creates complete dataset (temporary)

**Step 1: Iterate Through Features**

For each feature $j$ with missing values:

1. **Set as target**: $y = X_j$ (observed values only)

2. **Use others as predictors**: $X_{-j} = [X_1, ..., X_{j-1}, X_{j+1}, ..., X_p]$

3. **Train model**: 
   $$f_j: X_{-j} \to X_j$$
   
   Using samples where $X_j$ is observed

4. **Predict missing**: 
   $$\hat{X}_j[\text{missing}] = f_j(X_{-j}[\text{missing}])$$

5. **Update**: Replace old imputed values with new predictions

**Step 2: Repeat Until Convergence**

Cycle through all features multiple times (iterations)

Stop when imputations stabilize (change < threshold)

**Typical**: 10 iterations

### Mathematical Formulation

**Feature $j$ with missing values**:

**Observed**: $X_j^{obs}$

**Missing**: $X_j^{mis}$

**Other features**: $X_{-j}$

**Model** for iteration $t$:
$$X_j^{mis,(t)} = f_j(X_{-j}^{(t-1)}) + \epsilon$$

Where:
- $f_j$: Regression model (linear, tree, etc.)
- $X_{-j}^{(t-1)}$: Other features from previous iteration
- $\epsilon$: Residual error

**Convergence**: 
$$\max_j ||X_j^{(t)} - X_j^{(t-1)}|| < \text{tolerance}$$

### Example: Step-by-Step

**Data**:
```
     Age   Income   Score
 0   25    50       80
 1   30    NaN      85
 2   NaN   60       90
 3   35    70       NaN
```

**Iteration 0: Initial Imputation** (mean)
```
     Age   Income   Score
 0   25    50       80
 1   30    60       85
 2   30    60       90
 3   35    70       85
```

**Iteration 1:**

**Feature: Income** (row 1 missing)
- Target: Income (rows 0, 2, 3 with observed Income)
- Predictors: Age, Score
- Train: Income = f(Age, Score)
- Predict: Income[1] = f(30, 85) = 62
- Update:
  ```
       Age   Income   Score
   1   30    62       85  ← Updated
  ```

**Feature: Age** (row 2 missing)
- Target: Age (rows 0, 1, 3)
- Predictors: Income, Score
- Train: Age = f(Income, Score)
- Predict: Age[2] = f(60, 90) = 32
- Update:
  ```
       Age   Income   Score
   2   32    60       90  ← Updated
  ```

**Feature: Score** (row 3 missing)
- Target: Score (rows 0, 1, 2)
- Predictors: Age, Income
- Train: Score = f(Age, Income)
- Predict: Score[3] = f(35, 70) = 88
- Update:
  ```
       Age   Income   Score
   3   35    70       88  ← Updated
  ```

**After Iteration 1**:
```
     Age   Income   Score
 0   25    50       80
 1   30    62       85
 2   32    60       90
 3   35    70       88
```

**Iteration 2:**
- Repeat process with updated values
- Predictions refined based on better estimates
- Continue until convergence (typically 10 iterations)

## Scikit-Learn IterativeImputer

### Basic Usage

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

X = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [np.nan, 8, 9],
    [10, 11, 12]
])

imputer = IterativeImputer(random_state=0)
X_imputed = imputer.fit_transform(X)

print(X_imputed)
# Imputed values based on learned relationships
```

**Note**: IterativeImputer is **experimental** in sklearn (as of 2024)
- `from sklearn.experimental import enable_iterative_imputer` required
- API may change in future versions

### Parameters

**estimator**: estimator object, default=BayesianRidge()
- Regression model to predict missing values
- Default: `BayesianRidge()` (regularized linear regression)
- Can be: `LinearRegression()`, `RandomForestRegressor()`, `ExtraTreesRegressor()`, etc.
- Must implement `fit()` and `predict()`

**missing_values**: int, float, str, np.nan, None, default=np.nan
- Placeholder for missing values

**sample_posterior**: bool, default=False
- If True, sample from posterior distribution (adds uncertainty)
- If False, use point estimate (mean prediction)
- Relevant for multiple imputation

**max_iter**: int, default=10
- Maximum number of imputation rounds
- More iterations: Better convergence, slower
- Typical: 10-20

**tol**: float, default=1e-3
- Convergence tolerance
- Stop if max change < tol
- Lower: Stricter convergence, more iterations

**n_nearest_features**: int, default=None
- Number of features to use for imputing each feature
- None: Use all features
- int: Use N most correlated features (faster for high-dim)

**initial_strategy**: str, default='mean'
- Initial imputation before iterations
- Options: 'mean', 'median', 'most_frequent', 'constant'

**imputation_order**: str, default='ascending'
- Order to impute features
- 'ascending': Low to high index
- 'descending': High to low
- 'roman': Left to right
- 'arabic': Right to left
- 'random': Random order each iteration

**skip_complete**: bool, default=False
- If True, skip features with no missing values
- If False, model them anyway (can still be updated)

**min_value**: float or array, default=-np.inf
- Minimum possible imputed value (clipping)

**max_value**: float or array, default=np.inf
- Maximum possible imputed value

**random_state**: int, default=None
- Seed for reproducibility

**add_indicator**: bool, default=False
- Add missing indicator features

### Example with Parameters

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Use Random Forest as estimator
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    max_iter=20,
    random_state=0,
    initial_strategy='median',
    imputation_order='random',
    add_indicator=True
)

X_imputed = imputer.fit_transform(X_train)
```

## Choosing the Estimator

### Linear Regression (Fast)

```python
from sklearn.linear_model import LinearRegression

imputer = IterativeImputer(estimator=LinearRegression())
```

**Pros**: Fast, simple

**Cons**: Assumes linearity

**Use when**: Linear relationships, many samples

### Bayesian Ridge (Default)

```python
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer(estimator=BayesianRidge())
```

**Pros**: Regularized (prevents overfitting), uncertainty quantification

**Cons**: Assumes linearity

**Use when**: Default choice, stable estimates

### Random Forest (Non-linear)

```python
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=0)
)
```

**Pros**: Handles non-linearity, feature interactions, robust

**Cons**: Slower, can overfit

**Use when**: Complex relationships, non-linear data

### Extra Trees (Fast Non-linear)

```python
from sklearn.ensemble import ExtraTreesRegressor

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=0)
)
```

**Pros**: Faster than RF, less overfitting

**Cons**: May be less accurate

**Use when**: Need speed + non-linearity

### Comparison

**Dataset**: 1000 samples, 10 features, 20% missing

**Results**:
```
LinearRegression:      Time=2s,  MAE=0.45
BayesianRidge:         Time=3s,  MAE=0.42  ← Default, good balance
RandomForestRegressor: Time=15s, MAE=0.38  ← Best accuracy
ExtraTreesRegressor:   Time=8s,  MAE=0.40
```

**Recommendation**: 
- Start with `BayesianRidge` (default)
- If non-linear, try `RandomForestRegressor`
- If slow, use `ExtraTreesRegressor`

## Convergence and Iterations

### Monitoring Convergence

**Imputed values should stabilize**:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

# Track imputed values across iterations
class MonitoredIterativeImputer(IterativeImputer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
    
    def _impute_one_feature(self, *args, **kwargs):
        result = super()._impute_one_feature(*args, **kwargs)
        self.history.append(result.copy())
        return result

imputer = MonitoredIterativeImputer(max_iter=20, random_state=0)
X_imputed = imputer.fit_transform(X)

# Check convergence
if len(imputer.history) > 1:
    changes = [np.abs(imputer.history[i] - imputer.history[i-1]).max() 
               for i in range(1, len(imputer.history))]
    print("Max change per iteration:", changes)
```

**Typical pattern**:
```
Iteration 1: 5.23  (large initial change)
Iteration 2: 2.14
Iteration 3: 0.89
Iteration 4: 0.34
Iteration 5: 0.12
Iteration 6: 0.04  (converged, change < tol)
```

### Choosing max_iter

**Rules of thumb**:
- Simple data: 5-10 iterations
- Complex data: 10-20 iterations
- High missingness: 15-30 iterations

**Cross-validation**:
```python
for max_iter in [5, 10, 15, 20]:
    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    X_imp = imputer.fit_transform(X_train)
    
    # Evaluate on downstream task
    model = RandomForestClassifier()
    model.fit(X_imp, y_train)
    score = model.score(X_test_imputed, y_test)
    
    print(f"max_iter={max_iter}: {score:.4f}")
```

**Diminishing returns**: Accuracy plateaus after enough iterations

## Imputation Order

### Strategies

**ascending** (default):
- Feature 0, 1, 2, ..., p-1
- Deterministic, reproducible

**descending**:
- Feature p-1, p-2, ..., 1, 0
- Reverse order

**random**:
- Random order each iteration
- Breaks dependency on order
- Can improve convergence

**roman** / **arabic**:
- Same as ascending/descending (legacy names)

### Impact

**Order matters** when features have different missingness patterns:

**Example**:
- Feature A: 10% missing
- Feature B: 30% missing

**ascending** (A then B):
- Impute A with better info (less missing)
- Impute B using improved A
- Often better

**random**:
- Averages over orders
- More robust
- Slightly slower (randomness overhead)

**Recommendation**: Use **random** for robustness

## Advanced Features

### n_nearest_features (Feature Selection)

**Problem**: High-dimensional data slows imputation

**Solution**: Use only most relevant features

```python
imputer = IterativeImputer(
    n_nearest_features=10,  # Use 10 most correlated features
    random_state=0
)
```

**How it works**: For each feature, select N most correlated features as predictors

**Example**: 100 features
- Without: Each feature predicted by 99 others (slow)
- With n_nearest_features=10: Each feature predicted by 10 most correlated (fast)

**Correlation calculation**: 
$$r_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}$$

Select top $N$ with highest $|r_{ij}|$

**Trade-off**: Speed vs accuracy

### min_value and max_value (Clipping)

**Ensure realistic imputations**:

```python
# Age must be 0-120
imputer = IterativeImputer(min_value=0, max_value=120)
```

**Per-feature clipping**:
```python
# Different bounds per feature
min_vals = [0, 0, -10]     # Min for features 0, 1, 2
max_vals = [120, 100, 10]  # Max for features 0, 1, 2

imputer = IterativeImputer(min_value=min_vals, max_value=max_vals)
```

**Post-prediction**:
$$\hat{x}_{clipped} = \max(\min\_value, \min(\hat{x}, \max\_value))$$

### sample_posterior (Multiple Imputation)

**Concept**: Capture uncertainty by generating multiple imputed datasets

**Single imputation** (sample_posterior=False):
- One imputed dataset
- Point estimate (mean prediction)

**Multiple imputation** (sample_posterior=True):
- Multiple imputed datasets (M times)
- Each samples from posterior distribution
- Accounts for imputation uncertainty

**Process**:
1. Generate M imputed datasets
2. Analyze each separately (fit model M times)
3. Pool results (average coefficients, combine variances)

**sklearn implementation**:
```python
M = 5  # Number of imputations
imputed_datasets = []

for m in range(M):
    imputer = IterativeImputer(sample_posterior=True, random_state=m)
    X_imp = imputer.fit_transform(X)
    imputed_datasets.append(X_imp)

# Train model on each
models = []
for X_imp in imputed_datasets:
    model = LogisticRegression()
    model.fit(X_imp, y)
    models.append(model)

# Pool predictions (average)
X_test_imp = imputer.transform(X_test)
predictions = np.mean([model.predict_proba(X_test_imp) for model in models], axis=0)
```

**Rubin's Rules** (combining estimates):

**Pooled estimate**:
$$\bar{\theta} = \frac{1}{M} \sum_{m=1}^M \hat{\theta}_m$$

**Total variance**:
$$T = \bar{U} + \left(1 + \frac{1}{M}\right) B$$

Where:
- $\bar{U}$: Within-imputation variance
- $B$: Between-imputation variance

## Complete Pipeline Example

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data_with_missing.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: Simple Imputation
baseline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"Baseline (Simple Imputation): {baseline_score:.4f}")

# Iterative Imputation (Linear)
iterative_linear = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

iterative_linear.fit(X_train, y_train)
linear_score = iterative_linear.score(X_test, y_test)
print(f"Iterative (BayesianRidge): {linear_score:.4f}")

# Iterative Imputation (Random Forest)
iterative_rf = Pipeline([
    ('imputer', IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
        max_iter=10,
        random_state=42
    )),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

iterative_rf.fit(X_train, y_train)
rf_score = iterative_rf.score(X_test, y_test)
print(f"Iterative (ExtraTrees): {rf_score:.4f}")

# Compare
print("\nImprovement over baseline:")
print(f"  Linear: {(linear_score - baseline_score)*100:.2f}%")
print(f"  ExtraTrees: {(rf_score - baseline_score)*100:.2f}%")

# Cross-validation
print("\nCross-validation scores:")
for name, pipeline in [('Baseline', baseline), 
                        ('Iterative-Linear', iterative_linear),
                        ('Iterative-RF', iterative_rf)]:
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

**Typical output**:
```
Baseline (Simple Imputation): 0.8345
Iterative (BayesianRidge): 0.8512
Iterative (ExtraTrees): 0.8623

Improvement over baseline:
  Linear: 1.67%
  ExtraTrees: 2.78%

Cross-validation scores:
Baseline: 0.8289 (+/- 0.0234)
Iterative-Linear: 0.8445 (+/- 0.0198)
Iterative-RF: 0.8567 (+/- 0.0211)
```

## Best Practices

### 1. Start Simple, Then Iterate

**Progression**:
1. Baseline: SimpleImputer (mean/median)
2. Upgrade: IterativeImputer with BayesianRidge (default)
3. Advanced: IterativeImputer with RandomForest

**Only advance if**: Significant improvement

### 2. Use Appropriate Estimator

**Linear relationships**: BayesianRidge, LinearRegression

**Non-linear**: ExtraTreesRegressor, RandomForestRegressor

**High-dimensional**: BayesianRidge (regularization helps)

### 3. Set Reasonable max_iter

**Default 10** is usually sufficient

**Increase if**: Convergence plots show non-convergence

**Don't overdo**: Diminishing returns after 15-20

### 4. Use random_state for Reproducibility

```python
iterativeImputer(random_state=42)
```

**Ensures**: Same imputations each run

### 5. Monitor Computational Cost

**IterativeImputer is slow**:
- $O(n \times p^2 \times \text{max\_iter})$
- Especially with tree-based estimators

**For large data**:
- Use n_nearest_features to reduce dimensionality
- Use simpler estimator (LinearRegression)
- Consider SimpleImputer or KNNImputer

### 6. Validate Imputation Quality

**Artificial missingness test**:
```python
# On complete subset
X_complete = X[X.notnull().all(axis=1)]

# Create artificial missing
mask = np.random.rand(*X_complete.shape) < 0.2
X_missing = X_complete.copy()
X_missing[mask] = np.nan

# Impute
imputer = IterativeImputer()
X_imputed = imputer.fit_transform(X_missing)

# Compare
MAE = np.abs(X_imputed[mask] - X_complete[mask]).mean()
print(f"Imputation MAE: {MAE:.4f}")
```

### 7. Combine with Missing Indicators

```python
IterativeImputer(add_indicator=True)
```

**Preserves**: Information about which values were imputed

## Summary

Iterative Imputer (MICE) is a sophisticated imputation method that models each feature with missing values as a function of others, cycling through features iteratively until convergence to produce highly accurate multivariate imputations.

**Key Concepts**:

**Algorithm (MICE)**:
1. Initial imputation (e.g., mean)
2. For each feature with missing:
   - Train regression: $X_j = f(X_{-j})$ on observed
   - Predict missing values
   - Update imputed values
3. Repeat for max_iter iterations or until convergence

**sklearn Implementation**:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X)
```

**Key Parameters**:
- **estimator**: Regression model (default=BayesianRidge)
- **max_iter**: Number of iterations (default=10)
- **initial_strategy**: Initial fill method (default='mean')
- **imputation_order**: Feature order (default='ascending', try 'random')
- **n_nearest_features**: Feature subset for speed
- **sample_posterior**: Multiple imputation (uncertainty)

**Estimator Choices**:
- **BayesianRidge**: Default, regularized linear, stable
- **LinearRegression**: Fast, simple, linear
- **RandomForestRegressor**: Non-linear, accurate, slow
- **ExtraTreesRegressor**: Non-linear, faster than RF

**Advantages**:
- Most accurate imputation method
- Models feature dependencies
- Handles complex relationships
- Flexible (any regressor)
- Multivariate approach
- Iterative refinement

**Limitations**:
- Computationally expensive
- Slow for large datasets
- Risk of overfitting (complex estimators)
- Assumes MAR mechanism
- Requires tuning (estimator, max_iter)
- Experimental in sklearn

**Best Practices**:
- Start with BayesianRidge, upgrade if needed
- Set random_state for reproducibility
- Use max_iter=10 as default, increase if needed
- Monitor computational cost
- Use n_nearest_features for high dimensions
- Validate imputation quality
- Compare to simpler baselines
- Consider missing indicators

**When to Use**:
- High accuracy needed
- Strong feature relationships
- Moderate dataset size
- Can afford computation
- MAR mechanism
- Critical features

**When to Avoid**:
- Very large datasets (n > 50,000)
- Real-time constraints
- Simple baseline sufficient
- MCAR data (simple methods work)

**Comparison**:
- **vs Simple**: More accurate, much slower
- **vs KNN**: More accurate, slower, better for high-dim
- **vs Model-based**: Similar accuracy, more systematic

Iterative Imputer represents the state-of-the-art in single imputation, providing a principled and highly accurate approach to handling missing data by leveraging the full multivariate structure of the dataset.
