# Function Transformer - Custom Transformations in Pipelines

## General Idea

FunctionTransformer is a scikit-learn utility that converts any Python function into a transformer that can be used in pipelines. It allows you to incorporate custom data transformations, domain-specific preprocessing, or mathematical operations into the standardized scikit-learn workflow without writing a full custom transformer class. Essentially, it bridges the gap between functional programming and object-oriented transformer APIs.

## Why Use FunctionTransformer?

1. **Simplicity**: Convert functions to transformers without boilerplate code
2. **Custom Logic**: Implement domain-specific transformations
3. **Pipeline Integration**: Use custom functions in sklearn pipelines
4. **Flexibility**: Apply any mathematical transformation
5. **Reusability**: Create transformer libraries from function collections
6. **Rapid Prototyping**: Test transformation ideas quickly
7. **Maintainability**: Separate transformation logic from pipeline structure
8. **Mathematical Operations**: Log, sqrt, polynomial, trigonometric transformations

## Role in Machine Learning

### Feature Engineering

Many ML tasks require custom transformations that aren't provided by sklearn:

**Examples**:
- Log transformation for skewed data
- Square root for variance stabilization
- Custom domain calculations (e.g., BMI from height/weight)
- Date feature extraction (year, month, day of week)
- Text cleaning functions
- Outlier capping/flooring

**FunctionTransformer enables**: Incorporating these into pipelines seamlessly

### Mathematical Transformations

**Common transformations**:

**1. Log Transform** (reduce skewness):
$$X' = \log(X + 1)$$

**2. Square Root** (stabilize variance):
$$X' = \sqrt{X}$$

**3. Reciprocal** (linearize relationships):
$$X' = \frac{1}{X + \epsilon}$$

**4. Power** (adjust distribution):
$$X' = X^\lambda$$

**5. Custom Formula**:
$$\text{BMI} = \frac{\text{weight}}{\text{height}^2}$$

All can be wrapped with FunctionTransformer

### Data Distribution Adjustment

**Problem**: Many algorithms assume normally distributed data

**Solution**: Transform to approximate normality

**Example - Right Skewed Data**:

Original: $X \sim \text{LogNormal}$

Transformed: $X' = \log(X) \sim \text{Normal}$

**Using FunctionTransformer**:
```python
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
```

**Result**: Better model performance on transformed data

## FunctionTransformer Syntax

### Basic Usage

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Create transformer from function
transformer = FunctionTransformer(func=np.log1p)

# Use like any sklearn transformer
transformer.fit(X)  # Usually does nothing (stateless)
X_transformed = transformer.transform(X)
```

### Parameters

**func**: Function to apply
- Main transformation function
- Signature: `func(X) -> X_transformed`
- Example: `np.log1p`, `np.sqrt`, custom function

**inverse_func**: Inverse transformation (optional)
- For reversing transformation
- Enables `inverse_transform()` method
- Example: `np.expm1` for `np.log1p`

**validate**: Whether to validate input (default=False)
- If True: Checks for NaN, inf
- If False: Faster, but no validation

**accept_sparse**: Whether to accept sparse matrices (default=False)
- If True: Can handle scipy sparse matrices

**check_inverse**: Whether to check inverse correctness (default=True)
- Verifies: $f^{-1}(f(X)) \approx X$

**kw_args**: Keyword arguments for func
- Pass parameters to transformation function
- Example: `kw_args={'base': 10}` for log base 10

**inv_kw_args**: Keyword arguments for inverse_func

### Full Syntax Example

```python
transformer = FunctionTransformer(
    func=np.log,
    inverse_func=np.exp,
    validate=True,
    accept_sparse=False,
    check_inverse=True,
    kw_args=None,
    inv_kw_args=None
)
```

## Creating Custom Transformations

### Log Transformation

**Purpose**: Reduce right skewness, compress large values

**Function**:
```python
def log_transform(X):
    return np.log1p(X)  # log(1 + X) to handle zeros

def inverse_log(X):
    return np.expm1(X)  # exp(X) - 1

log_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log
)
```

**Mathematical form**:
$$f(x) = \log(1 + x)$$
$$f^{-1}(y) = e^y - 1$$

**When to use**:
- Right-skewed data (income, prices, counts)
- Multiplicative relationships
- Wide value ranges (1 to 1,000,000)

### Square Root Transformation

**Purpose**: Moderate skewness reduction, variance stabilization

**Function**:
```python
def sqrt_transform(X):
    return np.sqrt(X)

def inverse_sqrt(X):
    return np.square(X)

sqrt_transformer = FunctionTransformer(
    func=sqrt_transform,
    inverse_func=inverse_sqrt
)
```

**Mathematical form**:
$$f(x) = \sqrt{x}$$
$$f^{-1}(y) = y^2$$

**When to use**:
- Poisson-distributed data (counts)
- Moderate skewness
- Non-negative values

### Reciprocal Transformation

**Purpose**: Linearize inverse relationships, handle large outliers

**Function**:
```python
def reciprocal_transform(X, epsilon=1e-6):
    return 1 / (X + epsilon)  # epsilon prevents division by zero

def inverse_reciprocal(X, epsilon=1e-6):
    return (1 / X) - epsilon

reciprocal_transformer = FunctionTransformer(
    func=reciprocal_transform,
    inverse_func=inverse_reciprocal,
    kw_args={'epsilon': 1e-6},
    inv_kw_args={'epsilon': 1e-6}
)
```

**Mathematical form**:
$$f(x) = \frac{1}{x + \epsilon}$$

**When to use**:
- Inverse relationships (speed vs time)
- Left-skewed data
- Rate transformations

### Custom Domain Functions

**Example: Calculate BMI**

```python
def calculate_bmi(X):
    # X has columns: [weight_kg, height_m]
    weight = X[:, 0]
    height = X[:, 1]
    bmi = weight / (height ** 2)
    return bmi.reshape(-1, 1)

bmi_transformer = FunctionTransformer(func=calculate_bmi)
```

**Mathematical form**:
$$\text{BMI} = \frac{\text{weight (kg)}}{\text{height (m)}^2}$$

**Example: Extract Date Features**

```python
import pandas as pd

def extract_date_features(X):
    # X is datetime column
    df = pd.DataFrame(X, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    features = pd.DataFrame({
        'year': df['date'].dt.year,
        'month': df['date'].dt.month,
        'day': df['date'].dt.day,
        'dayofweek': df['date'].dt.dayofweek,
        'quarter': df['date'].dt.quarter
    })
    return features.values

date_transformer = FunctionTransformer(func=extract_date_features)
```

### Clipping/Capping Outliers

**Purpose**: Limit extreme values to reduce their influence

**Function**:
```python
def clip_outliers(X, lower_percentile=5, upper_percentile=95):
    lower = np.percentile(X, lower_percentile, axis=0)
    upper = np.percentile(X, upper_percentile, axis=0)
    return np.clip(X, lower, upper)

clipper = FunctionTransformer(
    func=clip_outliers,
    kw_args={'lower_percentile': 5, 'upper_percentile': 95}
)
```

**Mathematical form**:
$$f(x) = \begin{cases}
L & \text{if } x < L \\
x & \text{if } L \leq x \leq U \\
U & \text{if } x > U
\end{cases}$$

Where $L = P_5(X)$, $U = P_{95}(X)$

## Using FunctionTransformer in Pipelines

### Simple Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Flow**:
$$X \xrightarrow{\log(1+x)} X_1 \xrightarrow{\text{scale}} X_2 \xrightarrow{\text{model}} \hat{y}$$

### Multiple Custom Transformations

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), ['price', 'income']),
    ('sqrt', FunctionTransformer(np.sqrt), ['count', 'age']),
    ('scale', StandardScaler(), ['height', 'weight'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

**Benefit**: Different transformations for different feature types

### Chaining Transformations

```python
pipeline = Pipeline([
    ('clip', FunctionTransformer(clip_outliers)),
    ('log', FunctionTransformer(np.log1p)),
    ('scale', StandardScaler()),
    ('model', Ridge())
])
```

**Order matters**:
1. Clip outliers first
2. Then log transform
3. Then scale

## Stateless vs Stateful Transformations

### Stateless (Most FunctionTransformers)

**Definition**: Transformation doesn't depend on training data statistics

**Examples**:
- $f(x) = \log(x)$: Same function always
- $f(x) = \sqrt{x}$: No learned parameters
- $f(x) = x^2$: Pure mathematical operation

**fit() method**: Does nothing (just returns self)

**Behavior**:
```python
transformer.fit(X_train)  # No learning
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)
# Both use same function
```

### Stateful (Custom FunctionTransformers)

**Definition**: Transformation depends on training data

**Examples**:
- Clipping at percentiles: $P_5(X_{train})$, $P_{95}(X_{train})$
- Normalization: $\min(X_{train})$, $\max(X_{train})$
- Z-score: $\mu_{train}$, $\sigma_{train}$

**Problem**: Standard FunctionTransformer is stateless

**Solution**: Use custom transformer class or capture in closure

**Workaround** (not recommended):
```python
class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=5, upper=95):
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        self.lower_bound_ = np.percentile(X, self.lower, axis=0)
        self.upper_bound_ = np.percentile(X, self.upper, axis=0)
        return self
    
    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)
```

**Better**: Use QuantileTransformer or custom class for stateful operations

## Inverse Transformations

### Purpose

**Use case**: Reverse transformation to original scale

**Example**: Model trained on log(price), predictions are log-scale

**Need**: Convert predictions back to original price scale

### Implementation

**With inverse function**:
```python
log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1
)

# Transform
X_log = log_transformer.transform(X)

# Inverse transform
X_original = log_transformer.inverse_transform(X_log)
# X_original ≈ X
```

**Mathematical verification**:
$$f^{-1}(f(x)) = e^{\log(1+x)} - 1 = (1+x) - 1 = x$$

### Pipeline Inverse Transform

**Full pipeline with inverse**:
```python
pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p, inverse_func=np.expm1)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred_scaled = pipeline.predict(X_test)

# To get predictions in original scale:
# 1. Get predictions (already log-scaled)
# 2. Inverse transform through pipeline steps
y_pred_original = pipeline.named_steps['log'].inverse_transform(y_pred_scaled)
```

**Note**: Only works if each step has inverse_transform

## Mathematical Examples

### Example 1: Log Transform for House Prices

**Problem**: House prices are right-skewed

**Data**: Prices range from $100K to $10M

**Transformation**:
```python
log_transformer = FunctionTransformer(np.log1p)
```

**Effect**:

Original: $[100000, 500000, 1000000, 5000000]$

Transformed: $[\log(100001), \log(500001), \log(1000001), \log(5000001)]$
           $\approx [11.51, 13.12, 13.82, 15.42]$

**Variance reduction**:
- Original std: $\approx 2,000,000$
- Transformed std: $\approx 1.5$

### Example 2: Square Root for Count Data

**Problem**: Website visit counts (Poisson-distributed)

**Data**: $[1, 4, 9, 16, 100]$

**Transformation**:
```python
sqrt_transformer = FunctionTransformer(np.sqrt)
```

**Effect**:

Original: $[1, 4, 9, 16, 100]$

Transformed: $[1, 2, 3, 4, 10]$

**Variance stabilization**:
- Poisson: $\text{Var}(X) = \mu$ (variance equals mean)
- After sqrt: $\text{Var}(\sqrt{X}) \approx \text{constant}$

### Example 3: Custom Polynomial Features

**Create interaction terms**:
```python
def polynomial_interactions(X):
    # X has 2 columns: x1, x2
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    features = np.column_stack([
        X,              # Original: x1, x2
        x1**2,          # x1^2
        x2**2,          # x2^2
        x1 * x2         # x1 * x2
    ])
    return features

poly_transformer = FunctionTransformer(polynomial_interactions)
```

**Mathematical form**:
$$f([x_1, x_2]) = [x_1, x_2, x_1^2, x_2^2, x_1 x_2]$$

**Input**: 2 features
**Output**: 5 features

## Validation and Error Handling

### validate Parameter

**Purpose**: Check for invalid values

**Enable validation**:
```python
transformer = FunctionTransformer(
    func=np.log,  # Fails for x <= 0
    validate=True
)
```

**Checks**:
- NaN values
- Infinite values
- Finite array

**Trade-off**:
- validate=True: Safer, slower
- validate=False: Faster, may propagate errors

### Handling Invalid Inputs

**Problem**: Function fails on certain inputs

**Example**: $\log(x)$ undefined for $x \leq 0$

**Solutions**:

**1. Use safe variant**:
```python
FunctionTransformer(np.log1p)  # log(1 + x), works for x >= -1
```

**2. Clip before transform**:
```python
def safe_log(X, epsilon=1e-10):
    return np.log(np.maximum(X, epsilon))

transformer = FunctionTransformer(safe_log)
```

**3. Filter invalid values**:
```python
def log_with_filter(X):
    X = X.copy()
    valid_mask = X > 0
    X[valid_mask] = np.log(X[valid_mask])
    X[~valid_mask] = 0  # or np.nan, or other handling
    return X
```

### check_inverse

**Purpose**: Verify inverse function correctness

**Default**: True

**Process**:
```python
transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=True
)

transformer.fit(X)  # Checks: inverse_func(func(X)) ≈ X
```

**Tolerance**: Small numerical errors allowed ($< 10^{-7}$)

**Disable if**: Inverse is approximate or costly to compute

## Common Patterns

### Pattern 1: Log Transform Skewed Features

```python
from sklearn.compose import ColumnTransformer

skewed_features = ['price', 'income', 'loan_amount']

preprocessor = ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), skewed_features),
    ('scale', StandardScaler(), normal_features)
])
```

### Pattern 2: Apply Same Function to All Features

```python
pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('scale', StandardScaler()),
    ('model', Ridge())
])
```

### Pattern 3: Custom Feature Engineering

```python
def create_features(X):
    # X: [age, income]
    age = X[:, 0]
    income = X[:, 1]
    
    return np.column_stack([
        age,
        income,
        age * income,           # Interaction
        np.log1p(income),       # Log income
        (age > 30).astype(int)  # Age indicator
    ])

pipeline = Pipeline([
    ('features', FunctionTransformer(create_features)),
    ('model', LogisticRegression())
])
```

### Pattern 4: Normalize Then Custom Transform

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('activation', FunctionTransformer(lambda x: np.tanh(x))),
    ('model', SVC())
])
```

**Effect**: Apply tanh activation to scaled features

## Comparison with Alternatives

### vs Custom Transformer Class

**FunctionTransformer**:
- **Pros**: Simple, quick, no boilerplate
- **Cons**: Limited to stateless, less flexible
- **Use when**: Simple mathematical transformation

**Custom Class**:
- **Pros**: Full control, can be stateful, better error handling
- **Cons**: More code, requires class structure
- **Use when**: Complex logic, need state, reusability

**Example**:
```python
# FunctionTransformer: 1 line
FunctionTransformer(np.log1p)

# Custom class: ~15 lines
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(X)
```

### vs PowerTransformer/QuantileTransformer

**FunctionTransformer**:
- Fixed transformation (log, sqrt, etc.)
- No parameter learning
- You control exact function

**PowerTransformer**:
- Learns optimal power parameter $\lambda$
- Box-Cox or Yeo-Johnson
- Automatic normalization

**QuantileTransformer**:
- Learns quantile mapping from data
- Transforms to uniform/normal
- Data-driven

**Choose FunctionTransformer when**: You know the exact transformation needed

## Best Practices

### 1. Use Safe Variants

**Avoid**:
```python
FunctionTransformer(np.log)  # Fails for x <= 0
```

**Prefer**:
```python
FunctionTransformer(np.log1p)  # Works for x >= 0
```

### 2. Provide Inverse When Possible

**Good**:
```python
FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1
)
```

**Why**: Enables inverse_transform, helpful for interpreting results

### 3. Document Custom Functions

```python
def custom_transform(X):
    """
    Apply domain-specific transformation:
    - Clips outliers at 5th and 95th percentiles
    - Takes square root
    - Adds constant offset
    
    Parameters:
    X : array, shape (n_samples, n_features)
    
    Returns:
    X_transformed : array, same shape
    """
    X_clipped = np.clip(X, np.percentile(X, 5), np.percentile(X, 95))
    return np.sqrt(X_clipped + 1)
```

### 4. Validate in Development, Disable in Production

**Development**:
```python
FunctionTransformer(func, validate=True)  # Catch errors
```

**Production** (after validation):
```python
FunctionTransformer(func, validate=False)  # Faster
```

### 5. Use ColumnTransformer for Selective Application

**Apply to specific columns**:
```python
ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), ['price']),
    ('sqrt', FunctionTransformer(np.sqrt), ['count']),
    ('identity', 'passthrough', ['category'])
])
```

**Better than**: Applying to all features indiscriminately

## Summary

FunctionTransformer enables incorporating custom Python functions into scikit-learn pipelines, providing flexibility for domain-specific transformations and mathematical operations.

**Key Concepts**:

**Core Functionality**:
- Converts functions to sklearn-compatible transformers
- Enables custom transformations in pipelines
- Simplifies code (no custom class needed)

**Common Transformations**:
- Log: $f(x) = \log(1+x)$ (reduce skewness)
- Square root: $f(x) = \sqrt{x}$ (stabilize variance)
- Reciprocal: $f(x) = 1/x$ (inverse relationships)
- Custom: Domain-specific calculations

**Syntax**:
```python
FunctionTransformer(
    func=transformation_function,
    inverse_func=inverse_function,
    validate=False,
    accept_sparse=False
)
```

**Pipeline Integration**:
```python
Pipeline([
    ('custom', FunctionTransformer(custom_func)),
    ('scaler', StandardScaler()),
    ('model', Estimator())
])
```

**Best Practices**:
- Use safe variants (log1p vs log)
- Provide inverse functions
- Document custom transformations
- Validate in development
- Apply selectively with ColumnTransformer
- Consider stateful alternatives when needed

**When to Use**:
- Simple mathematical transformations
- Domain-specific feature engineering
- Quick prototyping
- Stateless operations

**When NOT to Use**:
- Need to learn parameters from data (use PowerTransformer, QuantileTransformer)
- Complex stateful logic (write custom transformer class)
- Operations already in sklearn (use existing transformers)

**Advantages**:
- Minimal code
- Pipeline compatible
- Flexible
- Easy to understand

**Limitations**:
- Typically stateless
- Less robust error handling than custom classes
- Limited parameter learning

FunctionTransformer is an essential tool for bridging custom data transformations with sklearn's standardized pipeline framework, enabling clean, maintainable feature engineering workflows.

---

**Video Link**: https://youtu.be/cTjj3LE8E90
