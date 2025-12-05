# Missing Indicator - Preserving Missingness Information

## General Idea

A Missing Indicator is a binary feature (0/1) that flags whether a value was originally missing before imputation. Even after filling missing values with imputation, the pattern of missingness itself may contain valuable information for prediction. Missing indicators preserve this information by creating additional columns that mark which values were imputed, allowing models to learn whether missingness is predictive of the target.

## Why Use Missing Indicators?

1. **Missingness May Be Informative**: Pattern of missing data can predict target
2. **Preserve Information**: Don't lose signal when imputing
3. **Handle MNAR**: When missingness depends on hidden values
4. **Improve Model Performance**: Additional predictive features
5. **Transparency**: Track which values were estimated
6. **Interaction Effects**: Combine with imputed values
7. **Simple Implementation**: Easy to add to pipeline
8. **No Assumptions**: Works regardless of missingness mechanism

## Role in Machine Learning

### The Information in Missingness

**Example 1: Income Missing**

**Pattern**: High earners don't report income (MNAR)

**Imputation**: Fill with median → underestimates high earners

**Missing indicator**:
```
Income    Income_Imputed    Income_was_missing
50000          50000              0
NaN → 60000   60000              1  ← Flag: originally missing
75000          75000              0
NaN → 60000   60000              1
```

**Model learns**: `Income_was_missing=1` correlates with higher actual values

**Result**: Better predictions despite imperfect imputation

**Example 2: Medical Data**

**Pattern**: Sicker patients skip optional tests (MNAR)

**Missing test result** itself indicates severity

**Missing indicator**: Captures this relationship

**Example 3: Survey Data**

**Pattern**: Younger people skip age question

**Missingness in age** predicts demographic group

**Model uses**: Both imputed age AND missing flag

### Mathematical Justification

**Original feature**: $X_i$

**Missingness indicator**: 
$$M_i = \begin{cases}
1 & \text{if } X_i \text{ was missing} \\
0 & \text{if } X_i \text{ was observed}
\end{cases}$$

**After imputation**: $\tilde{X}_i$ (imputed value)

**Model input**: Both $\tilde{X}_i$ and $M_i$

**Prediction**:
$$\hat{y} = f(\tilde{X}_1, M_1, \tilde{X}_2, M_2, ..., \tilde{X}_p, M_p)$$

**Interpretation**:
- $\tilde{X}_i$: Best guess of value
- $M_i$: Uncertainty flag / missingness pattern

**Combined information**: More complete than either alone

### When Missingness Is Predictive

**MNAR (Missing Not At Random)**:
- Missingness depends on unobserved value
- Example: High debt → hide debt amount
- Missing indicator captures this signal

**MAR (Missing At Random)**:
- Missingness depends on observed features
- Example: Young people skip income question
- Missing indicator still helpful (interaction effects)

**Even MCAR**:
- Random missingness generally not predictive
- But indicator doesn't hurt (model can ignore)

## Creating Missing Indicators

### Manual Creation (Pandas)

**Before imputation**:
```python
import pandas as pd
import numpy as np

# Create binary indicator
df['Income_was_missing'] = df['Income'].isnull().astype(int)

# Then impute
df['Income'] = df['Income'].fillna(df['Income'].median())
```

**Result**:
```
Income    Income_was_missing
50000            0
60000            1  ← This was NaN
75000            0
60000            1  ← This was NaN
```

### For Multiple Features

```python
# Create indicators for all numeric columns with missing
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().any():
        df[f'{col}_missing'] = df[col].isnull().astype(int)

# Then impute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

## Scikit-Learn MissingIndicator

### Standalone Usage

```python
from sklearn.impute import MissingIndicator
import numpy as np

X = np.array([
    [10, 20, 30],
    [np.nan, 25, 35],
    [15, np.nan, 40],
    [20, 30, np.nan]
])

indicator = MissingIndicator()
indicator.fit(X)
X_missing = indicator.transform(X)

print(X_missing)
# [[False False False]
#  [True  False False]
#  [False True  False]
#  [False False True ]]
```

**Output**: Boolean array indicating missingness

### Parameters

**missing_values**: number, string, np.nan (default=np.nan)
- Placeholder for missing values
- Can be: np.nan, 0, -1, 'missing', etc.

**features**: 'missing-only' (default) or 'all'
- `'missing-only'`: Create indicators only for features with missing values
- `'all'`: Create indicators for all features (even if complete)

**sparse**: bool, default=True
- If True, return sparse matrix
- If False, return dense array

**error_on_new**: bool, default=True
- If True, error when transform sees new missing pattern
- If False, silently ignore

### Fitted Attributes

**features_**: array of indices
- Which features have indicators
- Example: `[0, 2, 5]` → indicators for features 0, 2, 5

### Example

```python
from sklearn.impute import MissingIndicator

X_train = np.array([
    [1, np.nan, 3],
    [4, 5, np.nan],
    [np.nan, 8, 9]
])

# Fit indicator
indicator = MissingIndicator(features='missing-only')
indicator.fit(X_train)

print(f"Features with missing: {indicator.features_}")  # [0, 1, 2] (all have missing)

# Transform
X_missing_train = indicator.transform(X_train)
print(X_missing_train)
# [[False  True False]
#  [False False  True]
#  [ True False False]]
```

## SimpleImputer with add_indicator

### Integrated Approach

**Best practice**: Use SimpleImputer's built-in indicator

```python
from sklearn.impute import SimpleImputer

# Impute AND create indicators
imputer = SimpleImputer(strategy='median', add_indicator=True)
X_imputed_with_indicators = imputer.fit_transform(X)
```

**Result**: Concatenated array `[imputed_features, missing_indicators]`

### Example

```python
import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [np.nan, 8, 9]
])

imputer = SimpleImputer(strategy='mean', add_indicator=True)
X_transformed = imputer.fit_transform(X)

print("Shape:", X_transformed.shape)  # (3, 5) = 3 original + 2 indicators
print(X_transformed)
# [[1.  2.  3.  0. 0.]  ← Original values, no missing
#  [4.  5.  6.  0. 1.]  ← Feature 1 imputed (mean=5), indicator for feature 1
#  [2.5 8.  9.  1. 0.]]  ← Feature 0 imputed (mean=2.5), indicator for feature 0
#   ^^^^^^^^^^^  ^^^^^
#   Imputed       Indicators (col 0 missing?, col 1 missing?)
```

**Columns**:
- 0-2: Original features (imputed)
- 3: Indicator for feature 0
- 4: Indicator for feature 1

**Note**: Only features with missing get indicators (feature 2 has no indicator)

### Accessing Indicator Object

```python
imputer = SimpleImputer(add_indicator=True)
imputer.fit(X)

# Access the indicator object
missing_indicator = imputer.indicator_

# See which features have indicators
print(missing_indicator.features_)  # [0, 1] (features 0 and 1 had missing)
```

## Using in Pipelines

### Basic Pipeline with Indicators

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Flow**:
1. Impute + create indicators
2. Scale all features (imputed + indicators)
3. Train classifier on augmented feature set

### Selective Indicators with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

numeric_features = ['Age', 'Income', 'Credit_Score']
categorical_features = ['City', 'Education']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', add_indicator=True), numeric_features),
    ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

**Result**: 
- Numeric: Imputed + missing indicators
- Categorical: Imputed, no indicators

### Manual Indicator + Imputation

**More control**:
```python
from sklearn.preprocessing import FunctionTransformer

def add_missing_indicators(X):
    X_df = pd.DataFrame(X)
    indicators = X_df.isnull().astype(int)
    return pd.concat([X_df, indicators], axis=1)

pipeline = Pipeline([
    ('indicators', FunctionTransformer(add_missing_indicators)),
    ('imputer', SimpleImputer()),  # Imputes only original features
    ('model', GradientBoostingClassifier())
])
```

## Impact on Model Performance

### Performance Gain Examples

**Scenario**: Predict loan default

**Feature**: Income (some missing)

**Without indicator**:
```python
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
model.fit(X_imputed, y)
accuracy_without = 0.82
```

**With indicator**:
```python
imputer = SimpleImputer(strategy='median', add_indicator=True)
X_imputed_ind = imputer.fit_transform(X)
model.fit(X_imputed_ind, y)
accuracy_with = 0.85  # +3% improvement!
```

**Why improvement?**: 
- Missing income correlated with financial instability
- Indicator captures this signal

### When Indicators Help Most

**Large improvement** when:
- MNAR data (missingness depends on value itself)
- Missingness has strong relationship with target
- Moderate missingness (10-40%)
- Tree-based models (capture interactions easily)

**Small/no improvement** when:
- MCAR data (truly random missingness)
- Missingness unrelated to target
- Very low missingness (<5%)
- Very high missingness (>60%, imputation unreliable)

### Feature Importance

**Check if indicators are important**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_with_indicators, y)

# Get feature names
original_features = ['Age', 'Income', 'Score']
indicator_features = ['Age_missing', 'Income_missing', 'Score_missing']
all_features = original_features + indicator_features

# Check importance
importances = model.feature_importances_
for feat, imp in zip(all_features, importances):
    print(f"{feat}: {imp:.4f}")

# Age: 0.2500
# Income: 0.3000
# Score: 0.1500
# Age_missing: 0.0200  ← Low importance (not predictive)
# Income_missing: 0.2500  ← High! Missingness is informative
# Score_missing: 0.0300
```

**Interpretation**: Income_missing has 25% importance → pattern of missing income strongly predicts target

## Mathematical Properties

### Dimension Increase

**Original**: $p$ features

**With indicators**: $p + k$ features

Where $k$ = number of features with missing values

**Example**:
- 10 features, 4 have missing
- With indicators: 10 + 4 = 14 features

**Concern**: Dimension increase

**Usually acceptable**: Small increase, high information gain

### Correlation Structure

**Indicator correlations**:

$$\text{Corr}(M_i, M_j) = \begin{cases}
> 0 & \text{if missingness co-occurs} \\
\approx 0 & \text{if independent} \\
< 0 & \text{if anti-correlated (rare)}
\end{cases}$$

**Example**: Medical data
- `Blood_Pressure_missing` and `Cholesterol_missing` correlated
- Patients who skip one test often skip both
- Indicator correlation captures this pattern

### Interaction Effects

**Model can learn**:
$$f(X, M) = \beta_0 + \beta_1 X + \beta_2 M + \beta_3 (X \times M)$$

**Interpretation**:
- $\beta_1$: Effect of value (for observed)
- $\beta_2$: Effect of missingness
- $\beta_3$: Interaction (missingness modifies value effect)

**Tree models**: Automatically capture these interactions via splits

## Best Practices

### 1. Use When Missingness Might Be Informative

**Consider**:
- Domain knowledge: Why is data missing?
- Exploratory analysis: Is missingness correlated with target?

```python
# Check if missingness predicts target
for col in df.columns:
    if df[col].isnull().any():
        missing_mask = df[col].isnull()
        target_mean_missing = df[missing_mask]['target'].mean()
        target_mean_observed = df[~missing_mask]['target'].mean()
        
        print(f"{col}:")
        print(f"  Target mean (missing): {target_mean_missing:.3f}")
        print(f"  Target mean (observed): {target_mean_observed:.3f}")
        print(f"  Difference: {abs(target_mean_missing - target_mean_observed):.3f}")
```

**If large difference** → Use indicator!

### 2. Don't Overdo It

**Avoid**: Indicators for features with <1% missing

**Reason**: Adds noise, minimal information

**Threshold**: Only create indicators if missing > 5%

```python
missing_threshold = 0.05

for col in df.columns:
    missing_pct = df[col].isnull().mean()
    if missing_pct > missing_threshold:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### 3. Use add_indicator in SimpleImputer

**Prefer**:
```python
SimpleImputer(add_indicator=True)
```

**Over**: Manual creation + separate imputation

**Reason**: 
- Cleaner
- Automatic
- Pipeline-friendly
- Consistent train/test

### 4. Check Indicator Importance

**After training**:
```python
# For tree models
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    # Identify which indicators are important
    
# For linear models
if hasattr(model, 'coef_'):
    coefs = model.coef_
    # Check coefficients of indicator features
```

**If indicators not important**: Consider removing (simplify model)

### 5. Combine with Appropriate Imputation

**Strategy**:
1. Choose imputation that preserves signal (e.g., median for skewed)
2. Add indicator to capture additional missingness pattern

**Example**:
```python
# Income is skewed, some high earners missing (MNAR)
imputer = SimpleImputer(
    strategy='median',  # Robust to skew
    add_indicator=True   # Capture high earner pattern
)
```

### 6. Handle Test Set Consistency

**Ensure**: Test set indicators match train

```python
# Fit on train
imputer = SimpleImputer(add_indicator=True)
imputer.fit(X_train)

# Transform both (uses same features for indicators)
X_train_t = imputer.transform(X_train)
X_test_t = imputer.transform(X_test)

# Same features get indicators
assert X_train_t.shape[1] == X_test_t.shape[1]
```

### 7. Interpret Indicator Coefficients

**Linear model example**:
```python
model = LogisticRegression()
model.fit(X_with_indicators, y)

for i, coef in enumerate(model.coef_[0]):
    feature_name = feature_names[i]
    if '_missing' in feature_name:
        print(f"{feature_name}: {coef:.3f}")
        if coef > 0:
            print("  → Missingness increases probability of positive class")
        else:
            print("  → Missingness decreases probability of positive class")
```

## Common Patterns

### Pattern 1: Selective Indicators for High-Missing Features

```python
# Only add indicators for features with >10% missing
high_missing_features = [col for col in df.columns if df[col].isnull().mean() > 0.1]

for col in high_missing_features:
    df[f'{col}_was_missing'] = df[col].isnull().astype(int)

imputer = SimpleImputer(strategy='median')
df[df.columns] = imputer.fit_transform(df)
```

### Pattern 2: Indicators + Multiple Imputation Strategies

```python
from sklearn.compose import ColumnTransformer

skewed_features = ['Income', 'Price']
normal_features = ['Age', 'Score']

preprocessor = ColumnTransformer([
    ('skewed', SimpleImputer(strategy='median', add_indicator=True), skewed_features),
    ('normal', SimpleImputer(strategy='mean', add_indicator=True), normal_features)
])
```

### Pattern 3: Domain-Specific Indicator Interpretation

```python
# Medical: Missing test results → patient didn't take test
df['BloodTest_NotTaken'] = df['BloodTest'].isnull().astype(int)

# Finance: Missing income → privacy concern or unemployment
df['Income_NotReported'] = df['Income'].isnull().astype(int)

# Then impute
df['BloodTest'] = df['BloodTest'].fillna(df['BloodTest'].median())
df['Income'] = df['Income'].fillna(df['Income'].median())
```

## Summary

Missing indicators are binary features that flag which values were originally missing, preserving the informational content of missingness patterns even after imputation.

**Key Concepts**:

**Definition**:
$$M_i = \begin{cases}
1 & \text{if } X_i \text{ was missing} \\
0 & \text{if } X_i \text{ was observed}
\end{cases}$$

**Purpose**: Preserve signal in missingness pattern

**When Useful**:
- MNAR data (missingness depends on value)
- Missingness correlated with target
- Moderate missingness (5-40%)
- Want to preserve all information

**Implementation**:

**Manual**:
```python
df['Feature_missing'] = df['Feature'].isnull().astype(int)
df['Feature'].fillna(median, inplace=True)
```

**sklearn**:
```python
SimpleImputer(add_indicator=True)
```

**Result**: Augmented feature set `[imputed_features, indicators]`

**Advantages**:
- Preserves missingness information
- Often improves model performance
- Simple to implement
- Works with any imputation strategy
- No assumptions about mechanism

**Considerations**:
- Increases dimensionality ($p \to p + k$)
- May add noise if missingness random
- Check indicator importance
- Only useful if missingness predictive

**Best Practices**:
- Use when missingness might be informative
- Set threshold (e.g., >5% missing)
- Combine with appropriate imputation
- Check indicator feature importance
- Interpret indicator coefficients
- Ensure train/test consistency

**Common Use Cases**:
- Medical: Missing test indicates patient condition
- Finance: Missing income indicates privacy/unemployment
- Surveys: Missing answers indicate sensitivity
- E-commerce: Missing fields indicate user behavior

**Integration**:
- Works in pipelines
- Compatible with all ML algorithms
- Especially powerful with tree models (capture interactions)

Missing indicators are a simple yet powerful technique to extract additional value from incomplete data by treating the pattern of missingness as a feature in its own right.
