# Handling Missing Categorical Data - Imputation Strategies

## General Idea

Missing categorical data requires different imputation strategies than numerical data because categories don't have mathematical properties like mean or median. Common approaches include replacing missing values with the most frequent category (mode), a constant placeholder value, treating missingness as its own category, or using predictive models. The choice depends on the nature of the categorical variable, missingness mechanism, and domain context.

## Why Handle Missing Categorical Data Differently?

1. **No Numerical Meaning**: Can't compute mean/median for categories
2. **Distinct Strategies Needed**: Mode, constant, or 'missing' category
3. **Cardinality Matters**: Low vs high cardinality affects approach
4. **Ordinal vs Nominal**: Different treatment based on type
5. **Missingness May Be Informative**: "Unknown" can be meaningful category
6. **Encoding Compatibility**: Must work with one-hot/ordinal encoding
7. **Preserve Information**: Keep signal in missing pattern
8. **Production Requirements**: Need consistent categorical levels

## Role in Machine Learning

### Categorical vs Numerical Missingness

**Numerical**:
```
Age: [25, NaN, 35, 40] → Impute with mean (33.3) or median (35)
```

**Categorical**:
```
City: ['NYC', NaN, 'LA', 'Chicago'] → Can't compute mean!
```

**Options for categorical**:
1. Most frequent: 'NYC' (if it appeared most)
2. Constant: 'Unknown'
3. Missing category: Treat NaN as 'Missing'

### Impact on Encoding

**After imputation**, categorical data must be encoded:

**One-hot encoding**:
```
City (after impute): ['NYC', 'Unknown', 'LA', 'NYC']
One-hot: 
  City_NYC     City_LA     City_Unknown
     1            0            0
     0            0            1
     0            1            0
     1            0            0
```

**Missing as separate category** creates additional encoded feature

**Ordinal encoding**:
```
Education: ['HS', 'Missing', 'Bachelor', 'HS']
Ordinal: [0, 3, 1, 0]  # 'Missing' gets numeric code
```

## Types of Categorical Variables

### Nominal (Unordered)

**Definition**: No inherent order

**Examples**: color, city, product type, country

**Imputation strategies**:
- Mode (most frequent)
- Constant ('Unknown', 'Other', 'Missing')
- Predictive (based on other features)

**Not applicable**: Mean, median (no order)

### Ordinal (Ordered)

**Definition**: Clear ordering

**Examples**: education level, satisfaction rating, size (S/M/L)

**Imputation strategies**:
- Mode
- Constant (often 'Unknown' or separate level)
- Median category (middle value)
- Forward/backward fill (time series)

**Consideration**: Preserve ordering in imputation

### Binary (Special Case)

**Definition**: Two categories

**Examples**: Gender (M/F), Has_Loan (Yes/No)

**Imputation strategies**:
- Mode (majority class)
- Random (sample from observed proportion)
- Constant (0 or 1, 'Unknown')

## Imputation Strategies

### 1. Most Frequent (Mode) Imputation

**Method**: Replace missing with most common category

**Formula**:
$$x_{\text{imputed}} = \text{argmax}_c \text{ count}(c)$$

Where $c$ ranges over observed categories

**Example**:
```
Original: ['Red', 'Blue', NaN, 'Red', 'Green', NaN, 'Red']
Mode: 'Red' (appears 3 times, most frequent)
Imputed: ['Red', 'Blue', 'Red', 'Red', 'Green', 'Red', 'Red']
```

**Pros**:
- Simple, intuitive
- Preserves mode
- Works for any categorical type

**Cons**:
- Overrepresents majority class
- Ignores relationships with other features
- Poor for balanced categories (no clear mode)
- Reduces diversity

**When to use**:
- Clear dominant category
- Low missingness
- Quick baseline
- Nominal categories

### 2. Constant Imputation ('Unknown', 'Missing')

**Method**: Replace with fixed placeholder

**Common values**:
- 'Unknown'
- 'Missing'
- 'Other'
- 'N/A'
- 'Not_Provided'

**Example**:
```
Original: ['NYC', 'LA', NaN, 'Chicago', NaN]
Constant: 'Unknown'
Imputed: ['NYC', 'LA', 'Unknown', 'Chicago', 'Unknown']
```

**Pros**:
- Preserves information about missingness
- Doesn't distort category frequencies
- Model can learn if missingness predictive
- Transparent (clear which were imputed)

**Cons**:
- Increases cardinality (+1 category)
- May create rare category
- Requires handling in encoding

**When to use**:
- Missingness might be informative
- Don't want to bias toward majority
- Interpretability important
- **Default recommendation** for many cases

### 3. Missing as Separate Category

**Method**: Treat NaN as distinct category

**Implementation**:
```python
df['Category'].fillna('Missing', inplace=True)
# Or keep NaN and encode as separate level
```

**Differs from constant**: Explicitly labels as "missing" vs generic "unknown"

**After one-hot encoding**:
```
Original categories: ['A', 'B', 'C']
With missing: ['A', 'B', 'C', 'Missing']
One-hot: 4 features (A, B, C, Missing)
```

**Model interpretation**: Coefficient for 'Missing' shows impact of missingness

**When to use**:
- Missingness is informative (MNAR)
- Want explicit missing indicator
- Using tree-based models (can split on missing)

### 4. Random Sampling

**Method**: Sample from observed category distribution

**Formula**:
$$P(x_{\text{imputed}} = c) = \frac{\text{count}(c)}{\text{total observed}}$$

**Example**:
```
Original: ['A', 'A', 'B', NaN, NaN]
Observed distribution: A=67%, B=33%
Random sample: NaN → 'A' (67% chance) or 'B' (33% chance)
Possible result: ['A', 'A', 'B', 'A', 'B']
```

**Pros**:
- Preserves category distribution
- Maintains variability
- No artificial overrepresentation

**Cons**:
- Non-deterministic (different each run)
- Ignores feature relationships
- Adds randomness/noise

**When to use**:
- Want to preserve distribution
- Multiple imputation framework
- Balanced categories

### 5. Forward/Backward Fill (Time Series)

**Method**: Use previous/next value in sequence

**Forward fill** (last observation carried forward):
```
Time: 1    2    3    4    5
Val:  'A'  'B'  NaN  NaN  'C'
FFill: 'A'  'B'  'B'  'B'  'C'
```

**Backward fill**:
```
BFill: 'A'  'B'  'C'  'C'  'C'
```

**When to use**:
- Time-ordered data
- Status/state variables (likely unchanged)
- Sequential measurements

**Assumption**: Value remains constant between observations

### 6. Predictive Imputation

**Method**: Predict missing category from other features

**Approach**:
1. Treat category as target
2. Use rows with observed category to train classifier
3. Predict missing category from other features

**Example**:
```python
# Predict missing 'Education' from Age, Income
known = df[df['Education'].notnull()]
unknown = df[df['Education'].isnull()]

X_train = known[['Age', 'Income']]
y_train = known['Education']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

X_pred = unknown[['Age', 'Income']]
y_pred = clf.predict(X_pred)

df.loc[df['Education'].isnull(), 'Education'] = y_pred
```

**Pros**:
- Uses feature relationships
- More sophisticated
- Can be more accurate

**Cons**:
- Complex
- Computationally expensive
- Risk of overfitting
- Requires complete other features

**When to use**:
- Strong relationships exist
- High missingness
- Critical feature

## Scikit-Learn SimpleImputer for Categorical

### Syntax

```python
from sklearn.impute import SimpleImputer

# Most frequent
imputer = SimpleImputer(strategy='most_frequent')

# Constant
imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

X_imputed = imputer.fit_transform(X)
```

### Parameters for Categorical

**strategy='most_frequent'**:
- Replaces with mode
- Works for any data type

**strategy='constant'**:
- Replaces with fill_value
- Specify: `fill_value='Unknown'`, `fill_value='Missing'`, etc.

**Important**: SimpleImputer works with string categories, but:
- Input can be DataFrame or array
- Output is numpy array (loses category dtype)
- May need to convert back

### Example

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Data
df = pd.DataFrame({
    'City': ['NYC', 'LA', None, 'Chicago', None, 'NYC'],
    'Education': ['HS', None, 'Bachelor', 'Master', 'HS', None]
})

# Most frequent imputation
imputer_mode = SimpleImputer(strategy='most_frequent')
X_mode = imputer_mode.fit_transform(df)

print("Mode imputation:")
print(pd.DataFrame(X_mode, columns=df.columns))

# Constant imputation
imputer_const = SimpleImputer(strategy='constant', fill_value='Unknown')
X_const = imputer_const.fit_transform(df)

print("\nConstant imputation:")
print(pd.DataFrame(X_const, columns=df.columns))
```

## Pandas-Specific Approaches

### fillna() Method

**Most frequent**:
```python
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])
```

**Constant**:
```python
df['Category'] = df['Category'].fillna('Unknown')
```

**Forward fill**:
```python
df['Category'] = df['Category'].fillna(method='ffill')
```

**Backward fill**:
```python
df['Category'] = df['Category'].fillna(method='bfill')
```

### Conditional Imputation

**Different values based on conditions**:
```python
# If Age < 18, impute Education as 'Student'
# Otherwise, use mode
mask = (df['Education'].isnull()) & (df['Age'] < 18)
df.loc[mask, 'Education'] = 'Student'

mode_education = df['Education'].mode()[0]
df['Education'] = df['Education'].fillna(mode_education)
```

### Group-Based Imputation

**Impute with mode per group**:
```python
# Impute City with most frequent city per State
df['City'] = df.groupby('State')['City'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
)
```

## Handling with Encoders

### OneHotEncoder - handle_unknown Parameter

**Problem**: Test set has category not in train

**Solution**:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
# Unknown categories → all zeros

# Or
encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
# Groups rare/unknown into 'infrequent' category
```

### OrdinalEncoder - handle_unknown Parameter

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# Unknown categories → -1
```

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_features),
    ('num', SimpleImputer(strategy='median'), numerical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Best Practices

### 1. Understand Category Semantics

**Question**: What does missing mean for this category?

**Examples**:
- Missing email domain: Customer didn't provide email → 'No_Email'
- Missing product category: Product miscategorized → 'Uncategorized'
- Missing country: International/Unknown → 'Unknown'

**Domain knowledge guides imputation**

### 2. Check Category Frequencies Before Imputing

```python
# Check distribution
print(df['Category'].value_counts())
print(f"\nMissing: {df['Category'].isnull().sum()} ({df['Category'].isnull().mean()*100:.1f}%)")

# Visualize
df['Category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.show()
```

**Decision**:
- Clear mode → Consider mode imputation
- Balanced distribution → Avoid mode (overrepresents), use constant
- High missingness → Consider 'Missing' as category

### 3. Preserve Imputation for Encoding

**Ensure imputed category included in encoder**:

```python
from sklearn.preprocessing import OneHotEncoder

# Impute
imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
X_imputed = imputer.fit_transform(X_train)

# Encode (Unknown will be included)
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_imputed)

# Check categories
print(encoder.categories_)
# [array(['Chicago', 'LA', 'NYC', 'Unknown'], dtype=object)]
```

### 4. Consider Missing as Informative

**Add binary indicator** even after imputation:

```python
df['City_was_missing'] = df['City'].isnull().astype(int)
df['City'] = df['City'].fillna('Unknown')
```

**Result**: Model has both imputed value AND missingness flag

### 5. Handle Ordinal Carefully

**For ordinal categories** (e.g., Low/Medium/High):

**Option 1**: Mode
```python
df['Priority'] = df['Priority'].fillna(df['Priority'].mode()[0])
```

**Option 2**: 'Unknown' level (often best)
```python
df['Priority'] = df['Priority'].fillna('Unknown')
# Then encode: Low=0, Medium=1, High=2, Unknown=3
```

**Option 3**: Median category (middle value)
```python
category_order = ['Low', 'Medium', 'High']
median_idx = len(category_order) // 2
df['Priority'] = df['Priority'].fillna(category_order[median_idx])  # 'Medium'
```

### 6. Validate Imputation

```python
# Check no NaN remain
assert df['Category'].isnull().sum() == 0, "Still have missing values!"

# Check imputed values are valid categories
valid_categories = ['A', 'B', 'C', 'Unknown']
assert df['Category'].isin(valid_categories).all(), "Invalid category introduced!"

# Check distribution change
print("Before imputation distribution:")
print(original_df['Category'].value_counts(normalize=True))
print("\nAfter imputation distribution:")
print(df['Category'].value_counts(normalize=True))
```

### 7. Document Imputation Choices

```python
imputation_log = {
    'City': {
        'strategy': 'constant',
        'fill_value': 'Unknown',
        'missing_count': df['City'].isnull().sum(),
        'missing_pct': df['City'].isnull().mean() * 100
    },
    'Education': {
        'strategy': 'most_frequent',
        'fill_value': df['Education'].mode()[0],
        'missing_count': df['Education'].isnull().sum()
    }
}
```

### 8. Test Set Consistency

**Ensure same imputation on test**:

```python
# Fit on train
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(X_train)

# Store learned mode
train_mode = imputer.statistics_[0]

# Transform both
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)  # Uses train mode

# Verify
print(f"Train mode used: {train_mode}")
```

## Common Patterns

### Pattern 1: Different Strategies per Category Type

```python
from sklearn.compose import ColumnTransformer

nominal_features = ['City', 'Color', 'Product']
ordinal_features = ['Education', 'Satisfaction']

preprocessor = ColumnTransformer([
    ('nominal_impute', SimpleImputer(strategy='constant', fill_value='Unknown'), nominal_features),
    ('ordinal_impute', SimpleImputer(strategy='most_frequent'), ordinal_features)
])
```

### Pattern 2: Impute + Encode Pipeline

```python
categorical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
```

### Pattern 3: High Cardinality Handling

**For many categories** (e.g., 100+ cities):

```python
# Group rare categories before imputing
from sklearn.preprocessing import OrdinalEncoder

# Count frequencies
freq = df['City'].value_counts()
rare_threshold = 10  # Appears < 10 times = rare

# Replace rare with 'Other'
rare_cities = freq[freq < rare_threshold].index
df['City'] = df['City'].replace(rare_cities, 'Other')

# Then impute
df['City'] = df['City'].fillna('Unknown')
```

### Pattern 4: Conditional Imputation

```python
# Domain-specific logic
def conditional_impute(row):
    if pd.isnull(row['Education']):
        if row['Age'] < 18:
            return 'High_School_or_Less'
        elif row['Age'] > 60:
            return df['Education'].mode()[0]
        else:
            return 'Unknown'
    return row['Education']

df['Education'] = df.apply(conditional_impute, axis=1)
```

## Summary

Handling missing categorical data requires strategies tailored to the non-numerical nature of categories, with common approaches being mode imputation, constant values, or treating missingness as a separate category.

**Key Concepts**:

**Imputation Strategies**:
1. **Most Frequent (Mode)**: Replace with most common category
2. **Constant**: Replace with 'Unknown', 'Missing', 'Other'
3. **Missing as Category**: Treat NaN as distinct level
4. **Random**: Sample from observed distribution
5. **Forward/Backward Fill**: For time series
6. **Predictive**: Model-based imputation

**sklearn Implementation**:
```python
# Mode
SimpleImputer(strategy='most_frequent')

# Constant
SimpleImputer(strategy='constant', fill_value='Unknown')
```

**Nominal vs Ordinal**:
- **Nominal**: Mode or constant (no order to preserve)
- **Ordinal**: Mode, constant, or median category (preserve order)

**Best Practices**:
- Understand category semantics
- Check frequencies before imputing
- Consider missingness as informative
- Use constant='Unknown' as safe default
- Handle ordinal carefully
- Ensure encoder compatibility
- Validate no NaN remain
- Document choices

**Common Patterns**:
- Impute → Encode pipeline
- Different strategies per type
- Group rare categories
- Conditional imputation

**Advantages**:
- Preserves sample size
- Simple implementation
- Flexible (multiple strategies)
- Works with encoding

**Considerations**:
- Mode overrepresents majority
- Constant increases cardinality
- May distort category distribution
- Ignores feature relationships (simple methods)

**Recommended Approach**:
1. Low missingness + clear mode → Most frequent
2. Missingness might be informative → Constant ('Unknown')
3. High cardinality → Group rare, then constant
4. Ordinal → Most frequent or 'Unknown' level
5. Time series → Forward/backward fill

**Integration with Encoding**:
- Impute first, then encode
- Use handle_unknown in encoders
- Ensure imputed categories included in training

Proper handling of missing categorical data ensures clean preprocessing that maintains data integrity while preparing categories for encoding and modeling.

---
