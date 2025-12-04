# Handling Mixed Variables - Processing Heterogeneous Data

## General Idea

Mixed variables (also called heterogeneous features) refer to datasets containing different types of variables: numerical (continuous or discrete), categorical (nominal or ordinal), binary, text, dates, and other data types. Handling mixed variables involves applying appropriate preprocessing techniques to each variable type while maintaining data integrity and preventing information leakage. This is one of the most common challenges in real-world machine learning, as clean, homogeneous datasets are rare.

## Why Handle Mixed Variables?

1. **Real-World Reality**: Most datasets have mixed types (age + category + text)
2. **Algorithm Requirements**: ML models need uniform numeric input
3. **Optimal Preprocessing**: Each type needs different treatment
4. **Prevent Errors**: Wrong preprocessing causes failures or poor performance
5. **Information Preservation**: Different encodings preserve different information
6. **Scale Appropriateness**: Categorical doesn't need scaling, numerical does
7. **Leakage Prevention**: Must handle train/test separately per type
8. **Production Pipelines**: Automated processing of diverse inputs

## Role in Machine Learning

### The Challenge

**Typical Real Dataset**:
```
Age (numeric)           : 25, 34, 45
Income (numeric)        : 50000, 75000, 60000
Education (ordinal)     : HS, Bachelor, Master
City (categorical)      : NYC, LA, Chicago
Has_Loan (binary)       : Yes, No, Yes
Signup_Date (datetime)  : 2020-01-15, 2021-03-22, ...
Review_Text (text)      : "Great service!", ...
```

**ML Model Requirement**: All features as numbers

**Solution**: Type-specific preprocessing pipeline

### Consequences of Improper Handling

**1. Treating Categorical as Numeric**:
```
City encoding: NYC=1, LA=2, Chicago=3
Implies: LA is "twice" NYC, Chicago > LA
Wrong: Cities have no inherent order
```

**2. Scaling Categorical**:
```
One-hot encoded: [1, 0, 0]
Scaled: [0.577, 0, 0]  # Meaningless!
```

**3. Encoding Numeric as Categorical**:
```
Age: 25, 26, 27, 28, ...
One-hot: 100+ features (one per age)
Result: Curse of dimensionality, lost ordinality
```

**4. Ignoring Ordinal Nature**:
```
Education: HS, Bachelor, Master, PhD
One-hot: Loses inherent ordering
Better: Ordinal encoding (HS=0, BS=1, MS=2, PhD=3)
```

## Types of Variables

### 1. Numerical Variables

**Continuous**: Real values, infinite possibilities
- Examples: height, weight, temperature, price
- Range: $x \in \mathbb{R}$ or bounded $x \in [a, b]$

**Discrete**: Integer values, countable
- Examples: number of children, page views, items purchased
- Range: $x \in \mathbb{Z}^+$ or $\{0, 1, 2, ...\}$

**Preprocessing**:
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Transformation**: PowerTransformer, QuantileTransformer, log
- **Imputation**: Mean, median, KNN imputer
- **Binning** (optional): KBinsDiscretizer

**No encoding needed**: Already numeric

### 2. Categorical Variables - Nominal

**Definition**: Unordered categories
- Examples: color, country, product type, city
- No inherent order: "Red" is not > "Blue"

**Cardinality**:
- **Low**: $< 10$ unique values (color: Red, Blue, Green)
- **Medium**: $10-50$ values (US states: 50)
- **High**: $> 50$ values (zip codes: 40,000+)

**Preprocessing**:
- **Low cardinality**: OneHotEncoder
- **High cardinality**: Target encoding, frequency encoding, hashing
- **Imputation**: Most frequent, constant value ('Missing')
- **No scaling**: Meaningless for categories

### 3. Categorical Variables - Ordinal

**Definition**: Ordered categories
- Examples: education level, satisfaction rating, size (S/M/L)
- Clear order: PhD > Master > Bachelor > HS

**Preprocessing**:
- **Encoding**: OrdinalEncoder (preserves order)
- **Alternative**: Custom mapping with domain knowledge
- **Scaling** (sometimes): After encoding, if needed
- **Imputation**: Most frequent, or forward/backward fill

**Key**: Preserve ordering in encoding

### 4. Binary Variables

**Definition**: Two possible values
- Examples: gender (M/F), has_loan (Yes/No), is_premium (True/False)
- Special case of categorical with 2 classes

**Preprocessing**:
- **Encoding**: LabelEncoder or manual {0, 1} mapping
- **No one-hot needed**: Already binary (though one-hot also works, creates 2 features)
- **Imputation**: Most frequent or domain logic

**Mathematical representation**: $x \in \{0, 1\}$

### 5. Datetime Variables

**Definition**: Date and/or time information
- Examples: signup_date, transaction_timestamp, birth_date

**Preprocessing**:
- **Feature extraction**: year, month, day, day_of_week, hour, minute
- **Derived features**: age (from birth_date), days_since (from signup_date)
- **Cyclical encoding**: hour, month (sin/cos transform)
- **Binning**: time_of_day (morning/afternoon/evening)

**Not directly usable**: Must convert to numeric features

### 6. Text Variables

**Definition**: Free-form text
- Examples: product reviews, customer comments, descriptions

**Preprocessing**:
- **Vectorization**: TfidfVectorizer, CountVectorizer
- **Embeddings**: Word2Vec, BERT, sentence transformers
- **Feature extraction**: length, word count, sentiment score
- **NLP pipelines**: Tokenization, stemming, lemmatization

**High dimensionality**: Often creates many features

## Preprocessing Strategies by Type

### Strategy 1: Separate Pipelines per Type

**Concept**: Different preprocessing for each variable type

**Implementation**: ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Define column types
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'occupation', 'product_type']
ordinal_features = ['education', 'satisfaction']

# Separate pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
    ('ord', ordinal_pipeline, ordinal_features)
])
```

**Flow**:
```
Numeric cols    → Impute → Scale           →\
Categorical     → Impute → OneHot         → Concatenate → Model
Ordinal         → Impute → OrdinalEncode →/
```

### Strategy 2: Automatic Type Detection

**Concept**: Automatically identify column types

**Implementation**: make_column_selector

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include='number')),
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object'))
])
```

**Benefit**: Adapts to different datasets automatically

**Limitation**: Can't distinguish ordinal from nominal

### Strategy 3: Manual Type Specification

**Concept**: Explicitly list columns by type

**Best for**: Complex datasets where automatic detection fails

**Example**:
```python
# Manually curated lists
scale_features = ['age', 'income']  # Need scaling
transform_features = ['price', 'loan_amount']  # Need log transform
onehot_features = ['city', 'color']  # Nominal categorical
ordinal_features = ['education', 'rating']  # Ordinal categorical
passthrough_features = ['id', 'binary_flag']  # Use as-is

preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), scale_features),
    ('transform', PowerTransformer(), transform_features),
    ('onehot', OneHotEncoder(), onehot_features),
    ('ordinal', OrdinalEncoder(), ordinal_features),
    ('pass', 'passthrough', passthrough_features)
])
```

## Common Patterns

### Pattern 1: Numeric + Categorical (Most Common)

```python
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

### Pattern 2: High Cardinality Categorical Handling

**Problem**: Too many unique categories (e.g., 1000+ zip codes)

**Solution**: Alternative encodings

```python
from category_encoders import TargetEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('low_card_cat', OneHotEncoder(), low_cardinality_cols),
    ('high_card_cat', TargetEncoder(), high_cardinality_cols)
])
```

**Alternatives**:
- **Frequency Encoding**: Encode by category frequency
- **Mean Target Encoding**: Encode by mean target value
- **Hashing**: Hash categories to fixed number of bins

### Pattern 3: Datetime Feature Engineering

```python
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def extract_datetime_features(X):
    X = pd.DataFrame(X, columns=['signup_date'])
    X['signup_date'] = pd.to_datetime(X['signup_date'])
    
    features = pd.DataFrame({
        'year': X['signup_date'].dt.year,
        'month': X['signup_date'].dt.month,
        'day': X['signup_date'].dt.day,
        'dayofweek': X['signup_date'].dt.dayofweek,
        'quarter': X['signup_date'].dt.quarter,
        'is_weekend': X['signup_date'].dt.dayofweek.isin([5, 6]).astype(int),
        'days_since': (pd.Timestamp.now() - X['signup_date']).dt.days
    })
    return features.values

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols),
    ('datetime', FunctionTransformer(extract_datetime_features), ['signup_date'])
])
```

### Pattern 4: Text + Structured Data

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Text pipeline
text_pipeline = Pipeline([
    ('selector', FunctionTransformer(lambda X: X['review_text'])),
    ('vectorizer', TfidfVectorizer(max_features=100))
])

# Structured data pipeline
structured_pipeline = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

# Combine
feature_union = FeatureUnion([
    ('text', text_pipeline),
    ('structured', structured_pipeline)
])

full_pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', LogisticRegression())
])
```

## Mathematical Considerations

### Dimensionality After Encoding

**Original**: $p$ features (mixed types)

**After preprocessing**:

**Numeric** ($n_{num}$ features): $\rightarrow n_{num}$ (unchanged)

**Categorical** ($n_{cat}$ features with $k_i$ categories each):
- One-hot: $\rightarrow \sum_{i=1}^{n_{cat}} k_i$ features
- Ordinal: $\rightarrow n_{cat}$ features (unchanged)

**Total dimensions**: $p' = n_{num} + \sum_{i=1}^{n_{cat}} k_i$ (for one-hot)

**Example**:
- 3 numeric: age, income, score
- 2 categorical: city (5 values), color (3 values)

**After one-hot**: $3 + 5 + 3 = 11$ features

**Issue**: High cardinality explodes dimensions
- Zip code (40,000 values) → 40,000 features!

### Scale Differences

**Before preprocessing**:
```
Age:    [18, 65]          (range: 47)
Income: [20000, 200000]   (range: 180000)
City:   ['NYC', 'LA', ...] (categorical)
```

**Problem**: Vastly different scales affect distance-based algorithms

**After preprocessing**:
```
Age_scaled:     [-1.5, 1.5]  (mean=0, std=1)
Income_scaled:  [-1.5, 1.5]  (mean=0, std=1)
City_NYC:       [0, 1]       (binary)
City_LA:        [0, 1]       (binary)
```

**Result**: Comparable scales, algorithm performs better

### Information Preservation

**Encoding Trade-offs**:

**Ordinal Encoding**: 
- Preserves: Order relationship
- Assumes: Equal spacing between levels
- Information: $\log_2(k)$ bits (where $k$ = categories)

**One-Hot Encoding**:
- Preserves: Category identity
- Assumes: No relationship between categories
- Information: $k$ bits (one per category)
- Space: $k$ features

**Target Encoding**:
- Preserves: Relationship with target
- Assumes: Categories differ by target mean
- Information: Continuous value per category
- Space: 1 feature
- Risk: Leakage if not done correctly

## Handling Edge Cases

### Mixed Numeric-Categorical Column

**Example**: Column contains "<5", "10-20", "30+", "unknown"

**Problem**: Neither purely numeric nor categorical

**Solutions**:

**1. Extract Numeric + Indicator**:
```python
def parse_mixed(X):
    numeric = X.str.extract(r'(\d+)').astype(float)  # Extract number
    is_range = X.str.contains('-').astype(int)      # Indicator for range
    is_unknown = (X == 'unknown').astype(int)       # Unknown flag
    return np.column_stack([numeric, is_range, is_unknown])
```

**2. Treat as Ordinal**:
```python
mapping = {'<5': 0, '10-20': 1, '30+': 2, 'unknown': -1}
X_encoded = X.map(mapping)
```

### Missing Values in Mixed Data

**Challenge**: Different strategies per type

**Numeric**: Impute with mean/median

```python
SimpleImputer(strategy='median')
```

**Categorical**: Impute with mode or 'missing' category

```python
SimpleImputer(strategy='most_frequent')
# or
SimpleImputer(strategy='constant', fill_value='Missing')
```

**Ordinal**: Forward/backward fill (if time-ordered) or mode

```python
SimpleImputer(strategy='most_frequent')
```

**Critical**: Must handle separately per type in ColumnTransformer

### Unknown Categories at Test Time

**Problem**: Test data has category not seen in training

**Example**:
- Train: City ∈ {NYC, LA, Chicago}
- Test: City = Houston (new!)

**Solution**: `handle_unknown='ignore'` in OneHotEncoder

```python
OneHotEncoder(handle_unknown='ignore')
```

**Behavior**: Unknown category → all zeros [0, 0, 0]

**Alternative**: `handle_unknown='infrequent_if_exist'` (groups rare categories)

## Best Practices

### 1. Understand Your Data First

**Before preprocessing**:
```python
# Check types
print(df.dtypes)

# Check cardinality
for col in df.select_dtypes(include='object'):
    print(f"{col}: {df[col].nunique()} unique values")

# Check missing
print(df.isnull().sum())

# Check distributions
df.describe()
```

### 2. Separate Pipelines for Each Type

**Always use ColumnTransformer**:
- Cleaner code
- Prevents errors
- Easier maintenance

### 3. Handle Missing Values Before Encoding

**Order**: Impute → Encode → Scale

**Reason**: Can't encode NaN values

```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler())
])
```

### 4. Don't Scale Categorical Variables

**Wrong**:
```python
StandardScaler().fit_transform(one_hot_encoded)
```

**Right**: Only scale numeric features

```python
ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)  # No scaling
])
```

### 5. Use Ordinal Encoding for Ordinal Variables

**If**: education = ['HS', 'Bachelor', 'Master', 'PhD']

**Do**: 
```python
OrdinalEncoder(categories=[['HS', 'Bachelor', 'Master', 'PhD']])
```

**Don't**: OneHotEncoder (loses ordering)

### 6. Handle High Cardinality Carefully

**If**: $k > 50$ unique categories

**Consider**:
- Target encoding
- Frequency encoding
- Grouping rare categories
- Hashing trick
- Embeddings (neural networks)

**Avoid**: One-hot (dimension explosion)

### 7. Extract Features from Datetime

**Don't**: Use raw datetime (not numeric)

**Do**: Extract meaningful features
```python
['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend', 'quarter']
```

### 8. Keep Track of Feature Names

```python
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()
print(feature_names)
# ['num__age', 'num__income', 'cat__city_NYC', 'cat__city_LA', ...]
```

**Benefit**: Interpretability, debugging

### 9. Validate Pipeline Output

```python
X_transformed = preprocessor.fit_transform(X_train)

# Check shape
print(f"Original: {X_train.shape}, Transformed: {X_transformed.shape}")

# Check for NaN/inf
assert not np.isnan(X_transformed).any()
assert not np.isinf(X_transformed).any()

# Check types
print(type(X_transformed))  # numpy array or sparse matrix?
```

### 10. Save Preprocessing Pipeline

```python
from joblib import dump

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', classifier)
])

full_pipeline.fit(X_train, y_train)
dump(full_pipeline, 'model_pipeline.joblib')
```

**Production**: Load and use same pipeline

## Common Mistakes

### Mistake 1: Preprocessing Before Train/Test Split

**Wrong**:
```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

**Problem**: Test statistics leaked into scaling

**Right**:
```python
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Mistake 2: Scaling One-Hot Encoded Features

**Wrong**: Applying StandardScaler after OneHotEncoder

**Why wrong**: Binary values [0, 1] become meaningless floats

**Right**: Only scale numeric features

### Mistake 3: Label Encoding Nominal Categorical

**Wrong**:
```python
LabelEncoder().fit_transform(['Red', 'Blue', 'Green'])
# Result: [2, 0, 1]  # Implies Blue < Green < Red
```

**Right**: Use OneHotEncoder for nominal categories

### Mistake 4: Ignoring handle_unknown Parameter

**Wrong**: Default `handle_unknown='error'`

**Problem**: Pipeline breaks on new categories in production

**Right**:
```python
OneHotEncoder(handle_unknown='ignore')
```

### Mistake 5: Not Specifying Ordinal Categories Order

**Wrong** (random order):
```python
OrdinalEncoder().fit(['Low', 'High', 'Medium'])
# May encode as: Low=0, High=1, Medium=2 (wrong order!)
```

**Right** (explicit order):
```python
OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
```

## Summary

Handling mixed variables is a critical preprocessing step that requires applying appropriate transformations to each variable type while maintaining data integrity and preventing leakage.

**Key Concepts**:

**Variable Types**:
- **Numeric**: Scale, transform, impute with mean/median
- **Categorical (Nominal)**: One-hot encode, impute with mode
- **Categorical (Ordinal)**: Ordinal encode, preserve order
- **Binary**: Simple {0,1} encoding
- **Datetime**: Extract features (year, month, day, etc.)
- **Text**: Vectorize (TF-IDF, embeddings)

**Preprocessing Strategy**:
```python
ColumnTransformer([
    ('num_pipeline', numeric_transformers, numeric_cols),
    ('cat_pipeline', categorical_transformers, categorical_cols),
    ('ord_pipeline', ordinal_transformers, ordinal_cols),
    ...
])
```

**Common Patterns**:
1. Separate pipelines per type
2. Impute → Encode → Scale (order matters)
3. High cardinality → alternative encodings
4. Datetime → feature extraction
5. Text + structured → FeatureUnion

**Best Practices**:
- Understand data types first
- Use ColumnTransformer
- Handle missing values before encoding
- Don't scale categorical
- Use ordinal encoding for ordered categories
- Handle high cardinality carefully
- Extract datetime features
- Validate pipeline output

**Common Mistakes to Avoid**:
- Preprocessing before split (leakage)
- Scaling one-hot encoded features
- Label encoding nominal categories
- Ignoring unknown categories
- Not specifying ordinal order

**Tools**:
- `ColumnTransformer`: Apply different preprocessing per column type
- `Pipeline`: Chain preprocessing and modeling
- `make_column_selector`: Automatic type detection
- `OneHotEncoder`: Nominal categorical
- `OrdinalEncoder`: Ordinal categorical
- `StandardScaler`: Numeric scaling
- `SimpleImputer`: Missing value handling

**Result**: Clean, properly preprocessed data ready for machine learning models, with type-appropriate transformations applied consistently across train and test sets.

Handling mixed variables effectively is essential for building robust, production-ready machine learning systems that can process the heterogeneous data found in real-world applications.

---

**Video Link**: https://youtu.be/9xiX-I5_LQY
