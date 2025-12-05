# Feature Construction and Feature Splitting

## Table of Contents
- [Introduction](#introduction)
- [What is Feature Engineering?](#what-is-feature-engineering)
- [Feature Construction](#feature-construction)
- [Feature Splitting](#feature-splitting)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Feature Engineering Matters](#why-feature-engineering-matters)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Types of Feature Construction](#types-of-feature-construction)
- [Types of Feature Splitting](#types-of-feature-splitting)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)

## Introduction

Feature engineering is often considered the most important skill in machine learning—more critical than choosing the right algorithm. It's the process of creating new features or transforming existing ones to better represent the underlying patterns in your data.

**Feature Construction** (also called **Feature Creation** or **Feature Generation**) involves creating new features from existing ones by combining, transforming, or extracting information.

**Feature Splitting** (also called **Feature Decomposition**) involves breaking down complex features into simpler, more meaningful components.

These techniques can dramatically improve model performance by:
1. Capturing non-linear relationships
2. Encoding domain knowledge
3. Reducing dimensionality
4. Making patterns more apparent to algorithms

## What is Feature Engineering?

Feature engineering is the art and science of transforming raw data into features that better represent the predictive patterns in the data to machine learning models.

### The Feature Engineering Process

```
Raw Data → Feature Engineering → Informative Features → Model Training → Better Predictions
```

### Types of Feature Engineering

1. **Feature Construction**: Creating new features
   - Combining existing features
   - Mathematical transformations
   - Domain-specific calculations

2. **Feature Splitting**: Breaking down features
   - Extracting components
   - Parsing structured data
   - Decomposing complex variables

3. **Feature Transformation**: Changing feature scale/distribution
   - Normalization
   - Standardization
   - Log transformation

4. **Feature Selection**: Choosing relevant features
   - Removing redundant features
   - Identifying important variables

## Feature Construction

Feature construction creates new variables that capture information not explicitly present in the original features.

### Methods of Feature Construction

#### 1. **Polynomial Features**

Create interaction terms and polynomial combinations:

$$f_{new} = f_1^a \times f_2^b \times ... \times f_n^c$$

For degree 2 with features $x_1, x_2$:
$$\phi(x_1, x_2) = \{1, x_1, x_2, x_1^2, x_1x_2, x_2^2\}$$

#### 2. **Arithmetic Operations**

Combine features using mathematical operations:

**Addition**: $f_{new} = f_1 + f_2$

**Subtraction**: $f_{new} = f_1 - f_2$

**Multiplication**: $f_{new} = f_1 \times f_2$

**Division**: $f_{new} = \frac{f_1}{f_2}$ (where $f_2 \neq 0$)

#### 3. **Statistical Aggregations**

Create features from grouped data:

**Mean**: $f_{new} = \frac{1}{n}\sum_{i=1}^{n} x_i$

**Standard Deviation**: $f_{new} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$

**Maximum/Minimum**: $f_{new} = \max(x_1, x_2, ..., x_n)$

**Range**: $f_{new} = \max(X) - \min(X)$

#### 4. **Domain-Specific Features**

Create features based on domain knowledge:

**Example (E-commerce)**:
- Customer Lifetime Value = Average Order Value × Purchase Frequency × Customer Lifespan
- Cart Abandonment Rate = Abandoned Carts / Total Carts

**Example (Finance)**:
- Debt-to-Income Ratio = Total Debt / Annual Income
- Return on Investment = (Gain - Cost) / Cost

## Feature Splitting

Feature splitting decomposes complex features into simpler components that may be more informative.

### Methods of Feature Splitting

#### 1. **Date/Time Decomposition**

Split datetime into components:

$$\text{DateTime} \rightarrow \{\text{Year, Month, Day, Hour, Minute, DayOfWeek, Quarter, ...}\}$$

#### 2. **Text Splitting**

Extract components from text:

**Full Name** → {First Name, Last Name}

**Email** → {Username, Domain}

**Address** → {Street, City, State, ZIP}

#### 3. **Categorical Decomposition**

Break complex categories into hierarchies:

**Product Code "ELEC-LAP-DEL-001"** →
- Category: Electronics
- Subcategory: Laptop
- Brand: Dell
- Model: 001

#### 4. **Numerical Binning**

Convert continuous variables into categorical bins:

$$f_{continuous} \rightarrow f_{categorical}$$

Example: Age → Age Group {Child, Teen, Adult, Senior}

## Mathematical Foundation

### Polynomial Feature Transformation

For a feature vector $\mathbf{x} = [x_1, x_2]$ and degree $d=2$:

$$\phi_2(\mathbf{x}) = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$$

General form for degree $d$:

$$\phi_d(\mathbf{x}) = \{x_1^{i_1} x_2^{i_2} ... x_n^{i_n} : i_1 + i_2 + ... + i_n \leq d\}$$

**Number of features generated**:

$$N_{features} = \binom{n + d}{d} = \frac{(n+d)!}{d! \cdot n!}$$

where $n$ is the number of original features and $d$ is the degree.

### Interaction Terms

For two features $x_1$ and $x_2$, the interaction term is:

$$x_{interaction} = x_1 \times x_2$$

This captures the **joint effect** of both features.

**Example in Linear Regression**:

Without interaction:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

With interaction:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \times x_2)$$

The interaction term $\beta_3 (x_1 \times x_2)$ allows the effect of $x_1$ to depend on the value of $x_2$.

### Ratio Features

Creating ratios can normalize relationships:

$$r = \frac{x_1}{x_2}$$

This is particularly useful when:
- Variables are related proportionally
- Need scale-invariant features
- Expressing relative comparisons

### Cyclical Feature Encoding

For cyclical features (hours, days, months), use sine/cosine transformation:

$$\sin\_component = \sin\left(\frac{2\pi \cdot value}{max\_value}\right)$$

$$\cos\_component = \cos\left(\frac{2\pi \cdot value}{max\_value}\right)$$

This preserves the cyclical nature (e.g., hour 23 is close to hour 0).

## Why Feature Engineering Matters

### 1. **Model Performance**

Well-engineered features can dramatically improve model accuracy:
- Better features > better algorithms
- Can turn a linear model into a powerful predictor

### 2. **Capture Non-Linearity**

Polynomial and interaction features allow linear models to capture non-linear relationships.

### 3. **Encode Domain Knowledge**

Domain-specific features incorporate expert knowledge:
- Medical diagnosis: BMI = weight / height²
- Finance: P/E ratio, debt-to-equity
- E-commerce: average order value, purchase frequency

### 4. **Reduce Dimensionality**

Good feature construction can capture complex patterns in fewer features.

### 5. **Improve Interpretability**

Meaningful features make models easier to understand and explain.

## Advantages and Disadvantages

### Advantages of Feature Construction

1. **Improved Performance**: Better features lead to better predictions
2. **Captures Complex Patterns**: Reveals hidden relationships
3. **Domain Integration**: Incorporates expert knowledge
4. **Model Simplification**: Can reduce need for complex models
5. **Handles Non-Linearity**: Allows linear models to fit non-linear data

### Disadvantages of Feature Construction

1. **Increased Dimensionality**: More features can lead to overfitting
2. **Computational Cost**: More features require more computation
3. **Manual Effort**: Requires domain expertise and creativity
4. **Risk of Leakage**: Poorly constructed features can leak target information
5. **Multicollinearity**: New features may be highly correlated

### Advantages of Feature Splitting

1. **Reveals Hidden Information**: Extracts latent components
2. **Improves Interpretability**: Simpler features are easier to understand
3. **Handles Structured Data**: Parses complex formats
4. **Temporal Patterns**: Extracts seasonality and trends
5. **Reduces Complexity**: Breaks down complex variables

### Disadvantages of Feature Splitting

1. **Information Loss**: May lose some context
2. **Increased Features**: Creates more variables
3. **Requires Preprocessing**: Needs careful parsing
4. **Domain Dependency**: Splitting logic varies by domain

## Types of Feature Construction

### 1. Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Example: x1=2, x2=3, degree=2
# Output: [1, 2, 3, 4, 6, 9]
#         [1, x1, x2, x1², x1·x2, x2²]
```

### 2. Interaction Features

```python
# Manual interaction
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Example: price × quantity = total_value
df['total_value'] = df['price'] * df['quantity']
```

### 3. Ratio Features

```python
# Ratio construction
df['ratio'] = df['numerator'] / (df['denominator'] + 1e-10)

# Example: conversion rate
df['conversion_rate'] = df['purchases'] / df['visits']
```

### 4. Aggregation Features

```python
# Group statistics
df['avg_purchase_by_user'] = df.groupby('user_id')['purchase_amount'].transform('mean')
df['total_purchases'] = df.groupby('user_id')['purchase_id'].transform('count')
```

### 5. Mathematical Transformations

```python
import numpy as np

# Log transformation
df['log_income'] = np.log1p(df['income'])

# Square root
df['sqrt_area'] = np.sqrt(df['area'])

# Exponential
df['exp_growth'] = np.exp(df['growth_rate'])
```

## Types of Feature Splitting

### 1. DateTime Splitting

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
```

### 2. Name/Text Splitting

```python
# Split full name
df[['first_name', 'last_name']] = df['full_name'].str.split(' ', n=1, expand=True)

# Split email
df['email_domain'] = df['email'].str.split('@').str[1]

# Extract from URL
df['url_domain'] = df['url'].str.extract(r'https?://([^/]+)')
```

### 3. Address Splitting

```python
# Split address
df[['street', 'city', 'state', 'zip']] = df['address'].str.split(',', expand=True)
```

### 4. Categorical Hierarchy Splitting

```python
# Product code: "ELEC-LAP-DEL-001"
df['category'] = df['product_code'].str.split('-').str[0]
df['subcategory'] = df['product_code'].str.split('-').str[1]
df['brand'] = df['product_code'].str.split('-').str[2]
```

## Mathematical Examples

### Example 1: Polynomial Features

**Original features**: $x_1 = 3, x_2 = 4$

**Degree 2 transformation**:
$$\phi_2(x_1, x_2) = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$$

**Result**:
$$\phi_2(3, 4) = [1, 3, 4, 9, 12, 16]$$

**Number of features**:
$$\binom{2 + 2}{2} = \binom{4}{2} = 6 \text{ features}$$

### Example 2: Interaction Terms

**Scenario**: Predicting house price with features:
- $x_1$: Square footage = 2000
- $x_2$: Number of bedrooms = 3

**Without interaction**:
$$\text{price} = 50000 + 100 \times 2000 + 20000 \times 3 = 310{,}000$$

**With interaction** (captures that larger bedrooms in bigger houses are more valuable):
$$\text{price} = 50000 + 100 \times 2000 + 20000 \times 3 + 10 \times (2000 \times 3)$$
$$= 50000 + 200000 + 60000 + 60000 = 370{,}000$$

### Example 3: DateTime Feature Extraction

**Original**: `2024-12-25 14:30:00`

**Extracted features**:
- Year: 2024
- Month: 12
- Day: 25
- Hour: 14
- Day of week: Wednesday (2)
- Is weekend: 0
- Quarter: 4
- Is holiday: 1
- Hour sin: $\sin(2\pi \cdot 14/24) = -0.866$
- Hour cos: $\cos(2\pi \cdot 14/24) = -0.5$

### Example 4: Domain-Specific Feature Construction

**E-commerce Customer Data**:

Original features:
- Total purchases: 10
- Total spent: $1000
- Account age: 365 days
- Last purchase: 30 days ago

Constructed features:
- Average order value: $1000 / 10 = $100
- Purchase frequency: 10 / 365 = 0.027 purchases/day
- Recency score: 30 days
- Customer lifetime value: $100 × 0.027 × 365 = $985/year
- Days since last purchase: 30
- Purchase velocity: 10 / 365 = 0.027 purchases/day

## Implementation in Python

### Polynomial Features

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Sample data
X = np.array([[2, 3],
              [4, 5],
              [6, 7]])

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(['x1', 'x2'])

print("Original shape:", X.shape)
print("Polynomial shape:", X_poly.shape)
print("\nFeature names:", feature_names)
print("\nTransformed data:")
print(pd.DataFrame(X_poly, columns=feature_names))
```

### Custom Interaction Features

```python
def create_interaction_features(df, features_list):
    """
    Create all pairwise interaction features
    """
    df_interactions = df.copy()
    
    for i in range(len(features_list)):
        for j in range(i+1, len(features_list)):
            feat1, feat2 = features_list[i], features_list[j]
            interaction_name = f"{feat1}_x_{feat2}"
            df_interactions[interaction_name] = df[feat1] * df[feat2]
            print(f"Created: {interaction_name}")
    
    return df_interactions

# Example
df = pd.DataFrame({
    'area': [1000, 1500, 2000],
    'rooms': [2, 3, 4],
    'age': [5, 10, 15]
})

df_enhanced = create_interaction_features(df, ['area', 'rooms', 'age'])
print(df_enhanced.head())
```

### Ratio Features

```python
def create_ratio_features(df, numerators, denominators, epsilon=1e-10):
    """
    Create ratio features from lists of numerator and denominator columns
    """
    df_ratios = df.copy()
    
    for num in numerators:
        for denom in denominators:
            if num != denom:
                ratio_name = f"{num}_per_{denom}"
                df_ratios[ratio_name] = df[num] / (df[denom] + epsilon)
                print(f"Created: {ratio_name}")
    
    return df_ratios

# Example
df = pd.DataFrame({
    'revenue': [100000, 150000, 200000],
    'employees': [10, 12, 15],
    'customers': [500, 600, 750]
})

df_with_ratios = create_ratio_features(df, 
                                        numerators=['revenue'], 
                                        denominators=['employees', 'customers'])
print(df_with_ratios)
```

### DateTime Feature Splitting

```python
def extract_datetime_features(df, datetime_column):
    """
    Extract comprehensive datetime features
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_column])
    
    # Basic components
    df[f'{datetime_column}_year'] = dt.dt.year
    df[f'{datetime_column}_month'] = dt.dt.month
    df[f'{datetime_column}_day'] = dt.dt.day
    df[f'{datetime_column}_hour'] = dt.dt.hour
    df[f'{datetime_column}_minute'] = dt.dt.minute
    df[f'{datetime_column}_dayofweek'] = dt.dt.dayofweek
    df[f'{datetime_column}_dayofyear'] = dt.dt.dayofyear
    df[f'{datetime_column}_weekofyear'] = dt.dt.isocalendar().week
    df[f'{datetime_column}_quarter'] = dt.dt.quarter
    
    # Binary indicators
    df[f'{datetime_column}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df[f'{datetime_column}_is_month_start'] = dt.dt.is_month_start.astype(int)
    df[f'{datetime_column}_is_month_end'] = dt.dt.is_month_end.astype(int)
    
    # Cyclical encoding for hour
    df[f'{datetime_column}_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df[f'{datetime_column}_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    # Cyclical encoding for month
    df[f'{datetime_column}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f'{datetime_column}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    # Cyclical encoding for day of week
    df[f'{datetime_column}_dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df[f'{datetime_column}_dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    
    return df

# Example
df = pd.DataFrame({
    'transaction_date': ['2024-01-15 10:30:00', '2024-06-20 14:45:00', '2024-12-25 18:00:00']
})

df_with_datetime = extract_datetime_features(df, 'transaction_date')
print(df_with_datetime.columns.tolist())
print(df_with_datetime.head())
```

### Complete Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('train.csv')

print("Original dataset shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())

# ===== FEATURE CONSTRUCTION =====

# 1. Polynomial features (for numerical columns)
numerical_cols = ['feature1', 'feature2']  # Replace with actual column names
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[numerical_cols])
poly_feature_names = poly.get_feature_names_out(numerical_cols)

# Add polynomial features to dataframe
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
df = pd.concat([df, df_poly], axis=1)

# 2. Interaction features
if 'price' in df.columns and 'quantity' in df.columns:
    df['total_value'] = df['price'] * df['quantity']

# 3. Ratio features
if 'revenue' in df.columns and 'employees' in df.columns:
    df['revenue_per_employee'] = df['revenue'] / (df['employees'] + 1)

# 4. Aggregation features (if grouping variable exists)
if 'category' in df.columns and 'sales' in df.columns:
    df['avg_sales_by_category'] = df.groupby('category')['sales'].transform('mean')
    df['sales_vs_category_avg'] = df['sales'] - df['avg_sales_by_category']

# ===== FEATURE SPLITTING =====

# 1. DateTime features
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

# 2. Text splitting
if 'full_name' in df.columns:
    df[['first_name', 'last_name']] = df['full_name'].str.split(' ', n=1, expand=True)

# 3. Categorical splitting
if 'product_code' in df.columns:
    split_code = df['product_code'].str.split('-', expand=True)
    df['category'] = split_code[0]
    df['subcategory'] = split_code[1]

print("\nEnhanced dataset shape:", df.shape)
print("New columns added:", df.shape[1] - len(df.columns))
print("\nSample of engineered features:")
print(df.head())

# ===== MODEL COMPARISON =====

# Prepare data for modeling
# (Assume 'target' is your target variable)
if 'target' in df.columns:
    X_original = df[numerical_cols]
    X_engineered = df.drop(['target', 'date'], axis=1)  # Use all features
    y = df['target']
    
    # Split data
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42)
    
    X_train_eng, X_test_eng, _, _ = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42)
    
    # Train models
    model_original = LinearRegression()
    model_original.fit(X_train_orig, y_train)
    
    model_engineered = LinearRegression()
    model_engineered.fit(X_train_eng, y_train)
    
    # Evaluate
    y_pred_orig = model_original.predict(X_test_orig)
    y_pred_eng = model_engineered.predict(X_test_eng)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<30} {'R² Score':<15} {'RMSE':<15}")
    print("-"*60)
    print(f"{'Original Features':<30} {r2_score(y_test, y_pred_orig):<15.4f} "
          f"{np.sqrt(mean_squared_error(y_test, y_pred_orig)):<15.2f}")
    print(f"{'Engineered Features':<30} {r2_score(y_test, y_pred_eng):<15.4f} "
          f"{np.sqrt(mean_squared_error(y_test, y_pred_eng)):<15.2f}")
```

## Practical Applications

### 1. **Predicting House Prices**

**Feature Construction**:
- Total square footage = indoor area + outdoor area
- Price per square foot = price / total area
- Room density = number of rooms / total area
- Age at sale = sale year - construction year

**Feature Splitting**:
- Sale date → year, month, season
- Address → neighborhood, city, state

### 2. **Customer Churn Prediction**

**Feature Construction**:
- Tenure = account_end_date - account_start_date
- Average monthly spend = total_spend / months_active
- Usage frequency = total_logins / days_active
- Customer lifetime value = avg_monthly_spend × tenure

**Feature Splitting**:
- Registration date → day, month, year, day_of_week
- Customer ID → customer type, region code

### 3. **Fraud Detection**

**Feature Construction**:
- Transaction velocity = number of transactions / time window
- Amount deviation = abs(amount - user_avg_amount)
- Distance from previous = geographic distance between transactions
- Time since last transaction

**Feature Splitting**:
- Transaction timestamp → hour, day, is_business_hours
- Card number → issuing bank, card type
- IP address → country, region

### 4. **Recommendation Systems**

**Feature Construction**:
- User-item interaction count
- User average rating - item average rating
- Cosine similarity between user vectors
- Recency-weighted engagement score

**Feature Splitting**:
- Product category → main category, subcategory
- User demographics → age group, income bracket

## Summary

Feature construction and feature splitting are powerful techniques that transform raw data into informative representations for machine learning models.

**Key Principles**:

1. **Domain Knowledge**: Leverage expertise to create meaningful features
2. **Start Simple**: Begin with basic transformations before complex ones
3. **Validate Impact**: Always measure if new features improve performance
4. **Avoid Leakage**: Ensure features don't contain target information
5. **Document Everything**: Keep track of feature engineering steps

**Common Transformations**:

- **Polynomial**: $\phi_d(\mathbf{x})$ for degree $d$
- **Interaction**: $x_i \times x_j$
- **Ratio**: $x_i / x_j$
- **Aggregation**: $\text{mean}, \text{sum}, \text{max}, \text{min}$
- **Cyclical**: $\sin(2\pi x/\text{max}), \cos(2\pi x/\text{max})$

**Best Practices**:

✅ Understand your data and domain
✅ Create features that make intuitive sense
✅ Test feature importance after creation
✅ Use cross-validation to prevent overfitting
✅ Document your feature engineering logic

❌ Don't create too many features without selection
❌ Don't leak future information into training data
❌ Don't ignore feature scaling after construction
❌ Don't forget to apply same transformations to test data

The art of feature engineering often makes the difference between a mediocre model and a state-of-the-art solution!
