# Outlier Detection Using Percentiles

## Table of Contents
- [Introduction](#introduction)
- [What are Percentiles?](#what-are-percentiles)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use Percentiles for Outlier Detection?](#why-use-percentiles-for-outlier-detection)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Percentile-Based Methods](#percentile-based-methods)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)
- [Comparison with Other Methods](#comparison-with-other-methods)

## Introduction

Percentile-based outlier detection is a flexible, distribution-free method that identifies extreme values by directly examining the tails of the data distribution. Unlike methods that rely on standard deviations or fixed multipliers, percentile-based approaches allow you to customize the outlier threshold based on your specific needs.

This method is particularly powerful because:
1. It works with any distribution shape
2. You can control exactly what percentage of data to flag as outliers
3. It's intuitive and easy to explain to stakeholders
4. It's computationally simple and fast

## What are Percentiles?

A **percentile** is a value below which a given percentage of observations fall. For example:
- The 10th percentile (P10) is the value below which 10% of the data lies
- The 90th percentile (P90) is the value below which 90% of the data lies
- The 50th percentile (P50) is the median

### Relationship to Quantiles

Percentiles are a specific type of quantile:
- **Quantile**: Divides data based on fraction (0 to 1)
- **Percentile**: Divides data based on percentage (0 to 100)

$$\text{Percentile}(p) = \text{Quantile}\left(\frac{p}{100}\right)$$

For example, the 75th percentile equals the 0.75 quantile.

## Mathematical Foundation

### Percentile Definition

For a dataset $X = \{x_1, x_2, ..., x_n\}$ sorted in ascending order, the $p$-th percentile $P_p$ is:

$$P_p = x_{(k)} + (k - \lfloor k \rfloor)(x_{(k+1)} - x_{(k)})$$

where:
- $k = \frac{p}{100} \times (n + 1)$
- $x_{(k)}$ is the value at position $k$
- $\lfloor k \rfloor$ is the floor (integer part) of $k$

### Common Percentiles

**Quartiles** (special percentiles):
- $Q1 = P_{25}$ (First quartile)
- $Q2 = P_{50}$ (Median)
- $Q3 = P_{75}$ (Third quartile)

**Deciles** (10% intervals):
- $D1 = P_{10}, D2 = P_{20}, ..., D9 = P_{90}$

**Percentile Rank**

The percentile rank of a value $x$ tells us what percentage of data falls below it:

$$\text{Percentile Rank}(x) = \frac{\text{Number of values} < x}{\text{Total number of values}} \times 100$$

### Interpolation Methods

Different interpolation methods exist for calculating percentiles:

1. **Linear Interpolation** (most common):
   $$P_p = (1-\alpha) x_{(j)} + \alpha x_{(j+1)}$$
   where $j = \lfloor h \rfloor$ and $\alpha = h - j$, with $h = \frac{p}{100}(n-1) + 1$

2. **Lower Value**: Always take the lower index
3. **Higher Value**: Always take the higher index
4. **Nearest**: Take the nearest index
5. **Midpoint**: Average of lower and upper values

## Why Use Percentiles for Outlier Detection?

### 1. **Direct Control Over Outlier Percentage**

You can specify exactly what percentage of data to consider as outliers:
- Flag the lowest 5% and highest 5% as outliers
- Flag only the extreme 1% on each tail
- Asymmetric thresholds: flag lowest 2% and highest 10%

### 2. **Distribution-Free**

Percentiles don't assume any specific distribution:
- Works for normal distributions
- Works for skewed distributions
- Works for multimodal distributions
- Works for discrete or continuous data

### 3. **Robust to Extreme Values**

The percentile itself is not affected by how extreme the outliers are, only by their position in the sorted data.

### 4. **Interpretability**

Easy to explain: "We consider the bottom 5% and top 5% as outliers" is more intuitive than "values beyond 1.5 times the IQR."

### 5. **Flexibility**

Can create asymmetric bounds when data is naturally skewed:
```python
lower_bound = percentile(data, 1)   # Lower 1%
upper_bound = percentile(data, 99)  # Upper 99%
```

## Advantages and Disadvantages

### Advantages

1. **Customizable Thresholds**: Set any percentile based on domain knowledge
2. **No Distribution Assumptions**: Works with any data distribution
3. **Simple to Understand**: Stakeholders easily grasp the concept
4. **Computationally Efficient**: Fast to calculate
5. **Handles Skewness**: Can use asymmetric thresholds
6. **Robust**: Not affected by extreme outlier values
7. **Consistent**: Always flags the specified percentage

### Disadvantages

1. **Arbitrary Cutoffs**: Choosing percentiles can be subjective
2. **Fixed Proportion**: Always removes a fixed percentage regardless of data quality
3. **Univariate**: Traditional approach examines features independently
4. **May Remove Valid Data**: In clean datasets, might flag legitimate extreme values
5. **No Statistical Justification**: Unlike methods based on statistical theory
6. **Sensitive to Sample Size**: Small samples may not have meaningful percentiles

## Percentile-Based Methods

### Method 1: Fixed Percentile Bounds

Define lower and upper percentile thresholds:

$$L_{bound} = P_{low}$$
$$U_{bound} = P_{high}$$

Common choices:
- **Conservative**: $P_1$ and $P_{99}$ (flag 2% total)
- **Moderate**: $P_5$ and $P_{95}$ (flag 10% total)
- **Aggressive**: $P_{10}$ and $P_{90}$ (flag 20% total)

**Outlier Definition**: A value $x$ is an outlier if:
$$x < P_{low} \quad \text{or} \quad x > P_{high}$$

### Method 2: Interpercentile Range (IPR)

Similar to IQR but using arbitrary percentiles:

$$IPR = P_{high} - P_{low}$$

For example, using $P_{10}$ and $P_{90}$:
$$IPR_{80} = P_{90} - P_{10}$$

Then apply a multiplier:
$$L_{bound} = P_{low} - k \times IPR$$
$$U_{bound} = P_{high} + k \times IPR$$

where $k$ is typically 1.5 or 2.0.

### Method 3: Percentile-Based Z-Score

Combine percentiles with standardization:

1. Calculate median and median absolute deviation from percentiles
2. Compute modified Z-scores
3. Flag values with high modified Z-scores

### Method 4: Adaptive Percentiles

Adjust percentiles based on data characteristics:
- For symmetric data: Use symmetric percentiles (e.g., $P_5$ and $P_{95}$)
- For right-skewed: Use $P_2$ and $P_{98}$ or $P_1$ and $P_{99}$
- For left-skewed: Adjust accordingly

## Mathematical Examples

### Example 1: Simple Percentile Bounds

**Dataset** (student heights in cm):
$$X = \{150, 152, 155, 158, 160, 162, 165, 168, 170, 172, 175, 178, 180, 195, 200\}$$

$n = 15$ observations

**Goal**: Flag the lowest 10% and highest 10% as outliers.

**Calculate 10th percentile ($P_{10}$)**:

Position: $k = \frac{10}{100} \times (15 + 1) = 1.6$

Interpolate between 1st and 2nd values:
$$P_{10} = 150 + 0.6(152 - 150) = 150 + 1.2 = 151.2$$

**Calculate 90th percentile ($P_{90}$)**:

Position: $k = \frac{90}{100} \times (15 + 1) = 14.4$

Interpolate between 14th and 15th values:
$$P_{90} = 195 + 0.4(200 - 195) = 195 + 2.0 = 197$$

**Outlier Detection**:
- $L_{bound} = 151.2$
- $U_{bound} = 197$

**Results**:
- Lower outlier: 150 (< 151.2) ✓
- Upper outliers: 200 (> 197) ✓
- **Total outliers detected**: 2 values (13.3% of data)

### Example 2: Asymmetric Bounds for Skewed Data

**Dataset** (income in thousands, right-skewed):
$$X = \{20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 60, 80, 150, 500\}$$

For right-skewed data, we might want:
- Lower bound: $P_5$ (to capture extreme low values)
- Upper bound: $P_{98}$ (to capture extreme high values)

**Calculate $P_5$**:

Position: $k = \frac{5}{100} \times 16 = 0.8$

$$P_5 \approx 20 + 0.8(22-20) = 21.6$$

**Calculate $P_{98}$**:

Position: $k = \frac{98}{100} \times 16 = 15.68$

$$P_{98} \approx 150 + 0.68(500-150) = 388$$

**Outliers**:
- Lower: 20 (< 21.6) ✓
- Upper: 500 (> 388) ✓

### Example 3: Interpercentile Range Method

**Dataset**:
$$X = \{5, 8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30, 100\}$$

Use $P_{25}$ and $P_{75}$ (same as IQR):

**Calculate percentiles**:
- $P_{25} = 12$
- $P_{75} = 22$

**Calculate IPR**:
$$IPR = P_{75} - P_{25} = 22 - 12 = 10$$

**Apply 1.5 multiplier**:
$$L_{bound} = P_{25} - 1.5 \times IPR = 12 - 15 = -3$$
$$U_{bound} = P_{75} + 1.5 \times IPR = 22 + 15 = 37$$

**Outliers**:
- 100 > 37 ✗ **OUTLIER**

All other values are within bounds.

## Implementation in Python

### Basic Percentile-Based Outlier Detection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_outliers_percentile(data, column, lower_percentile=5, upper_percentile=95):
    """
    Detect outliers using percentile bounds
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input dataset
    column : str
        Column name to analyze
    lower_percentile : float
        Lower percentile threshold (0-100)
    upper_percentile : float
        Upper percentile threshold (0-100)
    
    Returns:
    --------
    outliers : pandas DataFrame
        Rows containing outliers
    cleaned_data : pandas DataFrame
        Data without outliers
    bounds : dict
        Dictionary with boundary values
    """
    # Calculate percentile bounds
    lower_bound = np.percentile(data[column], lower_percentile)
    upper_bound = np.percentile(data[column], upper_percentile)
    
    print(f"Lower Bound (P{lower_percentile}): {lower_bound:.2f}")
    print(f"Upper Bound (P{upper_percentile}): {upper_bound:.2f}")
    
    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    # Clean data
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    print(f"\nTotal observations: {len(data)}")
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
    print(f"Clean observations: {len(cleaned_data)}")
    
    bounds = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile
    }
    
    return outliers, cleaned_data, bounds

# Example usage
df = pd.DataFrame({
    'height': [150, 152, 155, 158, 160, 162, 165, 168, 170, 172, 175, 178, 180, 195, 200]
})

outliers, cleaned, bounds = detect_outliers_percentile(df, 'height', 
                                                         lower_percentile=10, 
                                                         upper_percentile=90)
print("\nOutlier values:")
print(outliers['height'].values)
```

### Asymmetric Percentile Bounds

```python
def detect_outliers_asymmetric(data, column, lower_p=1, upper_p=99):
    """
    Detect outliers with asymmetric percentile bounds
    Useful for skewed distributions
    """
    lower_bound = np.percentile(data[column], lower_p)
    upper_bound = np.percentile(data[column], upper_p)
    
    outliers_lower = data[data[column] < lower_bound]
    outliers_upper = data[data[column] > upper_bound]
    
    print(f"Lower {lower_p}% percentile: {lower_bound:.2f}")
    print(f"Upper {upper_p}% percentile: {upper_bound:.2f}")
    print(f"Lower outliers: {len(outliers_lower)}")
    print(f"Upper outliers: {len(outliers_upper)}")
    
    all_outliers = pd.concat([outliers_lower, outliers_upper])
    cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return all_outliers, cleaned

# Example with skewed data
income_data = pd.DataFrame({
    'income': [20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 60, 80, 150, 500]
})

outliers, cleaned = detect_outliers_asymmetric(income_data, 'income', 
                                                 lower_p=5, upper_p=98)
```

### Interpercentile Range (IPR) Method

```python
def detect_outliers_ipr(data, column, lower_p=25, upper_p=75, multiplier=1.5):
    """
    Detect outliers using Interpercentile Range
    Similar to IQR but with customizable percentiles
    
    Parameters:
    -----------
    lower_p : int
        Lower percentile (default 25 for Q1)
    upper_p : int
        Upper percentile (default 75 for Q3)
    multiplier : float
        Multiplier for IPR (default 1.5)
    """
    # Calculate percentiles
    P_lower = np.percentile(data[column], lower_p)
    P_upper = np.percentile(data[column], upper_p)
    
    # Calculate IPR
    IPR = P_upper - P_lower
    
    # Calculate bounds
    lower_bound = P_lower - multiplier * IPR
    upper_bound = P_upper + multiplier * IPR
    
    print(f"P{lower_p}: {P_lower:.2f}")
    print(f"P{upper_p}: {P_upper:.2f}")
    print(f"IPR (P{upper_p} - P{lower_p}): {IPR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    
    # Detect outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    print(f"\nOutliers: {len(outliers)}")
    
    return outliers, cleaned

# Example
df = pd.DataFrame({'value': [5, 8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30, 100]})
outliers, cleaned = detect_outliers_ipr(df, 'value', lower_p=25, upper_p=75, multiplier=1.5)
```

### Visualization Function

```python
def visualize_percentile_outliers(data, column, lower_p=5, upper_p=95):
    """
    Visualize outliers with percentile bounds
    """
    lower_bound = np.percentile(data[column], lower_p)
    upper_bound = np.percentile(data[column], upper_p)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram with percentile lines
    axes[0, 0].hist(data[column], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(lower_bound, color='red', linestyle='--', linewidth=2, 
                       label=f'P{lower_p}: {lower_bound:.2f}')
    axes[0, 0].axvline(upper_bound, color='red', linestyle='--', linewidth=2,
                       label=f'P{upper_p}: {upper_bound:.2f}')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution with Percentile Bounds')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(data[column], vert=False)
    axes[0, 1].axvline(lower_bound, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(upper_bound, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel(column)
    axes[0, 1].set_title('Box Plot with Percentile Bounds')
    axes[0, 1].grid(alpha=0.3)
    
    # Scatter plot with colors
    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    colors = ['red' if outlier else 'blue' for outlier in outliers_mask]
    
    axes[1, 0].scatter(range(len(data)), data[column], c=colors, alpha=0.6)
    axes[1, 0].axhline(lower_bound, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axhline(upper_bound, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel(column)
    axes[1, 0].set_title('Scatter Plot (Red = Outliers)')
    axes[1, 0].grid(alpha=0.3)
    
    # Percentile plot
    percentiles = list(range(0, 101, 5))
    percentile_values = [np.percentile(data[column], p) for p in percentiles]
    
    axes[1, 1].plot(percentiles, percentile_values, marker='o', linewidth=2)
    axes[1, 1].axhline(lower_bound, color='red', linestyle='--', linewidth=2,
                       label=f'P{lower_p}')
    axes[1, 1].axhline(upper_bound, color='red', linestyle='--', linewidth=2,
                       label=f'P{upper_p}')
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel(column)
    axes[1, 1].set_title('Percentile Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example
visualize_percentile_outliers(df, 'height', lower_p=10, upper_p=90)
```

### Complete Real-World Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('weight-height.csv')

print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Analyze 'Height' column for outliers
column = 'Height'

print(f"\n{'='*60}")
print(f"Percentile-Based Outlier Analysis for '{column}'")
print(f"{'='*60}")

# Calculate various percentiles
percentiles_to_check = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\nPercentile Values:")
for p in percentiles_to_check:
    value = np.percentile(df[column], p)
    print(f"  P{p:2d}: {value:.2f}")

# Method 1: 5th and 95th percentile
print(f"\n{'='*60}")
print("Method 1: 5th and 95th Percentile Bounds")
print(f"{'='*60}")

lower_bound = np.percentile(df[column], 5)
upper_bound = np.percentile(df[column], 95)

outliers_method1 = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

print(f"Lower Bound (P5): {lower_bound:.2f}")
print(f"Upper Bound (P95): {upper_bound:.2f}")
print(f"Outliers detected: {len(outliers_method1)} ({len(outliers_method1)/len(df)*100:.2f}%)")

# Method 2: 1st and 99th percentile (more conservative)
print(f"\n{'='*60}")
print("Method 2: 1st and 99th Percentile Bounds (Conservative)")
print(f"{'='*60}")

lower_bound_conservative = np.percentile(df[column], 1)
upper_bound_conservative = np.percentile(df[column], 99)

outliers_method2 = df[(df[column] < lower_bound_conservative) | 
                      (df[column] > upper_bound_conservative)]

print(f"Lower Bound (P1): {lower_bound_conservative:.2f}")
print(f"Upper Bound (P99): {upper_bound_conservative:.2f}")
print(f"Outliers detected: {len(outliers_method2)} ({len(outliers_method2)/len(df)*100:.2f}%)")

# Compare statistics before and after removal
df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

print(f"\n{'='*60}")
print("Statistics Comparison (Method 1: P5-P95)")
print(f"{'='*60}")
print(f"{'Metric':<20} {'Original':<15} {'Cleaned':<15} {'Change':<15}")
print(f"{'-'*65}")

metrics = ['mean', 'median', 'std', 'min', 'max']
for metric in metrics:
    orig_val = getattr(df[column], metric)()
    clean_val = getattr(df_cleaned[column], metric)()
    change = clean_val - orig_val
    print(f"{metric.capitalize():<20} {orig_val:<15.2f} {clean_val:<15.2f} {change:+.2f}")

# Visualize
visualize_percentile_outliers(df, column, lower_p=5, upper_p=95)
```

## Practical Applications

### 1. **E-commerce**: Product Pricing
Remove extremely low or high prices that may be data entry errors:
```python
# Keep middle 98% of prices
outliers, clean_prices = detect_outliers_percentile(df, 'price', 
                                                      lower_percentile=1, 
                                                      upper_percentile=99)
```

### 2. **Healthcare**: Patient Vital Signs
Flag abnormal measurements:
```python
# Heart rate: Flag lowest 2% and highest 2%
outliers, normal_hr = detect_outliers_percentile(patient_data, 'heart_rate',
                                                   lower_percentile=2,
                                                   upper_percentile=98)
```

### 3. **Manufacturing**: Quality Control
Identify products outside specification limits:
```python
# Product weight: Keep middle 95%
outliers, acceptable = detect_outliers_percentile(production_data, 'weight',
                                                    lower_percentile=2.5,
                                                    upper_percentile=97.5)
```

### 4. **Finance**: Fraud Detection
Flag unusual transaction amounts:
```python
# Asymmetric for right-skewed transaction data
outliers, normal_trans = detect_outliers_asymmetric(transactions, 'amount',
                                                      lower_p=0.5, 
                                                      upper_p=99.5)
```

### 5. **Education**: Exam Score Analysis
Identify exceptionally low or high performers:
```python
# Flag bottom 10% and top 10%
low_perform, high_perform, normal = detect_outliers_percentile(scores, 'test_score',
                                                                 lower_percentile=10,
                                                                 upper_percentile=90)
```

## Comparison with Other Methods

### Percentiles vs IQR

| Aspect | Percentile Method | IQR Method |
|--------|------------------|------------|
| **Threshold** | Any percentile (P1-P99) | Fixed: Q1-1.5×IQR, Q3+1.5×IQR |
| **Flexibility** | Very flexible | Standard multiplier (1.5) |
| **% Flagged** | Exactly as specified | Varies by distribution |
| **Interpretation** | Direct percentage | Based on spread |
| **Best for** | Custom thresholds | Standard box plot analysis |

### Percentiles vs Z-Score

| Aspect | Percentile Method | Z-Score Method |
|--------|------------------|----------------|
| **Assumption** | Distribution-free | Assumes normality |
| **Robustness** | Very robust | Sensitive to outliers |
| **Calculation** | Rank-based | Mean and std dev |
| **Skewed Data** | Works well | Poor performance |
| **Threshold** | Percentile values | Usually ±3σ |

### When to Use Percentile Method

✅ **Use Percentiles When**:
- You need exact control over the % of data flagged
- Data distribution is unknown or non-normal
- Data is heavily skewed
- Stakeholders want simple, interpretable thresholds
- You have domain knowledge about acceptable ranges

❌ **Avoid Percentiles When**:
- You have strong theoretical reasons for other methods
- Dataset is very small (< 30 observations)
- You need multivariate outlier detection
- Outlier definition should be based on statistical theory

## Summary

Percentile-based outlier detection offers a powerful, flexible, and intuitive approach to identifying extreme values in data. Its distribution-free nature and direct interpretability make it particularly valuable in real-world applications.

**Key Formulas**:
1. **Percentile**: $P_p = x_{(k)} + (k - \lfloor k \rfloor)(x_{(k+1)} - x_{(k)})$ where $k = \frac{p}{100}(n+1)$
2. **Outlier Bounds**: $x < P_{low}$ or $x > P_{high}$
3. **IPR Method**: $L = P_{low} - k \times (P_{high} - P_{low})$, $U = P_{high} + k \times (P_{high} - P_{low})$

**Best Practices**:
- Choose percentiles based on domain knowledge
- Consider data distribution (symmetric vs skewed)
- Visualize before and after outlier removal
- Document your choice of thresholds
- Validate that removed values are truly anomalous
- Consider using asymmetric bounds for skewed data

The percentile method's greatest strength is its flexibility—you can adapt it to virtually any situation while maintaining clarity and interpretability.
