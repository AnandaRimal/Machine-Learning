# Outlier Removal Using IQR Method

## Table of Contents
- [Introduction](#introduction)
- [What is the IQR Method?](#what-is-the-iqr-method)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use IQR for Outlier Detection?](#why-use-iqr-for-outlier-detection)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Step-by-Step Algorithm](#step-by-step-algorithm)
- [Mathematical Example](#mathematical-example)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)
- [Comparison with Other Methods](#comparison-with-other-methods)

## Introduction

The Interquartile Range (IQR) method is one of the most robust and widely used statistical techniques for detecting and removing outliers from datasets. Unlike methods that rely on mean and standard deviation (which are sensitive to outliers themselves), the IQR method uses quartiles, making it resistant to extreme values.

Outliers can significantly distort statistical analyses, machine learning models, and data visualizations. The IQR method provides a systematic, distribution-free approach to identify data points that fall unusually far from the central tendency of the data.

## What is the IQR Method?

The Interquartile Range (IQR) is a measure of statistical dispersion that represents the middle 50% of your data. It's calculated as the difference between the third quartile (Q3, 75th percentile) and the first quartile (Q1, 25th percentile).

**Key Concept**: Any data point that falls below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$ is considered an outlier.

This 1.5 × IQR rule was popularized by John Tukey and is the basis for box plot whiskers.

## Mathematical Foundation

### Quartiles

For a dataset with $n$ observations arranged in ascending order:

**First Quartile (Q1)**: The 25th percentile
$$Q1 = \text{Value at position} \left\lceil \frac{n+1}{4} \right\rceil$$

**Second Quartile (Q2)**: The median (50th percentile)
$$Q2 = \text{Median}$$

**Third Quartile (Q3)**: The 75th percentile
$$Q3 = \text{Value at position} \left\lceil \frac{3(n+1)}{4} \right\rceil$$

### Interquartile Range (IQR)

$$IQR = Q3 - Q1$$

The IQR represents the spread of the middle 50% of the data.

### Outlier Boundaries

**Lower Bound**: 
$$L_{bound} = Q1 - 1.5 \times IQR$$

**Upper Bound**: 
$$U_{bound} = Q3 + 1.5 \times IQR$$

**Outlier Definition**: A data point $x$ is an outlier if:
$$x < L_{bound} \quad \text{or} \quad x > U_{bound}$$

### Extreme Outliers

For more extreme outliers, some analysts use a factor of 3.0 instead of 1.5:

$$L_{extreme} = Q1 - 3.0 \times IQR$$
$$U_{extreme} = Q3 + 3.0 \times IQR$$

## Why Use IQR for Outlier Detection?

### 1. **Robustness to Outliers**
The quartiles themselves are not affected by extreme values. Even if you have several outliers, Q1 and Q3 remain stable because they depend on the position of data points, not their values.

### 2. **Distribution-Free**
The IQR method doesn't assume any specific distribution (unlike the Z-score method which assumes normality). It works equally well for:
- Normal distributions
- Skewed distributions
- Multimodal distributions

### 3. **Visual Interpretation**
The IQR method aligns perfectly with box plots, making it easy to visualize outliers:
```
   |----[=====|=====]----| ← Normal range
   ↑     ↑    ↑    ↑     ↑
   Min   Q1   Q2   Q3   Max
   (whisker)        (whisker)
```

### 4. **Intuitive Threshold**
The 1.5 × IQR rule is well-established and provides a good balance between detecting true outliers and avoiding false positives.

## Advantages and Disadvantages

### Advantages

1. **Robust**: Not influenced by extreme values
2. **Simple**: Easy to understand and implement
3. **No Assumptions**: Works with any distribution
4. **Visual**: Integrates well with box plots
5. **Standardized**: The 1.5 × IQR rule is widely accepted
6. **Univariate and Multivariate**: Can be applied to each feature independently
7. **Interpretable**: Clear definition of what constitutes an outlier

### Disadvantages

1. **Fixed Threshold**: The 1.5 multiplier may not be optimal for all datasets
2. **Univariate**: Traditional IQR looks at each variable separately, may miss multivariate outliers
3. **May Remove Valid Data**: In small datasets, legitimate extreme values might be flagged
4. **Context-Insensitive**: Doesn't consider domain knowledge
5. **Symmetric Bounds**: Uses the same multiplier for both upper and lower bounds (may not suit asymmetric distributions)

## Step-by-Step Algorithm

### Algorithm for Outlier Removal Using IQR

**Input**: Dataset $X = \{x_1, x_2, ..., x_n\}$

**Output**: Cleaned dataset $X'$ without outliers

**Steps**:

1. **Sort the data** in ascending order
2. **Calculate Q1** (25th percentile)
3. **Calculate Q3** (75th percentile)
4. **Compute IQR**: $IQR = Q3 - Q1$
5. **Calculate bounds**:
   - $L_{bound} = Q1 - 1.5 \times IQR$
   - $U_{bound} = Q3 + 1.5 \times IQR$
6. **Identify outliers**: Flag all $x_i$ where $x_i < L_{bound}$ or $x_i > U_{bound}$
7. **Remove outliers**: Create $X'$ containing only non-outlier values
8. **Return** cleaned dataset $X'$

### Pseudocode

```
function remove_outliers_iqr(data):
    Q1 = percentile(data, 25)
    Q3 = percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = []
    for value in data:
        if lower_bound <= value <= upper_bound:
            cleaned_data.append(value)
    
    return cleaned_data
```

## Mathematical Example

Let's work through a complete example with a small dataset.

### Dataset
Consider student placement salaries (in thousands):

$$X = \{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 50, 55\}$$

Notice the last two values (50, 55) appear to be potential outliers.

### Step 1: Sort Data (already sorted)
$$X = \{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 50, 55\}$$

### Step 2: Calculate Q1
Position of Q1: $\frac{n+1}{4} = \frac{12+1}{4} = 3.25$

Interpolating between 3rd and 4th values:
$$Q1 = 7 + 0.25(8-7) = 7.25$$

### Step 3: Calculate Q3
Position of Q3: $\frac{3(n+1)}{4} = \frac{3(13)}{4} = 9.75$

Interpolating between 9th and 10th values:
$$Q3 = 13 + 0.75(14-13) = 13.75$$

### Step 4: Compute IQR
$$IQR = Q3 - Q1 = 13.75 - 7.25 = 6.5$$

### Step 5: Calculate Boundaries
$$L_{bound} = Q1 - 1.5 \times IQR = 7.25 - 1.5(6.5) = 7.25 - 9.75 = -2.5$$

$$U_{bound} = Q3 + 1.5 \times IQR = 13.75 + 1.5(6.5) = 13.75 + 9.75 = 23.5$$

### Step 6: Identify Outliers
Check each value:
- $5 \geq -2.5$ ✓ (not an outlier)
- $6 \geq -2.5$ ✓ (not an outlier)
- ...
- $14 \leq 23.5$ ✓ (not an outlier)
- $50 > 23.5$ ✗ **OUTLIER**
- $55 > 23.5$ ✗ **OUTLIER**

### Result
**Outliers detected**: {50, 55}

**Cleaned dataset**: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

## Implementation in Python

### Basic Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_outliers_iqr(data, column_name):
    """
    Remove outliers using the IQR method
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input dataset
    column_name : str
        Name of the column to check for outliers
    
    Returns:
    --------
    cleaned_data : pandas DataFrame
        Dataset with outliers removed
    outliers : pandas DataFrame
        Removed outlier rows
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    
    # Identify outliers
    outliers = data[(data[column_name] < lower_bound) | 
                    (data[column_name] > upper_bound)]
    
    # Remove outliers
    cleaned_data = data[(data[column_name] >= lower_bound) & 
                        (data[column_name] <= upper_bound)]
    
    print(f"\nOriginal data size: {len(data)}")
    print(f"Outliers detected: {len(outliers)}")
    print(f"Cleaned data size: {len(cleaned_data)}")
    
    return cleaned_data, outliers

# Example usage
data = pd.DataFrame({
    'salary': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 50, 55]
})

cleaned_data, outliers = remove_outliers_iqr(data, 'salary')
```

### Multivariate Implementation

```python
def remove_outliers_multivariate(data, columns):
    """
    Remove outliers from multiple columns
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input dataset
    columns : list
        List of column names to check for outliers
    
    Returns:
    --------
    cleaned_data : pandas DataFrame
        Dataset with outliers removed from all specified columns
    """
    cleaned_data = data.copy()
    
    for column in columns:
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep only rows without outliers in this column
        cleaned_data = cleaned_data[
            (cleaned_data[column] >= lower_bound) & 
            (cleaned_data[column] <= upper_bound)
        ]
        
        print(f"{column}: Removed {len(data) - len(cleaned_data)} outliers")
    
    return cleaned_data

# Example
df = pd.read_csv('placement.csv')
cleaned_df = remove_outliers_multivariate(df, ['cgpa', 'placement_exam_marks'])
```

### Visualization Function

```python
def visualize_outliers(data, column_name):
    """
    Visualize outliers using box plot and histogram
    """
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot(data[column_name], vert=False)
    axes[0].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
    axes[0].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
    axes[0].set_xlabel(column_name)
    axes[0].set_title('Box Plot with IQR Boundaries')
    axes[0].legend()
    
    # Histogram with boundaries
    axes[1].hist(data[column_name], bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(lower_bound, color='r', linestyle='--', 
                    linewidth=2, label=f'Lower: {lower_bound:.2f}')
    axes[1].axvline(upper_bound, color='r', linestyle='--', 
                    linewidth=2, label=f'Upper: {upper_bound:.2f}')
    axes[1].axvline(Q1, color='g', linestyle=':', label=f'Q1: {Q1:.2f}')
    axes[1].axvline(Q3, color='g', linestyle=':', label=f'Q3: {Q3:.2f}')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution with Quartiles')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Example
visualize_outliers(data, 'salary')
```

### Complete Example with Real Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('placement.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for outliers in 'cgpa' column
column = 'cgpa'

# Before outlier removal
print(f"\n{'='*50}")
print(f"Analyzing outliers in '{column}'")
print(f"{'='*50}")

Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")

# Identify outliers
outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
print(f"\nNumber of outliers: {len(outliers)}")
print(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")

if len(outliers) > 0:
    print("\nOutlier values:")
    print(outliers[column].values)

# Remove outliers
df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

print(f"\nOriginal dataset size: {len(df)}")
print(f"Cleaned dataset size: {len(df_cleaned)}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")

# Compare statistics
print(f"\n{'='*50}")
print("Statistics Comparison")
print(f"{'='*50}")
print(f"{'Metric':<20} {'Original':<15} {'Cleaned':<15}")
print(f"{'-'*50}")
print(f"{'Mean':<20} {df[column].mean():<15.2f} {df_cleaned[column].mean():<15.2f}")
print(f"{'Median':<20} {df[column].median():<15.2f} {df_cleaned[column].median():<15.2f}")
print(f"{'Std Dev':<20} {df[column].std():<15.2f} {df_cleaned[column].std():<15.2f}")
print(f"{'Min':<20} {df[column].min():<15.2f} {df_cleaned[column].min():<15.2f}")
print(f"{'Max':<20} {df[column].max():<15.2f} {df_cleaned[column].max():<15.2f}")
```

## Practical Applications

### 1. **Financial Data**
Removing extreme transaction amounts that could indicate fraud or data entry errors.

### 2. **Sensor Data**
Filtering out sensor readings that are physically impossible or indicate sensor malfunction.

### 3. **Medical Research**
Identifying abnormal lab results that may indicate measurement errors.

### 4. **Machine Learning**
Preprocessing data before training models to improve performance:
- Linear Regression
- K-Means Clustering
- Neural Networks

### 5. **Quality Control**
Detecting defective products in manufacturing based on measurements.

## Comparison with Other Methods

### IQR vs Z-Score

| Aspect | IQR Method | Z-Score Method |
|--------|-----------|----------------|
| **Assumption** | Distribution-free | Assumes normal distribution |
| **Robustness** | Very robust | Sensitive to outliers |
| **Threshold** | 1.5 × IQR | Usually ±3σ |
| **Best for** | Skewed data | Normal data |
| **Computation** | Quartiles | Mean & std dev |

**Z-Score Formula**: $z = \frac{x - \mu}{\sigma}$

An observation is an outlier if $|z| > 3$

### IQR vs Modified Z-Score

The modified Z-score uses median and MAD (Median Absolute Deviation):

$$M_i = \frac{0.6745(x_i - \text{median})}{\text{MAD}}$$

where $\text{MAD} = \text{median}(|x_i - \text{median}|)$

Both IQR and modified Z-score are robust, but IQR is simpler.

### When to Use IQR

- ✅ Data is skewed
- ✅ Data contains outliers affecting mean/std
- ✅ Distribution is unknown
- ✅ Quick visual interpretation needed
- ✅ Small to medium-sized datasets

### When NOT to Use IQR

- ❌ Data truly follows normal distribution (Z-score might be better)
- ❌ Extreme values are scientifically important
- ❌ Very small datasets (< 20 observations)
- ❌ Multivariate outliers need detection (use Mahalanobis distance)

## Summary

The IQR method provides a robust, intuitive, and widely-applicable technique for outlier detection and removal. Its resistance to extreme values and lack of distributional assumptions make it particularly valuable for real-world datasets that often violate normality assumptions.

**Key Takeaways**:
1. IQR = Q3 - Q1 (middle 50% spread)
2. Outliers: $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$
3. Works with any distribution
4. Visualized perfectly with box plots
5. May need adjustment for specific domains

The choice of whether to remove outliers should always consider:
- **Domain knowledge**: Are these values scientifically plausible?
- **Data size**: Can you afford to lose observations?
- **Analysis goals**: Will outliers bias your results?
- **Downstream impact**: How will removal affect model performance?
