# Understanding Your Data - Descriptive Statistics

## General Idea

Descriptive statistics are mathematical tools that summarize and describe the main features of a dataset. They provide simple quantitative measures that characterize the central tendency, dispersion, and shape of data distributions. In machine learning, descriptive statistics form the foundation of Exploratory Data Analysis (EDA), helping practitioners understand data before building models.

## Why Use Descriptive Statistics?

1. **Data Understanding**: Quickly grasp dataset characteristics
2. **Quality Assessment**: Identify data quality issues, anomalies, errors
3. **Feature Selection**: Understand which features vary and which are constant
4. **Assumption Validation**: Check if data meets model assumptions (normality, etc.)
5. **Outlier Detection**: Identify unusual values that may need handling
6. **Communication**: Summarize findings to stakeholders
7. **Baseline Metrics**: Establish benchmarks for model performance

## Role in Machine Learning

### Pre-Modeling Phase

- **Data Profiling**: Understand data types, ranges, distributions
- **Feature Engineering**: Identify transformation needs (scaling, encoding)
- **Data Cleaning**: Detect missing values, outliers, errors
- **Sampling Strategy**: Determine if data is balanced, representative
- **Feature Selection**: Identify low-variance or redundant features

### Model Building Phase

- **Hyperparameter Tuning**: Understand feature scales for regularization
- **Performance Baseline**: Compare against simple statistics (mean prediction)
- **Validation**: Check if train/test splits are representative
- **Debugging**: Investigate model failures through data inspection

### Post-Modeling Phase

- **Error Analysis**: Understand prediction errors through residual statistics
- **Model Interpretation**: Relate model coefficients to feature statistics
- **Monitoring**: Track data drift in production using statistical tests

## Measures of Central Tendency

Central tendency describes the "center" or "typical value" of a distribution.

### 1. Mean (Arithmetic Average)

**Formula**:
$$\bar{x} = \mu = \frac{1}{n}\sum_{i=1}^{n}x_i$$

**Properties**:
- **Sensitive to outliers**: Extreme values pull the mean
- **Unique**: Only one mean exists
- **Uses all data**: Every value contributes equally
- **Minimizes squared deviations**: $\sum(x_i - \bar{x})^2$ is minimal

**When to Use**:
- Symmetric distributions
- No extreme outliers
- Interval or ratio data

**Example**: Test scores $[85, 90, 88, 92, 95]$
$$\bar{x} = \frac{85+90+88+92+95}{5} = \frac{450}{5} = 90$$

### 2. Median

**Definition**: Middle value when data is sorted

**Formula**: For sorted data $x_1 \leq x_2 \leq ... \leq x_n$:
$$\text{Median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd}\\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even}
\end{cases}$$

**Properties**:
- **Robust to outliers**: Not affected by extreme values
- **Unique**: Only one median
- **50th percentile**: Half the data below, half above
- **Minimizes absolute deviations**: $\sum|x_i - \text{median}|$ is minimal

**When to Use**:
- Skewed distributions
- Presence of outliers
- Ordinal data or higher

**Example**: Income data $[30k, 35k, 40k, 45k, 200k]$
- Mean: $70k$ (pulled by outlier)
- Median: $40k$ (resistant to outlier)

### 3. Mode

**Definition**: Most frequently occurring value

**Properties**:
- **Can be multiple**: Bimodal, multimodal distributions
- **Can be none**: All values unique
- **Works for categorical**: Only measure for nominal data
- **Not unique**: Multiple modes possible

**When to Use**:
- Categorical data
- Identifying most common category
- Discrete distributions

**Example**: Shirt sizes $[S, M, M, M, L, L, XL]$
- Mode: $M$ (appears 3 times)

### Relationship: Mean, Median, Mode

**Symmetric Distribution**: Mean ≈ Median ≈ Mode

**Right-Skewed** (positive skew): Mode < Median < Mean
- Tail extends to right
- Example: Income, house prices

**Left-Skewed** (negative skew): Mean < Median < Mode
- Tail extends to left
- Example: Test scores with ceiling effect

**Mathematical Skewness**:
$$\text{Skewness} = \frac{E[(X-\mu)^3]}{\sigma^3}$$

- Skewness > 0: Right-skewed
- Skewness < 0: Left-skewed
- Skewness ≈ 0: Symmetric

## Measures of Dispersion (Spread)

Dispersion describes how spread out or varied the data is.

### 1. Range

**Formula**:
$$\text{Range} = \max(x) - \min(x)$$

**Properties**:
- **Simple**: Easy to compute
- **Sensitive to outliers**: Uses extreme values only
- **Non-robust**: One outlier changes range dramatically
- **Scale-dependent**: Larger for larger scales

**When to Use**:
- Quick assessment
- Small datasets
- When extremes matter (min-max scaling)

**Example**: Test scores $[65, 75, 80, 85, 95]$
$$\text{Range} = 95 - 65 = 30$$

### 2. Interquartile Range (IQR)

**Definition**: Range of middle 50% of data

**Formula**:
$$IQR = Q_3 - Q_1$$

Where:
- $Q_1$: 25th percentile (first quartile)
- $Q_3$: 75th percentile (third quartile)

**Properties**:
- **Robust**: Not affected by outliers
- **Describes middle**: Ignores tails
- **Used for outlier detection**: Values beyond $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$

**Example**: Data $[10, 15, 20, 25, 30, 35, 40, 45, 50]$
- $Q_1 = 17.5$, $Q_3 = 42.5$
- $IQR = 42.5 - 17.5 = 25$

### 3. Variance

**Population Variance**:
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

**Sample Variance** (Bessel's correction):
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Properties**:
- **Non-negative**: Always ≥ 0
- **Zero only if all equal**: $s^2 = 0 \iff x_1 = x_2 = ... = x_n$
- **Units squared**: If $x$ in meters, $\sigma^2$ in meters²
- **Sensitive to outliers**: Squaring amplifies deviations

**Degrees of Freedom**: $n-1$ instead of $n$
- Unbiased estimator of population variance
- Accounts for using sample mean instead of true mean

### 4. Standard Deviation

**Formula**:
$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$

**Sample Standard Deviation**:
$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Properties**:
- **Same units as data**: If $x$ in meters, $\sigma$ in meters
- **Interpretable**: Average distance from mean
- **68-95-99.7 Rule** (for normal distribution):
  - ~68% within 1σ of mean
  - ~95% within 2σ
  - ~99.7% within 3σ

**Example**: Heights $[160, 165, 170, 175, 180]$ cm
$$\bar{x} = 170, \quad s = \sqrt{\frac{(160-170)^2 + ... + (180-170)^2}{4}} = 7.91 \text{ cm}$$

### 5. Coefficient of Variation (CV)

**Formula**:
$$CV = \frac{\sigma}{\mu} \times 100\%$$

**Properties**:
- **Dimensionless**: No units
- **Relative measure**: Compares variation across different scales
- **Useful for comparison**: Compare variability of different variables

**Example**: 
- Dataset A: $\mu=100$, $\sigma=10$ → $CV = 10\%$
- Dataset B: $\mu=10$, $\sigma=1$ → $CV = 10\%$
- Equal relative variation despite different scales

## Measures of Shape

### 1. Skewness

**Formula** (Pearson's moment coefficient):
$$\text{Skewness} = \frac{n}{(n-1)(n-2)}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3$$

**Interpretation**:
- **-0.5 to 0.5**: Approximately symmetric
- **0.5 to 1** or **-1 to -0.5**: Moderately skewed
- **> 1** or **< -1**: Highly skewed

**Impact on ML**:
- Many algorithms assume symmetry
- Skewed features may need transformation (log, Box-Cox)
- Affects outlier detection

**Transformation for Right-Skew**:
- Log: $\log(x)$ or $\log(x+1)$
- Square root: $\sqrt{x}$
- Cube root: $\sqrt[3]{x}$
- Box-Cox: $\frac{x^\lambda - 1}{\lambda}$

### 2. Kurtosis

**Formula** (excess kurtosis):
$$\text{Kurtosis} = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

**Interpretation**:
- **Kurtosis = 0**: Mesokurtic (normal-like tails)
- **Kurtosis > 0**: Leptokurtic (heavy tails, more outliers)
- **Kurtosis < 0**: Platykurtic (light tails, fewer outliers)

**Impact on ML**:
- High kurtosis indicates more outliers
- Affects outlier detection strategies
- Influences model robustness requirements

## Percentiles and Quantiles

**Percentile**: Value below which $p\%$ of data falls

**Quantile**: Generic term for division points
- Quartiles: 4 divisions (Q1, Q2, Q3)
- Deciles: 10 divisions (D1, D2, ..., D9)
- Percentiles: 100 divisions (P1, P2, ..., P99)

**Formula** (linear interpolation):
For percentile $p$ (0 to 100):
1. Sort data: $x_1 \leq x_2 \leq ... \leq x_n$
2. Compute position: $k = \frac{p}{100} \times (n - 1) + 1$
3. If $k$ is integer: $P_p = x_k$
4. If $k$ fractional: Interpolate between $x_{\lfloor k \rfloor}$ and $x_{\lceil k \rceil}$

**Key Percentiles**:
- P25 (Q1): First quartile
- P50 (Q2): Median
- P75 (Q3): Third quartile
- P90, P95, P99: Common outlier thresholds

## Five-Number Summary

Concise description of distribution:
1. **Minimum**: Smallest value
2. **Q1**: 25th percentile
3. **Median**: 50th percentile
4. **Q3**: 75th percentile
5. **Maximum**: Largest value

**Visualization**: Box plot (box-and-whisker plot)

```
    |-------|     |-------|
    min    Q1    Q2/Med  Q3    max
```

**Outlier Detection** (Tukey's fences):
- Lower fence: $Q_1 - 1.5 \times IQR$
- Upper fence: $Q_3 + 1.5 \times IQR$
- Values beyond fences considered outliers

## Correlation and Covariance

### Covariance

Measures how two variables vary together:

**Population Covariance**:
$$\sigma_{XY} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_X)(y_i - \mu_Y)$$

**Sample Covariance**:
$$s_{XY} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Interpretation**:
- **Positive**: Variables increase together
- **Negative**: One increases, other decreases
- **Zero**: No linear relationship
- **Scale-dependent**: Hard to interpret magnitude

### Correlation (Pearson's)

**Formula**:
$$r = \frac{s_{XY}}{s_X \cdot s_Y} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$$

**Properties**:
- **Range**: $-1 \leq r \leq 1$
- **Scale-invariant**: Dimensionless
- **r = 1**: Perfect positive linear relationship
- **r = -1**: Perfect negative linear relationship
- **r = 0**: No linear relationship (may still have nonlinear)

**Interpretation**:
- $|r| < 0.3$: Weak correlation
- $0.3 \leq |r| < 0.7$: Moderate correlation
- $|r| \geq 0.7$: Strong correlation

**Coefficient of Determination**: $R^2 = r^2$
- Proportion of variance explained
- Example: $r = 0.8$ → $R^2 = 0.64$ (64% of variance explained)

**Limitations**:
- Only measures linear relationships
- Sensitive to outliers
- Correlation ≠ Causation

## Missing Value Analysis

### Detecting Missing Values

**Counts**:
- Absolute count: Number of missing values per column
- Percentage: Proportion of missing values

**Pattern Analysis**:
- **MCAR** (Missing Completely at Random): No pattern
- **MAR** (Missing at Random): Depends on observed data
- **MNAR** (Missing Not at Random): Depends on unobserved data

### Implications for ML

- **Complete Case Analysis**: Drop rows with any missing (can bias)
- **Imputation**: Fill with statistics (mean, median, mode)
- **Indicator**: Create binary "is_missing" feature
- **Model-based**: Predict missing values

**Missingness Threshold**:
- < 5%: Generally safe to impute or drop
- 5-20%: Careful imputation needed
- > 20%: Consider dropping feature or specialized methods

## Data Type Analysis

### Continuous Variables

- **Integer**: Counts (number of purchases)
- **Float**: Measurements (temperature, price)
- **Ratio**: True zero exists (height, weight)
- **Interval**: No true zero (temperature in Celsius)

**Statistics**: Mean, median, variance, percentiles

### Categorical Variables

- **Nominal**: No order (color, country)
- **Ordinal**: Ordered (education level, rating)
- **Binary**: Two categories (yes/no, true/false)

**Statistics**: Mode, frequency counts, proportions

### DateTime Variables

- **Date**: Year-month-day
- **Time**: Hour-minute-second
- **Datetime**: Combined
- **Timezone**: Localized time

**Analysis**:
- Time trends
- Seasonality
- Day of week effects
- Time since reference

## Practical Workflow

### 1. Initial Inspection

- Shape: Number of rows and columns
- Data types: Categorical vs continuous
- Memory usage: Dataset size
- First/last rows: Preview data

### 2. Summary Statistics

- Central tendency: Mean, median, mode
- Dispersion: Std, variance, range, IQR
- Shape: Skewness, kurtosis
- Missing values: Count and percentage

### 3. Distribution Analysis

- Histograms: Shape of distribution
- Box plots: Outliers and spread
- QQ plots: Normality assessment
- Density plots: Smooth distribution

### 4. Relationship Analysis

- Correlation matrix: Pairwise correlations
- Scatter plots: Bivariate relationships
- Pair plots: Multiple scatter plots

### 5. Outlier Detection

- Z-score method: $|z| > 3$
- IQR method: Beyond fences
- Isolation Forest: Algorithm-based
- Visual inspection: Box plots, scatter plots

## Example Interpretation

**Dataset**: House Prices

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean Price | $250k | Average house price |
| Median Price | $220k | Middle value, lower than mean |
| Std Dev | $80k | Large variation in prices |
| Skewness | 1.2 | Right-skewed (expensive outliers) |
| Kurtosis | 2.5 | Heavy tails (more extreme values) |
| Min | $100k | Cheapest house |
| Max | $800k | Most expensive house |
| IQR | $120k | Middle 50% span $120k |
| Missing | 2% | Few missing values |

**Insights**:
- Distribution is right-skewed (median < mean)
- Presence of expensive outliers (high max, high kurtosis)
- Consider log transformation for modeling
- Missing values minimal, can impute or drop
- Large variability suggests diverse market

## Summary

Descriptive statistics are the foundation of data understanding in machine learning. They provide quantitative measures that reveal data characteristics, quality issues, and patterns. Mastering these concepts enables practitioners to:

**Before Modeling**:
- Understand data distributions and relationships
- Identify quality issues and outliers
- Select appropriate preprocessing techniques
- Validate assumptions

**During Modeling**:
- Choose suitable algorithms based on data properties
- Set appropriate hyperparameters
- Debug unexpected model behavior

**After Modeling**:
- Interpret model performance
- Analyze prediction errors
- Monitor production data drift

**Key Takeaways**:
- Use mean for symmetric data, median for skewed
- Variance/std measure spread in original units
- Skewness and kurtosis describe distribution shape
- Correlation measures linear relationships only
- Five-number summary provides robust overview
- Always visualize alongside statistics

Understanding descriptive statistics is not just about calculating numbers—it's about developing intuition for data behavior, which guides every subsequent decision in the machine learning pipeline.

---
