# Univariate Analysis

## General Idea

Univariate analysis is the examination of a single variable at a time to understand its distribution, central tendency, dispersion, and patterns. It's the simplest form of statistical analysis, focusing on describing and finding patterns in individual features without considering relationships with other variables. Univariate analysis is the first step in exploratory data analysis (EDA).

## Why Use Univariate Analysis?

1. **Individual Feature Understanding**: Learn each feature's characteristics independently
2. **Distribution Identification**: Determine if data is normal, skewed, uniform, etc.
3. **Outlier Detection**: Identify unusual values within single variables
4. **Data Quality Check**: Find missing values, impossible values, data entry errors
5. **Feature Selection**: Identify low-variance or uninformative features
6. **Transformation Planning**: Determine which features need scaling, encoding, or transformation
7. **Baseline Establishment**: Create simple models (mean prediction, mode classification)

## Role in Machine Learning

### Data Preprocessing

- **Identify transformations needed**: Log, square root, Box-Cox for skewed data
- **Detect constant/low-variance features**: Remove uninformative features
- **Plan scaling strategy**: Standardization vs normalization based on distribution
- **Handle missing values**: Choose imputation strategy per feature
- **Outlier treatment**: Decide on outlier removal or capping

### Feature Engineering

- **Binning decisions**: Convert continuous to categorical based on distribution
- **Encoding strategy**: Choose encoding method for categorical variables
- **Feature creation**: Generate polynomial features, interactions based on understanding
- **Target transformation**: Transform target variable if needed for regression

### Model Selection

- **Algorithm choice**: Some algorithms assume normality, others don't
- **Hyperparameter setting**: Initialize based on feature scales and distributions
- **Validation strategy**: Stratification for imbalanced categorical target

## Continuous Variable Analysis

### 1. Distribution Visualization

#### Histogram

**Purpose**: Show frequency distribution of continuous variable

**Components**:
- **Bins**: Intervals that divide range
- **Frequency**: Count of observations in each bin
- **Density**: Frequency normalized to sum to 1

**Bin Selection**:
Sturges' Rule: $k = \lceil \log_2(n) + 1 \rceil$

Rice Rule: $k = \lceil 2n^{1/3} \rceil$

Square Root Rule: $k = \lceil \sqrt{n} \rceil$

Where $n$ is sample size

**Interpretation**:
- **Shape**: Symmetric, skewed, bimodal
- **Center**: Where bulk of data lies
- **Spread**: Width of distribution
- **Gaps**: Missing ranges
- **Outliers**: Isolated bars far from main distribution

#### Density Plot (KDE - Kernel Density Estimation)

**Purpose**: Smooth estimate of probability density function

**Formula**:
$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x - x_i}{h}\right)$$

Where:
- $K$: Kernel function (usually Gaussian)
- $h$: Bandwidth (smoothing parameter)
- $n$: Sample size

**Bandwidth Selection**:
- Small $h$: Undersmoothed (too much detail, noise)
- Large $h$: Oversmoothed (misses features)
- Optimal: Scott's rule: $h = 1.06\sigma n^{-1/5}$

**Advantages over histogram**:
- Smooth representation
- No bin boundary artifacts
- Better for comparing distributions

#### Box Plot

**Purpose**: Show five-number summary and outliers

**Components**:
- **Box**: IQR (Q1 to Q3)
- **Line in box**: Median
- **Whiskers**: Extend to non-outlier extremes
- **Points**: Outliers beyond whiskers

**Whisker Calculation**:
- Lower: $\max(\text{min}, Q_1 - 1.5 \times IQR)$
- Upper: $\min(\text{max}, Q_3 + 1.5 \times IQR)$

**Interpretation**:
- **Box width**: IQR (spread of middle 50%)
- **Box position**: Central tendency
- **Whisker length**: Range excluding outliers
- **Symmetry**: Median position in box
- **Outliers**: Points beyond whiskers

### 2. Statistical Measures

#### Central Tendency

**Mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$
- Best for symmetric distributions
- Sensitive to outliers

**Median**: Middle value when sorted
- Robust to outliers
- Best for skewed distributions

**Mode**: Most frequent value
- Useful for discrete or categorical
- May not exist or be multiple

#### Dispersion

**Variance**: $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$

**Standard Deviation**: $s = \sqrt{s^2}$
- Same units as data
- Measures average distance from mean

**Range**: $\max - \min$
- Simple but outlier-sensitive

**IQR**: $Q_3 - Q_1$
- Robust measure of spread
- Describes middle 50%

**Coefficient of Variation**: $CV = \frac{s}{\bar{x}} \times 100\%$
- Relative dispersion
- Compares variability across scales

#### Shape

**Skewness**: $\gamma_1 = \frac{n}{(n-1)(n-2)}\sum\left(\frac{x_i - \bar{x}}{s}\right)^3$

Interpretation:
- $\gamma_1 > 0$: Right-skewed (tail to right)
- $\gamma_1 < 0$: Left-skewed (tail to left)
- $|\gamma_1| < 0.5$: Approximately symmetric

**Kurtosis**: $\gamma_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum\left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$

Interpretation:
- $\gamma_2 > 0$: Heavy tails (more outliers)
- $\gamma_2 < 0$: Light tails (fewer outliers)
- $\gamma_2 \approx 0$: Normal-like tails

### 3. Normality Tests

Many ML algorithms assume normality. Testing helps decide if transformation needed.

#### Visual Tests

**Q-Q Plot (Quantile-Quantile)**:
- Plot sample quantiles vs theoretical normal quantiles
- Straight line indicates normality
- Deviations show where distribution differs

**Interpretation**:
- **S-curve**: Skewed distribution
- **Bow-shape**: Heavy or light tails
- **Straight line**: Normal distribution

#### Statistical Tests

**Shapiro-Wilk Test**:
- Null hypothesis: Data is normally distributed
- Test statistic $W$, ranges 0 to 1
- Closer to 1 indicates normality
- $p < 0.05$: Reject normality

**Formula**: Based on correlation between data and normal scores

**Kolmogorov-Smirnov Test**:
- Compares empirical CDF to normal CDF
- Maximum distance: $D = \max|F_n(x) - F_0(x)|$
- $p < 0.05$: Reject normality

**Anderson-Darling Test**:
- Weighted version of K-S test
- More sensitive to tails
- Critical values depend on sample size

### 4. Outlier Detection

**Z-Score Method**:
$$z_i = \frac{x_i - \bar{x}}{s}$$

Outlier if $|z_i| > 3$ (or 2.5)

**Advantages**: Simple, assumes normality
**Disadvantages**: Sensitive to outliers (mean and std affected)

**Modified Z-Score** (using median):
$$M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}$$

Where $MAD = \text{median}(|x_i - \tilde{x}|)$ (median absolute deviation)

Outlier if $|M_i| > 3.5$

**More robust** than standard z-score

**IQR Method**:
- Lower fence: $Q_1 - 1.5 \times IQR$
- Upper fence: $Q_3 + 1.5 \times IQR$
- Outliers beyond fences

**Extreme outliers**: Use $3 \times IQR$ instead

**Percentile Method**:
- Mark bottom 1% and top 1% as outliers
- Or use 5th and 95th percentiles

**Flexible** threshold based on domain knowledge

## Categorical Variable Analysis

### 1. Frequency Distribution

**Frequency Table**:
| Category | Count | Percentage |
|----------|-------|------------|
| A        | 50    | 25%        |
| B        | 120   | 60%        |
| C        | 30    | 15%        |

**Cumulative Frequency**: Running total of frequencies

### 2. Visualization

#### Bar Chart

- **x-axis**: Categories
- **y-axis**: Frequency or percentage
- **Order**: Alphabetical, by frequency, or natural order

**Horizontal bar chart**: Better for many categories or long names

#### Pie Chart

- **Slices**: Proportional to category frequency
- **Angle**: $\theta_i = 360° \times \frac{f_i}{\sum f}$

**Use**: When showing parts of whole (< 7 categories)
**Avoid**: Hard to compare similar-sized slices

#### Count Plot

- Combined bar chart with counts
- Useful for comparing multiple categorical variables

### 3. Statistical Measures

**Mode**: Most frequent category

**Cardinality**: Number of unique categories
- High cardinality: Many unique values (problematic for encoding)
- Low cardinality: Few unique values

**Entropy**: Measure of uncertainty/diversity
$$H = -\sum_{i=1}^{k}p_i\log_2(p_i)$$

Where $p_i$ is proportion of category $i$

- High entropy: Even distribution (max: $\log_2(k)$)
- Low entropy: Concentrated in few categories (min: 0)

**Gini Impurity**: Alternative to entropy
$$G = 1 - \sum_{i=1}^{k}p_i^2$$

- Range: [0, 1-1/k]
- Used in decision trees

### 4. Balance Analysis

**Imbalance Ratio**: $\frac{\text{majority class count}}{\text{minority class count}}$

**Interpretation**:
- < 1.5: Well balanced
- 1.5 - 3: Slight imbalance
- 3 - 10: Moderate imbalance
- > 10: Severe imbalance

**Implications for ML**:
- Imbalanced data biases models toward majority
- Need stratified sampling, resampling, or class weights

## Distribution Types

### Common Distributions

#### Normal (Gaussian)

**PDF**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$

**Characteristics**:
- Symmetric, bell-shaped
- Mean = Median = Mode
- 68-95-99.7 rule
- Completely described by $\mu$ and $\sigma$

**Examples**: Height, measurement errors, test scores (with large $n$)

#### Uniform

**PDF**: $f(x) = \frac{1}{b-a}$ for $a \leq x \leq b$

**Characteristics**:
- Constant probability across range
- All values equally likely
- Mean: $\frac{a+b}{2}$
- Variance: $\frac{(b-a)^2}{12}$

**Examples**: Random number generators, dice rolls

#### Exponential

**PDF**: $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$

**Characteristics**:
- Right-skewed
- Describes time between events
- Memoryless property
- Mean: $\frac{1}{\lambda}$

**Examples**: Time until failure, time between arrivals

#### Log-Normal

**PDF**: If $\log(X) \sim N(\mu, \sigma^2)$, then $X$ is log-normal

**Characteristics**:
- Right-skewed
- Positive values only
- Appears after multiplicative processes

**Examples**: Income, stock prices, file sizes

#### Binomial

**PMF**: $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$

**Characteristics**:
- Discrete: $k = 0, 1, ..., n$
- Sum of Bernoulli trials
- Mean: $np$, Variance: $np(1-p)$

**Examples**: Number of successes in $n$ trials

#### Poisson

**PMF**: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$

**Characteristics**:
- Discrete, non-negative integers
- Models count data
- Mean = Variance = $\lambda$

**Examples**: Number of events in fixed time/space

## Transformations

### When to Transform

- **Skewness**: Make distribution more symmetric
- **Heteroscedasticity**: Stabilize variance
- **Linearity**: Create linear relationships
- **Normality**: Meet algorithm assumptions

### Common Transformations

#### Log Transformation

$$y = \log(x) \quad \text{or} \quad y = \log(x + 1)$$

**Effect**: Reduces right skew
**Requirement**: $x > 0$ (or $x \geq 0$ with +1)
**Interpretation**: Changes multiplicative to additive

**Example**: Income $[30k, 50k, 100k, 500k]$
- Highly skewed
- After log: More symmetric

#### Square Root

$$y = \sqrt{x}$$

**Effect**: Reduces right skew (milder than log)
**Requirement**: $x \geq 0$
**Use**: Count data, Poisson-distributed

#### Box-Cox Transformation

$$y = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\
\log(x) & \lambda = 0
\end{cases}$$

**Optimal $\lambda$**: Chosen to maximize normality (e.g., via likelihood)

**Common values**:
- $\lambda = -1$: Reciprocal
- $\lambda = -0.5$: Reciprocal square root
- $\lambda = 0$: Log
- $\lambda = 0.5$: Square root
- $\lambda = 1$: No transformation

#### Yeo-Johnson Transformation

Extension of Box-Cox for all real numbers (including negative)

$$y = \begin{cases}
\frac{(x+1)^\lambda - 1}{\lambda} & \lambda \neq 0, x \geq 0\\
\log(x+1) & \lambda = 0, x \geq 0\\
-\frac{(-x+1)^{2-\lambda} - 1}{2-\lambda} & \lambda \neq 2, x < 0\\
-\log(-x+1) & \lambda = 2, x < 0
\end{cases}$$

### Inverse Transformations

For left-skewed data:
- Reflect: $y = \max(x) - x$
- Then apply standard transformation

## Practical Workflow

### 1. Initial Inspection

```python
# Shape
print(data.shape)
# Data types
print(data.dtypes)
# Basic info
print(data.info())
```

### 2. Summary Statistics

```python
# Continuous variables
print(data.describe())
# Include categorical
print(data.describe(include='all'))
# Specific statistics
print(data['column'].mean(), data['column'].median())
```

### 3. Visualization

```python
# Histogram
data['column'].hist(bins=30)
# Density plot
data['column'].plot(kind='density')
# Box plot
data.boxplot(column='column')
```

### 4. Distribution Tests

```python
# Shapiro-Wilk
from scipy.stats import shapiro
stat, p = shapiro(data['column'])
# Q-Q plot
from scipy.stats import probplot
probplot(data['column'], dist="norm", plot=plt)
```

### 5. Outlier Detection

```python
# Z-score
from scipy.stats import zscore
z = zscore(data['column'])
outliers = data[abs(z) > 3]
# IQR
Q1, Q3 = data['column'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = data[(data['column'] < Q1 - 1.5*IQR) | (data['column'] > Q3 + 1.5*IQR)]
```

### 6. Categorical Analysis

```python
# Value counts
print(data['category'].value_counts())
# Proportions
print(data['category'].value_counts(normalize=True))
# Bar plot
data['category'].value_counts().plot(kind='bar')
```

## Example Analysis

**Dataset**: Customer Age

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 42.5 | Average age |
| Median | 40.0 | Middle age, lower than mean |
| Std | 12.3 | Moderate variability |
| Min | 18 | Youngest customer |
| Max | 85 | Oldest customer |
| Skewness | 0.8 | Moderately right-skewed |
| Kurtosis | 0.3 | Slightly heavy-tailed |

**Visual**: Histogram shows right skew with peak at 35-45

**Actions**:
1. **Check outliers**: Ages > 75 are rare but valid
2. **Consider binning**: Create age groups (18-25, 26-35, etc.)
3. **Transformation**: Log transform might improve symmetry for modeling
4. **Feature engineering**: Create "is_senior" (age > 65) binary feature

## Summary

Univariate analysis is the essential first step in understanding data for machine learning. By analyzing each feature individually, we gain insights into:

**Distribution characteristics**: Shape, center, spread
**Data quality**: Missing values, outliers, errors
**Transformation needs**: Skewness, scale, encoding
**Feature properties**: Variance, cardinality, balance

**Key Practices**:
- Always visualize (histograms, box plots, bar charts)
- Compute summary statistics (mean, median, std, percentiles)
- Test assumptions (normality, outliers)
- Document findings (inform preprocessing decisions)

**For Continuous Variables**:
- Use histograms, density plots, box plots
- Check for skewness, outliers, normality
- Consider transformations (log, Box-Cox)

**For Categorical Variables**:
- Use bar charts, count plots
- Check cardinality, balance, mode
- Plan encoding strategy

Univariate analysis builds intuition about individual features, which is crucial before examining relationships (bivariate/multivariate analysis) and building predictive models. It ensures data quality, guides preprocessing, and prevents costly mistakes in later stages of the ML pipeline.

---
