# Mathematical Foundations of Machine Learning
## A Comprehensive Theory Guide for Days 15-68

![Nano Banana visualization of ML mathematical landscape](https://via.placeholder.com/1200x600?text=Mathematical+ML+Landscape+by+Nano+Banana)

---

# TABLE OF CONTENTS

## PART I: DATA FUNDAMENTALS (Days 15-23)
1. CSV Parsing Theory & Algorithms
2. JSON/SQL Relational Algebra
3. API Design & HTTP Protocol
4. Web Scraping & DOM Theory
5. Descriptive Statistics & Probability
6. Univariate Analysis & Distribution Theory
7. Bivariate Analysis & Correlation
8. Data Profiling Algorithms
9. Visualization Theory

## PART II: PREPROCESSING (Days 24-35)
10. Standardization (Z-Score) Mathematics
11. Normalization (Min-Max) Theory
12. Ordinal Encoding & Monotonicity
13. One-Hot Encoding & Linear Independence
14. Column Transformer Composition
15. Pipeline Theory & DAGs
16. Function Transformers & Monotonic Functions
17. Power Transforms (Box-Cox, Yeo-Johnson)
18. Binning & Discretization Theory
19. Mixed Variable Parsing
20. Time Series Feature Engineering
21. Missing Data Theory (MCAR/MAR/MNAR)

## PART III: IMPUTATION (Days 36-40)
22. Mean/Median Imputation Statistics
23. Categorical Imputation (Mode)
24. Missing Indicators & Information Theory
25. KNN Imputation & Distance Metrics
26. MICE (Iterative Imputation) Algorithm

## PART IV: OUTLIER DETECTION (Days 42-44)
27. Z-Score Method & Gaussian Assumptions
28. IQR Method & Robust Statistics
29. Percentile-Based Methods

## PART V: FEATURE ENGINEERING (Days 45-47)
30. Feature Construction Algebra
31. PCA: Eigenvalue Decomposition Theory

## PART VI: REGRESSION (Days 48-57)
32. Simple Linear Regression (OLS)
33. Regression Metrics (MAE, MSE, RMSE, R²)
34. Multiple Linear Regression (Matrix Form)
35. Gradient Descent Optimization Theory
36. Batch/Mini-Batch/Stochastic GD
37. Polynomial Regression & Basis Functions
38. Ridge Regression (L2 Regularization)
39. Lasso Regression (L1 Regularization)
40. ElasticNet (Combined Penalties)

## PART VII: CLASSIFICATION (Days 58-66, 68)
41. Logistic Regression & Log-Odds
42. Classification Metrics (Precision, Recall, F1)
43. Multiclass Logistic (Softmax)
44. Random Forest Theory
45. AdaBoost Algorithm
46. Stacking & Blending Ensembles

---

# PART I: DATA FUNDAMENTALS

## 1. CSV Parsing Theory

### Formal Grammar
```
file ::= record (NEWLINE record)*
record ::= field (DELIMITER field)*  
field ::= escaped | non_escaped
escaped ::= QUOTE (TEXT | DELIMITER | QUOTE QUOTE)* QUOTE
```

### Memory Model
For $n$ rows, $m$ columns, average cell size $s$ bytes:
$$M = n \times m \times s \times \alpha$$
where $\alpha \in [1.2, 2.0]$ is overhead factor.

### Chunking Algorithm
Process file in blocks of size $C$:
- $C = \min(\frac{\\text{RAM}}{4}, 10^6)$ rows
- Aggregation: $\text{Result} = \bigoplus_{i=1}^P f(\text{Chunk}_i)$

**Example - Mean in Chunks**:
- Chunk 1: $\bar{x}_1 = 50, n_1=100K$
- Chunk 2: $\bar{x}_2 = 52, n_2=100K$
- Global: $\bar{x} = \frac{50 \times 100K + 52 \times 100K}{200K} = 51$

---

## 2. Relational Algebra (SQL)

### Operations
1. **Selection** ($\sigma$): $\sigma_{\text{condition}}(R)$
2. **Projection** ($\pi$): $\pi_{\text{cols}}(R)$
3. **Join** ($\bowtie$): $R \bowtie_{\\text{key}} S$

### Normalization Theory
**1NF**: Atomic values (no arrays)
**2NF**: No partial dependencies
**3NF**: No transitive dependencies

**Example**:
Bad: `{Student: "John", Courses: "Math,Physics"}`
Good: Two tables with FK relationship

---

## 3. Descriptive Statistics

### Measures of Central Tendency
- **Mean**: $\mu = \frac{1}{n}\sum_{i=1}^n x_i$
- **Median**: $M = x_{(\lceil n/2 \rceil)}$ (sorted)
- **Mode**: $\arg\max_x f(x)$ (most frequent)

### Measures of Dispersion
- **Variance**: $\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2$
- **Standard Deviation**: $\sigma = \sqrt{\sigma^2}$
- **IQR**: $Q_3 - Q_1$

### Moments
- **Skewness**: $\gamma_1 = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3}$
  - $\gamma_1 > 0$: Right-skewed (tail on right)
  - $\gamma_1 < 0$: Left-skewed
- **Kurtosis**: $\gamma_2 = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4}$
  - Measures tail heaviness

**Example**:
Data: [1,2,2,3,3,3,4,4,5]
- Mean: 3
- Median: 3
- Mode: 3
- Skew: 0 (symmetric)

---

## 4. Correlation & Covariance

### Covariance
$$\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)] = \frac{1}{n}\sum (x_i - \bar{x})(y_i - \bar{y})$$

### Pearson Correlation
$$r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$$

**Properties**:
- $r \in [-1, 1]$
- $r=1$: Perfect positive linear
- $r=-1$: Perfect negative linear
- $r=0$: No linear relationship (but may have non-linear!)

**Example**:
| X | Y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |

$r = 1$ (perfect line $Y = 2X$)

---

# PART II: PREPROCESSING

## 5. Standardization (Z-Score)

### Formula
$$z = \frac{x - \mu}{\sigma}$$

### Properties
After standardization:
- $\mathbb{E}[Z] = 0$
- $\text{Var}(Z) = 1$

### Why It Helps
**Gradient Descent**: Error surface becomes spherical instead of elongated.

**Contour Before**:
```
   |\      <- Long narrow valley
   | \     <- GD oscillates
   |  \
```

**Contour After**:
```
   ( )     <- Circular
   ( )     <- GD converges fast
```

### Distance Metrics
Euclidean distance with unscaled features:
$$d = \sqrt{(x_1^{(a)} - x_1^{(b)})^2 + (x_2^{(a)} - x_2^{(b)})^2}$$

If $x_1 \in [0,1]$ and $x_2 \in [0,10000]$, then $x_2$ dominates!

After scaling, both contribute equally.

---

## 6. Normalization (Min-Max)

### Formula
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Result: $x' \in [0,1]$

### Generalized Form
$$x' = a + \frac{(x - x_{\min})(b-a)}{x_{\max} - x_{\min}}$$
for range $[a, b]$.

### Outlier Sensitivity
**Problem**: Single outlier stretches range.

**Example**:
Data: [10, 20, 30, 1000]
- $x_{\min} = 10, x_{\max} = 1000$
- First three values squashed into [0, 0.02]!

**Solution**: Winsorization (cap at percentiles)

---

## 7. One-Hot Encoding Mathematics

### Linear Independence
For $k$ categories, creating $k$ binary columns:
$$\mathbf{1}_{\{c_1\}} + \mathbf{1}_{\{c_2\}} + \cdots + \mathbf{1}_{\{c_k\}} = \mathbf{1}$$
(always sum to 1)

This is **linear dependence**: $\text{rank}(\mathbf{X}) < k$

**Solution**: Drop one column (dummy variable trap).

### Example
Categories: {Red, Green, Blue}

**Before Drop**:
| Red | Green | Blue |
|-----|-------|------|
| 1   | 0     | 0    |
| 0   | 1     | 0    |
| 0   | 0     | 1    |

Columns sum to [1,1,1].

**After Drop** (drop Blue):
| Red | Green |
|-----|-------|
| 1   | 0     |
| 0   | 1     |
| 0   | 0     |

Now linearly independent. Blue is implicit: $\text{Blue} = 1 - \text{Red} - \text{Green}$.

---

## 8. Power Transforms

### Box-Cox
$$y(\lambda) = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\
\ln x & \lambda = 0
\end{cases}$$

**Constraint**: $x > 0$

**Special Cases**:
- $\lambda = 1$: $y = x - 1$ (linear shift)
- $\lambda = 0.5$: $y \approx \sqrt{x}$
- $\lambda = 0$: $y = \ln x$
- $\lambda = -1$: $y = 1/x$

### Yeo-Johnson
Extends to $x \leq 0$:
$$y = \begin{cases}
\frac{(x+1)^\lambda - 1}{\lambda} & x \geq 0, \lambda \neq 0 \\
\ln(x+1) & x \geq 0, \lambda = 0 \\
-\frac{(-x+1)^{2-\lambda} - 1}{2-\lambda} & x < 0, \lambda \neq 2 \\
-\ln(-x+1) & x < 0, \lambda = 2
\end{cases}$$

### Maximum Likelihood
Find $\lambda$ that maximizes:
$$\mathcal{L}(\lambda) = -\frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum y_i^2 + (\lambda - 1)\sum \ln x_i$$

Iterative search over $\lambda \in [-5, 5]$.

---

# PART III: IMPUTATION

## 9. Missing Data Taxonomy

### MCAR (Missing Completely At Random)
$$P(M=1|X,Y) = P(M=1)$$
Missingness independent of data.

**Example**: Survey responses lost due to server crash.

### MAR (Missing At Random)
$$P(M=1|X,Y) = P(M=1|X)$$
Missingness depends on observed variables.

**Example**: High-income respondents skip salary question (but we observe their education level).

### MNAR (Missing Not At Random)
$$P(M=1|Y)$$
Missingness depends on the missing value itself.

**Example**: People with very low income refuse to report it.

---

## 10. KNN Imputation

### Algorithm
1. For row $i$ with missing value in feature $j$:
2. Compute distance to all other rows using observed features:
   $$d(i, k) = \sqrt{\sum_{j' \neq j} (x_{ij'} - x_{kj'})^2}$$
3. Find $K$ nearest neighbors: $\mathcal{N}_K(i)$
4. Impute: 
   $$\hat{x}_{ij} = \frac{1}{K}\sum_{k \in \mathcal{N}_K(i)} x_{kj}$$

### Weighted Variant
$$\hat{x}_{ij} = \frac{\sum_{k \in \mathcal{N}_K} w_k x_{kj}}{\sum_{k \in \mathcal{N}_K} w_k}$$
where $w_k = \frac{1}{d(i,k)}$ (closer neighbors have more weight).

---

## 11. MICE (Iterative Imputer)

### Algorithm
1. **Initialize**: Fill all missing with mean
2. **Iterate**:
   - For each column $j$ with missing:
     - Set $\mathbf{y} = X_j$ (observed values)
     - Set $\mathbf{X}_{-j}$ = all other columns
     - Train model: $f: \mathbf{X}_{-j} \to \mathbf{y}$
     - Predict missing: $\hat{X}_j = f(\mathbf{X}_{-j})$
3. **Repeat** until convergence (typically 5-10 iterations)

### Convergence Criterion
$$|\hat{X}^{(t)} - \hat{X}^{(t-1)}|_F < \epsilon$$
Frobenius norm of change.

---

# PART IV: OUTLIER DETECTION

## 12. Z-Score Method

### Definition
$$z_i = \frac{x_i - \mu}{\sigma}$$

**Rule**: Flag if $|z_i| > k$ (typically $k=3$)

### Assumption
Data is **Gaussian**. If not, many false positives.

### Example
Data: [10, 12, 11, 13, 100]
- $\mu = 29.2, \sigma = 35.6$
- $z_{100} = \frac{100-29.2}{35.6} = 1.99$ (NOT flagged!)

Problem: Outlier inflates $\sigma$, masking itself.

**Solution**: Use **robust statistics** (MAD).

---

## 13. IQR Method

### Formula
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
where $\text{IQR} = Q_3 - Q_1$.

### Why 1.5?
Tukey's original choice. For Gaussian:
- Captures 99.3% of data
- ~0.7% flagged as outliers

### Advantage
**Robust**: Not affected by extreme values (quartiles are percentiles).

---

# PART V: REGRESSION

## 14. Simple Linear Regression (OLS)

### Model
$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

### Objective
Minimize SSR:
$$\mathcal{L}(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$$

### Closed Form Solution
$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$$
$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

### Geometric Interpretation
$\hat{y} = \mathbf{X}\boldsymbol{\beta}$ is the **orthogonal projection** of $\mathbf{y}$ onto $\text{col}(\mathbf{X})$.

---

## 15. Multiple Linear Regression

### Matrix Form
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
where:
- $\mathbf{y} \in \mathbb{R}^n$
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ (includes intercept column)
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$

### Normal Equations
$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

### Solution
$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Condition**: $\mathbf{X}^T\mathbf{X}$ must be invertible (no multicollinearity).

---

## 16. Gradient Descent

### Objective
Minimize $\mathcal{L}(\boldsymbol{\beta})$.

### Update Rule
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \nabla \mathcal{L}(\boldsymbol{\beta}^{(t)})$$

where $\alpha$ is learning rate.

### For Linear Regression
$$\nabla \mathcal{L} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

### Convergence
Guaranteed if $\alpha < \frac{2}{\lambda_{\max}(\mathbf{X}^T\mathbf{X})}$.

---

## 17. Ridge Regression (L2)

### Objective
$$\mathcal{L} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2$$

### Solution
$$\boldsymbol{\beta}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

### Effect
- **Shrinks** coefficients toward zero
- **Stabilizes** when $\mathbf{X}^T\mathbf{X}$ is near-singular

---

## 18. Lasso Regression (L1)

### Objective
$$\mathcal{L} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

### Solution
No closed form. Use:
- Coordinate Descent
- Proximal Gradient

### Effect
- **Sparsity**: Sets some $\beta_j$ exactly to zero
- **Feature Selection**: Automatic

---

# PART VI: CLASSIFICATION

## 19. Logistic Regression

### Model
$$P(y=1|x) = \sigma(\beta_0 + \beta_1 x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

### Log-Odds (Logit)
$$\log\frac{P(y=1)}{P(y=0)} = \beta_0 + \beta_1 x$$

### Loss Function
$$\mathcal{L} = -\sum [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

(Cross-Entropy)

---

## 20. Multiclass (Softmax)

### Model
$$P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
where $z_k = \beta_k^T x$.

---

## 21. Random Forest

### Bagging
Train $B$ trees on bootstrap samples.

### Prediction
$$\hat{y} = \frac{1}{B}\sum_{b=1}^B f_b(x)$$

### Out-of-Bag Error
For each sample $i$, average predictions from trees where $i$ was NOT in training set.

---

## 22. AdaBoost

### Algorithm
1. Initialize weights: $w_i = 1/n$
2. For $t = 1$ to $T$:
   - Train weak learner on weighted data
   - Compute error: $\epsilon_t = \sum_{i: y_i \neq \hat{y}_i} w_i$
   - Compute $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
   - Update weights: $w_i \leftarrow w_i \exp(\alpha_t \mathbb{1}\{y_i \neq \hat{y}_i\})$
3. Final: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$

---

# CONCLUSION

This document provides the mathematical scaffolding for understanding ML. Each concept builds on prior foundations—from basic statistics to advanced ensemble methods. Mastery requires both theoretical understanding and practical application.

The journey from CSV parsing to AdaBoost is one of increasing abstraction and sophistication, but the core principles remain:
1. **Formalize** the problem mathematically
2. **Optimize** a well-defined objective
3. **Validate** assumptions and diagnose failures
4. **Iterate** toward better models

---

*Generated by Nano Banana Model - Mathematical ML Foundations*
