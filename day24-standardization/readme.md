# Standardization (Z-Score Normalization)

## General Idea

Standardization is a feature scaling technique that transforms data to have a mean of zero and a standard deviation of one. Also known as Z-score normalization, it rescales features by removing the mean and scaling to unit variance. This transformation converts data into a standard normal distribution (if the original was normal), making features comparable regardless of their original scales.

## Why Use Standardization?

1. **Scale Independence**: Makes features with different units comparable
2. **Gradient Descent Optimization**: Faster convergence in gradient-based algorithms
3. **Distance-Based Algorithms**: Essential for KNN, K-Means, SVM with RBF kernel
4. **Regularization**: L1/L2 penalties equally weighted across features
5. **Neural Networks**: Improves training stability and speed
6. **Outlier Preservation**: Maintains outlier information (unlike normalization)
7. **Algorithm Assumptions**: Many algorithms perform better with standardized data

## Role in Machine Learning

### Critical For

- **Support Vector Machines (SVM)**: Features with larger scales dominate the decision boundary
- **K-Nearest Neighbors (KNN)**: Distance calculations biased by scale
- **K-Means Clustering**: Euclidean distance sensitive to scale
- **Principal Component Analysis (PCA)**: Variance is scale-dependent
- **Gradient Descent**: Features on different scales cause slow, zigzag convergence
- **Ridge/Lasso Regression**: Regularization penalty scale-dependent
- **Neural Networks**: Accelerates convergence, prevents saturation

### Not Required For

- **Tree-Based Models**: Decision trees, Random Forest, Gradient Boosting
  - Split points are scale-invariant
- **Naive Bayes**: Works on probabilities, not distances

### Beneficial But Optional

- **Linear/Logistic Regression** (without regularization): Doesn't change predictions, aids interpretation

## Mathematical Formula

### Standardization (Z-Score)

For a feature $X$ with values $x_1, x_2, ..., x_n$:

$$z_i = \frac{x_i - \mu}{\sigma}$$

Where:
- $z_i$: Standardized value
- $x_i$: Original value
- $\mu$: Mean of the feature
- $\sigma$: Standard deviation of the feature

**Mean**:
$$\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$$

**Standard Deviation**:
$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$$

Or with Bessel's correction (sample std):
$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

### Properties After Standardization

**Mean**: $E[Z] = 0$

Proof:
$$E[Z] = E\left[\frac{X - \mu}{\sigma}\right] = \frac{1}{\sigma}(E[X] - \mu) = \frac{1}{\sigma}(\mu - \mu) = 0$$

**Variance**: $Var(Z) = 1$

Proof:
$$Var(Z) = Var\left(\frac{X - \mu}{\sigma}\right) = \frac{1}{\sigma^2}Var(X - \mu) = \frac{\sigma^2}{\sigma^2} = 1$$

**Standard Deviation**: $\sigma_Z = 1$

### Interpretation of Z-Score

$z_i$ represents how many standard deviations $x_i$ is from the mean:

- $z = 0$: Value equals mean
- $z = 1$: One standard deviation above mean
- $z = -1$: One standard deviation below mean
- $|z| > 3$: Potential outlier (for normal distribution)

**68-95-99.7 Rule** (for normal distribution):
- ~68% of data within $z \in [-1, 1]$
- ~95% of data within $z \in [-2, 2]$
- ~99.7% of data within $z \in [-3, 3]$

## Example Calculation

**Original Data**: $X = [10, 20, 30, 40, 50]$

**Step 1**: Calculate mean
$$\mu = \frac{10 + 20 + 30 + 40 + 50}{5} = \frac{150}{5} = 30$$

**Step 2**: Calculate standard deviation
$$\sigma = \sqrt{\frac{(10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2}{5}}$$
$$= \sqrt{\frac{400 + 100 + 0 + 100 + 400}{5}} = \sqrt{\frac{1000}{5}} = \sqrt{200} \approx 14.14$$

**Step 3**: Standardize each value
$$z_1 = \frac{10 - 30}{14.14} \approx -1.41$$
$$z_2 = \frac{20 - 30}{14.14} \approx -0.71$$
$$z_3 = \frac{30 - 30}{14.14} = 0$$
$$z_4 = \frac{40 - 30}{14.14} \approx 0.71$$
$$z_5 = \frac{50 - 30}{14.14} \approx 1.41$$

**Standardized Data**: $Z \approx [-1.41, -0.71, 0, 0.71, 1.41]$

**Verification**:
- Mean$(Z) \approx 0$ ✓
- Std$(Z) \approx 1$ ✓

## Standardization in Practice

### Training Phase

1. **Compute statistics** on training data:
   - $\mu_{train} = \frac{1}{n_{train}}\sum x_i^{train}$
   - $\sigma_{train} = \sqrt{\frac{1}{n_{train}}\sum (x_i^{train} - \mu_{train})^2}$

2. **Transform training data**:
   - $z_i^{train} = \frac{x_i^{train} - \mu_{train}}{\sigma_{train}}$

3. **Save parameters**: $\mu_{train}$, $\sigma_{train}$ for later use

### Test/Production Phase

**Critical**: Use training statistics, not test statistics

$$z_i^{test} = \frac{x_i^{test} - \mu_{train}}{\sigma_{train}}$$

**Why?**: 
- Prevents data leakage
- Ensures consistent transformation
- Test data may have different distribution

**Consequence**: Test data may not have mean=0, std=1

## Standardization vs Other Scaling Methods

### Standardization vs Min-Max Normalization

| Aspect | Standardization | Min-Max Normalization |
|--------|----------------|----------------------|
| **Formula** | $z = \frac{x - \mu}{\sigma}$ | $x' = \frac{x - \min}{\max - \min}$ |
| **Range** | Unbounded (typically -3 to 3) | [0, 1] or custom [a, b] |
| **Outlier Sensitivity** | Robust (outliers affect σ less) | Very sensitive (outliers affect range) |
| **Mean** | 0 | Depends on data |
| **Std** | 1 | Depends on data |
| **Use Case** | Normal-ish data, when outliers important | Bounded features, neural networks |
| **Preserves Shape** | Yes | Yes |

### Standardization vs Robust Scaling

**Robust Scaler**:
$$x' = \frac{x - \text{median}}{IQR}$$

Where $IQR = Q_3 - Q_1$

**Comparison**:
- **Robust**: Less affected by outliers (uses median, IQR)
- **Standardization**: Uses mean, std (more affected by outliers)
- **Use robust when**: Data has extreme outliers

### When to Choose Standardization

**Choose Standardization when**:
- Data is approximately normally distributed
- Outliers are informative (not errors)
- Algorithm uses Euclidean distance or gradient descent
- Want interpretable z-scores (standard deviations from mean)
- Features span very different ranges

**Choose Min-Max Normalization when**:
- Data is uniformly distributed or bounded
- Need specific range [0, 1] for algorithm
- No significant outliers
- Neural networks with sigmoid/tanh activation

**Choose Robust Scaling when**:
- Data has many outliers
- Distribution is heavily skewed
- Outliers are errors or noise

## Impact on Machine Learning Algorithms

### Distance-Based Algorithms

**K-Nearest Neighbors (KNN)**:

Without standardization, features with larger scales dominate distance:

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

**Example**:
- Feature 1 (age): 20-80, range = 60
- Feature 2 (income): 20k-200k, range = 180k

Income dominates distance calculation even if age is more predictive

**After standardization**: Both features contribute equally to distance

### Gradient-Based Algorithms

**Gradient Descent**:

Cost function landscape depends on feature scales:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

**Without standardization**:
- Elongated, elliptical contours
- Gradient points away from minimum
- Slow, zigzag path
- Requires careful learning rate tuning
- May not converge

**With standardization**:
- Circular contours
- Gradient points toward minimum
- Fast, direct path
- Easier learning rate selection
- Faster convergence

**Convergence Speed**: Can be 100x faster with standardization

### Regularized Models

**Ridge Regression**:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}\theta_j^2$$

**Lasso Regression**:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}|\theta_j|$$

**Without standardization**:
- Penalty $\lambda$ affects features unequally
- Features with larger scales get smaller coefficients
- Regularization biased toward features with smaller scales

**With standardization**:
- Penalty equally weighted across features
- Fair comparison of feature importance
- Coefficients represent importance (same scale)

### Support Vector Machines

**SVM with RBF Kernel**:

$$K(x, x') = \exp\left(-\gamma||x - x'||^2\right)$$

**Without standardization**:
- Kernel dominated by large-scale features
- Hyperparameter $\gamma$ hard to tune (scale-dependent)
- Poor generalization

**With standardization**:
- All features contribute equally to kernel
- Easier $\gamma$ selection
- Better performance

### Neural Networks

**Activation Functions**:

**Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
**Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

**Without standardization**:
- Large inputs cause saturation (gradient ≈ 0)
- Vanishing gradient problem
- Slow training
- Poor weight initialization

**With standardization**:
- Inputs in linear region of activation
- Non-zero gradients
- Faster training
- Better weight updates

**Weight Initialization**: Assumes standardized inputs for Xavier/He initialization

## Outlier Handling with Standardization

### Outlier Detection Using Z-Score

**Rule**: Flag as outlier if $|z| > 3$ (or 2.5)

**Rationale**: For normal distribution, P($|z| > 3$) ≈ 0.3%

**Example**:
- Data: [10, 12, 14, 16, 18, 100]
- Mean = 28.33, Std = 35.73
- Z-score for 100: $z = \frac{100 - 28.33}{35.73} \approx 2.01$

May not detect outlier if std is inflated by outlier itself

**Alternative**: Modified Z-score using median and MAD (more robust)

### Standardization Preserves Outliers

Unlike min-max normalization:
- Outliers compress main data into small range [0, 0.1]
- Outlier mapped to 1.0

Standardization:
- Outliers get large z-scores (e.g., z = 5)
- Main data stays in reasonable range (z ∈ [-2, 2])
- Outlier information preserved

**Trade-off**:
- Preserves outlier info: Good if outliers meaningful
- Outliers affect mean/std: Bad if outliers are noise

## Practical Considerations

### Multiple Features

Standardize each feature independently:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Where:
- $i$: Sample index
- $j$: Feature index
- $\mu_j$, $\sigma_j$: Mean and std of feature $j$

**Do NOT** standardize across samples (rows)

### Constant Features

If $\sigma_j = 0$ (all values identical):
- Division by zero issue
- Feature is uninformative
- **Solution**: Remove feature or set to 0

### Sparse Data

For sparse matrices (many zeros):
- Standardization densifies matrix (mean ≠ 0)
- Memory and computation increase
- **Alternative**: Use `with_mean=False` (only scale by std)

$$z = \frac{x}{\sigma}$$

Preserves sparsity but mean ≠ 0

### Categorical Variables

**Do NOT standardize categorical variables**:
- One-hot encoded: Already 0/1, standardization meaningless
- Ordinal encoded: Distances between categories arbitrary

**Standardize only continuous features**

### Time Series Data

**Caution**:
- Training statistics may not apply to test period
- Non-stationary data: Mean/std change over time
- **Alternative**: Rolling standardization, differencing

## When NOT to Standardize

1. **Tree-Based Models**: Random Forest, XGBoost, Decision Trees
   - Scale-invariant splits
   - Standardization unnecessary (doesn't hurt, but no benefit)

2. **Already Scaled Data**: 
   - Features already on comparable scales (e.g., all percentages)

3. **Interpretability Priority**:
   - Original units more meaningful to stakeholders
   - Example: Age, income for business users

4. **Algorithm Outputs Probabilities**:
   - Naive Bayes (uses probabilities)

5. **Extremely Skewed Data**:
   - Consider log transform first, then standardize
   - Or use robust scaling

## Inverse Transformation

To convert standardized values back to original scale:

$$x = z \cdot \sigma + \mu$$

**Use case**: Interpret model predictions in original units

**Example**:
- Predicted z-score: 1.5
- Original mean: 100, std: 15
- Predicted value: $x = 1.5 \times 15 + 100 = 122.5$

## Summary

Standardization (Z-score normalization) is a fundamental preprocessing technique that transforms features to have mean=0 and std=1. It's essential for many machine learning algorithms, particularly those based on distance metrics or gradient descent.

**Key Points**:

**Mathematical Foundation**:
- Formula: $z = \frac{x - \mu}{\sigma}$
- Results in mean=0, std=1
- Preserves distribution shape

**When to Use**:
- Gradient-based optimization (faster convergence)
- Distance-based algorithms (equal feature contribution)
- Regularized models (fair penalty application)
- Features on vastly different scales

**Advantages**:
- Makes features comparable
- Improves algorithm performance
- Faster training
- Better weight initialization
- Interpretable z-scores

**Considerations**:
- Use training statistics for test data
- Sensitive to outliers (use robust scaling if needed)
- Not needed for tree-based models
- Don't standardize categorical variables
- Preserve sparsity if needed

**Best Practice**:
- Always fit on training data, transform both train and test
- Include in pipeline for automatic handling
- Document scaling parameters for production deployment
- Consider robust alternatives for outlier-heavy data

Standardization is a simple yet powerful technique that often makes the difference between a model that struggles to converge and one that performs optimally. Understanding when and how to apply it is fundamental to successful machine learning practice.

---

**Video Link**: https://youtu.be/1Yw9sC0PNwY
