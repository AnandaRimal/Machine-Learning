# Regularized Linear Models

## Overview and Core Concept

Regularized Linear Models are extensions of ordinary linear regression that add a penalty term to the loss function to prevent overfitting and handle multicollinearity. The fundamental idea is to constrain the magnitude of coefficients, trading some bias for reduced variance, which often improves generalization to new data.

Standard linear regression minimizes only the sum of squared errors (SSE). Regularized models add a penalty for large coefficients:

$$\text{Loss} = \text{SSE} + \text{Penalty}$$

This penalty discourages complex models with large coefficients, implementing Occam's Razor: among models with similar training performance, prefer simpler ones. The three main types are Ridge (L2), Lasso (L1), and Elastic Net (combination of L1 and L2).

Regularization is crucial when:
- Number of predictors approaches or exceeds number of observations ($p \approx n$ or $p > n$)
- Predictors are highly correlated (multicollinearity)
- Model overfits training data
- You need automatic feature selection (Lasso)
- Seeking better generalization over perfect training fit

## Why Use Regularized Linear Models?

1. **Combat Overfitting**: Prevents model from fitting noise in training data
2. **Handle Multicollinearity**: Stabilizes coefficients when predictors are correlated
3. **Feature Selection**: Lasso automatically sets irrelevant coefficients to zero
4. **High Dimensions**: Works when $p > n$ (ordinary regression fails)
5. **Improved Generalization**: Better test performance through bias-variance tradeoff
6. **Numerical Stability**: Better-conditioned matrix inversion
7. **Interpretability**: Sparse solutions (Lasso) easier to interpret
8. **No Manual Selection**: Automatic handling of many features
9. **Theoretical Guarantees**: Statistical learning theory supports approach
10. **Versatility**: Applicable to regression and classification

## Mathematical Formulas

### Ordinary Least Squares (OLS) - Baseline

**Objective**:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 = \min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$$

**Solution**:

$$\hat{\boldsymbol{\beta}}_{OLS} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Problem**: When $\mathbf{X}^T\mathbf{X}$ is near-singular (multicollinearity) or $p > n$, solution is unstable or doesn't exist.

### Ridge Regression (L2 Regularization)

**Objective**:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}\beta_j^2$$

**Matrix form**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2_2$$

Where:
- $\lambda \geq 0$: Regularization parameter (hyperparameter)
- $||\boldsymbol{\beta}||^2_2 = \sum_{j=1}^{p}\beta_j^2$: L2 norm squared

**Closed-Form Solution**:

$$\hat{\boldsymbol{\beta}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Key Properties**:
- **Shrinkage**: All coefficients shrunk toward zero
- **Never exactly zero**: All features retained (unless $\lambda = \infty$)
- **Smooth penalty**: Differentiable everywhere
- **Invertibility**: $\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$ always invertible for $\lambda > 0$

**Effect of $\lambda$**:
- $\lambda = 0$: OLS (no regularization)
- $\lambda \to \infty$: $\hat{\boldsymbol{\beta}} \to 0$ (maximum shrinkage)
- **Small $\lambda$**: Slight shrinkage, similar to OLS
- **Large $\lambda$**: Heavy shrinkage, high bias

### Lasso Regression (L1 Regularization)

**Objective**:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}|\beta_j|$$

**Matrix form**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$$

Where:
- $||\boldsymbol{\beta}||_1 = \sum_{j=1}^{p}|\beta_j|$: L1 norm

**No Closed-Form Solution**: Must use iterative algorithms:
- Coordinate descent
- LARS (Least Angle Regression)
- Proximal gradient methods

**Key Properties**:
- **Sparsity**: Sets some coefficients exactly to zero
- **Feature Selection**: Automatically selects important features
- **Non-differentiable**: At $\beta_j = 0$ (absolute value kink)
- **Interpretability**: Sparse solutions easier to explain

**Soft-Thresholding Operator** (for coordinate descent):

$$\hat{\beta}_j = \text{sign}(\tilde{\beta}_j)\max(|\tilde{\beta}_j| - \lambda, 0)$$

Where $\tilde{\beta}_j$ is the unpenalized estimate.

### Elastic Net

**Objective**: Combines L1 and L2 penalties:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda_1||\boldsymbol{\beta}||_1 + \lambda_2||\boldsymbol{\beta}||^2_2$$

**Alternative Parameterization**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda\left[\alpha||\boldsymbol{\beta}||_1 + (1-\alpha)||\boldsymbol{\beta}||^2_2\right]$$

Where:
- $\lambda$: Overall regularization strength
- $\alpha \in [0,1]$: Mixing parameter
  - $\alpha = 0$: Pure Ridge
  - $\alpha = 1$: Pure Lasso
  - $0 < \alpha < 1$: Elastic Net

**Key Properties**:
- **Best of both**: Sparsity from L1, stability from L2
- **Grouped selection**: Tends to select/drop correlated features together
- **Better than Lasso**: When $p > n$ or features highly correlated

### Constraint Formulation (Equivalent)

Regularized regression can be expressed as constrained optimization:

**Ridge**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 \quad \text{subject to} \quad ||\boldsymbol{\beta}||^2_2 \leq t$$

**Lasso**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 \quad \text{subject to} \quad ||\boldsymbol{\beta}||_1 \leq t$$

Where $t$ corresponds to specific $\lambda$ (smaller $t$ = larger $\lambda$).

**Geometric Interpretation**:
- **Ridge**: Constraint region is a sphere (smooth)
- **Lasso**: Constraint region is a diamond (corners at axes)
- **Corners**: Where coefficients become exactly zero in Lasso

## Detailed Worked Example

**Problem**: Predict house price with 3 correlated features

**Data** (5 houses):

| Size $(x_1)$ | Age $(x_2)$ | Rooms $(x_3)$ | Price $(y)$ |
|--------------|-------------|---------------|-------------|
| 1500         | 10          | 5             | 250         |
| 1800         | 5           | 6             | 320         |
| 2000         | 15          | 5             | 280         |
| 2200         | 8           | 7             | 360         |
| 2500         | 3           | 8             | 400         |

**Note**: Size and Rooms highly correlated (larger houses have more rooms)

### Step 1: Standardize Features

**Why**: Regularization is scale-dependent. Must standardize first.

**Mean and Standard Deviation**:
- Size: $\mu_1 = 2000$, $\sigma_1 = 360$
- Age: $\mu_2 = 8.2$, $\sigma_2 = 4.5$
- Rooms: $\mu_3 = 6.2$, $\sigma_3 = 1.3$

**Standardized features**:

$$x_{ij}^{std} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

**Design Matrix** (standardized, intercept removed for simplicity):

$$\mathbf{X}_{std} = \begin{bmatrix} -1.39 & 0.40 & -0.92 \\ -0.56 & -0.71 & -0.15 \\ 0 & 1.51 & -0.92 \\ 0.56 & -0.04 & 0.62 \\ 1.39 & -1.16 & 1.38 \end{bmatrix}$$

**Target** (centered): $\mathbf{y}_c = [250, 320, 280, 360, 400] - 322 = [-72, -2, -42, 38, 78]$

### Step 2: Ordinary Least Squares

**Calculate** $(\mathbf{X}^T\mathbf{X})^{-1}$:

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 4 & -0.5 & 3.8 \\ -0.5 & 4 & -1.2 \\ 3.8 & -1.2 & 4 \end{bmatrix}$$

**Determinant**: Small (near-singular due to correlation between $x_1$ and $x_3$)

**OLS Solution** (unstable):

$$\hat{\boldsymbol{\beta}}_{OLS} \approx \begin{bmatrix} 85 \\ -15 \\ -60 \end{bmatrix}$$

**Problem**: Large, unstable coefficients. Negative coefficient for Rooms counterintuitive (multicollinearity effect).

### Step 3: Ridge Regression

**Choose** $\lambda = 1$

**Ridge Solution**:

$$\hat{\boldsymbol{\beta}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

$$\mathbf{X}^T\mathbf{X} + \mathbf{I} = \begin{bmatrix} 5 & -0.5 & 3.8 \\ -0.5 & 5 & -1.2 \\ 3.8 & -1.2 & 5 \end{bmatrix}$$

**Solution**:

$$\hat{\boldsymbol{\beta}}_{Ridge} \approx \begin{bmatrix} 42 \\ -12 \\ -25 \end{bmatrix}$$

**Observation**: Coefficients shrunk (smaller magnitude), more stable.

### Step 4: Lasso Regression

**Choose** $\lambda = 5$

**Lasso Solution** (computed via coordinate descent):

$$\hat{\boldsymbol{\beta}}_{Lasso} \approx \begin{bmatrix} 38 \\ 0 \\ 0 \end{bmatrix}$$

**Observation**: Age and Rooms coefficients set to exactly zero! Only Size retained (automatic feature selection).

**Interpretation**: Among correlated features (Size, Rooms), Lasso picked the most informative one.

### Step 5: Elastic Net

**Choose** $\lambda = 5, \alpha = 0.5$

**Elastic Net Solution**:

$$\hat{\boldsymbol{\beta}}_{EN} \approx \begin{bmatrix} 35 \\ -8 \\ -10 \end{bmatrix}$$

**Observation**: Compromise between Ridge and Lasso. Some shrinkage, partial sparsity.

### Step 6: Compare Predictions

**New house**: Size=1900, Age=7, Rooms=6

**Standardized**: $x_1^{std} = -0.28, x_2^{std} = -0.27, x_3^{std} = -0.15$

**OLS**: $\hat{y} = 322 + 85(-0.28) - 15(-0.27) - 60(-0.15) = 322 - 24 + 4 + 9 = 311$

**Ridge**: $\hat{y} = 322 + 42(-0.28) - 12(-0.27) - 25(-0.15) = 322 - 12 + 3 + 4 = 317$

**Lasso**: $\hat{y} = 322 + 38(-0.28) + 0 + 0 = 322 - 11 = 311$

**Elastic Net**: $\hat{y} = 322 + 35(-0.28) - 8(-0.27) - 10(-0.15) = 322 - 10 + 2 + 2 = 316$

**On test data**: Ridge and Elastic Net likely generalize better than OLS.

## Bias-Variance Tradeoff

### Conceptual Understanding

**Ordinary Least Squares**:
- **Bias**: Low (unbiased estimator)
- **Variance**: High (especially with multicollinearity or $p \approx n$)
- **Total Error**: Can be high due to variance

**Regularized Models**:
- **Bias**: Higher (coefficients shrunk/biased toward zero)
- **Variance**: Lower (stable estimates)
- **Total Error**: Often lower (variance reduction > bias increase)

### Mathematical Expression

**Mean Squared Error Decomposition**:

$$\text{MSE}(\hat{\boldsymbol{\beta}}) = \text{Bias}^2(\hat{\boldsymbol{\beta}}) + \text{Var}(\hat{\boldsymbol{\beta}}) + \sigma^2$$

**OLS**:
- $\text{Bias}(\hat{\boldsymbol{\beta}}_{OLS}) = 0$
- $\text{Var}(\hat{\boldsymbol{\beta}}_{OLS}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$ (can be very large)

**Ridge**:
- $\text{Bias}(\hat{\boldsymbol{\beta}}_{Ridge}) \neq 0$ (biased toward zero)
- $\text{Var}(\hat{\boldsymbol{\beta}}_{Ridge})$ smaller than OLS

**Optimal $\lambda$**: Minimizes total MSE by balancing bias and variance

### Regularization Path

As $\lambda$ increases:
- Coefficients shrink toward zero
- Training error increases
- Test error first decreases (variance reduction), then increases (bias dominates)
- Optimal $\lambda$ at minimum test error

## Choosing Regularization Parameter $\lambda$

### Cross-Validation (Recommended)

**K-Fold CV**:

1. Divide data into $K$ folds
2. For each $\lambda$ in grid:
   - For each fold $k$:
     - Train on other $K-1$ folds
     - Validate on fold $k$
   - Average validation error across folds
3. Select $\lambda$ with minimum average error

**Typical K**: 5 or 10

**Lambda Grid**: Logarithmic scale, e.g., $[10^{-3}, 10^{-2}, ..., 10^2, 10^3]$

### Information Criteria

**AIC** (Akaike Information Criterion):

$$\text{AIC} = n\log(\text{SSE}/n) + 2 \cdot \text{df}$$

**BIC** (Bayesian Information Criterion):

$$\text{BIC} = n\log(\text{SSE}/n) + \log(n) \cdot \text{df}$$

Where $\text{df}$ (effective degrees of freedom) depends on $\lambda$.

**For Ridge**: $\text{df}(\lambda) = \text{tr}[\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T]$

### Theoretical Bounds

For high-dimensional settings, theory suggests:

$$\lambda \propto \sigma\sqrt{\frac{\log p}{n}}$$

Where $\sigma$ is noise level.

### Validation Curve

Plot training and validation error vs. $\log(\lambda)$:
- **Too small $\lambda$**: Overfitting (high validation error)
- **Too large $\lambda$**: Underfitting (high training and validation error)
- **Optimal**: Valley in validation curve

## Assumptions

### Inherited from Linear Regression

1. **Linearity**: $E[y|\mathbf{X}] = \mathbf{X}\boldsymbol{\beta}$
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant error variance
4. **Normality**: Errors normally distributed (for inference)

### Additional for Regularization

5. **Standardization**: Features should be standardized (same scale)
   - **Why**: Penalty depends on magnitude of coefficients
   - **Effect**: Unscaled features with larger values penalized more

6. **Many Features or Multicollinearity**: Regularization most beneficial when:
   - $p$ is large relative to $n$
   - Features are correlated

7. **Sparse Truth** (for Lasso): True model has many zero coefficients
   - Lasso recovers true model under certain conditions

## Advantages

### Ridge Regression

1. **Handles Multicollinearity**: Stabilizes correlated features
2. **Always Solvable**: Works when $p > n$
3. **Closed-Form**: Fast computation
4. **Smooth Shrinkage**: All features contribute (no abrupt selection)
5. **Grouped Effects**: Correlated features get similar coefficients
6. **Numerical Stability**: Well-conditioned matrix inversion
7. **Differentiable**: Smooth optimization landscape

### Lasso Regression

1. **Feature Selection**: Automatic, sets coefficients exactly to zero
2. **Interpretability**: Sparse models easier to understand
3. **Handles High Dimensions**: Effective when $p \gg n$
4. **Automatic Relevance**: Identifies important features
5. **Computational Efficiency**: For prediction, only need non-zero features
6. **Embedded Method**: Selection and estimation simultaneously

### Elastic Net

1. **Best of Both Worlds**: Sparsity + stability
2. **Grouped Selection**: Selects correlated features together
3. **Superior to Lasso**: When $p > n$ or strong correlations
4. **Flexible**: $\alpha$ parameter tunes L1/L2 balance

### General Advantages

8. **Better Generalization**: Lower test error than OLS
9. **Prevents Overfitting**: Built-in complexity control
10. **Theoretical Guarantees**: Statistical learning theory support

## Disadvantages

### Ridge Regression

1. **No Feature Selection**: Retains all features (interpretability suffers)
2. **Biased Estimates**: Coefficients shrunk (not unbiased)
3. **Hyperparameter Tuning**: Requires selecting $\lambda$

### Lasso Regression

1. **Arbitrary Selection**: Among correlated features, may pick any one
2. **Unstable**: Small data changes can change selected features
3. **Grouped Variables**: Tends to select only one from correlated group
4. **$n < p$ Limit**: Selects at most $n$ features (even if more relevant)
5. **Non-Differentiable**: Optimization more complex than Ridge

### Elastic Net

1. **Two Hyperparameters**: Must tune both $\lambda$ and $\alpha$
2. **Computational Cost**: Slower than Ridge, more complex than Lasso

### General Disadvantages

7. **Requires Standardization**: Extra preprocessing step
8. **Interpretation**: Bias makes coefficient interpretation less direct
9. **Linearity**: Still assumes linear relationships
10. **Hyperparameter Dependence**: Performance sensitive to $\lambda$ choice

## When to Use Regularized Linear Models

### Ridge Regression

**Use When**:
1. **Multicollinearity**: Features highly correlated
2. **Keep All Features**: Theoretical reasons to include all
3. **Prediction Focus**: Don't need feature selection
4. **Smooth Shrinkage**: Want all features to contribute
5. **Grouped Features**: Related features should all be included

**Example Scenarios**:
- Genomics: Many correlated gene expressions, all potentially relevant
- Economics: Correlated macroeconomic indicators
- Image pixels: Neighboring pixels correlated

### Lasso Regression

**Use When**:
1. **Feature Selection Needed**: Want to identify important features
2. **Sparse Truth**: Believe many coefficients are truly zero
3. **Interpretability**: Need simple, explainable model
4. **High Dimensions**: $p \gg n$
5. **Computational Constraints**: Fewer features for deployment

**Example Scenarios**:
- Text analysis: Most words irrelevant for specific task
- Biomarker discovery: Find small set of diagnostic genes
- Sensor networks: Identify critical sensors

### Elastic Net

**Use When**:
1. **$p > n$**: More features than observations
2. **Correlated Predictors**: Multiple related features
3. **Grouped Selection**: Want to keep/drop correlated features together
4. **Lasso Instability**: Lasso selection too variable
5. **Best Performance**: Willing to tune two parameters for optimal results

**Example Scenarios**:
- Genomics with $p > n$: More genes than samples
- Marketing: Many correlated customer behavior features
- Climate modeling: Correlated environmental variables

## When NOT to Use

1. **Few Features, Large n**: If $p$ small and $n$ large, OLS sufficient
2. **No Multicollinearity**: Well-conditioned $\mathbf{X}^T\mathbf{X}$, OLS fine
3. **Non-Linear Relationships**: Need non-linear models (polynomial, trees, etc.)
4. **Unbiased Estimates Required**: Statistical inference demands unbiased coefficients
5. **Feature Importance Ranking**: Coefficient magnitudes can be misleading in regularized models
6. **Categorical Target**: Use regularized logistic regression instead
7. **Complex Interactions**: Tree-based methods may be better
8. **Time Series**: Autocorrelation requires specialized methods
9. **Perfect Separation** (classification): Regularization may harm if clear decision boundary

## Related Methods

### Regularization Variants

**Adaptive Lasso**
- Weighted L1 penalty: $\sum_{j=1}^{p}w_j|\beta_j|$
- Different penalties per coefficient
- Oracle properties (asymptotically selects true model)

**Group Lasso**
- Penalty on groups of coefficients
- Selects/drops entire groups together
- Useful for categorical variables (one-hot encoded)

**Fused Lasso**
- Penalizes differences between adjacent coefficients
- Useful for ordered features (time, spatial)

**SCAD** (Smoothly Clipped Absolute Deviation)
- Non-convex penalty
- Reduced bias for large coefficients
- Oracle properties

**Bridge Regression**
- $L_q$ penalty: $\sum_{j=1}^{p}|\beta_j|^q$ for $0 < q < 2$
- Generalizes Ridge ($q=2$) and Lasso ($q=1$)

### Related Techniques

**Principal Component Regression (PCR)**
- Project features to principal components
- Regress on top PCs
- Dimension reduction, addresses multicollinearity

**Partial Least Squares (PLS)**
- Find directions maximizing covariance with target
- Supervised dimension reduction

**Bayesian Linear Regression**
- Prior on coefficients (Gaussian for Ridge, Laplace for Lasso)
- Regularization emerges from Bayesian framework

**Robust Regression**
- Resistant to outliers
- Can combine with regularization

### Modern Extensions

**Regularized GLMs**
- Logistic regression: Ridge/Lasso for classification
- Poisson regression: For count data

**Regularized Neural Networks**
- L1/L2 penalties on weights
- Dropout as implicit regularization

**Kernel Ridge Regression**
- Non-linear version using kernel trick

## Summary

Regularized Linear Models extend ordinary least squares by adding penalties on coefficient magnitudes, providing crucial tools for high-dimensional data, multicollinearity, and overfitting prevention.

**Three Main Types**:

**Ridge (L2)**: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2_2$
- Shrinks all coefficients
- No feature selection
- Best for multicollinearity

**Lasso (L1)**: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$
- Sparse solutions (sets coefficients to zero)
- Automatic feature selection
- Best for high-dimensional sparse models

**Elastic Net**: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda[\alpha||\boldsymbol{\beta}||_1 + (1-\alpha)||\boldsymbol{\beta}||^2_2]$
- Combines L1 and L2
- Grouped feature selection
- Best for $p > n$ with correlations

**Core Principle**: Trade bias for variance to improve generalization

**Key Requirement**: Standardize features before applying regularization

**Hyperparameter Selection**: Use cross-validation to choose optimal $\lambda$ (and $\alpha$ for Elastic Net)

**When to Use**:
- High-dimensional data ($p$ large, $p > n$)
- Multicollinearity present
- Overfitting observed
- Feature selection desired (Lasso/Elastic Net)

**Advantages**: Better generalization, handles multicollinearity, feature selection (Lasso), works with $p > n$

**Limitations**: Introduces bias, requires hyperparameter tuning, still assumes linearity

Regularized linear models are essential tools in modern statistical learning, bridging classical regression and modern machine learning. They provide principled approach to complexity control while maintaining much of linear regression's interpretability and computational efficiency.
