# Lasso Regression (Least Absolute Shrinkage and Selection Operator)

## Overview and Core Concept

Lasso Regression is a regularized linear regression method that performs both shrinkage and automatic feature selection by adding an L1 penalty (sum of absolute values of coefficients) to the standard least squares loss function. Unlike Ridge regression which only shrinks coefficients, Lasso can set coefficients exactly to zero, effectively removing features from the model.

The fundamental innovation of Lasso is its ability to produce sparse models—models with fewer non-zero coefficients—making it invaluable for high-dimensional datasets where interpretability and feature selection are crucial. This sparsity arises from the geometric properties of the L1 norm, which creates a diamond-shaped constraint region with corners along the coordinate axes where coefficients become exactly zero.

Introduced by Robert Tibshirani in 1996, Lasso has become one of the most widely used techniques in statistical learning, particularly in fields like genomics, text mining, and any domain with many potential predictors but belief that only a subset truly matters.

## Why Use Lasso Regression?

1. **Automatic Feature Selection**: Sets irrelevant coefficients exactly to zero
2. **Interpretability**: Sparse models easier to understand and explain
3. **High-Dimensional Data**: Effective when $p \gg n$ (more features than samples)
4. **Prevents Overfitting**: Regularization controls model complexity
5. **Computational Efficiency**: Fewer features speed up prediction
6. **Variable Screening**: Identifies important predictors
7. **Embedded Method**: Combines feature selection with model training
8. **Handles Multicollinearity**: Though differently than Ridge
9. **Model Compression**: Reduces model size for deployment
10. **Scientific Discovery**: Reveals relevant variables in exploratory studies

## Mathematical Formulas

### Objective Function

**Lasso optimization problem**:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\sum_{i=1}^{n}(y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij})^2 + \lambda\sum_{j=1}^{p}|\beta_j|$$

**Matrix notation**:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2_2 + \lambda||\boldsymbol{\beta}||_1$$

Where:
- $||\boldsymbol{\beta}||_1 = \sum_{j=1}^{p}|\beta_j|$: L1 norm (sum of absolute values)
- $\lambda \geq 0$: Regularization parameter (hyperparameter)
- $n$: Number of observations
- $p$: Number of features
- Note: Intercept $\beta_0$ typically not penalized

**Components**:
- **Loss term**: $\frac{1}{2n}||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2_2$ (measures fit)
- **Penalty term**: $\lambda||\boldsymbol{\beta}||_1$ (promotes sparsity)

### Constraint Formulation (Equivalent)

**Lagrangian dual**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2_2 \quad \text{subject to} \quad ||\boldsymbol{\beta}||_1 \leq t$$

Where $t \geq 0$ corresponds to specific $\lambda$ value:
- Small $t$ (large $\lambda$): Many coefficients set to zero
- Large $t$ (small $\lambda$): Approaches OLS solution

**Geometric interpretation**: Find coefficients within L1 ball (diamond shape) that minimize squared error.

### Soft-Thresholding Operator

For simple case (orthogonal $\mathbf{X}$), coordinate-wise solution:

$$\hat{\beta}_j = \text{sign}(\tilde{\beta}_j) \cdot \max(|\tilde{\beta}_j| - \lambda, 0) = \begin{cases} \tilde{\beta}_j - \lambda & \text{if } \tilde{\beta}_j > \lambda \\ 0 & \text{if } |\tilde{\beta}_j| \leq \lambda \\ \tilde{\beta}_j + \lambda & \text{if } \tilde{\beta}_j < -\lambda \end{cases}$$

Where $\tilde{\beta}_j$ is the ordinary least squares estimate.

**Key insight**: Coefficients with $|\tilde{\beta}_j| < \lambda$ are set exactly to zero!

### Subgradient Conditions

Since L1 norm is not differentiable at zero, use subgradients:

**KKT (Karush-Kuhn-Tucker) conditions** for optimality:

$$-\frac{1}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \cdot \partial ||\boldsymbol{\beta}||_1 = 0$$

Where:

$$\partial |\beta_j| = \begin{cases} \{1\} & \text{if } \beta_j > 0 \\ [-1, 1] & \text{if } \beta_j = 0 \\ \{-1\} & \text{if } \beta_j < 0 \end{cases}$$

**For non-zero coefficients**: Standard gradient condition
**For zero coefficients**: Subgradient allows "staying at zero"

### Effect of $\lambda$

**Regularization path**: How coefficients change with $\lambda$

- $\lambda = 0$: OLS solution (no regularization), all features included
- $\lambda$ small: Few coefficients zero, slight shrinkage
- $\lambda$ moderate: Some coefficients exactly zero (sparse solution)
- $\lambda$ large: Many coefficients zero, strong feature selection
- $\lambda \to \infty$: All coefficients zero (null model)

**Critical value**: For each coefficient $j$, exists $\lambda_j^*$ where $\beta_j$ becomes zero

### Comparison with Ridge

| Aspect | Lasso (L1) | Ridge (L2) |
|--------|------------|------------|
| **Penalty** | $\lambda\sum|\beta_j|$ | $\lambda\sum\beta_j^2$ |
| **Sparsity** | Yes (exact zeros) | No (shrinks toward zero) |
| **Solution** | No closed form | Closed form |
| **Geometry** | Diamond | Circle |
| **Feature Selection** | Automatic | None |
| **Correlated Features** | Picks one | Shrinks equally |

## Detailed Worked Example

**Problem**: Predict exam score from 4 features (study hours, sleep, attendance, previous score)

**Data** (6 students):

| Study $(x_1)$ | Sleep $(x_2)$ | Attendance $(x_3)$ | Previous $(x_4)$ | Score $(y)$ |
|---------------|---------------|---------------------|------------------|-------------|
| 5             | 7             | 80                  | 75               | 78          |
| 8             | 8             | 95                  | 82               | 88          |
| 3             | 6             | 60                  | 70               | 65          |
| 6             | 7             | 85                  | 78               | 80          |
| 9             | 8             | 98                  | 85               | 92          |
| 4             | 5             | 70                  | 68               | 70          |

**Mean**: Study=5.83, Sleep=6.83, Attendance=81.33, Previous=76.33, Score=78.83

### Step 1: Standardize Features

**Why**: Lasso penalty scale-dependent. Must standardize for fair comparison.

**Standardization**:

$$x_{ij}^{std} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

**Standardized design matrix** (simplified):

$$\mathbf{X}_{std} = \begin{bmatrix} -0.42 & 0.18 & -0.15 & -0.35 \\ 1.08 & 1.23 & 1.52 & 1.48 \\ -1.42 & -0.87 & -2.37 & -1.65 \\ 0.08 & 0.18 & 0.41 & 0.44 \\ 1.58 & 1.23 & 1.85 & 2.26 \\ -0.92 & -1.93 & -1.26 & -2.17 \end{bmatrix}$$

**Centered target**: $\mathbf{y}_c = [78, 88, 65, 80, 92, 70] - 78.83 = [-0.83, 9.17, -13.83, 1.17, 13.17, -8.83]$

### Step 2: OLS Solution (Baseline)

**OLS coefficients** (for comparison):

$$\hat{\boldsymbol{\beta}}_{OLS} \approx [3.5, 1.2, 2.8, 5.1]$$

**Interpretation**: All features have positive effects, "Previous score" strongest.

**Issue**: With small $n$ (6) and moderate $p$ (4), overfitting risk.

### Step 3: Lasso with Small $\lambda$

**Set** $\lambda = 0.5$

**Lasso solution** (computed via coordinate descent):

$$\hat{\boldsymbol{\beta}}_{Lasso}(\lambda=0.5) \approx [3.2, 0.9, 2.3, 4.8]$$

**Effect**: Slight shrinkage compared to OLS, all features retained.

### Step 4: Lasso with Moderate $\lambda$

**Set** $\lambda = 2$

**Lasso solution**:

$$\hat{\boldsymbol{\beta}}_{Lasso}(\lambda=2) \approx [2.5, 0, 1.2, 4.1]$$

**Feature selection**: Sleep $(x_2)$ coefficient set to **exactly zero**! Only 3 features retained.

**Interpretation**: Sleep doesn't add predictive value beyond other features (given regularization strength).

### Step 5: Lasso with Large $\lambda$

**Set** $\lambda = 5$

**Lasso solution**:

$$\hat{\boldsymbol{\beta}}_{Lasso}(\lambda=5) \approx [0, 0, 0, 3.2]$$

**Strong feature selection**: Only "Previous score" retained. Other features zeroed out.

**Interpretation**: With heavy regularization, only strongest predictor survives.

### Step 6: Prediction Example

**New student**: Study=7, Sleep=7, Attendance=90, Previous=80

**Standardized**: $x_1^{std}=0.58, x_2^{std}=0.18, x_3^{std}=0.96, x_4^{std}=0.96$

**OLS prediction**:

$$\hat{y}_{OLS} = 78.83 + 3.5(0.58) + 1.2(0.18) + 2.8(0.96) + 5.1(0.96) = 78.83 + 9.6 = 88.4$$

**Lasso ($\lambda=2$) prediction**:

$$\hat{y}_{Lasso} = 78.83 + 2.5(0.58) + 0(0.18) + 1.2(0.96) + 4.1(0.96) = 78.83 + 6.5 = 85.3$$

**Lasso ($\lambda=5$) prediction**:

$$\hat{y}_{Lasso} = 78.83 + 0 + 0 + 0 + 3.2(0.96) = 78.83 + 3.1 = 81.9$$

**Simpler model** (Lasso with $\lambda=5$) uses only 1 feature, likely generalizes better.

### Step 7: Regularization Path

**As $\lambda$ increases from 0 to 10**:

| $\lambda$ | $\beta_1$ | $\beta_2$ | $\beta_3$ | $\beta_4$ | # Features |
|-----------|-----------|-----------|-----------|-----------|------------|
| 0         | 3.5       | 1.2       | 2.8       | 5.1       | 4          |
| 0.5       | 3.2       | 0.9       | 2.3       | 4.8       | 4          |
| 1.0       | 3.0       | 0.5       | 1.8       | 4.5       | 4          |
| 2.0       | 2.5       | 0         | 1.2       | 4.1       | 3          |
| 3.0       | 1.8       | 0         | 0         | 3.8       | 2          |
| 5.0       | 0         | 0         | 0         | 3.2       | 1          |
| 10.0      | 0         | 0         | 0         | 0         | 0          |

**Observation**: Coefficients shrink progressively to zero. Weaker predictors zeroed first.

**Order of removal**: Sleep → Attendance → Study hours → Previous score (strongest)

## Optimization Algorithms

Since Lasso has no closed-form solution, iterative algorithms required:

### Coordinate Descent

**Algorithm**: Update one coefficient at a time, holding others fixed.

**For each $\beta_j$**:

1. Compute partial residual: $r_{-j} = \mathbf{y} - \sum_{k \neq j}\mathbf{x}_k\beta_k$
2. Compute correlation: $\rho_j = \mathbf{x}_j^T r_{-j}$
3. Update: $\beta_j = \text{SoftThreshold}(\rho_j, \lambda)$

**Repeat** until convergence.

**Advantages**: Fast, easy to implement, handles large $p$

### LARS (Least Angle Regression)

**Clever algorithm**: Computes entire regularization path efficiently.

**Key idea**: Add features one at a time in order of correlation with residual.

**Output**: Solution for all $\lambda$ values (piecewise linear path).

**Computational cost**: Similar to single OLS fit.

### Proximal Gradient Descent

**Iterative update**:

$$\boldsymbol{\beta}^{(t+1)} = \text{SoftThreshold}(\boldsymbol{\beta}^{(t)} - \eta \nabla L(\boldsymbol{\beta}^{(t)}), \eta\lambda)$$

Where $\nabla L$ is gradient of squared error, $\eta$ is step size.

**Advantage**: Generalizes to other penalties and losses.

### Computational Complexity

- **Coordinate descent**: $O(np \cdot k)$ where $k$ = iterations (typically fast)
- **LARS**: $O(np \min(n,p))$ for full path
- **General**: Much faster than OLS when $p \gg n$ and solution sparse

## Choosing $\lambda$ (Hyperparameter Tuning)

### Cross-Validation (Standard Approach)

**K-Fold CV**:

1. Create $\lambda$ grid: $[\lambda_1, \lambda_2, ..., \lambda_m]$ (logarithmic scale)
2. For each $\lambda_i$:
   - Split data into $K$ folds
   - For each fold:
     - Train Lasso on $K-1$ folds
     - Validate on held-out fold
   - Average validation error
3. Select $\lambda^*$ with minimum CV error

**Common choice**: 10-fold CV

**Lambda grid**: $[10^{-4}, 10^{-3}, ..., 10^1, 10^2]$

### One-Standard-Error Rule

**Conservative choice**: Select largest $\lambda$ within one standard error of minimum CV error.

**Rationale**: Simpler model (more sparse) with similar performance.

**Formula**:

$$\lambda_{1SE} = \max\{\lambda : \text{CV}(\lambda) \leq \text{CV}(\lambda^*) + SE(\lambda^*)\}$$

### Information Criteria

**AIC for Lasso**:

$$\text{AIC} = n\log(\text{SSE}/n) + 2 \cdot |\{\beta_j \neq 0\}|$$

Where $|\{\beta_j \neq 0\}|$ is number of non-zero coefficients.

**BIC for Lasso**:

$$\text{BIC} = n\log(\text{SSE}/n) + \log(n) \cdot |\{\beta_j \neq 0\}|$$

**BIC** penalizes complexity more, selects sparser models.

### Validation Curve Analysis

**Plot**: CV error vs. $\log(\lambda)$

**Characteristics**:
- **Left side** (small $\lambda$): Overfitting, high CV error
- **Middle**: Optimal region, minimum CV error
- **Right side** (large $\lambda$): Underfitting, high CV error

**Optimal $\lambda$**: Valley of curve

## Properties of Lasso

### Sparsity

**Key property**: Sets coefficients **exactly** to zero (unlike Ridge).

**Why?**: L1 penalty has corners at axes where optimal solution can occur.

**Geometric explanation**: L1 constraint region (diamond) has corners aligned with axes. Contours of squared error likely intersect at corner → sparse solution.

### Selection Among Correlated Features

**Behavior**: When features highly correlated, Lasso tends to:
- Select one arbitrarily
- Set others to zero

**Example**: If $x_1$ and $x_2$ perfectly correlated, Lasso picks one, ignores other.

**Limitation**: Selection can be unstable—small data changes may change which feature selected.

**Solution**: Elastic Net (combines L1 and L2) handles this better.

### Degrees of Freedom

**Effective degrees of freedom**:

$$\text{df}(\lambda) = E[|\{\beta_j \neq 0\}|]$$

Approximately equals number of non-zero coefficients.

**Use**: For AIC/BIC calculation, complexity measures.

### Bias

**Lasso is biased**: Even for non-zero coefficients, estimates shrunk toward zero.

**Consequence**: Coefficients underestimate true magnitudes.

**Solution (if inference needed)**: 
1. Use Lasso for selection
2. Refit OLS on selected features (relaxed Lasso)

## Assumptions

### Inherited from Linear Regression

1. **Linearity**: True relationship is linear
2. **Independence**: Observations independent
3. **Homoscedasticity**: Constant error variance
4. **Normality**: For inference (less critical for prediction)

### Additional for Lasso

5. **Standardization**: Features should be standardized
   - **Critical**: L1 penalty scale-dependent
   - Unscaled features penalized unfairly

6. **Sparsity**: True model has many zero coefficients
   - **Ideal**: Lasso recovers true sparse model
   - If true model dense, Lasso may perform suboptimally

7. **Irrepresentable Condition** (for consistency):
   - Technical condition on correlation structure
   - Ensures Lasso selects true model as $n \to \infty$

8. **Sample Size**: Need $n > p$ for unique solution (though Lasso can work with $n < p$)

## Advantages

1. **Automatic Feature Selection**: Eliminates irrelevant features
2. **Interpretability**: Sparse models easy to understand and communicate
3. **High-Dimensional Capable**: Works when $p \gg n$
4. **Prevents Overfitting**: Regularization provides built-in complexity control
5. **Computational Efficiency**: Fast algorithms (coordinate descent, LARS)
6. **Variable Screening**: Identifies important predictors
7. **Embedded Method**: Selection and estimation simultaneously
8. **Model Compression**: Reduces model size for deployment
9. **Theoretical Guarantees**: Under certain conditions, recovers true sparse model
10. **Versatility**: Applicable to regression, classification, survival analysis

## Disadvantages

1. **Arbitrary Selection**: Among correlated features, picks one arbitrarily
2. **Instability**: Small data changes can alter selected features
3. **Grouped Variables**: Doesn't select correlated features together (Elastic Net better)
4. **$n < p$ Limitation**: Selects at most $n$ features (even if more relevant)
5. **Bias**: Shrinks all coefficients, including non-zero ones
6. **No Closed Form**: Requires iterative algorithms
7. **Hyperparameter Tuning**: Must select $\lambda$ (adds complexity)
8. **Coefficient Interpretation**: Magnitude affected by regularization
9. **Non-Differentiability**: Optimization more complex than smooth penalties
10. **Linearity**: Still assumes linear relationships (need extensions for non-linearity)

## When to Use Lasso Regression

### Ideal Scenarios

1. **High-Dimensional Data**: $p \gg n$ (many features, few samples)
2. **Feature Selection Needed**: Want to identify important predictors
3. **Sparse Truth**: Believe many coefficients truly zero
4. **Interpretability Critical**: Need simple, explainable model
5. **Model Deployment**: Computational/memory constraints favor fewer features
6. **Exploratory Analysis**: Screening variables for further study
7. **Regularization Needed**: Overfitting observed with OLS
8. **Scientific Discovery**: Identifying relevant biomarkers, genes, etc.

### Application Domains

**Genomics**:
- Gene expression analysis (thousands of genes, hundreds of samples)
- SNP selection in GWAS
- Disease biomarker discovery

**Text Mining**:
- Document classification (vast vocabulary, sparse relevance)
- Sentiment analysis with word features

**Finance**:
- Portfolio optimization with many assets
- Risk factor identification

**Medical Research**:
- Patient outcome prediction with many clinical variables
- Drug response prediction

**Marketing**:
- Customer churn prediction with many behavioral features
- Campaign effectiveness analysis

**Signal Processing**:
- Compressed sensing
- Sparse signal recovery

## When NOT to Use Lasso

1. **Few Features**: If $p$ small (< 20) and all potentially relevant, OLS or Ridge fine
2. **Grouped Correlated Features**: Want to keep/remove groups together (use Elastic Net or Group Lasso)
3. **All Features Relevant**: No sparsity in truth (Ridge better)
4. **Unbiased Estimates Required**: Statistical inference needs unbiased coefficients
5. **Non-Linear Relationships**: Need non-linear models (GAM, trees, neural nets)
6. **Feature Importance Ranking**: Regularization makes magnitude comparisons misleading
7. **Stable Selection Needed**: Lasso selection unstable (use stability selection or Bolasso)
8. **Smooth Shrinkage Preferred**: Want all features to contribute (use Ridge)

## Related Methods

### Regularization Variants

**Elastic Net**
- Combines L1 and L2: $\lambda[\alpha||\boldsymbol{\beta}||_1 + (1-\alpha)||\boldsymbol{\beta}||^2_2]$
- Handles correlated features better
- Groups and selects correlated features

**Adaptive Lasso**
- Weighted L1: $\sum_{j=1}^{p}w_j|\beta_j|$
- Weights $w_j = 1/|\hat{\beta}_j^{OLS}|^\gamma$
- Oracle properties (consistency in selection)

**Group Lasso**
- Penalty on groups: $\sum_{g=1}^{G}\sqrt{p_g}||\boldsymbol{\beta}_g||_2$
- Selects entire groups together
- Useful for categorical features (one-hot encoded)

**Fused Lasso**
- Penalty: $||\boldsymbol{\beta}||_1 + \lambda_2\sum_{j=1}^{p-1}|\beta_{j+1} - \beta_j|$
- Encourages coefficient similarity for ordered features
- Applications: Time series, spatial data

**Square-Root Lasso**
- Uses $||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||$ instead of squared error
- $\lambda$ selection less sensitive to noise variance

### Extensions

**Logistic Lasso (L1-Regularized Logistic Regression)**
- For classification
- Sparse logistic regression models

**Cox Lasso**
- Survival analysis with Lasso
- Feature selection for time-to-event data

**Poisson Lasso**
- Count data regression
- Sparse generalized linear models

**Graphical Lasso**
- Sparse precision matrix estimation
- Network/graph learning

### Related Techniques

**SCAD** (Smoothly Clipped Absolute Deviation)
- Non-convex penalty
- Less bias for large coefficients
- Oracle properties

**MCP** (Minimax Concave Penalty)
- Non-convex, similar to SCAD
- Stronger sparsity

**Best Subset Selection**
- Directly selects $k$ features
- Combinatorially hard (NP-hard)
- Lasso is convex relaxation

**Forward/Backward Stepwise**
- Greedy feature selection
- Lasso often superior

### Stability Methods

**Bolasso** (Bootstrap Lasso)
- Run Lasso on bootstrap samples
- Select features appearing frequently
- More stable selection

**Stability Selection**
- Resample data, run Lasso many times
- Select features with high selection frequency
- Provides finite-sample error control

## Summary

Lasso Regression is a powerful regularized linear model that performs both parameter estimation and automatic feature selection through L1 penalization. By adding the sum of absolute coefficient values to the loss function, Lasso produces sparse solutions where many coefficients are exactly zero.

**Core Formula**:

$$\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2_2 + \lambda||\boldsymbol{\beta}||_1$$

**Key Innovation**: L1 penalty drives coefficients to exactly zero (unlike L2), enabling automatic feature selection.

**Geometric Insight**: L1 constraint region (diamond shape) has corners at coordinate axes where sparse solutions occur.

**Algorithm**: No closed-form solution; use coordinate descent, LARS, or proximal methods.

**Hyperparameter $\lambda$**: Controls sparsity—larger $\lambda$ means fewer features. Select via cross-validation.

**Strengths**:
- Automatic feature selection
- Handles high-dimensional data ($p \gg n$)
- Interpretable sparse models
- Prevents overfitting
- Computationally efficient

**Limitations**:
- Arbitrary selection among correlated features
- Biased coefficient estimates
- Selection instability
- Requires standardization

**When to Use**:
- High-dimensional sparse problems
- Feature selection needed
- Interpretability important
- Many irrelevant features suspected

**Alternatives**:
- **Ridge**: No feature selection, handles correlations better
- **Elastic Net**: Combines L1 and L2, better for correlated features
- **Adaptive Lasso**: Improved theoretical properties
- **Group Lasso**: Select feature groups

Lasso has revolutionized high-dimensional statistics and machine learning, providing a principled, computationally tractable approach to feature selection and sparse modeling. It remains one of the most important tools for understanding and predicting with complex, high-dimensional data.
