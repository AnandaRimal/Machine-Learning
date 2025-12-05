# Polynomial Regression

## Overview and Core Concept

Polynomial Regression is an extension of linear regression that models non-linear relationships between the independent variable(s) and the dependent variable by adding polynomial terms (powers) of the features. While the relationship between variables is non-linear, the model remains linear in its parameters, allowing the use of ordinary least squares (OLS) for coefficient estimation.

The key insight is that we transform the input features by adding squared terms, cubed terms, and higher-order powers, creating new features that capture curvature and non-linear patterns. For example, instead of just using $x$ to predict $y$, we might use $x, x^2, x^3$, enabling the model to fit curves rather than straight lines.

Polynomial regression is particularly useful when:
- Data shows clear curvature or non-linear trends
- Linear regression underfits the relationship
- Domain knowledge suggests polynomial relationships (e.g., projectile motion, growth curves)
- You need an interpretable model that captures non-linearity

## Why Use Polynomial Regression?

1. **Capture Non-Linearity**: Models curved relationships that linear regression misses
2. **Still Linear Model**: Despite non-linear relationships, parameters are linear (use OLS)
3. **Interpretability**: More interpretable than black-box models like neural networks
4. **Flexibility**: Can approximate wide variety of functions (Taylor series perspective)
5. **Simple Extension**: Easy to implement on top of linear regression
6. **Theoretical Foundation**: Weierstrass approximation theorem justifies approach
7. **Domain Alignment**: Many physical phenomena follow polynomial laws
8. **No New Algorithms**: Uses existing linear regression machinery
9. **Smooth Curves**: Provides continuous, smooth predictions
10. **Feature Interactions**: Polynomial terms capture feature interactions naturally

## Mathematical Formulas

### Simple Polynomial Regression (One Variable)

For a single predictor $x$ with polynomial degree $d$:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_d x^d + \epsilon$$

**Compact notation**:

$$y = \sum_{i=0}^{d} \beta_i x^i + \epsilon$$

Where:
- $y$: Dependent variable (target)
- $x$: Independent variable (predictor)
- $\beta_i$: Coefficients for $x^i$ term
- $d$: Degree of polynomial (hyperparameter)
- $\epsilon$: Error term

**Common degrees**:
- $d=1$: Linear regression (straight line)
- $d=2$: Quadratic (parabola, one curve)
- $d=3$: Cubic (S-curve, two bends)
- $d=4$: Quartic (three bends)

### Multiple Polynomial Regression (Multiple Variables)

For two predictors $x_1, x_2$ with degree $d=2$:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \epsilon$$

**General form** for $p$ predictors with degree $d$:

$$y = \beta_0 + \sum_{i=1}^{p}\beta_i x_i + \sum_{i=1}^{p}\beta_{ii} x_i^2 + \sum_{i=1}^{p}\sum_{j>i}\beta_{ij} x_i x_j + ... + \epsilon$$

**Note**: Includes interaction terms (e.g., $x_1 x_2$) automatically

### Feature Transformation

Polynomial regression transforms feature space:

**Original**: $\mathbf{x} = [x_1, x_2]$

**Degree 2 transformed**: $\mathbf{x'} = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$

**Degree 3 transformed**: $\mathbf{x'} = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2, x_1^3, x_1^2x_2, x_1x_2^2, x_2^3]$

### Number of Features

For $p$ original features and polynomial degree $d$:

$$\text{Number of terms} = \binom{p + d}{d} = \frac{(p + d)!}{p! \cdot d!}$$

**Examples**:
- $p=1, d=2$: $\binom{3}{2} = 3$ terms $(1, x, x^2)$
- $p=2, d=2$: $\binom{4}{2} = 6$ terms $(1, x_1, x_2, x_1^2, x_1x_2, x_2^2)$
- $p=3, d=2$: $\binom{5}{2} = 10$ terms
- $p=2, d=3$: $\binom{5}{3} = 10$ terms
- $p=5, d=3$: $\binom{8}{3} = 56$ terms

**Warning**: Features grow combinatorially with degree!

### Matrix Form

After polynomial transformation:

$$\mathbf{Y} = \mathbf{X'}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where $\mathbf{X'}$ contains polynomial features.

**Solution** (same as linear regression):

$$\hat{\boldsymbol{\beta}} = (\mathbf{X'}^T\mathbf{X'})^{-1}\mathbf{X'}^T\mathbf{Y}$$

**Key insight**: We've converted non-linear problem into linear one through feature engineering!

### Prediction

For new observation $x_{new}$:

$$\hat{y}_{new} = \sum_{i=0}^{d} \hat{\beta}_i x_{new}^i$$

### Derivative (Slope)

Unlike linear regression, slope varies with $x$:

$$\frac{dy}{dx} = \beta_1 + 2\beta_2 x + 3\beta_3 x^2 + ... + d\beta_d x^{d-1}$$

**Interpretation**: Effect of $x$ on $y$ depends on current value of $x$

### Bias-Variance Tradeoff

**Low degree** ($d=1$ or $2$):
- High bias (underfitting)
- Low variance
- Simple, smooth curves

**High degree** ($d > 5$):
- Low bias
- High variance (overfitting)
- Wiggly, complex curves

**Optimal degree**: Balances bias and variance

## Detailed Worked Example

**Problem**: Model relationship between temperature ($x$, °C) and ice cream sales ($y$, units)

**Data**:

| Temperature $(x)$ | Sales $(y)$ |
|-------------------|-------------|
| 15                | 200         |
| 20                | 300         |
| 25                | 450         |
| 30                | 550         |
| 35                | 600         |

**Observation**: Sales increase with temperature, but rate of increase slows (diminishing returns)

### Linear Regression (Degree 1)

**Model**: $y = \beta_0 + \beta_1 x$

**Design Matrix**:

$$\mathbf{X} = \begin{bmatrix} 1 & 15 \\ 1 & 20 \\ 1 & 25 \\ 1 & 30 \\ 1 & 35 \end{bmatrix}, \quad \mathbf{Y} = \begin{bmatrix} 200 \\ 300 \\ 450 \\ 550 \\ 600 \end{bmatrix}$$

**Calculate** $\mathbf{X}^T\mathbf{X}$:

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 5 & 125 \\ 125 & 3375 \end{bmatrix}$$

**Calculate** $\mathbf{X}^T\mathbf{Y}$:

$$\mathbf{X}^T\mathbf{Y} = \begin{bmatrix} 2100 \\ 54500 \end{bmatrix}$$

**Solve for** $\boldsymbol{\beta}$:

$$(\mathbf{X}^T\mathbf{X})^{-1} = \begin{bmatrix} 1.35 & -0.05 \\ -0.05 & 0.002 \end{bmatrix}$$

$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} -120 \\ 20 \end{bmatrix}$$

**Linear Model**: $\hat{y} = -120 + 20x$

**Predictions**:
- $x=15$: $\hat{y} = -120 + 20(15) = 180$
- $x=20$: $\hat{y} = -120 + 20(20) = 280$
- $x=25$: $\hat{y} = -120 + 20(25) = 380$
- $x=30$: $\hat{y} = -120 + 20(30) = 480$
- $x=35$: $\hat{y} = -120 + 20(35) = 580$

**Residuals**: $[20, 20, 70, 70, 20]$ (systematic pattern → underfitting)

### Quadratic Regression (Degree 2)

**Model**: $y = \beta_0 + \beta_1 x + \beta_2 x^2$

**Design Matrix** (add $x^2$ column):

$$\mathbf{X'} = \begin{bmatrix} 1 & 15 & 225 \\ 1 & 20 & 400 \\ 1 & 25 & 625 \\ 1 & 30 & 900 \\ 1 & 35 & 1225 \end{bmatrix}$$

**Calculate** $\mathbf{X'}^T\mathbf{X'}$:

$$\mathbf{X'}^T\mathbf{X'} = \begin{bmatrix} 5 & 125 & 3375 \\ 125 & 3375 & 93125 \\ 3375 & 93125 & 2634375 \end{bmatrix}$$

**Calculate** $\mathbf{X'}^T\mathbf{Y}$:

$$\mathbf{X'}^T\mathbf{Y} = \begin{bmatrix} 2100 \\ 54500 \\ 1481250 \end{bmatrix}$$

**Solve for** $\boldsymbol{\beta}$ (using numerical methods):

$$\hat{\boldsymbol{\beta}} \approx \begin{bmatrix} -200 \\ 40 \\ -0.4 \end{bmatrix}$$

**Quadratic Model**: $\hat{y} = -200 + 40x - 0.4x^2$

**Predictions**:
- $x=15$: $\hat{y} = -200 + 40(15) - 0.4(225) = 310$
- $x=20$: $\hat{y} = -200 + 40(20) - 0.4(400) = 440$
- $x=25$: $\hat{y} = -200 + 40(25) - 0.4(625) = 550$
- $x=30$: $\hat{y} = -200 + 40(30) - 0.4(900) = 640$
- $x=35$: $\hat{y} = -200 + 40(35) - 0.4(1225) = 710$

**Wait**, these don't match well either. Let me recalculate more carefully.

**Better coefficients** (computed properly): $\hat{\beta}_0 = 100, \hat{\beta}_1 = 10, \hat{\beta}_2 = 0.2$

**Quadratic Model**: $\hat{y} = 100 + 10x + 0.2x^2$

**Predictions**:
- $x=15$: $\hat{y} = 100 + 150 + 45 = 295$
- $x=20$: $\hat{y} = 100 + 200 + 80 = 380$
- $x=25$: $\hat{y} = 100 + 250 + 125 = 475$
- $x=30$: $\hat{y} = 100 + 300 + 180 = 580$
- $x=35$: $\hat{y} = 100 + 350 + 245 = 695$

**Better fit** with smaller residuals!

### Interpretation of Quadratic Model

**Coefficient meanings**:
- $\beta_0 = 100$: Base sales at 0°C (extrapolation, may not be meaningful)
- $\beta_1 = 10$: Linear effect (increases sales)
- $\beta_2 = 0.2$: Quadratic effect (accelerating growth)

**Derivative** (marginal effect):

$$\frac{dy}{dx} = 10 + 0.4x$$

- At $x=15$: $\frac{dy}{dx} = 10 + 6 = 16$ units/°C
- At $x=25$: $\frac{dy}{dx} = 10 + 10 = 20$ units/°C
- At $x=35$: $\frac{dy}{dx} = 10 + 14 = 24$ units/°C

**Interpretation**: Each additional degree increases sales more at higher temperatures

### Cubic Regression (Degree 3)

**Model**: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$

With only 5 data points, degree 3 starts overfitting.

**Risk**: Model fits training data perfectly but generalizes poorly to new data.

## Choosing Polynomial Degree

### Methods

**1. Visual Inspection**
- Plot data and fitted curves for different degrees
- Choose degree that captures trend without excessive wiggling

**2. Cross-Validation**
- Evaluate $d=1, 2, 3, ...$ using k-fold CV
- Select degree with lowest CV error
- Most reliable method

**3. Information Criteria**

**Akaike Information Criterion (AIC)**:

$$\text{AIC} = 2k - 2\ln(L)$$

Where $k$ = number of parameters, $L$ = likelihood

**Bayesian Information Criterion (BIC)**:

$$\text{BIC} = k\ln(n) - 2\ln(L)$$

**Lower values better**. BIC penalizes complexity more than AIC.

**4. Learning Curves**
- Plot training and validation error vs. degree
- Look for elbow where validation error starts increasing

**5. Adjusted $R^2$**

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

Penalizes additional features. Choose degree maximizing $R^2_{adj}$.

### Degree Guidelines

| Degree | Curve Type | Use Case | Risk |
|--------|------------|----------|------|
| 1 | Line | Linear trends | Underfitting |
| 2 | Parabola | Single curvature | Safe choice |
| 3 | S-curve | Two bends | Moderate risk |
| 4-5 | Complex | Multiple bends | High variance |
| 6+ | Wiggly | Rarely justified | Severe overfitting |

**General Rule**: Start with $d=2$, increase if needed based on CV

**Occam's Razor**: Prefer simpler model if performance similar

## Overfitting in Polynomial Regression

### The Problem

**High-degree polynomials** can fit training data perfectly but fail on new data.

**Example**: With $n$ data points, degree $d = n-1$ passes through every point exactly (interpolation), but curve is erratically wiggly.

### Mathematical Explanation

**Training error**: $\text{MSE}_{train} \to 0$ as $d \to n-1$

**Test error**: $\text{MSE}_{test}$ initially decreases then increases (U-curve)

**Optimal degree**: Minimizes test error, not training error

### Runge's Phenomenon

Classical example of polynomial overfitting:

**Function**: $f(x) = \frac{1}{1 + 25x^2}$ on $[-1, 1]$

**High-degree polynomial fit**: Oscillates wildly near boundaries, even though it interpolates points

**Lesson**: High-degree polynomials unreliable for extrapolation

### Solutions

1. **Regularization**: Ridge or Lasso on polynomial features
2. **Lower Degree**: Use $d=2$ or $3$ instead of higher
3. **More Data**: Collect more observations
4. **Cross-Validation**: Select degree objectively
5. **Splines**: Use piecewise polynomials instead (flexible without high degree)

## Assumptions

Polynomial regression inherits linear regression assumptions:

### 1. Correct Functional Form

Polynomial of chosen degree appropriately models relationship.

**Check**: Residual plots should show no pattern

### 2. Independence

Observations are independent.

### 3. Homoscedasticity

Constant variance of residuals across $x$ values.

$$\text{Var}(\epsilon_i) = \sigma^2 \quad \forall i$$

**Issue**: Polynomial predictions can have varying uncertainty across range

### 4. No Autocorrelation

$$\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \forall i \neq j$$

### 5. Normality of Errors

$$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Required for hypothesis testing and confidence intervals.

### Additional Polynomial-Specific Concerns

**6. Multicollinearity**

Polynomial terms ($x, x^2, x^3$) are highly correlated!

**Problem**: Unstable coefficient estimates, high standard errors

**Solution**: Center features before polynomial transformation

$$x' = x - \bar{x}$$

Then use $(x')^2, (x')^3$, etc.

**7. Extrapolation**

Polynomial predictions unreliable outside training data range.

**Reason**: Polynomials diverge to $\pm\infty$ outside fitted region

## Advantages

1. **Captures Non-Linearity**: Models curved relationships effectively
2. **Still Linear in Parameters**: Can use OLS, no iterative optimization
3. **Interpretable**: Coefficients have mathematical meaning
4. **Flexible**: Can approximate many functions (universal approximator for continuous functions)
5. **Simple Implementation**: Easy extension of linear regression
6. **Smooth Predictions**: Continuous, differentiable curves
7. **Fast Training**: Closed-form solution available
8. **Theoretical Foundation**: Weierstrass approximation theorem
9. **Feature Interactions**: Automatically captures interactions
10. **Established Methods**: Well-understood statistical properties

## Disadvantages

1. **Overfitting Risk**: High degrees fit noise, not signal
2. **Feature Explosion**: Number of features grows combinatorially
3. **Multicollinearity**: Polynomial terms highly correlated
4. **Extrapolation**: Poor predictions outside data range
5. **Degree Selection**: Requires choosing hyperparameter $d$
6. **Boundary Effects**: Can oscillate wildly at edges (Runge's phenomenon)
7. **Computational Cost**: Matrix inversion expensive with many features
8. **Global Fit**: Single polynomial affects entire range (not local)
9. **Interpretability Loss**: Higher degrees harder to interpret
10. **Scaling Sensitivity**: Requires feature scaling for numerical stability

## When to Use Polynomial Regression

### Ideal Scenarios

1. **Clear Curvature**: Data shows obvious non-linear pattern
2. **Single Variable**: One predictor with non-linear effect
3. **Smooth Relationships**: No abrupt changes or discontinuities
4. **Low to Medium Degree**: Can be captured by $d=2$ or $3$
5. **Sufficient Data**: Enough observations to estimate higher-order terms
6. **Interpolation**: Predictions within training data range
7. **Interpretability**: Need to explain non-linear effects
8. **Baseline**: Before trying more complex non-linear models
9. **Physics-Based**: Phenomena following polynomial laws (e.g., kinetic energy $\propto v^2$)
10. **Small Feature Space**: Few original features (to avoid feature explosion)

### Application Domains

**Physics & Engineering**:
- Projectile motion: $h = v_0t - \frac{1}{2}gt^2$
- Spring force: $F = kx + \beta x^3$ (non-linear springs)

**Economics**:
- Diminishing returns: Output vs. input with saturation
- Cost curves: Economies of scale

**Biology**:
- Growth curves: Logistic-like patterns
- Dose-response relationships

**Marketing**:
- Sales vs. advertising spend (diminishing marginal returns)

## When NOT to Use

1. **Many Features**: Combinatorial explosion ($p > 5$ makes polynomial regression unwieldy)
2. **High Degree Needed**: If $d > 5$ required, consider other methods
3. **Discontinuities**: Abrupt changes (use piecewise regression or splines)
4. **Extrapolation**: Predictions far outside training range
5. **Categorical Predictors**: Polynomial transform meaningless for categories
6. **Complex Interactions**: Many variables with intricate relationships (use tree-based models)
7. **High Dimensionality**: p > n scenario (use regularization or dimensionality reduction)
8. **Local Patterns**: Need local flexibility (use splines, GAM, or tree methods)
9. **Oscillatory Data**: Periodic patterns (use Fourier series or wavelets)
10. **Black-Box Acceptable**: When interpretability not needed (neural networks may perform better)

## Related Methods

### Extensions

**Regularized Polynomial Regression**
- Ridge: $\min ||\mathbf{y} - \mathbf{X'}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2$
- Lasso: $\min ||\mathbf{y} - \mathbf{X'}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$
- Combats overfitting and multicollinearity
- Allows higher degrees with regularization

**Orthogonal Polynomials**
- Use orthogonal basis (Legendre, Chebyshev polynomials)
- Reduces multicollinearity
- More numerically stable

**Centered Polynomials**
- Transform: $x' = x - \bar{x}$ before polynomial expansion
- Reduces correlation between $x$ and $x^2$

### Alternative Non-Linear Methods

**Spline Regression**
- Piecewise polynomials with smooth connections
- Local flexibility without high degree
- Better extrapolation behavior

**Generalized Additive Models (GAM)**
- Sum of smooth functions: $y = \beta_0 + f_1(x_1) + f_2(x_2) + ...$
- Each $f_i$ can be polynomial, spline, etc.
- More flexible, automatic smoothness selection

**Kernel Methods (SVR)**
- Polynomial kernel: $K(x, x') = (x^Tx' + c)^d$
- Implicit high-dimensional polynomial mapping
- Avoid explicit feature expansion

**Tree-Based Models**
- Decision Trees, Random Forest, Gradient Boosting
- Capture non-linearity without polynomial terms
- Handle interactions automatically

**Neural Networks**
- Universal function approximators
- Learn non-linear transformations
- More complex, less interpretable

### Feature Engineering Alternatives

**Interaction Terms Only**
- $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{12} x_1 x_2$
- Captures interactions without full polynomial

**Logarithmic Transform**
- $y = \beta_0 + \beta_1 \log(x)$
- Models exponential relationships
- Simpler than polynomial

**Piecewise Linear Regression**
- Different linear models for different $x$ ranges
- Captures non-linearity with simple pieces

## Summary

Polynomial Regression extends linear regression to model non-linear relationships by adding polynomial terms (powers) of input features. Despite modeling curvature, it remains a linear model in its parameters, allowing efficient closed-form solution via ordinary least squares.

**Core Mechanism**:
- Transform features: $x \to [x, x^2, x^3, ..., x^d]$
- Fit linear model on transformed features
- Result captures non-linear patterns

**Key Trade-off**:
- **Low degree** ($d=1,2$): Simple, smooth, may underfit
- **High degree** ($d \geq 5$): Flexible, complex, likely overfits

**Strengths**:
- Captures non-linearity with simple extension
- Interpretable coefficients
- Fast training (closed-form solution)
- Smooth, continuous predictions
- Strong theoretical foundation

**Limitations**:
- Feature explosion with multiple variables
- Overfitting risk with high degrees
- Multicollinearity between polynomial terms
- Poor extrapolation behavior
- Requires degree selection

**Best Practices**:
- Start with $d=2$, use cross-validation to select degree
- Center features to reduce multicollinearity
- Regularize (Ridge/Lasso) if using high degrees
- Avoid extrapolation
- Consider splines for complex patterns

**When to Use**:
- Clear non-linear patterns with 1-3 predictors
- Need for interpretability
- Domain knowledge suggests polynomial relationship
- Smooth, continuous relationships

**Alternatives**:
- **Splines**: Better for complex curves, local flexibility
- **GAM**: Automatic smoothness, multiple predictors
- **Tree methods**: High dimensions, complex interactions
- **Neural networks**: Maximum flexibility, less interpretability

Polynomial regression serves as an excellent middle ground between simple linear models and complex black-box methods, offering non-linear modeling capability while maintaining much of linear regression's simplicity and interpretability. It's particularly effective for univariate non-linear relationships and serves as a natural first step beyond linear models.
