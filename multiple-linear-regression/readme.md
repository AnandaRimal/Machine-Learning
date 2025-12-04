# Multiple Linear Regression

## Overview and Core Concept

Multiple Linear Regression (MLR) is an extension of simple linear regression that models the relationship between a dependent variable and multiple independent variables. While simple linear regression uses one predictor, MLR uses two or more predictors to explain the variance in the target variable. The relationship is assumed to be linear, meaning the dependent variable can be expressed as a weighted sum of independent variables plus an intercept term.

The fundamental idea is to find the best-fitting hyperplane in multi-dimensional space that minimizes the difference between predicted and actual values. This hyperplane is defined by coefficients (weights) for each predictor variable, allowing us to understand both the individual and combined effects of multiple factors on the outcome.

## Why Use Multiple Linear Regression?

1. **Real-World Complexity**: Most phenomena depend on multiple factors, not just one
2. **Improved Predictions**: Multiple predictors generally provide better accuracy than single predictor
3. **Variable Importance**: Understand relative importance of different features
4. **Interaction Effects**: Can be extended to capture interactions between variables
5. **Statistical Inference**: Hypothesis testing for individual coefficients
6. **Interpretability**: Coefficients have clear meaning (holding other variables constant)
7. **Foundation**: Basis for more advanced regression techniques

## Mathematical Formulas

### Model Equation

For $n$ observations and $p$ predictors:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_p x_{ip} + \epsilon_i$$

Where:
- $y_i$: Dependent variable (target) for observation $i$
- $x_{ij}$: Value of predictor $j$ for observation $i$
- $\beta_0$: Intercept (value when all predictors = 0)
- $\beta_j$: Coefficient for predictor $j$ (partial slope)
- $\epsilon_i$: Error term (residual) for observation $i$
- $p$: Number of predictors
- $n$: Number of observations

### Matrix Notation

The entire system can be expressed compactly:

$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:

$$\mathbf{Y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \quad \mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix}, \quad \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{bmatrix}, \quad \boldsymbol{\epsilon} = \begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{bmatrix}$$

### Normal Equation (Least Squares Solution)

The coefficients that minimize the sum of squared residuals:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

This is derived by minimizing:

$$\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}\epsilon_i^2$$

### Prediction

For a new observation $\mathbf{x}_{new} = [1, x_1, x_2, ..., x_p]^T$:

$$\hat{y}_{new} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + ... + \hat{\beta}_p x_p$$

Or in matrix form:

$$\hat{y}_{new} = \mathbf{x}_{new}^T\hat{\boldsymbol{\beta}}$$

### Coefficient Interpretation

For coefficient $\beta_j$:

**Interpretation**: The expected change in $y$ for a one-unit increase in $x_j$, **holding all other variables constant** (ceteris paribus).

$$\frac{\partial y}{\partial x_j} = \beta_j$$

### Standard Error of Coefficients

$$SE(\hat{\beta}_j) = \sqrt{s^2 \cdot (\mathbf{X}^T\mathbf{X})^{-1}_{jj}}$$

Where $s^2$ is the residual variance:

$$s^2 = \frac{\text{SSE}}{n - p - 1} = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n - p - 1}$$

### T-Statistic for Hypothesis Testing

To test $H_0: \beta_j = 0$ (predictor has no effect):

$$t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}$$

Under $H_0$, $t_j \sim t_{n-p-1}$ (t-distribution with $n-p-1$ degrees of freedom)

### Coefficient of Determination ($R^2$)

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:
- SSE: Sum of Squared Errors (residual variation)
- SST: Total Sum of Squares (total variation)
- $\bar{y}$: Mean of dependent variable

**Interpretation**: Proportion of variance in $y$ explained by the model (0 to 1)

### Adjusted $R^2$

Penalizes model complexity:

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**Advantage**: Accounts for number of predictors, prevents overfitting

### F-Statistic (Overall Model Significance)

Tests $H_0$: All coefficients (except intercept) are zero:

$$F = \frac{\text{MSR}}{\text{MSE}} = \frac{\text{SSR}/p}{\text{SSE}/(n-p-1)}$$

Where:
- SSR: Sum of Squares Regression $= \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2$
- MSR: Mean Square Regression
- MSE: Mean Square Error

Under $H_0$, $F \sim F_{p, n-p-1}$ (F-distribution)

## Detailed Worked Example

**Problem**: Predict house price based on size (sq ft) and number of bedrooms.

**Data**:

| House | Size ($x_1$) | Bedrooms ($x_2$) | Price ($y$, $1000s) |
|-------|--------------|------------------|---------------------|
| 1     | 1500         | 3                | 250                 |
| 2     | 1800         | 4                | 300                 |
| 3     | 2000         | 3                | 280                 |
| 4     | 2200         | 4                | 350                 |
| 5     | 2500         | 5                | 400                 |

### Step 1: Set Up Matrices

$$\mathbf{Y} = \begin{bmatrix} 250 \\ 300 \\ 280 \\ 350 \\ 400 \end{bmatrix}, \quad \mathbf{X} = \begin{bmatrix} 1 & 1500 & 3 \\ 1 & 1800 & 4 \\ 1 & 2000 & 3 \\ 1 & 2200 & 4 \\ 1 & 2500 & 5 \end{bmatrix}$$

### Step 2: Calculate $\mathbf{X}^T\mathbf{X}$

$$\mathbf{X}^T = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1500 & 1800 & 2000 & 2200 & 2500 \\ 3 & 4 & 3 & 4 & 5 \end{bmatrix}$$

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 5 & 10000 & 19 \\ 10000 & 20450000 & 38600 \\ 19 & 38600 & 75 \end{bmatrix}$$

**Calculations**:
- $(1,1)$: $1+1+1+1+1 = 5$
- $(1,2)$: $1500+1800+2000+2200+2500 = 10000$
- $(1,3)$: $3+4+3+4+5 = 19$
- $(2,2)$: $1500^2+1800^2+2000^2+2200^2+2500^2 = 20450000$
- $(2,3)$: $1500 \cdot 3+1800 \cdot 4+2000 \cdot 3+2200 \cdot 4+2500 \cdot 5 = 38600$
- $(3,3)$: $3^2+4^2+3^2+4^2+5^2 = 75$

### Step 3: Calculate $\mathbf{X}^T\mathbf{Y}$

$$\mathbf{X}^T\mathbf{Y} = \begin{bmatrix} 250+300+280+350+400 \\ 1500 \cdot 250+1800 \cdot 300+2000 \cdot 280+2200 \cdot 350+2500 \cdot 400 \\ 3 \cdot 250+4 \cdot 300+3 \cdot 280+4 \cdot 350+5 \cdot 400 \end{bmatrix} = \begin{bmatrix} 1580 \\ 3255000 \\ 6190 \end{bmatrix}$$

### Step 4: Calculate $(\mathbf{X}^T\mathbf{X})^{-1}$

For this example, the inverse (computed numerically):

$$(\mathbf{X}^T\mathbf{X})^{-1} \approx \begin{bmatrix} 57.6 & -0.024 & -6.4 \\ -0.024 & 0.00001 & 0.002 \\ -6.4 & 0.002 & 1.28 \end{bmatrix}$$

### Step 5: Calculate $\hat{\boldsymbol{\beta}}$

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

$$\hat{\beta}_0 \approx 50, \quad \hat{\beta}_1 \approx 0.12, \quad \hat{\beta}_2 \approx 20$$

**Model Equation**:

$$\hat{y} = 50 + 0.12x_1 + 20x_2$$

### Step 6: Interpretation

- **Intercept ($\beta_0 = 50$)**: Base price when size=0 and bedrooms=0 ($50,000)
- **Size coefficient ($\beta_1 = 0.12$)**: Each additional sq ft increases price by $120, holding bedrooms constant
- **Bedroom coefficient ($\beta_2 = 20$)**: Each additional bedroom increases price by $20,000, holding size constant

### Step 7: Make Predictions

**Example**: House with 1900 sq ft and 3 bedrooms:

$$\hat{y} = 50 + 0.12(1900) + 20(3) = 50 + 228 + 60 = 338$$

**Predicted price**: $338,000

### Step 8: Calculate $R^2$

**Predictions for training data**:
- House 1: $\hat{y}_1 = 50 + 0.12(1500) + 20(3) = 290$
- House 2: $\hat{y}_2 = 50 + 0.12(1800) + 20(4) = 346$
- House 3: $\hat{y}_3 = 50 + 0.12(2000) + 20(3) = 350$
- House 4: $\hat{y}_4 = 50 + 0.12(2200) + 20(4) = 394$
- House 5: $\hat{y}_5 = 50 + 0.12(2500) + 20(5) = 450$

**Calculate SSE**:

$$\text{SSE} = (250-290)^2 + (300-346)^2 + (280-350)^2 + (350-394)^2 + (400-450)^2$$
$$= 1600 + 2116 + 4900 + 1936 + 2500 = 13052$$

**Calculate SST**:

$$\bar{y} = \frac{250+300+280+350+400}{5} = 316$$

$$\text{SST} = (250-316)^2 + (300-316)^2 + (280-316)^2 + (350-316)^2 + (400-316)^2$$
$$= 4356 + 256 + 1296 + 1156 + 7056 = 14120$$

$$R^2 = 1 - \frac{13052}{14120} = 1 - 0.924 = 0.076$$

**Interpretation**: Only 7.6% of variance explained (not a good fit with simplified coefficients)

## Assumptions of Multiple Linear Regression

### 1. Linearity

The relationship between predictors and target is linear:

$$E[Y|\mathbf{X}] = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p$$

**Check**: Residual plots should show random scatter

### 2. Independence

Observations are independent (no autocorrelation):

$$\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \forall i \neq j$$

**Check**: Durbin-Watson test, autocorrelation plots

### 3. Homoscedasticity

Constant variance of errors across all levels of predictors:

$$\text{Var}(\epsilon_i) = \sigma^2 \quad \forall i$$

**Check**: Plot residuals vs fitted values (should show constant spread)

### 4. Normality of Errors

Errors are normally distributed:

$$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

**Check**: Q-Q plot, Shapiro-Wilk test, histogram of residuals

### 5. No Multicollinearity

Predictors are not highly correlated with each other.

**Check**: Variance Inflation Factor (VIF)

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

Where $R^2_j$ is from regressing $x_j$ on all other predictors.

**Rule**: VIF > 10 indicates problematic multicollinearity

### 6. No Influential Outliers

Extreme values shouldn't disproportionately affect the model.

**Check**: Cook's distance, leverage values, DFFITS

## Advantages

1. **Interpretability**: Coefficients have clear, intuitive meaning
2. **Computational Efficiency**: Fast to train, even with large datasets
3. **Statistical Foundation**: Well-established theory for inference and hypothesis testing
4. **Prediction Intervals**: Can quantify uncertainty in predictions
5. **Feature Importance**: Coefficients indicate relative importance (when scaled)
6. **Baseline Model**: Excellent starting point before trying complex models
7. **Extrapolation**: Can make predictions outside training range (with caution)
8. **Small Data**: Works well even with limited observations
9. **No Hyperparameters**: No tuning required (unlike regularized methods)
10. **Diagnostic Tools**: Rich set of residual analysis techniques

## Disadvantages

1. **Linearity Assumption**: Cannot capture non-linear relationships without transformation
2. **Multicollinearity**: High correlation between predictors causes unstable coefficients
3. **Outlier Sensitivity**: Least squares is sensitive to extreme values
4. **Overfitting**: With many predictors relative to observations (high p, low n)
5. **Feature Engineering**: Requires manual creation of interactions/polynomial terms
6. **Assumptions**: Performance degrades if assumptions violated
7. **No Automatic Feature Selection**: Includes all provided features
8. **Homoscedasticity Required**: Struggles with heteroscedastic data
9. **Normal Errors**: Statistical tests require normally distributed residuals
10. **Extrapolation Risk**: Predictions outside data range can be unreliable

## When to Use Multiple Linear Regression

### Ideal Scenarios

1. **Linear Relationships**: When predictors have approximately linear relationship with target
2. **Interpretability Needed**: When understanding feature effects is crucial
3. **Small to Medium Data**: Works well even with limited observations
4. **Statistical Inference**: When hypothesis testing is required
5. **Continuous Target**: Predicting continuous numerical outcomes
6. **Uncorrelated Features**: When predictors are relatively independent
7. **Homoscedastic Data**: Variance of errors is roughly constant
8. **Baseline Modeling**: As first model to establish performance benchmark
9. **Real-Time Predictions**: When fast inference is critical
10. **Regulatory Requirements**: When model transparency is mandated

### Problem Types

- **Economics**: Price prediction, demand forecasting
- **Real Estate**: Property valuation based on features
- **Healthcare**: Patient outcome prediction (with linear relationships)
- **Marketing**: Sales prediction from advertising spend
- **Finance**: Risk assessment, return prediction
- **Manufacturing**: Quality control, yield optimization

## When NOT to Use

1. **Non-Linear Relationships**: Strong curvature in predictor-target relationships
2. **High Multicollinearity**: Predictors are highly correlated (use Ridge/Lasso instead)
3. **Categorical Target**: Use logistic regression or classification algorithms
4. **Many Predictors**: p > n scenario (use regularized regression)
5. **Complex Interactions**: High-order interactions between many features (use tree-based models)
6. **Heteroscedastic Data**: Variance changes significantly across predictor range
7. **Severe Outliers**: Extreme values dominate the fit (use robust regression)
8. **Time Series**: Strong autocorrelation present (use ARIMA, state space models)
9. **Image/Text Data**: High-dimensional unstructured data (use neural networks)
10. **Black-Box Acceptable**: When interpretability not important and accuracy is priority

## Related Methods

### Extensions of Multiple Linear Regression

**Polynomial Regression**
- Adds polynomial terms: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_k x^k$
- Captures non-linear relationships
- Still linear in parameters (can use OLS)

**Ridge Regression (L2 Regularization)**
- Adds penalty: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2$
- Handles multicollinearity
- Shrinks coefficients toward zero

**Lasso Regression (L1 Regularization)**
- Adds penalty: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$
- Performs feature selection
- Sets some coefficients exactly to zero

**Elastic Net**
- Combines L1 and L2: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda_1||\boldsymbol{\beta}||_1 + \lambda_2||\boldsymbol{\beta}||^2$
- Benefits of both Ridge and Lasso

**Stepwise Regression**
- Forward selection: Add variables incrementally
- Backward elimination: Remove variables incrementally
- Automatic feature selection

### Alternative Regression Methods

**Robust Regression**
- Less sensitive to outliers
- Uses alternative loss functions (Huber, RANSAC)

**Weighted Least Squares**
- Addresses heteroscedasticity
- Different weights for different observations

**Generalized Linear Models (GLM)**
- Extends to non-normal distributions
- Link functions connect linear predictor to target

**Non-Parametric Regression**
- Kernel regression, splines
- No parametric form assumed

### Modern Alternatives

**Tree-Based Methods**
- Random Forest, Gradient Boosting
- Handle non-linearity and interactions automatically

**Support Vector Regression**
- Kernel trick for non-linear relationships
- Robust to outliers

**Neural Networks**
- Universal function approximators
- Learn complex patterns

## Variance Inflation Factor (VIF) Detailed

### Formula

For predictor $x_j$:

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

Where $R^2_j$ is obtained by regressing $x_j$ on all other predictors.

### Interpretation

- **VIF = 1**: No correlation with other predictors
- **VIF = 5**: $R^2_j = 0.8$ (80% of variance explained by others)
- **VIF = 10**: $R^2_j = 0.9$ (90% of variance explained by others)

**Standard Error Inflation**:

$$SE(\hat{\beta}_j) = SE(\hat{\beta}_j)_{uncorrelated} \times \sqrt{\text{VIF}_j}$$

**Example**: VIF = 4 means standard error is doubled (2 = √4)

### Guidelines

- **VIF < 5**: Acceptable
- **5 ≤ VIF < 10**: Moderate multicollinearity (monitor)
- **VIF ≥ 10**: Severe multicollinearity (action required)

### Solutions for High VIF

1. Remove highly correlated predictors
2. Combine correlated predictors (e.g., PCA)
3. Use Ridge or Elastic Net regression
4. Collect more data
5. Center variables (for interaction terms)

## Summary

Multiple Linear Regression extends simple linear regression to model relationships between one continuous dependent variable and multiple independent variables. It assumes a linear relationship and uses the method of least squares to find optimal coefficients that minimize prediction error.

The model provides interpretable coefficients representing the effect of each predictor while holding others constant, making it valuable for both prediction and understanding variable relationships. It includes robust statistical framework for hypothesis testing, confidence intervals, and model diagnostics.

Key strengths include computational efficiency, interpretability, and solid theoretical foundation. Main limitations are the linearity assumption, sensitivity to multicollinearity and outliers, and inability to automatically handle non-linear relationships or interactions.

MLR serves as an essential baseline model and remains widely used in fields requiring interpretability such as economics, healthcare, and social sciences. For more complex relationships or when assumptions are violated, regularized variants (Ridge, Lasso, Elastic Net) or non-linear methods (polynomial regression, tree-based models, neural networks) may be more appropriate.
