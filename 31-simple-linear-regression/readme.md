# Simple Linear Regression

## Overview

Simple Linear Regression is the most fundamental supervised learning algorithm that models the relationship between a single independent variable (predictor) and a dependent variable (response) by fitting a linear equation to observed data. It establishes a straight-line relationship that best predicts the output based on the input.

## The Core Concept

The goal is to find the "best-fit line" through a scatter plot of data points. This line is defined by two parameters: the slope (how steep the line is) and the intercept (where it crosses the y-axis). Once we find these parameters, we can make predictions for new input values.

### Real-World Analogy

Imagine you're studying the relationship between hours studied and exam scores. Simple linear regression helps you answer: "If I study for X hours, what score can I expect?" It finds the line that best represents this relationship based on historical data.

## Mathematical Foundation

### The Linear Equation

The relationship between input $x$ and output $y$ is modeled as:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $y$: Dependent variable (target, response)
- $x$: Independent variable (predictor, feature)
- $\beta_0$: Intercept (value of $y$ when $x = 0$)
- $\beta_1$: Slope (change in $y$ for unit change in $x$)
- $\epsilon$: Error term (random noise, unexplained variation)

**Prediction equation** (without error):

$$\hat{y} = \beta_0 + \beta_1 x$$

where $\hat{y}$ (y-hat) represents the predicted value.

### Interpretation of Parameters

**Slope** ($\beta_1$):
- Represents the rate of change
- $\beta_1 = 5$: For each unit increase in $x$, $y$ increases by 5
- $\beta_1 = -2$: For each unit increase in $x$, $y$ decreases by 2
- $\beta_1 = 0$: No relationship between $x$ and $y$

**Intercept** ($\beta_0$):
- Value of $y$ when $x = 0$
- Starting point of the line
- May not always have practical interpretation (if $x = 0$ is outside data range)

## Finding the Best-Fit Line

### Least Squares Method

The most common approach is **Ordinary Least Squares (OLS)**, which minimizes the sum of squared residuals.

**Residual** (error) for data point $i$:

$$e_i = y_i - \hat{y}_i = y_i - (\beta_0 + \beta_1 x_i)$$

**Sum of Squared Errors (SSE)** or **Residual Sum of Squares (RSS)**:

$$SSE = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

**Goal**: Find $\beta_0$ and $\beta_1$ that minimize SSE.

### Analytical Solution

Taking partial derivatives and setting them to zero:

$$\frac{\partial SSE}{\partial \beta_0} = 0, \quad \frac{\partial SSE}{\partial \beta_1} = 0$$

This yields the **normal equations**:

$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{Cov(X, Y)}{Var(X)}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

Where:
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ (mean of $x$)
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ (mean of $y$)
- $Cov(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$
- $Var(X) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$

**Alternative formula for slope**:

$$\beta_1 = r \cdot \frac{s_y}{s_x}$$

Where:
- $r$: Correlation coefficient between $x$ and $y$
- $s_y$: Standard deviation of $y$
- $s_x$: Standard deviation of $x$

## Detailed Example

**Dataset**: Study hours ($x$) vs. Exam score ($y$)

| Hours ($x$) | Score ($y$) |
|-------------|-------------|
| 2           | 50          |
| 3           | 60          |
| 4           | 65          |
| 5           | 75          |
| 6           | 85          |

**Step 1**: Calculate means

$$\bar{x} = \frac{2 + 3 + 4 + 5 + 6}{5} = \frac{20}{5} = 4$$

$$\bar{y} = \frac{50 + 60 + 65 + 75 + 85}{5} = \frac{335}{5} = 67$$

**Step 2**: Calculate slope ($\beta_1$)

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|-------|-------|-----------------|-----------------|----------------------------------|---------------------|
| 2     | 50    | -2              | -17             | 34                               | 4                   |
| 3     | 60    | -1              | -7              | 7                                | 1                   |
| 4     | 65    | 0               | -2              | 0                                | 0                   |
| 5     | 75    | 1               | 8               | 8                                | 1                   |
| 6     | 85    | 2               | 18              | 36                               | 4                   |
| **Sum** |     |                 |                 | **85**                           | **10**              |

$$\beta_1 = \frac{85}{10} = 8.5$$

**Interpretation**: For each additional hour of study, the exam score increases by 8.5 points.

**Step 3**: Calculate intercept ($\beta_0$)

$$\beta_0 = 67 - 8.5 \times 4 = 67 - 34 = 33$$

**Interpretation**: A student who studies 0 hours is predicted to score 33 points (baseline knowledge).

**Step 4**: Final equation

$$\hat{y} = 33 + 8.5x$$

**Step 5**: Make predictions

- Study 7 hours: $\hat{y} = 33 + 8.5(7) = 33 + 59.5 = 92.5$ points
- Study 3.5 hours: $\hat{y} = 33 + 8.5(3.5) = 33 + 29.75 = 62.75$ points

## Model Evaluation

### R-squared ($R^2$) - Coefficient of Determination

Measures the proportion of variance in $y$ explained by $x$:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Where:
- $SS_{res} = \sum_i (y_i - \hat{y}_i)^2$: Residual sum of squares
- $SS_{tot} = \sum_i (y_i - \bar{y})^2$: Total sum of squares

**Range**: $0 \leq R^2 \leq 1$ (can be negative for poor models)

**Interpretation**:
- $R^2 = 0$: Model explains 0% of variance (no better than mean)
- $R^2 = 0.7$: Model explains 70% of variance
- $R^2 = 1$: Model perfectly predicts all points

**For simple linear regression**: $R^2 = r^2$ (square of correlation)

### Other Metrics

**Mean Squared Error (MSE)**:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**:

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE)**:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

## Assumptions of Linear Regression

### 1. Linearity
The relationship between $x$ and $y$ is linear.

**Check**: Scatter plot should show roughly linear pattern.

### 2. Independence
Observations are independent of each other.

**Violation**: Time series data with autocorrelation.

### 3. Homoscedasticity
Constant variance of errors across all levels of $x$.

$$Var(\epsilon_i) = \sigma^2 \text{ for all } i$$

**Check**: Residual plot should show random scatter (no funnel shape).

### 4. Normality of Errors
Errors follow normal distribution: $\epsilon \sim N(0, \sigma^2)$

**Check**: Q-Q plot or histogram of residuals.

## Hypothesis Testing

### Testing Slope Significance

**Null hypothesis**: $H_0: \beta_1 = 0$ (no relationship)

**Alternative**: $H_a: \beta_1 \neq 0$ (significant relationship)

**Test statistic**:

$$t = \frac{\beta_1 - 0}{SE(\beta_1)}$$

Where $SE(\beta_1)$ is the standard error of the slope.

**Decision**: If $|t| > t_{critical}$ or p-value $< \alpha$, reject $H_0$.

### Confidence Interval for Slope

$$(1-\alpha)100\% \text{ CI}: \beta_1 \pm t_{\alpha/2, n-2} \cdot SE(\beta_1)$$

## Advantages

1. **Simplicity**: Easy to understand and interpret
2. **Computational Efficiency**: Fast to train, even with large datasets
3. **Interpretability**: Clear relationship between variables
4. **Baseline Model**: Good starting point before complex models
5. **Analytical Solution**: Closed-form solution (no iterative optimization needed)
6. **Low Overfitting Risk**: With few parameters (only 2)
7. **Inference**: Provides statistical tests and confidence intervals
8. **Extrapolation**: Can make predictions outside training range (with caution)

## Disadvantages

1. **Linearity Assumption**: Cannot capture non-linear relationships
2. **Outlier Sensitivity**: Single outlier can drastically affect the line
3. **Single Predictor**: Limited to one independent variable
4. **Homoscedasticity Required**: Assumes constant variance
5. **No Interaction Effects**: Cannot model relationships between predictors
6. **Limited Complexity**: Cannot model complex real-world patterns
7. **Extrapolation Risk**: Predictions far from data range unreliable

## When to Use Simple Linear Regression

### Appropriate When:
- Clear linear relationship exists
- Single predictor variable
- Need for interpretability is high
- Quick baseline model needed
- Dataset is small
- Assumptions are reasonably met

### Not Appropriate When:
- Relationship is non-linear (use polynomial or non-parametric)
- Multiple predictors available (use multiple regression)
- Outliers heavily present (use robust regression)
- Classification problem (use logistic regression)

## Extensions and Related Methods

### Generalizations
- **Multiple Linear Regression**: Multiple predictors
- **Polynomial Regression**: Non-linear relationships using powers of $x$
- **Ridge/Lasso Regression**: Regularized linear models

### Robust Alternatives
- **RANSAC**: Robust to outliers
- **Huber Regression**: Robust loss function
- **Theil-Sen Estimator**: Median-based robust estimator

### Non-parametric Alternatives
- **LOWESS/LOESS**: Locally weighted regression
- **Spline Regression**: Piecewise polynomial fits

## Practical Considerations

### Feature Scaling
Not required for simple linear regression (doesn't change predictions), but can help with:
- Interpretation when comparing different models
- Numerical stability in some implementations

### Data Preparation
1. **Check for linearity**: Scatter plot
2. **Handle outliers**: Investigate and possibly remove
3. **Transform variables**: Log, square root if needed
4. **Check for influential points**: Cook's distance

### Model Diagnostics

**Residual Plots**:
- Plot residuals vs. fitted values
- Should show random scatter
- Patterns indicate assumption violations

**Q-Q Plot**:
- Check normality of residuals
- Points should follow diagonal line

**Leverage and Influence**:
- Identify points with high leverage (far from mean $x$)
- Check Cook's distance for influential points

## Mathematical Properties

### Best Linear Unbiased Estimator (BLUE)
Under the Gauss-Markov assumptions, OLS is the BLUE:
- **Best**: Minimum variance among all linear unbiased estimators
- **Linear**: Linear function of $y$ values
- **Unbiased**: $E[\hat{\beta}_1] = \beta_1$

### Variance of Estimators

$$Var(\beta_1) = \frac{\sigma^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$Var(\beta_0) = \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}\right)$$

Where $\sigma^2$ is the variance of errors.

**Implications**:
- More spread in $x$ → smaller variance of $\beta_1$ (more precise estimate)
- Larger sample size → smaller variances (more precision)

## Summary

Simple Linear Regression is the foundation of statistical modeling and machine learning. While limited to linear relationships with a single predictor, it provides:
- Intuitive interpretation
- Statistical inference capabilities
- Fast computation
- Transparent decision-making

Understanding simple linear regression is crucial before moving to more complex models like multiple regression, polynomial regression, or non-linear methods. It serves as both a practical tool for appropriate problems and a conceptual foundation for advanced techniques.

---

**Key Takeaway**: Simple linear regression finds the straight line that best fits the data by minimizing the sum of squared vertical distances from points to the line. The slope tells you the rate of change, and R² tells you how well the line fits the data.
