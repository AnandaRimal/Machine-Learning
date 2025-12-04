# Regression Metrics

## Overview and Core Concept

Regression metrics are quantitative measures used to evaluate the performance of regression models—algorithms that predict continuous numerical outcomes. Unlike classification metrics which measure categorical predictions, regression metrics assess how close predicted values are to actual values, quantifying the prediction error in various ways.

The choice of metric profoundly impacts model selection, hyperparameter tuning, and understanding of model performance. Different metrics emphasize different aspects of prediction quality: some are sensitive to outliers, others are scale-dependent, and some provide interpretable units while others are unitless. Understanding these characteristics is essential for selecting the appropriate metric for your specific problem domain and requirements.

Common regression metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared ($R^2$), Adjusted R-squared, Mean Absolute Percentage Error (MAPE), and others. Each has specific mathematical properties, advantages, and ideal use cases.

## Why Understanding Regression Metrics is Important

1. **Model Selection**: Compare different algorithms objectively
2. **Hyperparameter Tuning**: Optimize model parameters based on chosen metric
3. **Business Communication**: Translate model performance to stakeholders
4. **Error Analysis**: Understand where and how model fails
5. **Metric-Specific Optimization**: Different losses optimize different metrics
6. **Outlier Sensitivity**: Some metrics robust, others sensitive to extremes
7. **Scale Dependency**: Understanding units vs. unitless metrics
8. **Interpretability**: Some metrics easier to explain than others
9. **Problem Alignment**: Match metric to business objective
10. **Model Comparison**: Fair evaluation across different approaches

## Mathematical Formulas and Detailed Explanations

### Mean Absolute Error (MAE)

**Formula**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Where:
- $n$: Number of observations
- $y_i$: Actual value for observation $i$
- $\hat{y}_i$: Predicted value for observation $i$
- $|y_i - \hat{y}_i|$: Absolute error

**Characteristics**:
- **Units**: Same as target variable (interpretable)
- **Range**: $[0, \infty)$, lower is better
- **Outlier Sensitivity**: **Robust** (linear penalty)
- **Differentiability**: Not differentiable at zero (affects optimization)

**Interpretation**: Average absolute prediction error

**Example**: If predicting house prices in thousands:
- MAE = 15 means average error is $15,000

**Advantages**:
- Easy to understand and communicate
- Robust to outliers (doesn't square errors)
- Same scale as target
- All errors weighted equally

**Disadvantages**:
- Not differentiable at zero (optimization challenges)
- Doesn't heavily penalize large errors
- Cannot be used for analytical solutions (no closed form for many models)

**When to Use**:
- Outliers are measurement errors (shouldn't dominate loss)
- Want error in interpretable units
- All errors equally important (no special penalty for large mistakes)
- Robust performance desired

### Mean Squared Error (MSE)

**Formula**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Characteristics**:
- **Units**: Squared units of target (less interpretable)
- **Range**: $[0, \infty)$, lower is better
- **Outlier Sensitivity**: **Sensitive** (quadratic penalty)
- **Differentiability**: Smooth, differentiable everywhere

**Interpretation**: Average squared prediction error

**Decomposition** (bias-variance):

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Advantages**:
- Differentiable (easy optimization via gradient descent)
- Penalizes large errors heavily (squared term)
- Analytical solutions exist (e.g., linear regression closed form)
- Connects to maximum likelihood under Gaussian errors
- Theoretical foundation in statistics

**Disadvantages**:
- Units are squared (hard to interpret: dollars² doesn't make intuitive sense)
- Very sensitive to outliers (large errors dominate)
- Scale-dependent (can't compare across different datasets easily)

**When to Use**:
- Large errors are particularly problematic (want quadratic penalty)
- Outliers are genuine extreme cases (not errors)
- Need differentiable loss for optimization
- Gaussian error assumption reasonable

### Root Mean Squared Error (RMSE)

**Formula**:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Characteristics**:
- **Units**: Same as target (interpretable, like MAE)
- **Range**: $[0, \infty)$, lower is better
- **Outlier Sensitivity**: **Sensitive** (inherits from MSE)
- **Differentiability**: Differentiable except at RMSE=0

**Interpretation**: Root mean squared error, roughly average magnitude of error

**Relationship to Standard Deviation**: RMSE is standard deviation of residuals

**Advantages**:
- Same units as target (interpretable)
- Penalizes large errors more than MAE
- More sensitive to variance than MAE
- Commonly reported and understood

**Disadvantages**:
- Sensitive to outliers
- Harder to interpret than MAE (square root of average squared error)
- Scale-dependent

**When to Use**:
- Want interpretable units like MAE but with MSE's outlier sensitivity
- Standard metric in many domains (weather forecasting, etc.)
- Large errors should be penalized more

**Comparison with MAE**:
$$\text{RMSE} \geq \text{MAE}$$

Equality holds only when all errors have same magnitude. Larger difference indicates more variability in errors.

### R-Squared ($R^2$, Coefficient of Determination)

**Formula**:

$$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:
- $\text{SS}_{res} = \sum(y_i - \hat{y}_i)^2$: Residual sum of squares
- $\text{SS}_{tot} = \sum(y_i - \bar{y})^2$: Total sum of squares
- $\bar{y} = \frac{1}{n}\sum y_i$: Mean of target

**Alternative formulation**:

$$R^2 = \frac{\text{SS}_{reg}}{\text{SS}_{tot}} = \frac{\sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where $\text{SS}_{reg}$: Explained sum of squares

**Characteristics**:
- **Range**: $(-\infty, 1]$ for general models; $[0, 1]$ for OLS with intercept
- **Units**: Unitless (percentage-like)
- **Interpretation**: Proportion of variance explained by model
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model no better than mean baseline
- $R^2 < 0$: Model worse than predicting mean

**Advantages**:
- Unitless, comparable across datasets
- Intuitive interpretation (percentage of variance explained)
- Widely reported and understood
- Scale-invariant

**Disadvantages**:
- **Always increases** with more features (even irrelevant ones)
- Can be misleading with overfitting
- Not suitable for non-linear models without intercept
- Can be negative for models that don't minimize SSE

**When to Use**:
- Comparing models on same dataset
- Communicating model quality to non-technical audience
- Linear regression models with intercept
- Want unitless metric

### Adjusted R-Squared ($R^2_{adj}$)

**Formula**:

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where:
- $n$: Number of observations
- $p$: Number of predictors
- $R^2$: Standard R-squared

**Alternative**:

$$R^2_{adj} = 1 - \frac{\text{SS}_{res}/(n-p-1)}{\text{SS}_{tot}/(n-1)}$$

**Characteristics**:
- **Penalizes** additional features
- Can **decrease** when adding irrelevant predictors
- Better for model comparison with different numbers of features
- $R^2_{adj} \leq R^2$ always

**Penalty for complexity**: Adding predictor increases $R^2_{adj}$ only if improvement exceeds penalty.

**Advantages**:
- Accounts for model complexity
- Prevents overfitting via feature proliferation
- Fairer comparison across models with different $p$
- Used in stepwise regression for variable selection

**Disadvantages**:
- Still unitless (like $R^2$)
- Less intuitive than $R^2$
- Still can increase with irrelevant features if they provide slight improvement

**When to Use**:
- Comparing models with different numbers of features
- Preventing overfitting in feature selection
- Balancing fit quality with model complexity

### Mean Absolute Percentage Error (MAPE)

**Formula**:

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Characteristics**:
- **Units**: Percentage (%)
- **Range**: $[0, \infty)$
- **Scale-independent**: Can compare across different datasets
- **Asymmetric**: Penalizes over-predictions and under-predictions differently

**Interpretation**: Average percentage error

**Advantages**:
- Intuitive (everyone understands percentages)
- Scale-independent (compare across datasets)
- Easy to communicate to non-technical stakeholders

**Disadvantages**:
- **Undefined** when $y_i = 0$
- **Asymmetric**: Over-predictions penalized less than under-predictions
- **Infinite** penalty when actual value is near zero
- Biased toward under-prediction
- Not suitable for data with zeros or small values

**When to Use**:
- All actual values significantly greater than zero
- Need scale-independent metric
- Percentage errors meaningful in domain
- Communicating to business stakeholders

**Warning**: Avoid when target can be zero or near-zero!

### Symmetric Mean Absolute Percentage Error (SMAPE)

**Formula** (most common version):

$$\text{SMAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

**Alternative formulation**:

$$\text{SMAPE} = \frac{200\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$

**Characteristics**:
- **Range**: $[0\%, 200\%]$
- More symmetric than MAPE
- Still has issues with zeros

**Advantages**:
- More balanced than MAPE
- Treats over and under-predictions more equally

**Disadvantages**:
- Multiple definitions in literature (ambiguity)
- Still problematic with zeros
- Can behave unexpectedly at low values

### Mean Squared Logarithmic Error (MSLE)

**Formula**:

$$\text{MSLE} = \frac{1}{n}\sum_{i=1}^{n}(\log(y_i + 1) - \log(\hat{y}_i + 1))^2$$

**Characteristics**:
- **Requires**: $y_i, \hat{y}_i \geq 0$ (non-negative targets)
- **Penalty**: Relative errors rather than absolute
- **Symmetry**: Under and over-predictions penalized similarly on log scale

**Advantages**:
- Robust to large values (compression via log)
- Penalizes under-prediction of small values more than RMSE
- Good for targets spanning orders of magnitude
- Relatively outlier-robust

**Disadvantages**:
- Only for non-negative targets
- Less interpretable than MAE/RMSE
- Biased toward under-prediction in original scale

**When to Use**:
- Target spans many orders of magnitude (e.g., 1 to 1,000,000)
- Care more about relative errors than absolute
- Predicting counts, amounts, prices that vary widely

### Median Absolute Error

**Formula**:

$$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, ..., |y_n - \hat{y}_n|)$$

**Characteristics**:
- **Robust**: Very resistant to outliers (50% breakdown point)
- **Units**: Same as target
- **Interpretation**: Median magnitude of errors

**Advantages**:
- Extremely robust to outliers
- Interpretable units
- Good for skewed error distributions

**Disadvantages**:
- Ignores magnitude of large errors (too robust for some applications)
- Not differentiable (optimization challenges)
- Less commonly used/understood

**When to Use**:
- Heavy outliers present
- Want robust central tendency of errors
- Median more meaningful than mean for your error distribution

## Detailed Worked Example

**Problem**: Predicting house prices (in $1000s)

**Data** (5 houses):

| House | Actual Price ($y$) | Predicted Price ($\hat{y}$) | Error | Abs Error | Squared Error |
|-------|--------------------|-----------------------------|-------|-----------|---------------|
| 1     | 250                | 240                         | 10    | 10        | 100           |
| 2     | 300                | 320                         | -20   | 20        | 400           |
| 3     | 200                | 195                         | 5     | 5         | 25            |
| 4     | 400                | 390                         | 10    | 10        | 100           |
| 5     | 350                | 380                         | -30   | 30        | 900           |

**Mean of actuals**: $\bar{y} = \frac{250+300+200+400+350}{5} = 300$

### Calculate MAE

$$\text{MAE} = \frac{1}{5}(10 + 20 + 5 + 10 + 30) = \frac{75}{5} = 15$$

**Interpretation**: On average, predictions are off by $15,000.

### Calculate MSE

$$\text{MSE} = \frac{1}{5}(100 + 400 + 25 + 100 + 900) = \frac{1525}{5} = 305$$

**Units**: $(1000s)^2$ – hard to interpret directly.

### Calculate RMSE

$$\text{RMSE} = \sqrt{305} \approx 17.46$$

**Interpretation**: Root mean squared error is about $17,460.

**Comparison**: RMSE (17.46) > MAE (15) indicates error variance (large error on house 5).

### Calculate $R^2$

**Residual sum of squares**:

$$\text{SS}_{res} = 100 + 400 + 25 + 100 + 900 = 1525$$

**Total sum of squares**:

$$\text{SS}_{tot} = (250-300)^2 + (300-300)^2 + (200-300)^2 + (400-300)^2 + (350-300)^2$$
$$= 2500 + 0 + 10000 + 10000 + 2500 = 25000$$

$$R^2 = 1 - \frac{1525}{25000} = 1 - 0.061 = 0.939$$

**Interpretation**: Model explains 93.9% of variance in house prices.

### Calculate Adjusted $R^2$

Assume $p = 3$ predictors (size, bedrooms, age):

$$R^2_{adj} = 1 - \frac{(1-0.939)(5-1)}{5-3-1} = 1 - \frac{0.061 \times 4}{1} = 1 - 0.244 = 0.756$$

**Interpretation**: After adjusting for 3 predictors, model explains 75.6% of variance.

**Note**: With small $n$ (5) and moderate $p$ (3), penalty is substantial.

### Calculate MAPE

$$\text{MAPE} = \frac{100\%}{5}\left(\frac{10}{250} + \frac{20}{300} + \frac{5}{200} + \frac{10}{400} + \frac{30}{350}\right)$$
$$= \frac{100\%}{5}(0.04 + 0.0667 + 0.025 + 0.025 + 0.0857)$$
$$= \frac{100\%}{5}(0.2424) = 4.85\%$$

**Interpretation**: On average, predictions are off by 4.85%.

### Calculate MedAE

**Sorted absolute errors**: $[5, 10, 10, 20, 30]$

**Median**: Middle value = 10

$$\text{MedAE} = 10$$

**Interpretation**: Median error is $10,000, more robust than MAE to the outlier (house 5).

### Summary Table

| Metric          | Value  | Interpretation                       |
|-----------------|--------|--------------------------------------|
| MAE             | 15     | Avg error: $15,000                   |
| MSE             | 305    | Hard to interpret (squared units)    |
| RMSE            | 17.46  | RMS error: $17,460                   |
| $R^2$           | 0.939  | Explains 93.9% of variance           |
| $R^2_{adj}$     | 0.756  | Adjusted for 3 predictors            |
| MAPE            | 4.85%  | Avg percentage error: 4.85%          |
| MedAE           | 10     | Median error: $10,000                |

**Insights**:
- $RMSE > MAE$: Indicates error variance (some large errors)
- High $R^2$ but lower $R^2_{adj}$: Many predictors relative to data
- MAPE low: Good relative performance
- MedAE < MAE: Outlier (house 5) pulls MAE up

## Choosing the Right Metric

### Decision Framework

**Consider these factors**:

1. **Outlier Sensitivity**:
   - **Robust needed**: MAE, MedAE
   - **Sensitive acceptable**: RMSE, MSE

2. **Units**:
   - **Interpretable units**: MAE, RMSE
   - **Unitless**: $R^2$, $R^2_{adj}$, MAPE

3. **Scale**:
   - **Scale-dependent**: MAE, RMSE, MSE
   - **Scale-independent**: $R^2$, MAPE, MSLE

4. **Business Objective**:
   - **Large errors catastrophic**: MSE, RMSE
   - **All errors equally bad**: MAE
   - **Percentage matters**: MAPE
   - **Variance explanation**: $R^2$

5. **Target Characteristics**:
   - **Can be zero**: Avoid MAPE
   - **Wide range**: MSLE, MAPE
   - **Outliers present**: MAE, MedAE

6. **Model Comparison**:
   - **Same dataset**: Any metric
   - **Different datasets**: $R^2$, MAPE

7. **Optimization**:
   - **Need differentiability**: MSE
   - **Robust loss**: MAE (use subgradients)

### Common Recommendations by Domain

**House Price Prediction**:
- Primary: RMSE (interpretable, penalizes large errors)
- Secondary: $R^2$ (for variance explanation)

**Sales Forecasting**:
- Primary: MAPE (percentage error intuitive for business)
- Secondary: MAE (absolute error for planning)

**Stock Price Prediction**:
- Primary: RMSE (sensitive to large deviations)
- Avoid: MAPE (can be near zero)

**Temperature Prediction**:
- Primary: MAE (robust to outliers)
- Secondary: RMSE (standard in meteorology)

**Medical Dosage**:
- Primary: MAE or RMSE depending on cost of errors
- Careful: Balance over-dosing vs under-dosing risks

## Advantages and Disadvantages Summary

### MAE

**Advantages**: Interpretable units, robust to outliers, simple
**Disadvantages**: Not differentiable at zero, doesn't heavily penalize large errors

### MSE

**Advantages**: Differentiable, penalizes large errors, theoretical foundation
**Disadvantages**: Squared units (uninterpretable), very sensitive to outliers

### RMSE

**Advantages**: Interpretable units, penalizes large errors, widely used
**Disadvantages**: Sensitive to outliers, more complex than MAE

### $R^2$

**Advantages**: Unitless, intuitive percentage interpretation, comparable across datasets
**Disadvantages**: Always increases with features, can mislead about overfitting

### $R^2_{adj}$

**Advantages**: Penalizes complexity, better for model comparison
**Disadvantages**: Less intuitive than $R^2$, still can mislead

### MAPE

**Advantages**: Percentage (intuitive), scale-independent
**Disadvantages**: Undefined for zeros, asymmetric, biased toward under-prediction

### MSLE

**Advantages**: Handles wide ranges, relative errors, robust to large values
**Disadvantages**: Only for non-negative, less interpretable

### MedAE

**Advantages**: Very robust, interpretable units
**Disadvantages**: Ignores outlier magnitude, less common

## Related Concepts

### Residual Analysis

**Residuals**: $e_i = y_i - \hat{y}_i$

**Ideal residuals**:
- Mean zero: $\bar{e} = 0$
- Constant variance (homoscedastic)
- Normally distributed
- No patterns vs. fitted values or features

### Bias and Variance

**Mean Bias**:

$$\text{Bias} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

**Variance of predictions**:

$$\text{Var}(\hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - \bar{\hat{y}})^2$$

### Information Criteria

**AIC** (Akaike Information Criterion):

$$\text{AIC} = n\log(\text{MSE}) + 2p$$

**BIC** (Bayesian Information Criterion):

$$\text{BIC} = n\log(\text{MSE}) + p\log(n)$$

Lower values indicate better models. Balance fit and complexity.

## Summary

Regression metrics are essential tools for evaluating and comparing regression models. Each metric captures different aspects of prediction quality and has specific strengths and weaknesses.

**Key Metrics**:

**MAE**: Robust, interpretable, treats all errors equally
**MSE/RMSE**: Sensitive to outliers, penalizes large errors, widely used
**$R^2$**: Variance explained, unitless, intuitive but increases with features
**$R^2_{adj}$**: Adjusts for model complexity
**MAPE**: Percentage-based, scale-independent, problematic with zeros
**MSLE**: For wide-range targets, relative errors
**MedAE**: Very robust to outliers

**Selection Guide**:
- **Outliers present**: MAE or MedAE
- **Large errors costly**: RMSE or MSE
- **Interpretability**: MAE or RMSE
- **Percentage context**: MAPE (if no zeros)
- **Wide target range**: MSLE
- **Variance explanation**: $R^2$
- **Model comparison (different p)**: $R^2_{adj}$

**Best Practice**: Report multiple metrics to provide comprehensive view of model performance. No single metric tells the complete story.

Understanding these metrics enables informed model selection, effective hyperparameter tuning, and clear communication of model performance to both technical and non-technical audiences.
