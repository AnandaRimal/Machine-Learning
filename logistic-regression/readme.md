# Logistic Regression

## Overview and Core Concept

Logistic Regression is a fundamental statistical and machine learning algorithm for binary classification that models the probability of an instance belonging to a particular class. Despite its name containing "regression," it is a classification algorithm that predicts probabilities constrained between 0 and 1 using the logistic (sigmoid) function.

The core idea is to model the log-odds (logit) of the probability as a linear combination of input features. By applying the logistic function to this linear combination, we obtain probabilities that naturally satisfy the constraint $0 \leq P(y=1|\mathbf{x}) \leq 1$. The decision boundary created is linear in feature space, making it a linear classifier.

Developed in the 1940s-1950s and refined over decades, logistic regression remains one of the most widely used classification algorithms due to its interpretability, computational efficiency, probabilistic output, and effectiveness as a baseline model.

## Why Use Logistic Regression?

1. **Probabilistic Output**: Provides probability estimates, not just class labels
2. **Interpretability**: Coefficients have clear meaning (log-odds ratios)
3. **Computational Efficiency**: Fast training and prediction
4. **Baseline Model**: Excellent starting point for classification
5. **Well-Calibrated**: Probability estimates generally reliable
6. **No Distribution Assumptions**: Unlike LDA, doesn't assume feature distributions
7. **Regularization**: Can add L1/L2 penalties for high dimensions
8. **Binary and Multi-Class**: Extends to multiple classes
9. **Online Learning**: Can update incrementally
10. **Feature Importance**: Coefficients indicate feature relevance

## Mathematical Formulas

### Logistic (Sigmoid) Function

**Definition**:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

**Properties**:
- **Domain**: $z \in (-\infty, \infty)$
- **Range**: $\sigma(z) \in (0, 1)$
- **Monotonic**: Strictly increasing
- **Symmetric**: $\sigma(-z) = 1 - \sigma(z)$
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

**Behavior**:
- $\sigma(0) = 0.5$
- $z \to \infty$: $\sigma(z) \to 1$
- $z \to -\infty$: $\sigma(z) \to 0$
- $z > 5$: $\sigma(z) > 0.99$
- $z < -5$: $\sigma(z) < 0.01$

### Logistic Regression Model

**Probability of positive class**:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

Where:
- $\mathbf{x} = [x_1, x_2, ..., x_p]^T$: Feature vector
- $\mathbf{w} = [w_1, w_2, ..., w_p]^T$: Weight vector (coefficients)
- $b$: Bias (intercept)
- $z = \mathbf{w}^T\mathbf{x} + b$: Linear combination (log-odds)

**Probability of negative class**:

$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = \frac{1}{1 + e^{\mathbf{w}^T\mathbf{x} + b}}$$

**Compact notation** (incorporating bias into $\mathbf{w}$):

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$$

### Log-Odds (Logit)

**Definition**:

$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = \log(\text{odds})$$

**For logistic regression**:

$$\log\left(\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})}\right) = \mathbf{w}^T\mathbf{x} + b$$

**Key insight**: Log-odds is linear in features!

**Odds**:

$$\text{odds} = \frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = e^{\mathbf{w}^T\mathbf{x} + b}$$

### Decision Boundary

**Classification rule** (threshold $t = 0.5$):

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Equivalently**:

$$\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w}^T\mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

**Decision boundary** (hyperplane where $P(y=1|\mathbf{x}) = 0.5$):

$$\mathbf{w}^T\mathbf{x} + b = 0$$

**Geometric interpretation**: Linear boundary separating classes in feature space.

### Loss Function (Log-Loss / Binary Cross-Entropy)

**For single sample** $(x_i, y_i)$:

$$L_i = -[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

Where $\hat{p}_i = P(y=1|\mathbf{x}_i)$ is the predicted probability.

**For entire dataset**:

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

**Negative log-likelihood formulation**:

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\log P(y_i|\mathbf{x}_i)$$

**Properties**:
- **Convex**: Guaranteed global minimum
- **Smooth**: Differentiable everywhere
- **Penalizes confidence**: Wrong predictions with high confidence penalized heavily

### Gradient

**Gradient of log-loss**:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)\mathbf{x}_i = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$$

**Remarkably simple form**: Similar to linear regression gradient!

**Gradient descent update**:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L(\mathbf{w}^{(t)})$$

Where $\eta$ is the learning rate.

### Maximum Likelihood Estimation (MLE)

**Likelihood** (assuming independence):

$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n}P(y_i|\mathbf{x}_i) = \prod_{i=1}^{n}\hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$$

**Log-likelihood**:

$$\ell(\mathbf{w}) = \sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

**MLE principle**: Find $\mathbf{w}$ that maximizes $\ell(\mathbf{w})$

**Equivalent to minimizing log-loss**:

$$\hat{\mathbf{w}}_{MLE} = \arg\max_{\mathbf{w}} \ell(\mathbf{w}) = \arg\min_{\mathbf{w}} L(\mathbf{w})$$

### Regularized Logistic Regression

**L2 Regularization (Ridge)**:

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)] + \lambda||\mathbf{w}||^2_2$$

**L1 Regularization (Lasso)**:

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)] + \lambda||\mathbf{w}||_1$$

**Elastic Net**:

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)] + \lambda[\alpha||\mathbf{w}||_1 + (1-\alpha)||\mathbf{w}||^2_2]$$

**Purpose**: Prevent overfitting, handle high dimensions, feature selection (L1).

## Detailed Worked Example

**Problem**: Predict whether a student passes (1) or fails (0) based on study hours

**Data** (10 students):

| Student | Study Hours $(x)$ | Pass $(y)$ |
|---------|-------------------|------------|
| 1       | 1                 | 0          |
| 2       | 2                 | 0          |
| 3       | 3                 | 0          |
| 4       | 4                 | 0          |
| 5       | 5                 | 1          |
| 6       | 6                 | 1          |
| 7       | 7                 | 1          |
| 8       | 8                 | 1          |
| 9       | 9                 | 1          |
| 10      | 10                | 1          |

### Step 1: Initialize Parameters

**Model**: $P(y=1|x) = \sigma(wx + b)$

**Initial values**: $w = 0, b = 0$

### Step 2: Compute Predictions (Iteration 0)

For all $x$, initial prediction:

$$P(y=1|x) = \sigma(0 \cdot x + 0) = \sigma(0) = 0.5$$

**Predicted probabilities**: All 0.5

### Step 3: Compute Loss (Iteration 0)

$$L = -\frac{1}{10}\sum_{i=1}^{10}[y_i\log(0.5) + (1-y_i)\log(0.5)]$$
$$= -\frac{1}{10} \cdot 10 \cdot \log(0.5) = -\log(0.5) \approx 0.693$$

### Step 4: Compute Gradients

**Gradient w.r.t. $w$**:

$$\frac{\partial L}{\partial w} = \frac{1}{10}\sum_{i=1}^{10}(0.5 - y_i) \cdot x_i$$
$$= \frac{1}{10}[(0.5-0) \cdot 1 + (0.5-0) \cdot 2 + ... + (0.5-1) \cdot 10]$$
$$= \frac{1}{10}[0.5(1+2+3+4) - 0.5(6+7+8+9+10)]$$
$$= \frac{1}{10}[0.5 \cdot 10 - 0.5 \cdot 40] = \frac{1}{10}[-15] = -1.5$$

**Gradient w.r.t. $b$**:

$$\frac{\partial L}{\partial b} = \frac{1}{10}\sum_{i=1}^{10}(0.5 - y_i) = \frac{1}{10}[0.5 \cdot 4 - 0.5 \cdot 6] = -0.1$$

### Step 5: Update Parameters

**Learning rate**: $\eta = 0.1$

$$w^{(1)} = w^{(0)} - \eta \cdot \frac{\partial L}{\partial w} = 0 - 0.1 \cdot (-1.5) = 0.15$$

$$b^{(1)} = b^{(0)} - \eta \cdot \frac{\partial L}{\partial b} = 0 - 0.1 \cdot (-0.1) = 0.01$$

### Step 6: New Predictions (Iteration 1)

For $x=1$: $P(y=1|x=1) = \sigma(0.15 \cdot 1 + 0.01) = \sigma(0.16) \approx 0.540$

For $x=5$: $P(y=1|x=5) = \sigma(0.15 \cdot 5 + 0.01) = \sigma(0.76) \approx 0.681$

For $x=10$: $P(y=1|x=10) = \sigma(0.15 \cdot 10 + 0.01) = \sigma(1.51) \approx 0.819$

**Progress**: Probabilities now vary with $x$ (higher for larger $x$).

### Step 7: Continue Iterations

**After convergence** (e.g., 1000 iterations):

$$w^* \approx 0.8, \quad b^* \approx -4.0$$

**Converged model**:

$$P(y=1|x) = \frac{1}{1 + e^{-(0.8x - 4)}}$$

### Step 8: Interpret Coefficients

**Weight $w = 0.8$**: 

Each additional study hour increases log-odds by 0.8.

**Odds ratio**: $e^{0.8} \approx 2.23$

Each hour multiplies odds of passing by 2.23.

**Bias $b = -4$**:

At $x=0$ (no study), log-odds = -4, so $P(y=1|x=0) = \sigma(-4) \approx 0.018$ (very low).

### Step 9: Decision Boundary

**Where $P(y=1|x) = 0.5$**:

$$0.8x - 4 = 0 \Rightarrow x = 5$$

**Interpretation**: Students with 5 hours have 50% chance. Below 5 → fail, above 5 → pass.

### Step 10: Predictions for New Students

**Student A** (3 hours):

$$P(pass|x=3) = \sigma(0.8 \cdot 3 - 4) = \sigma(-1.6) \approx 0.168$$

**Prediction**: Fail (probability < 0.5)

**Student B** (7 hours):

$$P(pass|x=7) = \sigma(0.8 \cdot 7 - 4) = \sigma(1.6) \approx 0.832$$

**Prediction**: Pass (probability > 0.5)

### Summary of Example

- **Converged model**: $P(y=1|x) = \sigma(0.8x - 4)$
- **Decision boundary**: $x = 5$ hours
- **Interpretation**: Each study hour multiplies odds of passing by ~2.23
- **Prediction**: Use probability threshold (typically 0.5) for classification

## Coefficient Interpretation

### Odds and Odds Ratios

**Odds** of event $y=1$ at feature value $x$:

$$\text{odds}(x) = \frac{P(y=1|x)}{P(y=0|x)} = e^{wx + b}$$

**Odds ratio** for one-unit increase in $x_j$:

$$\text{OR} = \frac{\text{odds}(x_j + 1)}{\text{odds}(x_j)} = \frac{e^{w_j(x_j+1)}}{e^{w_j x_j}} = e^{w_j}$$

**Interpretation**:
- $w_j > 0$: $\text{OR} > 1$ → Feature increases odds of positive class
- $w_j < 0$: $\text{OR} < 1$ → Feature decreases odds
- $w_j = 0$: $\text{OR} = 1$ → Feature has no effect

**Example**: If $w_{\text{age}} = 0.05$:
- $\text{OR} = e^{0.05} \approx 1.051$
- Each year increases odds by 5.1%

### Marginal Effects

**Marginal effect** of $x_j$ on probability:

$$\frac{\partial P(y=1|\mathbf{x})}{\partial x_j} = w_j \cdot P(y=1|\mathbf{x}) \cdot P(y=0|\mathbf{x})$$

**Key insight**: Effect depends on current probability! Maximum effect at $P = 0.5$.

**Average marginal effect** (AME):

$$\text{AME}_j = \frac{1}{n}\sum_{i=1}^{n}w_j \cdot \hat{p}_i(1-\hat{p}_i)$$

### Statistical Significance

**Standard errors**: Computed from Hessian of log-likelihood (Fisher information matrix)

**Z-statistic**:

$$z_j = \frac{\hat{w}_j}{SE(\hat{w}_j)}$$

**P-value**: Test $H_0: w_j = 0$ (feature has no effect)

**Confidence interval**:

$$\hat{w}_j \pm z_{\alpha/2} \cdot SE(\hat{w}_j)$$

## Assumptions

### 1. Binary Outcome

Target variable is binary (0 or 1).

**Extensions**: Multinomial logistic regression for multi-class.

### 2. Independence

Observations are independent.

**Violation**: Time series, spatial data → use specialized models.

### 3. Linearity in Log-Odds

Log-odds is linear in features:

$$\log\left(\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})}\right) = \mathbf{w}^T\mathbf{x} + b$$

**Check**: Plot log-odds vs. features (should be roughly linear).

**Solution if violated**: Add polynomial terms, interactions, or use non-linear classifiers.

### 4. No Multicollinearity

Features should not be too highly correlated.

**Problem**: Unstable coefficients, large standard errors.

**Check**: VIF (Variance Inflation Factor).

**Solution**: Remove correlated features, use regularization (L1/L2).

### 5. Large Sample Size

MLE is asymptotic—requires sufficient data.

**Rule of thumb**: At least 10-15 observations per feature for each class.

**Small samples**: Consider regularization or simpler models.

### 6. No Extreme Outliers

Outliers in feature space can be influential.

**Check**: Leverage, Cook's distance.

**Solution**: Remove outliers, use robust methods.

### 7. Complete Separation (Avoid)

**Perfect separation**: Feature(s) perfectly separate classes.

**Problem**: MLE undefined (coefficients → $\pm\infty$).

**Symptoms**: Very large coefficients, optimization doesn't converge.

**Solutions**:
- Use regularization (L1 or L2)
- Remove perfectly separating features
- Use penalized maximum likelihood

## Advantages

1. **Probability Estimates**: Provides calibrated probabilities, not just labels
2. **Interpretable**: Coefficients represent log-odds ratios
3. **Efficient**: Fast training and prediction
4. **No Distribution Assumptions**: Unlike LDA/QDA
5. **Robust**: Generally stable with proper regularization
6. **Extensions**: Easily extends to multi-class, regularization
7. **Online Learning**: Supports incremental updates
8. **Well-Studied**: Extensive statistical theory
9. **Baseline**: Excellent first model
10. **Feature Importance**: Coefficients indicate relevance

## Disadvantages

1. **Linear Decision Boundary**: Can't capture complex non-linear relationships
2. **Feature Engineering**: Requires manual creation of interactions/polynomials
3. **Assumption Violations**: Performance degrades if assumptions not met
4. **Multicollinearity**: Sensitive to highly correlated features
5. **Outliers**: Can be influenced by extreme values
6. **Complete Separation**: Fails with perfect class separation (without regularization)
7. **Imbalanced Data**: May bias toward majority class
8. **Not Best for Complex**: Outperformed by trees/neural nets on complex patterns
9. **Standardization**: Benefits from feature scaling
10. **Interpretability vs. Performance**: Simple model may sacrifice accuracy

## When to Use Logistic Regression

### Ideal Scenarios

1. **Binary Classification**: Two-class problems
2. **Probability Needed**: When probability estimates important (not just labels)
3. **Interpretability**: When understanding feature effects is crucial
4. **Linearly Separable**: Classes roughly separable by hyperplane
5. **Baseline**: As first model before trying complex methods
6. **Small to Medium Data**: Works well without massive datasets
7. **Fast Inference**: Real-time prediction requirements
8. **Statistical Inference**: Need hypothesis testing, confidence intervals
9. **Regularization**: High-dimensional sparse problems (with L1)
10. **Benchmark**: Standard comparison point

### Application Domains

**Medicine**:
- Disease diagnosis (presence/absence)
- Patient risk stratification
- Clinical trial outcome prediction

**Finance**:
- Credit default prediction
- Fraud detection
- Churn prediction

**Marketing**:
- Customer conversion
- Email click-through prediction
- Ad response modeling

**HR**:
- Employee attrition
- Hiring success prediction

**E-commerce**:
- Purchase likelihood
- Recommendation click probability

## When NOT to Use Logistic Regression

1. **Complex Non-Linear**: Intricate decision boundaries (use trees, neural nets, SVM with non-linear kernel)
2. **Many Interactions**: High-order interactions between features (tree-based methods)
3. **Image/Text Raw Features**: High-dimensional unstructured data (deep learning)
4. **Perfectly Separable**: Without regularization, will have convergence issues
5. **Multi-Class with Structure**: Hierarchical classes (specialized methods)
6. **Severe Class Imbalance**: Without proper handling (resampling, cost-sensitive learning)
7. **Time Series**: Strong temporal dependencies (use LSTM, RNN, ARIMA)
8. **Spatial Data**: Spatial autocorrelation (use spatial models)

## Related Methods

### Extensions

**Multinomial Logistic Regression (Softmax Regression)**
- Multi-class generalization (K > 2 classes)
- Softmax function: $P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^T\mathbf{x}}}$

**Ordinal Logistic Regression**
- For ordered categories (e.g., low/medium/high)
- Respects ordinal structure

**Regularized Logistic Regression**
- L1 (Lasso): Sparse models, feature selection
- L2 (Ridge): Handle multicollinearity
- Elastic Net: Combination

**Polynomial Logistic Regression**
- Add polynomial features for non-linearity
- Still linear in parameters

### Related Classifiers

**Linear Discriminant Analysis (LDA)**
- Also linear classifier
- Assumes Gaussian features with equal covariance
- Can be more efficient if assumptions met

**Support Vector Machine (SVM)**
- Maximum margin classifier
- Linear SVM: Similar decision boundary
- Kernel SVM: Non-linear boundaries
- No probability estimates by default

**Naive Bayes**
- Probabilistic classifier
- Assumes feature independence
- Very fast, good for high dimensions

**Perceptron**
- Simplest neural network
- Linear classifier (no probabilities)
- Online learning

### Non-Linear Alternatives

**Decision Trees**
- Non-linear, axis-aligned boundaries
- Interpretable, no scaling needed

**Random Forest**
- Ensemble of trees
- Handles non-linearity, interactions
- Less interpretable

**Gradient Boosting (XGBoost, LightGBM)**
- State-of-art for tabular data
- Complex non-linear patterns

**Neural Networks**
- Universal function approximators
- For complex patterns, large data

## Summary

Logistic Regression is a fundamental probabilistic linear classifier that models the probability of binary outcomes using the logistic (sigmoid) function. It learns a linear decision boundary while providing interpretable probability estimates.

**Core Formula**:

$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Key Properties**:
- **Output**: Probabilities between 0 and 1
- **Decision boundary**: Linear hyperplane
- **Training**: Maximum likelihood via gradient descent
- **Loss**: Log-loss (binary cross-entropy), convex
- **Interpretation**: Coefficients are log-odds ratios

**Strengths**:
- Probability estimates
- Interpretable coefficients
- Computationally efficient
- Well-calibrated probabilities
- Regularization support

**Limitations**:
- Linear decision boundary only
- Requires feature engineering for non-linearity
- Sensitive to multicollinearity
- Assumption violations impact performance

**When to Use**:
- Binary classification
- Need probabilities
- Interpretability important
- Baseline modeling
- Linearly separable data

**Alternatives**:
- **LDA**: If Gaussian assumptions met
- **SVM**: For maximum margin, kernel tricks
- **Trees/Forests**: Non-linear patterns
- **Neural Networks**: Complex, large-scale

Logistic regression remains a cornerstone of machine learning, balancing simplicity, interpretability, and effectiveness. It serves as an essential tool for understanding classification fundamentals and provides an excellent baseline against which more complex models are compared.
