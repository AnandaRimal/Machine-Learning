# Gradient Boosting

## Table of Contents
- [Introduction](#introduction)
- [What is Gradient Boosting?](#what-is-gradient-boosting)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use Gradient Boosting?](#why-use-gradient-boosting)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Algorithm Details](#algorithm-details)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Practical Applications](#practical-applications)

## Introduction

Gradient Boosting is one of the most powerful machine learning techniques available today. It builds an ensemble of weak learners (typically decision trees) sequentially, where each new tree corrects the errors of the previous ensemble by fitting to the **residuals** (errors).

The key innovation: Instead of adjusting sample weights (like AdaBoost), Gradient Boosting fits new models to the **negative gradient** of the loss function. This makes it a general framework that works with any differentiable loss function.

Popular implementations:
- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)
- **CatBoost** (Categorical Boosting)
- **Scikit-learn's GradientBoostingClassifier/Regressor**

## What is Gradient Boosting?

Gradient Boosting is an ensemble method that:

1. Builds models sequentially
2. Each new model fits the **residuals** (negative gradient of loss)
3. Combines models through additive modeling
4. Uses gradient descent in function space

### Key Concepts

**Additive Model**: Final prediction is sum of weak learners

$$F_M(\mathbf{x}) = \sum_{m=0}^{M} \nu h_m(\mathbf{x})$$

where $\nu$ is the learning rate and $h_m$ are weak learners.

**Gradient Descent in Function Space**: Instead of optimizing parameters, optimize the function itself

**Residual Fitting**: Each tree fits the residuals of the previous ensemble

$$\text{residual}_i = y_i - F_{m-1}(\mathbf{x}_i)$$

**Shrinkage (Learning Rate)**: Scale each tree's contribution

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu h_m(\mathbf{x})$$

## Mathematical Foundation

### Objective Function

Minimize loss over training data:

$$\min_{F} \mathcal{L}(F) = \sum_{i=1}^{n} L(y_i, F(\mathbf{x}_i))$$

where:
- $L(\cdot, \cdot)$ is a differentiable loss function
- $F(\mathbf{x})$ is the ensemble model

### Gradient Descent Analogy

**Standard Gradient Descent** (parameter space):
$$\theta_{t+1} = \theta_t - \nu \nabla_\theta L(\theta_t)$$

**Gradient Boosting** (function space):
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) - \nu \nabla_F L(y, F_{m-1}(\mathbf{x}))$$

### Negative Gradient (Pseudo-Residuals)

The negative gradient of the loss with respect to predictions:

$$r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$$

This is what we fit the next tree to!

### Common Loss Functions

**Regression**:

1. **Mean Squared Error (L2)**:
   $$L(y, F) = \frac{1}{2}(y - F)^2$$
   $$r = -(y - F) = y - F$$
   (Simple residuals!)

2. **Mean Absolute Error (L1)**:
   $$L(y, F) = |y - F|$$
   $$r = \text{sign}(y - F)$$

**Classification** (binary):

1. **Log Loss (Logistic)**:
   $$L(y, F) = \log(1 + e^{-yF})$$
   $$r = \frac{y}{1 + e^{yF}}$$

2. **Exponential Loss** (AdaBoost):
   $$L(y, F) = e^{-yF}$$
   $$r = ye^{-yF}$$

### Gradient Boosting Algorithm (Regression)

**Input**:
- Training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$
- Loss function $L(y, F)$
- Number of iterations $M$
- Learning rate $\nu$

**Algorithm**:

1. **Initialize** with constant value:
   $$F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$$
   
   For MSE: $F_0 = \bar{y}$ (mean)

2. **For $m = 1$ to $M$**:

   a. Compute pseudo-residuals:
   $$r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$$
   
   For MSE: $r_{im} = y_i - F_{m-1}(\mathbf{x}_i)$
   
   b. Fit tree $h_m$ to residuals $\{(\mathbf{x}_i, r_{im})\}_{i=1}^n$
   
   c. **Line search**: Find optimal step size for each leaf region $R_{jm}$:
   $$\gamma_{jm} = \arg\min_\gamma \sum_{\mathbf{x}_i \in R_{jm}} L(y_i, F_{m-1}(\mathbf{x}_i) + \gamma)$$
   
   d. Update model:
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{1}(\mathbf{x} \in R_{jm})$$

3. **Output**: $F_M(\mathbf{x})$

### Learning Rate Trade-off

Small learning rate $\nu$ (e.g., 0.01-0.1):
- **Pros**: Better generalization, smoother learning
- **Cons**: Needs more trees, slower training

Large learning rate (e.g., 0.5-1.0):
- **Pros**: Faster convergence, fewer trees needed
- **Cons**: Risk of overfitting, less smooth

**Optimal strategy**: Use small $\nu$ with early stopping.

## Why Use Gradient Boosting?

### 1. **State-of-the-Art Performance**
Often wins Kaggle competitions and achieves best results on tabular data.

### 2. **Handles Any Differentiable Loss**
Can optimize custom loss functions for specific problems.

### 3. **Captures Complex Patterns**
Non-linear relationships without manual feature engineering.

### 4. **Built-in Feature Importance**
Identifies most predictive features.

### 5. **Robust to Outliers**
Especially with robust loss functions (MAE, Huber).

### 6. **Handles Missing Data**
Modern implementations handle missing values natively.

### 7. **No Feature Scaling Needed**
Tree-based models are scale-invariant.

## Advantages and Disadvantages

### Advantages

1. **Excellent Performance**: Often best on structured/tabular data
2. **Flexible**: Works with various loss functions
3. **Feature Importance**: Built-in interpretability
4. **Handles Mixed Data**: Numerical and categorical features
5. **Robust**: Resistant to outliers (with appropriate loss)
6. **No Preprocessing**: Minimal data preparation needed
7. **Regularization Options**: L1, L2, dropout, subsampling
8. **Missing Data**: Native handling in XGBoost, LightGBM
9. **Versatile**: Classification, regression, ranking, etc.

### Disadvantages

1. **Sequential Training**: Cannot parallelize across trees
2. **Sensitive to Hyperparameters**: Requires tuning
3. **Overfitting Risk**: Easy to overfit with too many trees
4. **Training Time**: Slower than Random Forest
5. **Less Interpretable**: Harder to explain than single tree
6. **Memory Usage**: Can be memory-intensive
7. **Not for Sparse High-D**: Better alternatives for text/images

## Algorithm Details

### Gradient Boosting for Regression (MSE Loss)

```
Initialize: F_0(x) = mean(y)

For m = 1 to M:
    1. Compute residuals:
       r_i = y_i - F_{m-1}(x_i)  for all i
    
    2. Fit regression tree h_m to {(x_i, r_i)}
       - Grow tree with max_depth, min_samples_leaf
       - Creates regions R_jm (leaf nodes)
    
    3. For each leaf j in tree m:
       - Compute optimal value:
         γ_jm = mean(r_i for x_i in R_jm)
    
    4. Update model:
       F_m(x) = F_{m-1}(x) + ν * h_m(x)

Return: F_M(x)
```

### Gradient Boosting for Classification (Log Loss)

```
Initialize: F_0(x) = log(p / (1-p))  where p = mean(y)

For m = 1 to M:
    1. Compute probabilities:
       p_i = 1 / (1 + exp(-F_{m-1}(x_i)))
    
    2. Compute pseudo-residuals:
       r_i = y_i - p_i
    
    3. Fit tree h_m to {(x_i, r_i)}
    
    4. For each leaf j:
       γ_jm = Σ r_i / Σ p_i(1-p_i)  for x_i in R_jm
    
    5. Update:
       F_m(x) = F_{m-1}(x) + ν * h_m(x)

Final prediction: p(x) = 1 / (1 + exp(-F_M(x)))
```

### Regularization Techniques

1. **Shrinkage (Learning Rate)**:
   $$F_m = F_{m-1} + \nu h_m, \quad 0 < \nu \leq 1$$

2. **Subsampling** (Stochastic Gradient Boosting):
   - Sample fraction of data for each tree
   - Reduces overfitting, speeds up training

3. **Tree Constraints**:
   - `max_depth`: Limit tree depth
   - `min_samples_split`: Minimum samples to split
   - `min_samples_leaf`: Minimum samples per leaf
   - `max_leaf_nodes`: Limit number of leaves

4. **Early Stopping**:
   - Monitor validation error
   - Stop when no improvement

## Mathematical Examples

### Example 1: Gradient Boosting for Regression (3 samples)

**Data**:
$$\begin{array}{|c|c|}
\hline
x & y \\
\hline
1 & 2 \\
2 & 4 \\
3 & 6 \\
\hline
\end{array}$$

**Parameters**: Learning rate $\nu = 0.1$, MSE loss

**Iteration 0** (Initialize):
$$F_0(x) = \bar{y} = \frac{2 + 4 + 6}{3} = 4$$

**Iteration 1**:

Residuals:
- $r_1 = 2 - 4 = -2$
- $r_2 = 4 - 4 = 0$
- $r_3 = 6 - 4 = 2$

Fit tree $h_1$ to $(x, r)$: Simple tree predicts residual mean per region

Suppose tree splits at $x = 2.5$:
- Left region ($x \leq 2.5$): $\gamma_1 = \text{mean}(-2, 0) = -1$
- Right region ($x > 2.5$): $\gamma_2 = \text{mean}(2) = 2$

Update:
$$F_1(x) = F_0(x) + 0.1 \times h_1(x)$$

- $F_1(1) = 4 + 0.1 \times (-1) = 3.9$
- $F_1(2) = 4 + 0.1 \times (-1) = 3.9$
- $F_1(3) = 4 + 0.1 \times 2 = 4.2$

**Iteration 2**:

New residuals:
- $r_1 = 2 - 3.9 = -1.9$
- $r_2 = 4 - 3.9 = 0.1$
- $r_3 = 6 - 4.2 = 1.8$

Fit tree $h_2$, update $F_2$, and continue...

**Observations**:
- Residuals shrink with each iteration
- Model gradually improves
- Learning rate controls step size

### Example 2: Loss Function Gradients

**MSE Loss**: $L = \frac{1}{2}(y - F)^2$

$$\frac{\partial L}{\partial F} = -(y - F)$$
$$r = -\frac{\partial L}{\partial F} = y - F$$

**MAE Loss**: $L = |y - F|$

$$\frac{\partial L}{\partial F} = -\text{sign}(y - F)$$
$$r = \text{sign}(y - F)$$

**Huber Loss**: Combines MSE and MAE

$$L = \begin{cases}
\frac{1}{2}(y - F)^2 & \text{if } |y - F| \leq \delta \\
\delta(|y - F| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

## Implementation in Python

### Basic Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=100,            # Number of boosting stages
    learning_rate=0.1,           # Shrinkage parameter
    max_depth=3,                 # Max depth of individual trees
    min_samples_split=2,         # Min samples to split
    min_samples_leaf=1,          # Min samples per leaf
    subsample=1.0,               # Fraction of samples for each tree
    loss='squared_error',        # Loss function
    random_state=42
)

# Train
gb_reg.fit(X_train, y_train)

# Predict
y_pred_train = gb_reg.predict(X_train)
y_pred_test = gb_reg.predict(X_test)

# Evaluate
print("="*60)
print("GRADIENT BOOSTING REGRESSOR")
print("="*60)
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# Feature importance
print("\nTop 5 Important Features:")
feature_importance = gb_reg.feature_importances_
top_indices = np.argsort(feature_importance)[-5:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
```

### Gradient Boosting Classifier

```python
# Generate classification data
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, 
                                    n_informative=5, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Create classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    subsample=0.8,              # Use 80% of samples per tree
    loss='log_loss',            # Logistic regression loss
    random_state=42
)

# Train
gb_clf.fit(X_train_clf, y_train_clf)

# Predict
y_pred_clf = gb_clf.predict(X_test_clf)
y_pred_proba = gb_clf.predict_proba(X_test_clf)

print("\n" + "="*60)
print("GRADIENT BOOSTING CLASSIFIER")
print("="*60)
print(f"Train Accuracy: {gb_clf.score(X_train_clf, y_train_clf):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}")
```

### Staged Predictions (Training Progress)

```python
# Track error over boosting iterations
train_errors = []
test_errors = []

for y_pred_train_stage in gb_reg.staged_predict(X_train):
    train_errors.append(mean_squared_error(y_train, y_pred_train_stage))

for y_pred_test_stage in gb_reg.staged_predict(X_test):
    test_errors.append(mean_squared_error(y_test, y_pred_test_stage))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_errors) + 1), train_errors, 
         label='Training MSE', linewidth=2, color='blue')
plt.plot(range(1, len(test_errors) + 1), test_errors, 
         label='Test MSE', linewidth=2, color='green')
plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Boosting: Error vs Iterations')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find optimal number of estimators
optimal_n_estimators = np.argmin(test_errors) + 1
print(f"\nOptimal number of estimators: {optimal_n_estimators}")
print(f"Best test MSE: {min(test_errors):.2f}")
```

### Learning Rate Comparison

```python
learning_rates = [0.01, 0.05, 0.1, 0.5]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(12, 6))

for lr, color in zip(learning_rates, colors):
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=lr, 
                                     max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    
    test_errors = [mean_squared_error(y_test, y_pred) 
                   for y_pred in gb.staged_predict(X_test)]
    
    plt.plot(range(1, len(test_errors) + 1), test_errors, 
             label=f'LR = {lr}', linewidth=2, color=color)

plt.xlabel('Number of Estimators')
plt.ylabel('Test MSE')
plt.title('Effect of Learning Rate on Gradient Boosting')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Early Stopping

```python
gb_early = GradientBoostingRegressor(
    n_estimators=1000,           # Set high
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.2,     # Use 20% for validation
    n_iter_no_change=10,         # Stop if no improvement for 10 iterations
    tol=1e-4,                    # Minimum improvement threshold
    random_state=42
)

gb_early.fit(X_train, y_train)

print(f"\nEarly stopping at estimator: {gb_early.n_estimators_}")
print(f"Total fitted estimators: {len(gb_early.estimators_)}")
```

### Complete Pipeline with XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,        # Fraction of features per tree
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(xgb_reg, X_train, y_train, 
                             cv=5, scoring='r2')

print("\n" + "="*60)
print("XGBOOST REGRESSOR")
print("="*60)
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train and evaluate
xgb_reg.fit(X_train, y_train)
print(f"Test R²: {xgb_reg.score(X_test, y_test):.4f}")
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **n_estimators**: Number of boosting stages (50-500)
2. **learning_rate**: Shrinkage (0.01-0.3)
3. **max_depth**: Tree depth (3-10)
4. **min_samples_split**: Min to split (2-20)
5. **min_samples_leaf**: Min per leaf (1-10)
6. **subsample**: Sample fraction (0.5-1.0)
7. **max_features**: Features per split ('sqrt', 'log2', or fraction)

### Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")
```

## Practical Applications

### 1. **Kaggle Competitions**
Gradient boosting (XGBoost, LightGBM) dominates tabular data competitions.

### 2. **Credit Scoring**
Predict loan default risk.

### 3. **Customer Churn**
Identify customers likely to leave.

### 4. **Demand Forecasting**
Predict product demand.

### 5. **Fraud Detection**
Detect anomalous transactions.

### 6. **Ranking**
Search engine result ranking.

## Summary

Gradient Boosting builds powerful ensembles by sequentially fitting trees to residuals.

**Key Concepts**:
1. **Residual Fitting**: $r_i = -\frac{\partial L}{\partial F}|_{F=F_{m-1}}$
2. **Additive Model**: $F_m = F_{m-1} + \nu h_m$
3. **Learning Rate**: Controls contribution of each tree
4. **Regularization**: Subsampling, tree constraints, early stopping

**Best Practices**:
✅ Use small learning rate (0.01-0.1) with many trees
✅ Enable early stopping on validation set
✅ Tune max_depth (3-7 typically good)
✅ Use subsampling (0.5-0.8) for regularization
✅ Monitor training vs validation error

❌ Don't use too large learning rate
❌ Don't skip validation monitoring
❌ Don't ignore overfitting signals
❌ Don't use with sparse high-dimensional data
❌ Don't expect fast training on large datasets

Gradient Boosting is the gold standard for structured/tabular data!
