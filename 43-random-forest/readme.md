# Random Forest

## Table of Contents
- [Introduction](#introduction)
- [What is Random Forest?](#what-is-random-forest)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use Random Forest?](#why-use-random-forest)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Algorithm Details](#algorithm-details)
- [Feature Importance](#feature-importance)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Practical Applications](#practical-applications)

## Introduction

Random Forest is one of the most powerful and widely used machine learning algorithms. It's an **ensemble learning** method that combines multiple decision trees to create a more robust and accurate model. The "forest" is built by training many decision trees on random subsets of the data and features, then aggregating their predictions.

Random Forest excels at:
- Handling non-linear relationships
- Working with high-dimensional data
- Reducing overfitting compared to single decision trees
- Providing feature importance insights
- Working with both classification and regression tasks

## What is Random Forest?

Random Forest is an ensemble of decision trees trained using **bagging** (Bootstrap Aggregating) and **random feature selection**.

###

 Key Concepts

**1. Ensemble Learning**: Combining multiple models to improve performance
$$\text{Ensemble Prediction} = \text{Aggregate}(\text{Model}_1, \text{Model}_2, ..., \text{Model}_n)$$

**2. Bagging (Bootstrap Aggregating)**:
- Create $B$ bootstrap samples from training data
- Train one decision tree on each bootstrap sample
- Aggregate predictions (voting for classification, averaging for regression)

**3. Random Feature Selection**:
- At each split, consider only a random subset of $m$ features (where $m < p$)
- Typically: $m = \sqrt{p}$ for classification, $m = p/3$ for regression
- Reduces correlation between trees

**4. Out-of-Bag (OOB) Samples**:
- Each bootstrap sample excludes ~37% of data
- Use these OOB samples for validation
- No need for separate validation set

## Mathematical Foundation

### Bootstrap Sampling

From dataset $\mathcal{D} = \{(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)\}$, create $B$ bootstrap samples:

$$\mathcal{D}_b = \{(\mathbf{x}_{i_1}, y_{i_1}), ..., (\mathbf{x}_{i_n}, y_{i_n})\}$$

where each $i_j$ is sampled uniformly with replacement from $\{1, 2, ..., n\}$.

**Probability of being selected**:
- Probability a sample is NOT selected in one draw: $1 - \frac{1}{n}$
- Probability NOT selected in $n$ draws: $\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$
- Probability selected at least once: $1 - 0.368 = 0.632$ (63.2%)
- Remaining 36.8% are out-of-bag (OOB)

### Random Forest for Classification

For $B$ trees $\{T_1, T_2, ..., T_B\}$, the prediction for sample $\mathbf{x}$ is:

$$\hat{y}(\mathbf{x}) = \text{mode}\{T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_B(\mathbf{x})\}$$

**Majority voting**: Choose class with most votes

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{b=1}^{B} \mathbb{1}(T_b(\mathbf{x}) = c)$$

where $\mathbb{1}(\cdot)$ is the indicator function.

### Random Forest for Regression

Average predictions from all trees:

$$\hat{y}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

### Decision Tree Split Criterion

At each node, choose the split that maximizes information gain or minimizes impurity.

**For Classification (Gini Impurity)**:

$$\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2$$

where $p_c$ is the proportion of class $c$ in set $S$.

**Information Gain**:

$$\text{Gain}(S, A) = \text{Impurity}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Impurity}(S_v)$$

**For Regression (Variance Reduction)**:

$$\text{Variance}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

### Variance Reduction Through Averaging

Individual tree variance: $\sigma^2$

Correlation between trees: $\rho$

Variance of random forest with $B$ trees:

$$\text{Var}(\text{RF}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

As $B \to \infty$:
$$\text{Var}(\text{RF}) \to \rho \sigma^2$$

**Key insight**: Random feature selection reduces $\rho$, leading to lower variance!

## Why Use Random Forest?

### 1. **Reduced Overfitting**
Single decision trees overfit easily. Random forest reduces overfitting by:
- Averaging many trees (reduces variance)
- Using random subsets (decorrelates trees)
- Bootstrap sampling (different data perspectives)

### 2. **Handles Non-Linearity**
Captures complex, non-linear relationships without explicit feature engineering.

### 3. **Feature Importance**
Provides insight into which features are most predictive.

### 4. **Robust to Outliers**
Individual trees may be affected, but ensemble is robust.

### 5. **No Feature Scaling Required**
Tree-based methods are invariant to monotonic transformations.

### 6. **Handles Missing Values**
Can handle missing data through surrogate splits.

### 7. **Parallel Training**
Trees are independent; can train in parallel.

## Advantages and Disadvantages

### Advantages

1. **High Accuracy**: Often achieves state-of-the-art performance
2. **Reduces Overfitting**: Better generalization than single trees
3. **Handles Large Datasets**: Scales well to big data
4. **Feature Importance**: Built-in feature ranking
5. **No Normalization Needed**: Works with raw features
6. **Handles Mixed Data**: Categorical and numerical features
7. **Robust**: Resistant to outliers and noise
8. **Low Hyperparameter Sensitivity**: Works well with defaults
9. **OOB Validation**: Built-in cross-validation
10. **Parallel Processing**: Fast training on modern hardware

### Disadvantages

1. **Black Box**: Less interpretable than single decision tree
2. **Memory Intensive**: Stores many trees
3. **Slow Prediction**: Must query all trees (mitigated by parallelization)
4. **Not for Extrapolation**: Can't predict beyond training range
5. **Biased to Dominant Classes**: In imbalanced datasets
6. **Large Model Size**: Many trees = large file size
7. **Not Ideal for Linear**: Overkill for simple linear relationships

## Algorithm Details

### Random Forest Algorithm

**Input**: 
- Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$
- Number of trees $B$
- Number of features to consider at each split $m$
- Minimum samples per leaf $n_{min}$

**Output**: Ensemble of $B$ decision trees

**Algorithm**:

```
For b = 1 to B:
    1. Create bootstrap sample D_b by sampling n instances 
       from D with replacement
    
    2. Grow a decision tree T_b on D_b:
        a. At each node, randomly select m features 
           from p total features
        b. Choose the best split among these m features
        c. Split the node
        d. Repeat until stopping criterion:
           - All samples in node are same class, OR
           - Node has < n_min samples, OR
           - Maximum depth reached
    
    3. Store tree T_b (do NOT prune)

Return: Ensemble {T_1, T_2, ..., T_B}
```

**Prediction**:

For classification:
```
prediction = majority_vote(T_1(x), T_2(x), ..., T_B(x))
```

For regression:
```
prediction = average(T_1(x), T_2(x), ..., T_B(x))
```

### Hyperparameters

**Key hyperparameters**:

1. **n_estimators** ($B$): Number of trees
   - Default: 100
   - More trees = better performance (diminishing returns)
   - Trade-off: accuracy vs computation time

2. **max_features** ($m$): Features per split
   - Classification: $\sqrt{p}$
   - Regression: $p/3$
   - Lower $m$ = less correlation, more diversity

3. **max_depth**: Maximum tree depth
   - Default: None (grow until pure)
   - Control overfitting

4. **min_samples_split**: Minimum samples to split
   - Default: 2
   - Higher = more conservative splits

5. **min_samples_leaf**: Minimum samples per leaf
   - Default: 1
   - Higher = smoother decision boundaries

6. **bootstrap**: Use bootstrap sampling
   - Default: True
   - If False, use whole dataset (loses randomness)

## Feature Importance

Random Forest provides two types of feature importance:

### 1. Mean Decrease in Impurity (MDI)

Also called **Gini Importance**. For each feature:

$$\text{Importance}(f) = \sum_{t=1}^{B} \sum_{s \in \text{Splits}_t(f)} \text{Gain}(s) \times p(s)$$

where:
- $\text{Splits}_t(f)$ = splits using feature $f$ in tree $t$
- $\text{Gain}(s)$ = impurity reduction at split $s$
- $p(s)$ = proportion of samples reaching split $s$

**Normalized**:
$$\text{Importance}(f) = \frac{\text{Importance}(f)}{\sum_{f'} \text{Importance}(f')}$$

### 2. Mean Decrease in Accuracy (MDA)

Also called **Permutation Importance**. For each feature:

1. Measure OOB accuracy with original data
2. Randomly permute feature $f$ in OOB samples
3. Measure new OOB accuracy
4. Importance = drop in accuracy

$$\text{Importance}(f) = \text{Accuracy}_{original} - \text{Accuracy}_{permuted}$$

**More reliable** than MDI, especially for:
- Correlated features
- High-cardinality categorical features

## Mathematical Examples

### Example 1: Simple Random Forest Ensemble

**Dataset**: 8 samples, binary classification

$$\begin{array}{|c|c|c|}
\hline
x_1 & x_2 & y \\
\hline
1 & 2 & 0 \\
2 & 3 & 0 \\
3 & 1 & 0 \\
4 & 4 & 1 \\
5 & 5 & 1 \\
6 & 3 & 1 \\
7 & 6 & 1 \\
8 & 5 & 0 \\
\hline
\end{array}$$

**Bootstrap Sample 1**: Indices [1, 1, 3, 4, 5, 7, 7, 8]
**Tree 1 prediction** for new point $(x_1=4, x_2=3)$: Class 1

**Bootstrap Sample 2**: Indices [2, 2, 3, 4, 5, 6, 6, 8]
**Tree 2 prediction** for $(x_1=4, x_2=3)$: Class 1

**Bootstrap Sample 3**: Indices [1, 2, 3, 3, 5, 6, 7, 8]
**Tree 3 prediction** for $(x_1=4, x_2=3)$: Class 0

**Random Forest Prediction** (majority vote):
$$\hat{y} = \text{mode}(1, 1, 0) = 1$$

Vote distribution: Class 0 (1 vote), Class 1 (2 votes) → **Predict Class 1**

### Example 2: Variance Reduction

**Single tree variance**: $\sigma^2 = 0.25$

**Correlation between trees**: $\rho = 0.1$

**Number of trees**: $B = 100$

**Random Forest variance**:
$$\text{Var}(\text{RF}) = 0.1 \times 0.25 + \frac{1-0.1}{100} \times 0.25$$
$$= 0.025 + \frac{0.9}{100} \times 0.25$$
$$= 0.025 + 0.00225 = 0.02725$$

**Variance reduction**: $\frac{0.25 - 0.02725}{0.25} = 0.891$ (89.1% reduction!)

### Example 3: Feature Importance Calculation

**Tree 1 splits**:
- Split on $x_1$ at root: Gini gain = 0.3, samples = 100% → contribution = 0.3
- Split on $x_2$ at node: Gini gain = 0.1, samples = 60% → contribution = 0.06

**Tree 2 splits**:
- Split on $x_2$ at root: Gini gain = 0.25, samples = 100% → contribution = 0.25
- Split on $x_1$ at node: Gini gain = 0.15, samples = 50% → contribution = 0.075

**Total importance** (unnormalized):
- $x_1$: 0.3 + 0.075 = 0.375
- $x_2$: 0.06 + 0.25 = 0.31

**Normalized importance**:
- Total = 0.375 + 0.31 = 0.685
- $x_1$: 0.375 / 0.685 = 0.547 (54.7%)
- $x_2$: 0.31 / 0.685 = 0.453 (45.3%)

## Implementation in Python

### Basic Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, 
                           n_informative=5, n_redundant=2,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=None,             # Grow trees until pure
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples per leaf
    max_features='sqrt',        # Features per split
    bootstrap=True,             # Use bootstrap sampling
    oob_score=True,             # Calculate OOB score
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

# Train model
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)

# Evaluation
print("="*60)
print("RANDOM FOREST CLASSIFIER")
print("="*60)
print(f"Training Accuracy: {rf_classifier.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf_classifier.oob_score_:.4f}")
print(f"\nNumber of trees: {rf_classifier.n_estimators}")
print(f"Number of features: {rf_classifier.n_features_in_}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance
feature_importance = rf_classifier.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

### OOB Score Analysis

```python
# Track OOB score vs number of trees
oob_scores = []
n_trees_range = range(1, 201, 10)

for n_trees in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, 
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, oob_scores, marker='o', linewidth=2, color='green')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Number of Trees')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nOptimal OOB Score: {max(oob_scores):.4f}")
print(f"Achieved with {n_trees_range[np.argmax(oob_scores)]} trees")
```

### Random Forest for Regression

```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                                noise=10, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = rf_regressor.predict(X_test_reg)

print("="*60)
print("RANDOM FOREST REGRESSOR")
print("="*60)
print(f"R² Score (Train): {rf_regressor.score(X_train_reg, y_train_reg):.4f}")
print(f"R² Score (Test): {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE (Test): {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")
print(f"OOB Score: {rf_regressor.oob_score_:.4f}")
```

### Complete Pipeline with Tuning

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline (RF doesn't need scaling, but shown for completeness)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Hyperparameter grid
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2']
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

print("\n" + "="*60)
print("GRID SEARCH RESULTS")
print("="*60)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Test best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print(f"\nTest Accuracy (Best Model): {accuracy_score(y_test, y_pred_best):.4f}")
```

## Hyperparameter Tuning

### Key Parameters to Tune

1. **n_estimators**: Start with 100-200, increase if needed
2. **max_depth**: Try None, 10, 20, 30
3. **min_samples_split**: Try 2, 5, 10
4. **min_samples_leaf**: Try 1, 2, 4
5. **max_features**: Try 'sqrt', 'log2', or float values

### Tuning Strategy

```python
# Quick tuning
param_grid_quick = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10]
}

# Comprehensive tuning
param_grid_full = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}
```

## Practical Applications

### 1. **Credit Risk Assessment**
Predict loan default probability based on applicant features.

### 2. **Medical Diagnosis**
Classify diseases based on patient symptoms and test results.

### 3. **Customer Churn Prediction**
Identify customers likely to leave based on behavior patterns.

### 4. **Fraud Detection**
Detect fraudulent transactions in banking systems.

### 5. **Image Classification**
Classify images based on extracted features.

### 6. **Stock Price Prediction**
Forecast stock movements using historical and fundamental data.

### 7. **Recommendation Systems**
Predict user preferences for products or content.

## Summary

Random Forest is a powerful ensemble method that combines multiple decision trees to achieve high accuracy and robustness.

**Key Mathematical Concepts**:
1. **Bootstrap Sampling**: $P(\text{selected}) \approx 0.632$
2. **Aggregation**: $\hat{y} = \frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})$ (regression) or majority vote (classification)
3. **Variance Reduction**: $\text{Var}(\text{RF}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$
4. **Feature Importance**: $\sum_{t,s} \text{Gain}(s) \times p(s)$

**Best Practices**:
✅ Use 100-200 trees for most problems
✅ Enable OOB scoring for validation
✅ Analyze feature importance
✅ Consider max_depth for large datasets
✅ Use n_jobs=-1 for parallel training

❌ Don't prune trees (let them grow fully)
❌ Don't normalize features (not needed)
❌ Don't use too few trees (< 50)
❌ Don't ignore class imbalance
❌ Don't expect good extrapolation

Random Forest is often a great starting point for many machine learning problems!
