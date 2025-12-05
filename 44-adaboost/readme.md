# AdaBoost (Adaptive Boosting)

## Table of Contents
- [Introduction](#introduction)
- [What is AdaBoost?](#what-is-adaboost)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use AdaBoost?](#why-use-adaboost)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Algorithm Details](#algorithm-details)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Practical Applications](#practical-applications)

## Introduction

AdaBoost (Adaptive Boosting) is a powerful ensemble learning algorithm that combines multiple weak learners to create a strong classifier. Introduced by Freund and Schapire in 1996, AdaBoost was one of the first successful boosting algorithms and remains widely used today.

The key idea: Train weak learners sequentially, where each new learner focuses on the mistakes of the previous ones by adjusting sample weights. Samples that are misclassified get higher weights, forcing the next learner to pay more attention to them.

## What is AdaBoost?

AdaBoost is a **boosting** algorithm that:

1. Trains weak learners sequentially (typically decision stumps)
2. Adjusts sample weights based on errors
3. Combines weak learners through weighted voting
4. Focuses on hard-to-classify examples

### Key Concepts

**Weak Learner**: A model slightly better than random guessing
- For binary classification: accuracy > 50%
- Often uses decision stumps (1-level decision trees)
- Simple, low-variance models

**Boosting**: Sequential training where each model corrects previous errors
- Unlike bagging (parallel training)
- Each learner depends on previous learners
- Reduces both bias and variance

**Adaptive Weights**: Misclassified samples get higher weights
- Forces next learner to focus on mistakes
- Eventually, hard examples dominate
- Creates specialized learners

## Mathematical Foundation

### AdaBoost Algorithm (Binary Classification)

**Input**:
- Training data: $\{(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)\}$ where $y_i \in \{-1, +1\}$
- Number of weak learners: $T$

**Initialize**: Sample weights
$$w_1^{(i)} = \frac{1}{n} \quad \text{for } i = 1, ..., n$$

All samples start with equal weight.

**For $t = 1$ to $T$:**

1. **Train weak learner** $h_t$ on weighted data
   - Minimize weighted error: $\epsilon_t = \sum_{i=1}^{n} w_t^{(i)} \mathbb{1}(h_t(\mathbf{x}_i) \neq y_i)$

2. **Calculate weak learner weight**:
   $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
   
   - If $\epsilon_t = 0.5$ (random): $\alpha_t = 0$ (no contribution)
   - If $\epsilon_t \to 0$ (perfect): $\alpha_t \to \infty$ (high weight)
   - If $\epsilon_t > 0.5$ (worse than random): $\alpha_t < 0$ (flip prediction)

3. **Update sample weights**:
   $$w_{t+1}^{(i)} = w_t^{(i)} \times \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$$
   
   Equivalently:
   $$w_{t+1}^{(i)} = \begin{cases}
   w_t^{(i)} e^{-\alpha_t} & \text{if } h_t(\mathbf{x}_i) = y_i \quad \text{(correct)}\\
   w_t^{(i)} e^{\alpha_t} & \text{if } h_t(\mathbf{x}_i) \neq y_i \quad \text{(incorrect)}
   \end{cases}$$

4. **Normalize weights**:
   $$w_{t+1}^{(i)} = \frac{w_{t+1}^{(i)}}{\sum_{j=1}^{n} w_{t+1}^{(j)}}$$

**Final Classifier**:
$$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$$

### Weight Update Intuition

**Correct classification**: $y_i h_t(\mathbf{x}_i) = +1$
- Weight multiplier: $e^{-\alpha_t} < 1$
- Weight decreases (easier sample)

**Incorrect classification**: $y_i h_t(\mathbf{x}_i) = -1$
- Weight multiplier: $e^{\alpha_t} > 1$
- Weight increases (harder sample)

### Training Error Bound

AdaBoost's training error has an upper bound:

$$\text{Error} \leq \prod_{t=1}^{T} \sqrt{2\epsilon_t(1-\epsilon_t)} = \prod_{t=1}^{T} \sqrt{1 - 4\gamma_t^2}$$

where $\gamma_t = \frac{1}{2} - \epsilon_t$ is the edge (how much better than random).

If each weak learner has edge $\gamma_t \geq \gamma > 0$:
$$\text{Error} \leq e^{-2\gamma^2 T}$$

**Key insight**: Error decreases exponentially with number of weak learners!

### Generalization Error

AdaBoost has good generalization despite building complex ensembles. The margin theory explains this:

**Margin** for sample $(\mathbf{x}_i, y_i)$:
$$\text{margin}(\mathbf{x}_i) = y_i \frac{\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x}_i)}{\sum_{t=1}^{T} \alpha_t}$$

Larger margin → more confident prediction → better generalization.

## Why Use AdaBoost?

### 1. **Combines Weak to Strong**
Turns multiple weak learners into a strong ensemble.

### 2. **Automatic Feature Selection**
Focuses on most informative features through weighting.

### 3. **Low Variance**
Unlike single decision trees, AdaBoost is less prone to overfitting (to a point).

### 4. **Simple Implementation**
Easy to understand and implement.

### 5. **Versatile**
Works with any weak learner (decision stumps, logistic regression, etc.).

### 6. **No Hyperparameter Tuning**
Often works well with defaults.

## Advantages and Disadvantages

### Advantages

1. **High Accuracy**: Often achieves excellent performance
2. **Simple to Implement**: Straightforward algorithm
3. **Versatile**: Works with any weak learner
4. **Feature Selection**: Implicitly selects important features
5. **Low Bias**: Reduces bias of weak learners
6. **No Data Preprocessing**: Handles raw features
7. **Interpretable**: Can analyze which weak learners matter
8. **Mathematically Elegant**: Strong theoretical foundations

### Disadvantages

1. **Sensitive to Noise**: Outliers get high weights
2. **Sensitive to Overfitting**: Can overfit with too many estimators
3. **Sequential Training**: Cannot be parallelized
4. **Slower Training**: Must train learners one by one
5. **Requires Good Weak Learners**: If $\epsilon_t \geq 0.5$, algorithm fails
6. **Not Suitable for Regression**: Designed for classification
7. **Imbalanced Data Issues**: May focus too much on minority class

## Algorithm Details

### AdaBoost.M1 (Original) - Binary Classification

```
Initialize: w_1(i) = 1/n for i = 1 to n

For t = 1 to T:
    1. Train classifier h_t using weights w_t
    
    2. Calculate weighted error:
       ε_t = Σ w_t(i) * I(h_t(x_i) ≠ y_i)
    
    3. If ε_t >= 0.5, stop (weak learner not good enough)
    
    4. Calculate classifier weight:
       α_t = 0.5 * ln((1 - ε_t) / ε_t)
    
    5. Update sample weights:
       w_{t+1}(i) = w_t(i) * exp(-α_t * y_i * h_t(x_i))
    
    6. Normalize:
       w_{t+1}(i) = w_{t+1}(i) / Σ w_{t+1}(j)

Final classifier:
H(x) = sign(Σ α_t * h_t(x))
```

### SAMME (Stagewise Additive Modeling) - Multiclass

For $K$ classes:

$$\alpha_t = \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1)$$

$$w_{t+1}^{(i)} = w_t^{(i)} \exp\left(\alpha_t \mathbb{1}(h_t(\mathbf{x}_i) \neq y_i)\right)$$

### SAMME.R - Multiclass with Probability Estimates

Uses class probabilities instead of hard predictions:

$$H(\mathbf{x}) = \arg\max_k \sum_{t=1}^{T} \log p_t(k | \mathbf{x})$$

where $p_t(k | \mathbf{x})$ is the probability estimate from weak learner $t$.

## Mathematical Examples

### Example 1: Simple AdaBoost (3 samples, 2 weak learners)

**Dataset**:
$$\begin{array}{|c|c|c|}
\hline
x & y & w_1 \\
\hline
1 & -1 & 1/3 \\
2 & +1 & 1/3 \\
3 & +1 & 1/3 \\
\hline
\end{array}$$

**Round 1**: Weak learner $h_1$

Decision rule: $h_1(x) = \begin{cases} -1 & \text{if } x < 1.5 \\ +1 & \text{otherwise} \end{cases}$

Predictions: $h_1(1) = -1$ ✓, $h_1(2) = +1$ ✓, $h_1(3) = +1$ ✓

**Weighted error**:
$$\epsilon_1 = \frac{1}{3} \times 0 + \frac{1}{3} \times 0 + \frac{1}{3} \times 0 = 0$$

Perfect learner! But this is unrealistic. Let's say it makes one mistake:

$h_1(1) = -1$ ✓, $h_1(2) = -1$ ✗, $h_1(3) = +1$ ✓

$$\epsilon_1 = \frac{1}{3} \times 0 + \frac{1}{3} \times 1 + \frac{1}{3} \times 0 = \frac{1}{3}$$

**Learner weight**:
$$\alpha_1 = \frac{1}{2} \ln\left(\frac{1 - 1/3}{1/3}\right) = \frac{1}{2} \ln(2) \approx 0.347$$

**Update weights**:

Sample 1 (correct): $w_2^{(1)} = \frac{1}{3} e^{-0.347} = \frac{1}{3} \times 0.707 = 0.236$

Sample 2 (incorrect): $w_2^{(2)} = \frac{1}{3} e^{0.347} = \frac{1}{3} \times 1.414 = 0.471$

Sample 3 (correct): $w_2^{(3)} = \frac{1}{3} e^{-0.347} = \frac{1}{3} \times 0.707 = 0.236$

**Normalize**: Total = 0.236 + 0.471 + 0.236 = 0.943

$$w_2 = [0.250, 0.500, 0.250]$$

Sample 2 now has double the weight!

**Round 2**: Weak learner $h_2$ trained with new weights

Will focus more on sample 2 due to higher weight.

### Example 2: Classifier Weight Calculation

**Scenario 1**: $\epsilon = 0.1$ (90% accuracy)
$$\alpha = \frac{1}{2} \ln\left(\frac{0.9}{0.1}\right) = \frac{1}{2} \ln(9) = 1.099$$

**Scenario 2**: $\epsilon = 0.3$ (70% accuracy)
$$\alpha = \frac{1}{2} \ln\left(\frac{0.7}{0.3}\right) = \frac{1}{2} \ln(2.33) = 0.424$$

**Scenario 3**: $\epsilon = 0.5$ (50% accuracy - random)
$$\alpha = \frac{1}{2} \ln\left(\frac{0.5}{0.5}\right) = \frac{1}{2} \ln(1) = 0$$

**Observation**: Better weak learners get exponentially higher weights!

## Implementation in Python

### Basic AdaBoost Classifier

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, 
                           n_informative=5, n_redundant=2,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create AdaBoost classifier
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50,              # Number of weak learners
    learning_rate=1.0,            # Contribution of each classifier
    algorithm='SAMME.R',          # SAMME or SAMME.R
    random_state=42
)

# Train model
ada_clf.fit(X_train, y_train)

# Predictions
y_pred = ada_clf.predict(X_test)
y_pred_proba = ada_clf.predict_proba(X_test)

# Evaluation
print("="*60)
print("ADABOOST CLASSIFIER")
print("="*60)
print(f"Training Accuracy: {ada_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature Importances:")
for i, importance in enumerate(ada_clf.feature_importances_):
    print(f"  Feature {i}: {importance:.4f}")
```

### Analyzing Weak Learner Weights

```python
import matplotlib.pyplot as plt

# Get estimator weights
estimator_weights = ada_clf.estimator_weights_
estimator_errors = ada_clf.estimator_errors_

print(f"\nNumber of estimators: {len(ada_clf.estimators_)}")
print(f"\nFirst 10 estimator weights (α):")
for i in range(min(10, len(estimator_weights))):
    print(f"  Estimator {i+1}: α={estimator_weights[i]:.4f}, "
          f"error={estimator_errors[i]:.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot estimator weights
ax1.bar(range(len(estimator_weights)), estimator_weights, 
        color='skyblue', edgecolor='black')
ax1.set_xlabel('Estimator Index')
ax1.set_ylabel('Weight (α)')
ax1.set_title('AdaBoost Estimator Weights')
ax1.grid(alpha=0.3)

# Plot estimator errors
ax2.bar(range(len(estimator_errors)), estimator_errors, 
        color='salmon', edgecolor='black')
ax2.set_xlabel('Estimator Index')
ax2.set_ylabel('Weighted Error (ε)')
ax2.set_title('AdaBoost Estimator Errors')
ax2.axhline(0.5, color='red', linestyle='--', label='Random Threshold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Staged Predictions (Error vs Number of Estimators)

```python
from sklearn.metrics import accuracy_score

# Get staged predictions
n_estimators = len(ada_clf.estimators_)
train_scores = []
test_scores = []

for y_pred_train in ada_clf.staged_predict(X_train):
    train_scores.append(accuracy_score(y_train, y_pred_train))

for y_pred_test in ada_clf.staged_predict(X_test):
    test_scores.append(accuracy_score(y_test, y_pred_test))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_estimators + 1), train_scores, 
         label='Training Accuracy', linewidth=2, color='blue')
plt.plot(range(1, n_estimators + 1), test_scores, 
         label='Test Accuracy', linewidth=2, color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost: Accuracy vs Number of Estimators')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nBest Test Accuracy: {max(test_scores):.4f}")
print(f"Achieved at estimator: {np.argmax(test_scores) + 1}")
```

### Comparison: Different Base Estimators

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Decision Stump (depth=1)
ada_stump = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

# Deeper Tree (depth=3)
ada_tree = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    random_state=42
)

# Train both
ada_stump.fit(X_train, y_train)
ada_tree.fit(X_train, y_train)

# Evaluate
print("="*60)
print("COMPARISON: Different Base Estimators")
print("="*60)
print(f"{'Base Estimator':<30} {'Train Acc':<12} {'Test Acc':<12}")
print("-"*60)
print(f"{'Decision Stump (depth=1)':<30} "
      f"{ada_stump.score(X_train, y_train):<12.4f} "
      f"{ada_stump.score(X_test, y_test):<12.4f}")
print(f"{'Decision Tree (depth=3)':<30} "
      f"{ada_tree.score(X_train, y_train):<12.4f} "
      f"{ada_tree.score(X_test, y_test):<12.4f}")
```

### Custom AdaBoost Implementation

```python
class SimpleAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Initialize weights
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weak learner
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=weights)
            
            # Predictions
            predictions = model.predict(X)
            
            # Calculate weighted error
            incorrect = (predictions != y)
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate alpha
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            
            # Store model and alpha
            self.models.append(model)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        # Weighted majority vote
        predictions = np.array([alpha * model.predict(X) 
                               for alpha, model in zip(self.alphas, self.models)])
        return np.sign(np.sum(predictions, axis=0))

# Test custom implementation
y_binary = 2 * y - 1  # Convert to {-1, +1}
custom_ada = SimpleAdaBoost(n_estimators=50)
custom_ada.fit(X_train, y_binary[:len(X_train)])
custom_pred = custom_ada.predict(X_test)
custom_accuracy = accuracy_score(y_binary[len(X_train):], custom_pred)

print(f"\nCustom AdaBoost Accuracy: {custom_accuracy:.4f}")
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **n_estimators**: Number of weak learners
   - Default: 50
   - Try: 10, 50, 100, 200
   - More estimators = better fit but risk of overfitting

2. **learning_rate**: Shrinks contribution of each classifier
   - Default: 1.0
   - Try: 0.01, 0.1, 0.5, 1.0
   - Lower rate = slower learning, needs more estimators

3. **base_estimator**: Type of weak learner
   - Default: DecisionTreeClassifier(max_depth=1)
   - Try different depths: 1, 2, 3

4. **algorithm**: SAMME or SAMME.R
   - SAMME: Discrete predictions
   - SAMME.R: Probability estimates (usually better)

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3]
}

grid_search = GridSearchCV(
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), 
                       random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")
```

## Practical Applications

### 1. **Face Detection**
Viola-Jones face detector uses AdaBoost with Haar features.

### 2. **Medical Diagnosis**
Combine multiple diagnostic tests for disease prediction.

### 3. **Credit Scoring**
Ensemble weak models for loan default prediction.

### 4. **Fraud Detection**
Detect fraudulent transactions by combining simple rules.

### 5. **Text Classification**
Classify documents into categories.

## Summary

AdaBoost adaptively combines weak learners into a strong classifier by focusing on hard examples.

**Key Formulas**:
1. **Weighted Error**: $\epsilon_t = \sum_i w_t^{(i)} \mathbb{1}(h_t(\mathbf{x}_i) \neq y_i)$
2. **Classifier Weight**: $\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
3. **Weight Update**: $w_{t+1}^{(i)} = w_t^{(i)} e^{-\alpha_t y_i h_t(\mathbf{x}_i)}$
4. **Final Prediction**: $H(\mathbf{x}) = \text{sign}\left(\sum_t \alpha_t h_t(\mathbf{x})\right)$

**Best Practices**:
✅ Use decision stumps as base learners
✅ Start with 50-100 estimators
✅ Use SAMME.R for better performance
✅ Monitor staged predictions for overfitting
✅ Handle noisy data carefully

❌ Don't use with very noisy data
❌ Don't use too complex base learners
❌ Don't ignore learning rate tuning
❌ Don't use when interpretability is critical
❌ Don't expect parallelization

AdaBoost remains a powerful, elegant algorithm with strong theoretical foundations!
