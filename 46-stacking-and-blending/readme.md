# Stacking and Blending

## Table of Contents
- [Introduction](#introduction)
- [What is Stacking?](#what-is-stacking)
- [What is Blending?](#what-is-blending)
- [Mathematical Foundation](#mathematical-foundation)
- [Why Use Stacking and Blending?](#why-use-stacking-and-blending)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Algorithm Details](#algorithm-details)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)

## Introduction

Stacking (Stacked Generalization) and Blending are advanced ensemble learning techniques that combine multiple models to achieve better predictive performance than any individual model. Unlike bagging or boosting, which combine models of the same type, stacking and blending can combine diverse model types (e.g., decision trees, neural networks, linear models).

**Key Idea**: Train a **meta-model** (also called meta-learner or blender) to optimally combine the predictions of multiple **base models**.

These techniques are popular in machine learning competitions (Kaggle) where winning solutions often use stacked ensembles.

## What is Stacking?

**Stacking** (Stacked Generalization) is a multi-level ensemble method:

### Architecture

```
Level 0 (Base Models):
  Model 1 (e.g., Random Forest)
  Model 2 (e.g., Gradient Boosting)
  Model 3 (e.g., SVM)
  Model 4 (e.g., Neural Network)
       ↓
  Predictions from each model
       ↓
Level 1 (Meta-Model):
  Meta-Learner (e.g., Logistic Regression)
       ↓
  Final Prediction
```

### Two-Stage Process

**Stage 1**: Train base models
- Split data into K folds
- For each fold:
  - Train base models on K-1 folds
  - Predict on holdout fold
- Collect out-of-fold predictions

**Stage 2**: Train meta-model
- Use out-of-fold predictions as features
- Train meta-model on these predictions
- Meta-model learns how to best combine base models

## What is Blending?

**Blending** is a simplified version of stacking:

### Architecture

```
Training Set → Validation Set Split

Training Set:
  → Train Base Models (1, 2, 3, ...)
     ↓
  Generate predictions on Validation Set
     ↓
  Train Meta-Model on validation predictions
```

### Key Difference from Stacking

- **Stacking**: Uses K-fold cross-validation (no data wasted)
- **Blending**: Uses holdout validation set (simpler but wastes data)

Blending is easier to implement but stacking typically performs better.

## Mathematical Foundation

### Stacking Formulation

**Base Models**: $f_1, f_2, ..., f_M$

**Base Predictions** for sample $\mathbf{x}_i$:
$$\mathbf{z}_i = [f_1(\mathbf{x}_i), f_2(\mathbf{x}_i), ..., f_M(\mathbf{x}_i)]$$

**Meta-Model**: $g(\mathbf{z})$

**Final Prediction**:
$$\hat{y}_i = g(\mathbf{z}_i) = g(f_1(\mathbf{x}_i), f_2(\mathbf{x}_i), ..., f_M(\mathbf{x}_i))$$

### Cross-Validation for Stacking

For K-fold CV:

1. Split data into K folds: $\mathcal{D} = \mathcal{D}_1 \cup \mathcal{D}_2 \cup ... \cup \mathcal{D}_K$

2. For each fold $k$:
   - Train base models on $\mathcal{D} \setminus \mathcal{D}_k$
   - Predict on $\mathcal{D}_k$

3. Concatenate all out-of-fold predictions:
   $$\mathbf{Z} = [\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_n]^T$$

4. Train meta-model:
   $$g^* = \arg\min_g \sum_{i=1}^{n} L(y_i, g(\mathbf{z}_i))$$

### Weighted Average (Simple Meta-Model)

If meta-model is weighted average:

$$\hat{y} = \sum_{m=1}^{M} w_m f_m(\mathbf{x})$$

subject to:
$$\sum_{m=1}^{M} w_m = 1, \quad w_m \geq 0$$

Optimal weights minimize loss:
$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{i=1}^{n} L\left(y_i, \sum_{m=1}^{M} w_m f_m(\mathbf{x}_i)\right)$$

### Diversity in Base Models

**Why diversity matters**: Combining similar models provides little benefit.

**Ensemble Error** for regression:
$$\mathbb{E}[(y - \bar{f})^2] = \bar{E} - \bar{A}$$

where:
- $\bar{E}$ = average error of individual models
- $\bar{A}$ = average ambiguity (disagreement between models)

**Key insight**: Higher ambiguity (diversity) → lower ensemble error!

### Bias-Variance Decomposition

For ensemble average $\bar{f} = \frac{1}{M}\sum_{m=1}^M f_m$:

$$\text{MSE}(\bar{f}) = \sigma^2 + \text{Bias}^2(\bar{f}) + \frac{1}{M}\text{Var}(f)$$

If models are uncorrelated:
- Bias stays same
- Variance reduces by factor of $M$

## Why Use Stacking and Blending?

### 1. **Combines Model Strengths**
Each model captures different patterns; meta-model learns best combination.

### 2. **Reduces Overfitting**
Meta-model uses out-of-fold predictions (not seen during base model training).

### 3. **Flexibility**
Can combine any type of models (trees, linear, neural nets, etc.).

### 4. **State-of-the-Art Performance**
Often achieves best results in competitions.

### 5. **Model Diversity**
Leverages complementary strengths of different algorithms.

### 6. **Automated Model Selection**
Meta-model learns which base models to trust.

## Advantages and Disadvantages

### Advantages

1. **Best Performance**: Often outperforms individual models
2. **Flexible**: Combine any model types
3. **Leverages Diversity**: Uses complementary model strengths
4. **Reduces Variance**: Through ensemble averaging
5. **Automated Weighting**: Meta-model learns optimal combination
6. **Handles Different Data Types**: Models can specialize on different features
7. **Competition-Winning**: Popular in Kaggle and ML contests

### Disadvantages

1. **Complexity**: More complex than single models
2. **Computational Cost**: Must train M+1 models
3. **Training Time**: Longer due to multiple models
4. **Overfitting Risk**: Meta-model can overfit to base predictions
5. **Interpretability**: Very difficult to interpret
6. **Deployment Complexity**: Must deploy all models
7. **Memory Usage**: Stores multiple models
8. **Data Requirements**: Needs more data to avoid overfitting

## Algorithm Details

### Stacking Algorithm (Classification)

**Input**:
- Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$
- Base models $\{f_1, ..., f_M\}$
- Meta-model $g$
- Number of folds $K$

**Algorithm**:

```
# Stage 1: Generate out-of-fold predictions
Z = zeros(n, M)  # Matrix to store base predictions

For each fold k = 1 to K:
    train_idx = all indices except fold k
    val_idx = indices in fold k
    
    For each base model m = 1 to M:
        1. Train f_m on D[train_idx]
        2. Predict on D[val_idx]: Z[val_idx, m] = f_m(X[val_idx])

# Stage 2: Train meta-model
meta_model = train(g, Z, y)

# For new prediction:
base_preds = [f_1(x_new), f_2(x_new), ..., f_M(x_new)]
final_pred = g(base_preds)
```

### Blending Algorithm

```
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y)

# Stage 1: Train base models
base_preds_val = []

For each base model m = 1 to M:
    1. Train f_m on X_train, y_train
    2. Predict on X_val: preds_m = f_m(X_val)
    3. Append preds_m to base_preds_val

# Stage 2: Train meta-model
Z_val = stack(base_preds_val)
meta_model = train(g, Z_val, y_val)

# For new prediction:
base_preds = [f_1(x_new), ..., f_M(x_new)]
final_pred = g(base_preds)
```

### Multi-Level Stacking

Can stack multiple layers:

```
Level 0: Base Models (10-20 diverse models)
    ↓
Level 1: Mid-tier Models (3-5 models)
    ↓
Level 2: Meta-Model (simple model like logistic regression)
    ↓
Final Prediction
```

## Mathematical Examples

### Example 1: Simple Stacking (2 base models, 3 samples)

**Data**:
$$\begin{array}{|c|c|c|}
\hline
\mathbf{x} & y & \text{Fold} \\
\hline
x_1 & 0 & 1 \\
x_2 & 1 & 2 \\
x_3 & 1 & 3 \\
\hline
\end{array}$$

**Base Models**: $f_1$ (Model 1), $f_2$ (Model 2)

**Fold 1** (holdout $x_1$):
- Train $f_1, f_2$ on $\{x_2, x_3\}$
- Predict: $f_1(x_1) = 0.2$, $f_2(x_1) = 0.3$
- Store: $\mathbf{z}_1 = [0.2, 0.3]$

**Fold 2** (holdout $x_2$):
- Train $f_1, f_2$ on $\{x_1, x_3\}$
- Predict: $f_1(x_2) = 0.7$, $f_2(x_2) = 0.8$
- Store: $\mathbf{z}_2 = [0.7, 0.8]$

**Fold 3** (holdout $x_3$):
- Train $f_1, f_2$ on $\{x_1, x_2\}$
- Predict: $f_1(x_3) = 0.9$, $f_2(x_3) = 0.85$
- Store: $\mathbf{z}_3 = [0.9, 0.85]$

**Meta-features**:
$$\mathbf{Z} = \begin{bmatrix} 
0.2 & 0.3 \\
0.7 & 0.8 \\
0.9 & 0.85
\end{bmatrix}$$

**Train meta-model** on $(\mathbf{Z}, \mathbf{y})$:

Suppose meta-model is logistic regression:
$$g(\mathbf{z}) = \sigma(w_1 z_1 + w_2 z_2 + b)$$

After training: $w_1 = 0.6, w_2 = 0.4, b = 0$

**New prediction** for $x_{new}$:
- $f_1(x_{new}) = 0.8$
- $f_2(x_{new}) = 0.7$
- $\mathbf{z}_{new} = [0.8, 0.7]$
- $g(\mathbf{z}_{new}) = \sigma(0.6 \times 0.8 + 0.4 \times 0.7) = \sigma(0.76) = 0.68$

### Example 2: Weighted Average Meta-Model

**3 base model predictions**: $f_1(x) = 0.6, f_2(x) = 0.7, f_3(x) = 0.8$

**Weights**: $w_1 = 0.5, w_2 = 0.3, w_3 = 0.2$ (sum = 1)

**Final prediction**:
$$\hat{y} = 0.5 \times 0.6 + 0.3 \times 0.7 + 0.2 \times 0.8$$
$$= 0.3 + 0.21 + 0.16 = 0.67$$

## Implementation in Python

### Basic Stacking with Scikit-Learn

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=42)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,                    # 5-fold cross-validation for base models
    stack_method='predict_proba',  # Use probabilities as meta-features
    n_jobs=-1
)

# Train
print("Training stacking ensemble...")
stacking_clf.fit(X_train, y_train)

# Predict
y_pred = stacking_clf.predict(X_test)

# Evaluate
print("="*60)
print("STACKING ENSEMBLE")
print("="*60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Compare with individual models
print("\nIndividual Model Performance:")
for name, model in base_models:
    model.fit(X_train, y_train)
    y_pred_base = model.predict(X_test)
    print(f"  {name}: {accuracy_score(y_test, y_pred_base):.4f}")
```

### Custom Stacking Implementation

```python
from sklearn.model_selection import KFold
from sklearn.base import clone

class CustomStacking:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Initialize meta-features matrix
        meta_features = np.zeros((n_samples, n_models))
        
        # K-Fold cross-validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"Training base model {i+1}/{n_models}: {name}")
            model_list = []
            
            for train_idx, val_idx in kfold.split(X):
                # Clone model for this fold
                model_clone = clone(model)
                
                # Train on fold
                model_clone.fit(X[train_idx], y[train_idx])
                
                # Predict on validation set
                if hasattr(model_clone, 'predict_proba'):
                    preds = model_clone.predict_proba(X[val_idx])[:, 1]
                else:
                    preds = model_clone.predict(X[val_idx])
                
                meta_features[val_idx, i] = preds
                model_list.append(model_clone)
            
            self.base_models_.append(model_list)
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, n_models))
        
        # Get predictions from all base models
        for i, model_list in enumerate(self.base_models_):
            # Average predictions from all folds
            fold_preds = []
            for model in model_list:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                fold_preds.append(preds)
            meta_features[:, i] = np.mean(fold_preds, axis=0)
        
        # Meta-model prediction
        return self.meta_model.predict(meta_features)

# Example usage
base_models_list = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

meta = LogisticRegression()

custom_stack = CustomStacking(base_models_list, meta, n_folds=5)
custom_stack.fit(X_train, y_train)
y_pred_custom = custom_stack.predict(X_test)

print(f"\nCustom Stacking Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
```

### Blending Implementation

```python
from sklearn.model_selection import train_test_split

class Blending:
    def __init__(self, base_models, meta_model, val_size=0.2):
        self.base_models = base_models
        self.meta_model = meta_model
        self.val_size = val_size
        
    def fit(self, X, y):
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=42
        )
        
        # Train base models on training set
        n_models = len(self.base_models)
        val_predictions = np.zeros((X_val.shape[0], n_models))
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"Training base model {i+1}/{n_models}: {name}")
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                val_predictions[:, i] = model.predict_proba(X_val)[:, 1]
            else:
                val_predictions[:, i] = model.predict(X_val)
        
        # Train meta-model on validation predictions
        print("Training meta-model...")
        self.meta_model.fit(val_predictions, y_val)
        
        return self
    
    def predict(self, X):
        n_models = len(self.base_models)
        predictions = np.zeros((X.shape[0], n_models))
        
        for i, (name, model) in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                predictions[:, i] = model.predict(X)
        
        return self.meta_model.predict(predictions)

# Example
blender = Blending(base_models_list, LogisticRegression(), val_size=0.2)
blender.fit(X_train, y_train)
y_pred_blend = blender.predict(X_test)

print(f"\nBlending Accuracy: {accuracy_score(y_test, y_pred_blend):.4f}")
```

### Multi-Level Stacking

```python
# Level 0: Multiple diverse models
level_0 = [
    ('rf1', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('rf2', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=43)),
    ('gb1', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('gb2', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=43)),
    ('svm', SVC(probability=True, random_state=42))
]

# Level 1: Mid-tier ensemble
level_1_base = [
    ('stack_rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('stack_gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

level_1 = StackingClassifier(
    estimators=level_0,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    cv=3
)

# Level 2: Final meta-model
multi_level_stack = StackingClassifier(
    estimators=[('level1', level_1)] + level_1_base,
    final_estimator=LogisticRegression(),
    cv=5
)

# Train
multi_level_stack.fit(X_train, y_train)
y_pred_multi = multi_level_stack.predict(X_test)

print(f"\nMulti-Level Stacking Accuracy: {accuracy_score(y_test, y_pred_multi):.4f}")
```

### Visualization: Model Comparison

```python
import matplotlib.pyplot as plt

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Stacking': stacking_clf
}

accuracies = []
model_names = []

for name, model in models.items():
    if name != 'Stacking':
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    model_names.append(name)
    print(f"{name}: {acc:.4f}")

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'], 
               edgecolor='black', alpha=0.7)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Individual vs Stacking')
plt.xticks(rotation=45, ha='right')
plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.02])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Practical Applications

### 1. **Kaggle Competitions**
Top solutions often use multi-level stacking with 20+ base models.

### 2. **Customer Churn Prediction**
Combine models that capture different customer behavior patterns.

### 3. **Credit Risk Assessment**
Ensemble models for robust default prediction.

### 4. **Fraud Detection**
Stack specialized models for different fraud types.

### 5. **Image Classification**
Combine CNN architectures (ResNet, VGG, Inception).

### 6. **Natural Language Processing**
Ensemble different text models (BERT, LSTM, transformers).

## Summary

Stacking and blending leverage model diversity to achieve superior performance.

**Key Concepts**:
1. **Base Models**: Diverse models capturing different patterns
2. **Meta-Model**: Learns optimal combination of base predictions
3. **Out-of-Fold Predictions**: Prevents overfitting in stacking
4. **Model Diversity**: Essential for ensemble success

**Stacking vs Blending**:
- **Stacking**: K-fold CV, uses all data, better performance
- **Blending**: Holdout validation, simpler, faster

**Best Practices**:
✅ Use diverse base models (different algorithms/hyperparameters)
✅ Keep meta-model simple (logistic regression, linear models)
✅ Use 5-10 fold CV for stacking
✅ Monitor for overfitting on meta-level
✅ Validate on separate holdout set

❌ Don't use only similar base models
❌ Don't over-complicate meta-model
❌ Don't skip validation
❌ Don't ignore computational cost
❌ Don't expect interpretability

Stacking is the secret weapon of competition winners!
