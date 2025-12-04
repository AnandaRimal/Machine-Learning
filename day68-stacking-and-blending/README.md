# Day 68: Stacking and Blending 🎯

## 📖 Overview

Stacking and Blending are advanced ensemble learning techniques that combine multiple models to create a more powerful predictor. Unlike bagging (Random Forest) or boosting (AdaBoost, Gradient Boosting), these methods use a meta-learner to combine predictions from base models.

## 🎯 Learning Objectives

- Understand the concept of meta-learning
- Implement Stacking with cross-validation
- Learn Blending techniques
- Compare Stacking vs Blending
- Build multi-layer stacked models
- Optimize ensemble performance
- Understand when to use each technique

## 📚 Theoretical Concepts

### What is Ensemble Learning?

**Ensemble Learning** combines multiple models to produce better predictions than any individual model. The key principle: *"Wisdom of the crowd"*.

### Stacking (Stacked Generalization)

**Stacking** trains a meta-model to combine predictions from multiple base models using cross-validation.

**Architecture**:
```
                    Final Prediction
                           ↑
                    [ Meta-Model ]
                     (Level 1)
                    /      |      \
            Predictions  Predictions  Predictions
                /          |           \
        [Model 1]      [Model 2]     [Model 3]
         (Base)         (Base)        (Base)
         Level 0        Level 0       Level 0
            \              |            /
                 Original Features
```

### Blending

**Blending** is a simplified version of stacking that uses a holdout validation set instead of cross-validation.

**Key Difference from Stacking**:
- **Stacking**: Uses cross-validation to generate out-of-fold predictions
- **Blending**: Uses a simple train/validation split

## 💻 Mathematical Foundation

### Stacking Process

**Step 1**: Train base models with k-fold cross-validation
For each base model $h_i$ and each fold $j$:
- Train on training folds
- Predict on validation fold
- Collect out-of-fold predictions

**Step 2**: Create meta-features
$$\mathbf{X}_{meta} = [h_1(\mathbf{X}), h_2(\mathbf{X}), ..., h_n(\mathbf{X})]$$

**Step 3**: Train meta-model
$$f_{meta}(h_1, h_2, ..., h_n) = y$$

**Final Prediction**:
$$\hat{y} = f_{meta}(h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_n(\mathbf{x}))$$

## 🔬 Implementation

### Basic Stacking with Scikit-learn

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base models (Level 0)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Define meta-model (Level 1)
meta_model = LogisticRegression()

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # 5-fold cross-validation
)

# Train
stacking_clf.fit(X_train, y_train)

# Predict
y_pred = stacking_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Accuracy: {accuracy:.4f}")
```

### Manual Stacking Implementation

```python
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

# Step 1: Define base models
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm': SVC(probability=True, random_state=42)
}

# Step 2: Generate out-of-fold predictions for training set
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    # Cross-validated predictions
    meta_features_train[:, i] = cross_val_predict(
        model, X_train, y_train,
        cv=5,
        method='predict_proba'
    )[:, 1]  # Probability of positive class

# Step 3: Train base models on full training set for test predictions
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    model.fit(X_train, y_train)
    meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

# Step 4: Train meta-model
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)

# Step 5: Make final predictions
final_predictions = meta_model.predict(meta_features_test)

# Evaluate
accuracy = accuracy_score(y_test, final_predictions)
print(f"Manual Stacking Accuracy: {accuracy:.4f}")
```

### Blending Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Step 1: Split data into train, validation, and test
X_train_blend, X_val, y_train_blend, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Step 2: Train base models on blend training set
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'lr': LogisticRegression(random_state=42)
}

# Train each base model
for name, model in base_models.items():
    model.fit(X_train_blend, y_train_blend)

# Step 3: Generate predictions on validation set
val_predictions = np.zeros((X_val.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    val_predictions[:, i] = model.predict_proba(X_val)[:, 1]

# Step 4: Train meta-model on validation predictions
meta_model = LogisticRegression()
meta_model.fit(val_predictions, y_val)

# Step 5: Generate predictions on test set
test_predictions = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_predictions[:, i] = model.predict_proba(X_test)[:, 1]

# Step 6: Make final predictions
final_predictions = meta_model.predict(test_predictions)

# Evaluate
accuracy = accuracy_score(y_test, final_predictions)
print(f"Blending Accuracy: {accuracy:.4f}")
```

### Multi-Layer Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Layer 0 (Base models)
layer_0 = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('nb', GaussianNB())
]

# Layer 1 (First level stacking)
layer_1_clf = StackingClassifier(
    estimators=layer_0,
    final_estimator=LogisticRegression(),
    cv=5
)

# Layer 2 (Second level stacking)
layer_2 = [
    ('stack_1', layer_1_clf),
    ('rf_2', RandomForestClassifier(n_estimators=200))
]

# Final stacked model
final_stacked_clf = StackingClassifier(
    estimators=layer_2,
    final_estimator=LogisticRegression(),
    cv=3
)

# Train
final_stacked_clf.fit(X_train, y_train)

# Predict
y_pred = final_stacked_clf.predict(X_test)
```

## 📊 Comparison: Stacking vs Blending

| Aspect | Stacking | Blending |
|--------|----------|----------|
| **Validation** | K-fold cross-validation | Simple train/val split |
| **Data Usage** | Uses all training data | Loses validation data |
| **Computational Cost** | Higher (k models × n base models) | Lower |
| **Overfitting Risk** | Lower (better generalization) | Higher |
| **Implementation** | More complex | Simpler |
| **Performance** | Generally better | Slightly worse |
| **Best For** | Production models | Quick prototyping |

## 🎯 When to Use Stacking/Blending

### Use Stacking When:
✅ You want maximum performance
✅ You have diverse base models
✅ You have sufficient computational resources
✅ You're working on important competitions/production

### Use Blending When:
✅ You need quick prototyping
✅ Computational resources are limited
✅ Dataset is very large
✅ Simple validation split is sufficient

## 🎓 Best Practices

### 1. **Diversity is Key**
```python
# Good: Diverse models
base_models = [
    ('tree', DecisionTreeClassifier()),      # Non-linear, interpretable
    ('svm', SVC(probability=True)),          # Kernel-based
    ('lr', LogisticRegression()),            # Linear
    ('nb', GaussianNB())                     # Probabilistic
]

# Bad: Similar models
base_models = [
    ('rf1', RandomForestClassifier(n_estimators=50)),
    ('rf2', RandomForestClassifier(n_estimators=100)),
    ('rf3', RandomForestClassifier(n_estimators=200))
]
```

### 2. **Choose Appropriate Meta-Model**
```python
# For classification
meta_models = {
    'Logistic Regression': LogisticRegression(),  # Fast, interpretable
    'Random Forest': RandomForestClassifier(),    # Handles non-linearity
    'XGBoost': XGBClassifier()                   # High performance
}

# For regression
meta_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}
```

### 3. **Feature Engineering for Meta-Model**
```python
# Include original features + base predictions
meta_features = np.column_stack([
    X_train,                    # Original features
    base_pred_1,                # Predictions from base model 1
    base_pred_2,                # Predictions from base model 2
    base_pred_1 * base_pred_2,  # Interaction terms
    np.abs(base_pred_1 - base_pred_2)  # Differences
])
```

### 4. **Prevent Data Leakage**
```python
# WRONG: Training and predicting on same data
model.fit(X_train, y_train)
meta_features = model.predict(X_train)  # LEAKAGE!

# CORRECT: Use cross-validation
meta_features = cross_val_predict(model, X_train, y_train, cv=5)
```

## 🔬 Advanced Techniques

### 1. **Weighted Blending**
```python
# Assign weights to base model predictions
weights = [0.5, 0.3, 0.2]  # Weights for 3 models
weighted_pred = sum(w * pred for w, pred in zip(weights, predictions))
```

### 2. **Stacking with Feature Engineering**
```python
# Add original features to meta-features
meta_features_enhanced = np.column_stack([
    meta_features,  # Base model predictions
    X_train         # Original features
])

meta_model.fit(meta_features_enhanced, y_train)
```

### 3. **Dynamic Model Selection**
```python
# Select best k models for stacking
from sklearn.model_selection import cross_val_score

# Evaluate each base model
scores = {}
for name, model in all_models.items():
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    scores[name] = score

# Select top 3 models
top_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
```

## 📈 Real-World Example: Kaggle Competition

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier

# Load competition data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop('target', axis=1)
y = train['target']

# Diverse base models with optimized hyperparameters
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Meta-model
meta_model = LogisticRegression(C=0.1, max_iter=1000)

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=10,  # 10-fold CV
    stack_method='predict_proba',  # Use probabilities
    n_jobs=-1  # Use all CPU cores
)

# Train
stacking_clf.fit(X, y)

# Predict on test set
predictions = stacking_clf.predict(test)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'target': predictions
})
submission.to_csv('submission.csv', index=False)
```

## 📁 Files in This Module

- `stacking_demo.ipynb` - Complete stacking implementation
- `blending_demo.ipynb` - Blending techniques
- `multi_layer_stacking.ipynb` - Advanced multi-layer approach
- `comparison.ipynb` - Performance comparison

## 🔗 Real-World Applications

1. **Kaggle Competitions**: Top solutions almost always use stacking
2. **Credit Scoring**: Combine multiple risk models
3. **Medical Diagnosis**: Ensemble of diagnostic models
4. **Fraud Detection**: Multiple fraud detection algorithms
5. **Recommendation Systems**: Combine collaborative and content-based filters

## 📚 Further Reading

- [Stacking Paper (Wolpert, 1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
- [Kaggle Ensemble Guide](https://mlwave.com/kaggle-ensembling-guide/)
- [Scikit-learn Stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacking)

## ➡️ Next Steps

- Experiment with different base model combinations
- Try multi-layer stacking
- Optimize hyperparameters of both base and meta-models
- Practice on Kaggle competitions

## 🎯 Key Takeaways

1. **Diversity Matters**: Use different types of models
2. **Prevent Leakage**: Always use cross-validation for stacking
3. **Start Simple**: Begin with 2-3 base models
4. **Computational Cost**: Stacking is expensive but powerful
5. **Diminishing Returns**: More models ≠ better performance

---

**Stacking and Blending are powerful techniques that can give you that extra 1-2% performance boost that wins competitions!**
