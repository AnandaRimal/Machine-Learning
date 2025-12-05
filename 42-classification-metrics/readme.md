# Classification Metrics

## Table of Contents
- [Introduction](#introduction)
- [The Confusion Matrix](#the-confusion-matrix)
- [Basic Metrics](#basic-metrics)
- [Advanced Metrics](#advanced-metrics)
- [Mathematical Foundation](#mathematical-foundation)
- [ROC Curve and AUC](#roc-curve-and-auc)
- [When to Use Which Metric](#when-to-use-which-metric)
- [Mathematical Examples](#mathematical-examples)
- [Implementation in Python](#implementation-in-python)
- [Practical Applications](#practical-applications)

## Introduction

Classification metrics are essential tools for evaluating the performance of classification models. Unlike regression (where we predict continuous values), classification involves predicting discrete class labels. Simply measuring "accuracy" is often insufficient and can be misleading, especially with imbalanced datasets.

Choosing the right metric depends on:
- The business problem
- Class distribution
- Cost of different types of errors
- Threshold requirements

## The Confusion Matrix

The confusion matrix is the foundation of all classification metrics. For binary classification:

$$\begin{array}{|c|c|c|}
\hline
 & \text{Predicted Positive} & \text{Predicted Negative} \\
\hline
\text{Actual Positive} & \text{TP (True Positive)} & \text{FN (False Negative)} \\
\hline
\text{Actual Negative} & \text{FP (False Positive)} & \text{TN (True Negative)} \\
\hline
\end{array}$$

### Definitions

**True Positive (TP)**: Correctly predicted positive class
- Model predicts positive, actual is positive
- Example: Predicted cancer, patient has cancer ✓

**True Negative (TN)**: Correctly predicted negative class
- Model predicts negative, actual is negative
- Example: Predicted healthy, patient is healthy ✓

**False Positive (FP)**: Incorrectly predicted positive (Type I Error)
- Model predicts positive, actual is negative
- Example: Predicted cancer, patient is healthy ✗
- Also called "False Alarm"

**False Negative (FN)**: Incorrectly predicted negative (Type II Error)
- Model predicts negative, actual is positive
- Example: Predicted healthy, patient has cancer ✗
- Also called "Miss"

### Total Predictions

$$\text{Total} = TP + TN + FP + FN$$

## Basic Metrics

### 1. Accuracy

The proportion of correct predictions:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Range**: [0, 1] or [0%, 100%]

**Interpretation**: Overall correctness of the model

**Pros**:
- Simple and intuitive
- Good for balanced datasets

**Cons**:
- Misleading for imbalanced data
- Treats all errors equally

**Example**: In a dataset with 95% class 0 and 5% class 1, a model that always predicts class 0 achieves 95% accuracy!

### 2. Error Rate

The proportion of incorrect predictions:

$$\text{Error Rate} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy}$$

### 3. Precision (Positive Predictive Value)

Of all positive predictions, how many were actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Range**: [0, 1]

**Interpretation**: How reliable are positive predictions?

**Use when**: Cost of false positives is high
- Email spam detection (don't want important emails marked as spam)
- Recommender systems (don't recommend irrelevant items)

**Perfect precision**: No false positives (FP = 0)

### 4. Recall (Sensitivity, True Positive Rate)

Of all actual positives, how many did we correctly identify?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Range**: [0, 1]

**Interpretation**: How complete is our positive detection?

**Use when**: Cost of false negatives is high
- Cancer detection (don't miss cancer cases)
- Fraud detection (catch all fraudulent transactions)
- Airport security (detect all threats)

**Perfect recall**: No false negatives (FN = 0)

### 5. Specificity (True Negative Rate)

Of all actual negatives, how many did we correctly identify?

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Range**: [0, 1]

**Interpretation**: How well do we identify negatives?

**Relationship**: $\text{Specificity} = 1 - \text{FPR}$

### 6. False Positive Rate (FPR)

Of all actual negatives, how many did we incorrectly classify as positive?

$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

### 7. False Negative Rate (FNR)

Of all actual positives, how many did we incorrectly classify as negative?

$$\text{FNR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$$

## Advanced Metrics

### 1. F1-Score

The harmonic mean of precision and recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**Why harmonic mean?** Gives more weight to lower values. If either precision or recall is low, F1 will be low.

**Range**: [0, 1]

**Use when**: Need balance between precision and recall

**Properties**:
- Reaches maximum (1) only when both precision and recall are perfect
- More sensitive to low values than arithmetic mean
- Useful for imbalanced datasets

### 2. F-Beta Score

Generalized F-score with adjustable weight:

$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

**Parameter $\beta$**:
- $\beta = 1$: Equal weight (F1-score)
- $\beta = 0.5$: More weight on precision (F0.5-score)
- $\beta = 2$: More weight on recall (F2-score)

**Interpretation**:
- $\beta > 1$: Recall more important than precision
- $\beta < 1$: Precision more important than recall

### 3. Matthews Correlation Coefficient (MCC)

A balanced measure that considers all four confusion matrix categories:

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

**Range**: [-1, 1]
- +1: Perfect prediction
- 0: Random prediction
- -1: Total disagreement

**Advantages**:
- Balanced even for imbalanced datasets
- Considers all four confusion matrix quadrants
- Single, comprehensive metric

### 4. Cohen's Kappa

Measures agreement between predictions and truth, accounting for chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where:
- $p_o$: Observed agreement (accuracy)
- $p_e$: Expected agreement by chance

**Range**: [-1, 1]

**Interpretation**:
- < 0: Less than chance agreement
- 0.01-0.20: Slight agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

## Mathematical Foundation

### Precision-Recall Trade-off

Increasing recall often decreases precision and vice versa:

**Extreme 1**: Predict everything as positive
- Recall = 1.0 (all positives caught)
- Precision = low (many false positives)

**Extreme 2**: Predict only when very confident
- Precision = high (few false positives)
- Recall = low (many false negatives)

### Relationship Between Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}$$

### Multiclass Extension

For multiclass classification, use one-vs-all approach:

**Macro Average**: Average metric across all classes
$$\text{Macro-Precision} = \frac{1}{C} \sum_{i=1}^{C} \text{Precision}_i$$

**Weighted Average**: Weight by class frequency
$$\text{Weighted-Precision} = \sum_{i=1}^{C} w_i \times \text{Precision}_i$$

where $w_i = \frac{n_i}{n_{total}}$

**Micro Average**: Calculate globally across all instances
$$\text{Micro-Precision} = \frac{\sum_{i=1}^{C} TP_i}{\sum_{i=1}^{C} (TP_i + FP_i)}$$

## ROC Curve and AUC

### Receiver Operating Characteristic (ROC) Curve

Plots True Positive Rate (Recall) vs False Positive Rate across different thresholds:

$$\text{TPR} = \frac{TP}{TP + FN} \quad (\text{Recall})$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

**Characteristics**:
- X-axis: FPR (0 to 1)
- Y-axis: TPR (0 to 1)
- Diagonal line: Random classifier
- Top-left corner: Perfect classifier

### Area Under ROC Curve (AUC-ROC)

The area under the ROC curve:

$$\text{AUC} = \int_0^1 \text{TPR}(FPR) \, d(FPR)$$

**Range**: [0, 1]

**Interpretation**:
- 0.5: Random classifier (no discrimination)
- 0.7-0.8: Acceptable discrimination
- 0.8-0.9: Excellent discrimination
- > 0.9: Outstanding discrimination
- 1.0: Perfect classifier

**Probabilistic Interpretation**: Probability that a randomly chosen positive instance ranks higher than a randomly chosen negative instance.

### Precision-Recall Curve

Alternative to ROC curve, plots Precision vs Recall:

**Use when**:
- Imbalanced datasets
- Positive class is rare
- Focus on positive class performance

**AUC-PR**: Area under Precision-Recall curve

## When to Use Which Metric

### By Problem Type

| Problem | Primary Metric | Reason |
|---------|---------------|---------|
| **Balanced classes** | Accuracy, F1 | Classes equally important |
| **Imbalanced classes** | Precision, Recall, F1, AUC | Accuracy misleading |
| **Spam detection** | Precision | Minimize false positives |
| **Cancer diagnosis** | Recall | Minimize false negatives |
| **General binary** | F1, AUC-ROC | Balanced view |
| **Rare events** | Precision-Recall AUC | Focus on positive class |

### By Business Cost

**High cost of false positives**: Use Precision
- Credit approval (don't approve bad loans)
- Content recommendation (don't recommend bad content)

**High cost of false negatives**: Use Recall
- Disease screening (don't miss sick patients)
- Security threats (don't miss dangers)
- Customer churn (don't miss at-risk customers)

**Equal costs**: Use F1-score or Accuracy

### By Class Distribution

**Balanced** (40-60% each class):
- Accuracy
- F1-Score
- AUC-ROC

**Moderately Imbalanced** (70-30):
- F1-Score
- Precision and Recall
- AUC-ROC

**Highly Imbalanced** (90-10 or worse):
- Precision-Recall AUC
- F-Beta (adjust β for preference)
- MCC (balanced metric)

## Mathematical Examples

### Example 1: Binary Classification Metrics

**Scenario**: Disease diagnosis model tested on 100 patients

**Confusion Matrix**:
$$\begin{array}{|c|c|c|}
\hline
 & \text{Predicted Sick} & \text{Predicted Healthy} \\
\hline
\text{Actually Sick} & 45 & 5 \\
\hline
\text{Actually Healthy} & 10 & 40 \\
\hline
\end{array}$$

**Values**: TP=45, FN=5, FP=10, TN=40

**Accuracy**:
$$\text{Accuracy} = \frac{45 + 40}{45 + 40 + 10 + 5} = \frac{85}{100} = 0.85 \text{ or } 85\%$$

**Precision**:
$$\text{Precision} = \frac{45}{45 + 10} = \frac{45}{55} = 0.818 \text{ or } 81.8\%$$

Interpretation: Of all patients we diagnosed as sick, 81.8% actually were sick.

**Recall**:
$$\text{Recall} = \frac{45}{45 + 5} = \frac{45}{50} = 0.90 \text{ or } 90\%$$

Interpretation: Of all actually sick patients, we correctly identified 90%.

**F1-Score**:
$$F_1 = 2 \times \frac{0.818 \times 0.90}{0.818 + 0.90} = 2 \times \frac{0.736}{1.718} = 0.857$$

**Specificity**:
$$\text{Specificity} = \frac{40}{40 + 10} = \frac{40}{50} = 0.80 \text{ or } 80\%$$

**MCC**:
$$MCC = \frac{45 \times 40 - 10 \times 5}{\sqrt{(45+10)(45+5)(40+10)(40+5)}}$$
$$= \frac{1800 - 50}{\sqrt{55 \times 50 \times 50 \times 45}}$$
$$= \frac{1750}{\sqrt{6{,}187{,}500}} = \frac{1750}{2487.47} = 0.703$$

### Example 2: Imbalanced Dataset

**Scenario**: Fraud detection with 1000 transactions (990 normal, 10 fraud)

**Model A** (Always predicts "Normal"):

Confusion Matrix:
$$\begin{array}{|c|c|c|}
\hline
 & \text{Pred Fraud} & \text{Pred Normal} \\
\hline
\text{Actual Fraud} & 0 & 10 \\
\hline
\text{Actual Normal} & 0 & 990 \\
\hline
\end{array}$$

TP=0, FN=10, FP=0, TN=990

- Accuracy: $\frac{0+990}{1000} = 99\%$ (looks great!)
- Precision: $\frac{0}{0+0} = \text{undefined}$
- Recall: $\frac{0}{0+10} = 0\%$ (terrible!)
- F1: 0%

**Model B** (Actual classifier):

Confusion Matrix:
$$\begin{array}{|c|c|c|}
\hline
 & \text{Pred Fraud} & \text{Pred Normal} \\
\hline
\text{Actual Fraud} & 8 & 2 \\
\hline
\text{Actual Normal} & 50 & 940 \\
\hline
\end{array}$$

TP=8, FN=2, FP=50, TN=940

- Accuracy: $\frac{8+940}{1000} = 94.8\%$ (lower than Model A)
- Precision: $\frac{8}{8+50} = 13.8\%$
- Recall: $\frac{8}{8+2} = 80\%$ (much better!)
- F1: $2 \times \frac{0.138 \times 0.80}{0.138 + 0.80} = 23.5\%$

**Conclusion**: Model B is better despite lower accuracy! It actually catches fraud.

### Example 3: F-Beta Score Comparison

**Confusion Matrix**: TP=70, FN=30, FP=20, TN=880

**Precision**: $\frac{70}{70+20} = 0.778$

**Recall**: $\frac{70}{70+30} = 0.70$

**F0.5** (favors precision):
$$F_{0.5} = (1 + 0.5^2) \times \frac{0.778 \times 0.70}{0.5^2 \times 0.778 + 0.70}$$
$$= 1.25 \times \frac{0.545}{0.194 + 0.70} = 1.25 \times \frac{0.545}{0.894} = 0.762$$

**F1** (balanced):
$$F_1 = 2 \times \frac{0.778 \times 0.70}{0.778 + 0.70} = 0.737$$

**F2** (favors recall):
$$F_2 = (1 + 2^2) \times \frac{0.778 \times 0.70}{2^2 \times 0.778 + 0.70}$$
$$= 5 \times \frac{0.545}{3.112 + 0.70} = 5 \times \frac{0.545}{3.812} = 0.715$$

Notice: As β increases (more weight on recall), F-score decreases because recall < precision in this example.

## Implementation in Python

### Basic Metrics

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)

# Example predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Extract TP, TN, FP, FN
tn, fp, fn, tp = cm.ravel()
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print()

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print()

# Comprehensive report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
```

### Custom Metric Calculations

```python
def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all classification metrics from scratch
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity and FPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # F-scores
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # F-beta scores
    beta = 0.5
    f_beta_05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
    
    beta = 2
    f_beta_2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
    
    # MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'FPR': fpr,
        'F1-Score': f1,
        'F0.5-Score': f_beta_05,
        'F2-Score': f_beta_2,
        'MCC': mcc
    }

# Example
y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

metrics = calculate_all_metrics(y_true, y_pred)

print("="*50)
print("ALL CLASSIFICATION METRICS")
print("="*50)
for metric, value in metrics.items():
    if metric in ['TP', 'TN', 'FP', 'FN']:
        print(f"{metric:<15}: {value}")
    else:
        print(f"{metric:<15}: {value:.4f}")
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Example with probability scores
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.75, 0.6, 0.05])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Alternative: direct AUC calculation
roc_auc_direct = roc_auc_score(y_true, y_scores)

print(f"AUC-ROC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Print threshold analysis
print("\nThreshold Analysis:")
print(f"{'Threshold':<12} {'FPR':<10} {'TPR':<10}")
print("-"*32)
for i in range(0, len(thresholds), max(1, len(thresholds)//10)):
    print(f"{thresholds[i]:<12.3f} {fpr[i]:<10.3f} {tpr[i]:<10.3f}")
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

print(f"Average Precision: {avg_precision:.4f}")

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()
```

### Multiclass Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Multiclass example
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 2, 1, 0]
y_pred_multi = [0, 2, 2, 0, 1, 1, 0, 2, 1, 0]

# Confusion matrix
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Multiclass')
plt.show()

# Classification report with macro/micro/weighted averages
print("\nMulticlass Classification Report:")
print(classification_report(y_true_multi, y_pred_multi, 
                           target_names=['Class 0', 'Class 1', 'Class 2']))
```

### Complete Model Evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.5 > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("="*70)
print("COMPLETE MODEL EVALUATION")
print("="*70)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# All metrics
print(f"\nAccuracy:           {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:          {precision_score(y_test, y_pred):.4f}")
print(f"Recall:             {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:           {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:            {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Average Precision:  {average_precision_score(y_test, y_pred_proba):.4f}")

# Matthews Correlation Coefficient
print(f"MCC:                {matthews_corrcoef(y_test, y_pred):.4f}")

# Cohen's Kappa
from sklearn.metrics import cohen_kappa_score
print(f"Cohen's Kappa:      {cohen_kappa_score(y_test, y_pred):.4f}")

# Full classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
```

## Practical Applications

### 1. Medical Diagnosis
**Metric**: Recall (Sensitivity)
**Reason**: Must catch all sick patients; false negatives are dangerous

### 2. Spam Detection
**Metric**: Precision
**Reason**: Don't want important emails in spam; false positives are costly

### 3. Fraud Detection
**Metric**: F2-Score or Recall
**Reason**: Missing fraud is expensive; can tolerate some false alarms

### 4. Content Recommendation
**Metric**: Precision@K
**Reason**: Only show relevant items; user sees limited recommendations

### 5. Credit Approval
**Metric**: Balanced F1 or Custom Cost Function
**Reason**: Both false positives (bad loans) and false negatives (lost customers) matter

## Summary

Classification metrics provide diverse perspectives on model performance. No single metric tells the complete story.

**Key Formulas**:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1**: $\frac{2 \times Precision \times Recall}{Precision + Recall}$
- **MCC**: $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

**Best Practices**:
✅ Always look at confusion matrix first
✅ Use multiple metrics, not just accuracy
✅ Consider business costs when choosing metrics
✅ Account for class imbalance
✅ Visualize ROC and PR curves

❌ Don't rely solely on accuracy for imbalanced data
❌ Don't ignore the confusion matrix
❌ Don't forget to set appropriate thresholds
❌ Don't use metrics without understanding trade-offs

Choose metrics that align with your business objectives!
