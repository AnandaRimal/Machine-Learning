# Complete Case Analysis - Listwise Deletion

## General Idea

Complete Case Analysis (CCA), also known as listwise deletion, is the simplest approach to handling missing data: remove any row (observation) that contains one or more missing values. Only complete cases (rows with no missing data) are retained for analysis. While straightforward, this method has significant implications for sample size, statistical power, and potential bias.

## Why Use Complete Case Analysis?

1. **Simplicity**: Easiest to implement and understand
2. **No Imputation Bias**: Doesn't introduce artificial values
3. **Standard Methods**: Can use any ML algorithm without modification
4. **Unbiased Estimates** (if MCAR): Valid inference when data Missing Completely At Random
5. **Quick Prototyping**: Fast baseline approach
6. **Small Missing Proportion**: Effective when <5% data missing
7. **Conservative**: Avoids assumptions about missing mechanism

## Role in Machine Learning

### The Missing Data Problem

**Real-world datasets** rarely complete:
```
Age   Income   Education   Target
25    50000    Bachelor    1
30    NaN      Master      0
NaN   60000    PhD         1
35    70000    NaN         0
```

**ML algorithms** require complete numeric input:
- Can't compute with NaN values
- Matrix operations fail
- Distance metrics undefined

**Solutions**:
1. **Deletion**: Remove incomplete rows (CCA)
2. **Imputation**: Fill missing values
3. **Special handling**: Missing as category

### Impact on Model Training

**Before CCA**: $n = 1000$ samples, 30% missing some value

**After CCA**: $n' \approx 700$ samples (if missing distributed)

**Consequences**:
- **Reduced sample size**: $n' < n$
- **Lost information**: Discarded partial data
- **Potential bias**: If missingness not random
- **Lower power**: Statistical tests less reliable

**Mathematical representation**:
$$n_{complete} = n \times \prod_{j=1}^{p} (1 - r_j)$$

Where $r_j$ is missingness rate in feature $j$ (assuming independence)

**Example**: 10 features, each 10% missing:
$$n_{complete} \approx n \times (0.9)^{10} \approx 0.35n$$

Only 35% of data retained!

## Missing Data Mechanisms

### 1. Missing Completely At Random (MCAR)

**Definition**: Missingness is independent of observed and unobserved data

**Mathematical**:
$$P(\text{Missing} | X_{obs}, X_{miss}) = P(\text{Missing})$$

**Example**: 
- Survey responses randomly lost due to server error
- Accidental deletion of entries
- Random sensor failures

**Characteristics**:
- Missingness pattern has no relationship to data values
- Missing and observed data have same distribution
- **CCA is unbiased** under MCAR

**Test**: Compare distributions of observed values for missing vs non-missing cases
- If identical → likely MCAR
- If different → not MCAR

### 2. Missing At Random (MAR)

**Definition**: Missingness depends on observed data, but not unobserved data

**Mathematical**:
$$P(\text{Missing} | X_{obs}, X_{miss}) = P(\text{Missing} | X_{obs})$$

**Example**:
- Younger people less likely to report income (depends on age, which is observed)
- Men less likely to answer weight question (depends on gender, which is observed)

**Characteristics**:
- Missingness can be explained by other variables in dataset
- **CCA may be biased** under MAR
- Imputation can be unbiased if conditioned on observed data

**Test**: Check if missingness probability differs across groups
```
Missing income:
- Age < 25: 40% missing
- Age 25-50: 20% missing
- Age > 50: 10% missing
→ MAR (depends on age)
```

### 3. Missing Not At Random (MNAR)

**Definition**: Missingness depends on the missing value itself

**Mathematical**:
$$P(\text{Missing} | X_{obs}, X_{miss}) = f(X_{miss})$$

**Example**:
- High earners don't report income (missingness depends on income value itself)
- Severely depressed patients don't complete depression surveys
- Faulty sensors fail more often at extreme temperatures

**Characteristics**:
- Missingness mechanism depends on unobserved values
- **CCA is biased** under MNAR
- Difficult to handle without domain knowledge

**Test**: Hard to detect (requires external information or domain knowledge)

### Implications for CCA

| Mechanism | CCA Biased? | Safe to Use? |
|-----------|-------------|-------------|
| **MCAR** | No | ✓ Yes (but power loss) |
| **MAR** | Yes | ✗ Generally no |
| **MNAR** | Yes | ✗ Definitely no |

## Mathematical Properties

### Sample Size Reduction

**Single feature** with missingness rate $r$:
$$n_{complete} = n \times (1 - r)$$

**Multiple features** (independent missingness):
$$n_{complete} = n \times \prod_{j=1}^{p} (1 - r_j)$$

**Example**:
- Original: $n = 1000$
- 5 features: $r_1 = 0.1, r_2 = 0.05, r_3 = 0.15, r_4 = 0.08, r_5 = 0.12$
- Complete cases: $1000 \times 0.9 \times 0.95 \times 0.85 \times 0.92 \times 0.88 \approx 589$
- **Lost**: 41% of data

### Statistical Power Loss

**Statistical power**: Probability of detecting true effect

$$\text{Power} \propto \sqrt{n}$$

**With CCA**:
$$\text{Power}_{CCA} = \text{Power}_{full} \times \sqrt{\frac{n_{complete}}{n}}$$

**Example**: 50% data loss → Power reduced by $\sqrt{0.5} \approx 0.71$ (29% reduction)

### Bias Under MAR/MNAR

**Under MCAR**: 
$$E[\hat{\theta}_{CCA}] = \theta_{true}$$
(Unbiased, but higher variance)

**Under MAR/MNAR**:
$$E[\hat{\theta}_{CCA}] \neq \theta_{true}$$
(Biased estimates)

**Example** (MNAR):
- True mean income: $\mu = 60000$
- High earners (>80k) don't report: Missing
- Observed mean after CCA: $\hat{\mu} = 48000$
- **Bias**: $-12000$ (underestimate)

## Implementing Complete Case Analysis

### Pandas - dropna()

**Drop rows with any missing**:
```python
import pandas as pd

df_complete = df.dropna()
```

**Drop rows with missing in specific columns**:
```python
df_complete = df.dropna(subset=['Age', 'Income'])
```

**Drop rows with all missing**:
```python
df_complete = df.dropna(how='all')
```

**Drop if more than threshold missing**:
```python
df_complete = df.dropna(thresh=8)  # Keep rows with at least 8 non-null values
```

### NumPy - Masking

```python
import numpy as np

# Remove rows with any NaN
mask = ~np.isnan(X).any(axis=1)
X_complete = X[mask]
y_complete = y[mask]
```

### Scikit-Learn - No Built-in CCA

**Manual approach**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Drop missing before split
df_complete = df.dropna()

X = df_complete.drop('target', axis=1)
y = df_complete['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
```

**Note**: Drop on training data only to avoid leakage in imputation approaches

## When to Use Complete Case Analysis

### Use When:

**1. Small Missingness (<5%)**:
- Minimal data loss
- Negligible bias even if not MCAR
- Quick and simple

**2. MCAR Confirmed**:
- Statistical tests support MCAR
- Domain knowledge indicates random missingness
- CCA is unbiased

**3. Missingness in Target Variable**:
- Can't impute target (supervised learning)
- Must remove cases with missing $y$

**4. Quick Baseline**:
- Initial exploration
- Prototype models
- Compare with imputation methods

**5. Many Observations, Few Features**:
- Large $n$, even 30% loss still sufficient
- Few features → less compound missingness

**6. Exploratory Analysis**:
- Understanding data patterns
- Not final model

### Don't Use When:

**1. High Missingness (>10-20%)**:
- Severe sample size reduction
- Low statistical power
- Inefficient use of data

**2. MAR or MNAR**:
- Introduces bias
- Misleading conclusions
- Better to impute

**3. Small Datasets**:
- Can't afford to lose observations
- Every data point valuable

**4. Systematic Missingness**:
- Certain groups more likely missing
- Biased sample remains

**5. Multiple Features with Missingness**:
- Compound effect: $\prod(1 - r_j)$ very small
- Excessive data loss

## Advantages and Disadvantages

### Advantages

**1. Simplicity**:
- One line of code: `df.dropna()`
- No parameter tuning
- Easy to explain

**2. No Imputation Bias**:
- Doesn't introduce artificial values
- Only uses real observed data

**3. Unbiased Under MCAR**:
- Valid statistical inference
- Correct parameter estimates (but higher variance)

**4. Works with Any Algorithm**:
- No special handling needed
- Standard ML pipeline

**5. Computationally Fast**:
- No imputation calculations
- Instant preprocessing

### Disadvantages

**1. Data Loss**:
- Wastes partial information
- Discards complete features in incomplete rows

**Example**:
```
Age  Income  Education  City
25   50000   Bachelor   NYC    ← Complete
30   NaN     Master     LA     ← Lost! (Education, City discarded)
```

**2. Reduced Sample Size**:
- Lower statistical power: $\text{Power} \propto \sqrt{n}$
- Wider confidence intervals
- Harder to detect effects

**3. Bias Under MAR/MNAR**:
- Systematic exclusion of subgroups
- Unrepresentative sample
- Wrong conclusions

**4. Inefficient**:
- Modern imputation methods often better
- Throws away information

**5. Compounds with Multiple Features**:
- Each feature's missingness multiplies
- Can lose majority of data

## Assessing Missingness Mechanism

### Visual Inspection

**Missingness heatmap**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize missing pattern
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missingness Pattern')
plt.show()
```

**Missingness correlation**:
```python
# Check if missingness in one variable relates to values in another
df['Income_missing'] = df['Income'].isnull()
df.groupby('Income_missing')['Age'].mean()
# If means differ significantly → MAR (missing income depends on age)
```

### Statistical Tests

**Little's MCAR Test**:
- Null hypothesis: Data is MCAR
- Tests if missingness pattern independent of observed values
- Available in R package: `naniar`

**t-tests comparing groups**:
```python
from scipy import stats

# Compare observed values between missing/non-missing groups
missing_group = df[df['Income'].isnull()]['Age']
non_missing_group = df[df['Income'].notnull()]['Age']

t_stat, p_value = stats.ttest_ind(missing_group, non_missing_group)
# If p < 0.05 → not MCAR (missingness depends on Age)
```

### Domain Knowledge

**Consider**:
- Why might data be missing?
- Survey design (optional questions)
- Measurement process (sensor failures)
- Data collection (dropout patterns)

**Example**: Medical data
- Sicker patients miss appointments → MNAR
- Random equipment failures → MCAR
- Young patients skip certain tests → MAR

## Best Practices

### 1. Analyze Missingness Before Deleting

```python
# Check missingness percentage
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct.sort_values(ascending=False))

# Identify rows to be deleted
rows_to_delete = df[df.isnull().any(axis=1)]
print(f"Deleting {len(rows_to_delete)} rows ({len(rows_to_delete)/len(df)*100:.1f}%)")
```

### 2. Document Data Loss

```python
original_n = len(df)
df_complete = df.dropna()
final_n = len(df_complete)

print(f"Original samples: {original_n}")
print(f"Complete cases: {final_n}")
print(f"Lost: {original_n - final_n} ({(original_n - final_n)/original_n*100:.1f}%)")
```

### 3. Compare Distributions

```python
# Check if complete cases differ from incomplete
for col in df.select_dtypes(include='number').columns:
    complete = df.dropna()[col]
    all_data = df[col].dropna()
    
    print(f"{col}: Complete mean={complete.mean():.2f}, All mean={all_data.mean():.2f}")
```

### 4. Use as Baseline

```python
# Try CCA first, then compare with imputation
from sklearn.metrics import accuracy_score

# CCA baseline
X_complete = df.dropna()
model_cca = LogisticRegression().fit(X_complete, y_complete)
score_cca = model_cca.score(X_test_complete, y_test_complete)

# Imputation alternative
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
model_imp = LogisticRegression().fit(X_imputed, y)
score_imp = model_imp.score(X_test_imputed, y_test)

print(f"CCA score: {score_cca:.3f}")
print(f"Imputation score: {score_imp:.3f}")
```

### 5. Delete on Training Data Only (Usually)

**For imputation comparison**:
```python
# Fit imputer on train, transform test
X_train, X_test, y_train, y_test = train_test_split(X, y)

imputer = SimpleImputer().fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

**For CCA**: Delete from both (since using only observed)
```python
X_train_complete = X_train.dropna()
X_test_complete = X_test.dropna()
```

### 6. Consider Selective Deletion

**Delete only rows missing critical features**:
```python
# Keep rows with missing non-critical features
critical_features = ['Age', 'Income']
df_selective = df.dropna(subset=critical_features)

# Then impute remaining features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
df_selective_imputed = pd.DataFrame(
    imputer.fit_transform(df_selective),
    columns=df_selective.columns
)
```

## Alternatives to Complete Case Analysis

### 1. Imputation Methods

**Mean/Median/Mode**:
- Replace with central tendency
- Simple, fast

**KNN Imputation**:
- Use similar observations
- Preserves relationships

**Iterative Imputation**:
- Model-based (MICE)
- Most sophisticated

### 2. Missing Indicator

**Add binary flag**:
```python
df['Income_was_missing'] = df['Income'].isnull().astype(int)
df['Income'].fillna(df['Income'].median(), inplace=True)
```

**Benefit**: Model can learn if missingness is predictive

### 3. Missingness as Category

**For categorical features**:
```python
df['Education'].fillna('Missing', inplace=True)
```

**Treat missing as separate category**

### 4. Model-Based Methods

**Algorithms that handle missing**:
- XGBoost: Native missing value handling
- LightGBM: Automatic optimal split for missing
- CatBoost: Built-in missing support

## Summary

Complete Case Analysis (listwise deletion) is the simplest missing data approach: delete rows with any missing values. While easy to implement, it has important statistical implications.

**Key Concepts**:

**Definition**: Remove observations with any missing values
$$n_{complete} = \{i : X_i \text{ has no missing values}\}$$

**Data Loss**:
$$n_{complete} = n \times \prod_{j=1}^{p} (1 - r_j)$$
Where $r_j$ = missingness rate in feature $j$

**Missing Mechanisms**:
1. **MCAR**: Missing completely at random → CCA unbiased
2. **MAR**: Missing at random (conditional) → CCA biased
3. **MNAR**: Missing not at random → CCA biased

**When to Use**:
- Small missingness (<5%)
- MCAR confirmed
- Large sample size
- Quick baseline
- Missing in target variable

**When NOT to Use**:
- High missingness (>10-20%)
- MAR or MNAR
- Small datasets
- Multiple features with missing

**Implementation**:
```python
df.dropna()  # Remove any row with missing
df.dropna(subset=['col1', 'col2'])  # Specific columns
```

**Advantages**:
- Simple, fast
- No imputation bias
- Unbiased under MCAR
- Works with any algorithm

**Disadvantages**:
- Data loss (information waste)
- Reduced power
- Biased under MAR/MNAR
- Inefficient

**Best Practices**:
- Analyze missingness first
- Document data loss
- Compare distributions
- Use as baseline
- Test MCAR assumption
- Consider alternatives

**Alternatives**: Imputation (mean, KNN, iterative), missing indicators, algorithms with native missing support

Complete Case Analysis remains a useful baseline approach, but modern imputation methods often provide better results by preserving sample size and reducing bias.

---
