# Scikit-Learn Pipelines - Streamlined ML Workflows

## General Idea

A Pipeline in scikit-learn is a sequential chain of data transformations and a final estimator, bundled into a single object. It ensures that all preprocessing steps are applied consistently during training, validation, and prediction, while preventing data leakage and simplifying code structure. Think of it as an assembly line where data passes through stages sequentially, each stage performing a specific transformation.

## Why Use Pipelines?

1. **Prevent Data Leakage**: Guarantees fit on train, transform on test
2. **Code Simplification**: Replace 10+ lines with single pipeline object
3. **Reproducibility**: Entire workflow captured in one object
4. **Cross-Validation**: Automatic proper handling of train/val splits
5. **Hyperparameter Tuning**: GridSearch across all pipeline steps
6. **Deployment**: Save/load complete workflow as single file
7. **Clarity**: Explicit, readable ML workflow
8. **Testing**: Easier to unit test complete pipelines

## Role in Machine Learning

### The Data Leakage Problem

**Without Pipeline** (INCORRECT):
```
1. scaler.fit(X_entire_dataset)  # Leakage!
2. X_scaled = scaler.transform(X_entire_dataset)
3. Split into train/test
4. model.fit(X_train_scaled)
5. evaluate(X_test_scaled)
```

**Problem**: Scaler learned statistics from test data!
- Mean/std computed from entire dataset
- Test data influenced preprocessing
- Overly optimistic performance estimates

**With Pipeline** (CORRECT):
```
1. Split into train/test
2. pipeline.fit(X_train, y_train)  # Fit scaler AND model on train only
3. pipeline.predict(X_test)        # Transform test, then predict
```

**Solution**: Test data never seen during fit

### Mathematical Guarantee

**Pipeline ensures**:
$$\mu_{scaler} = \mu_{train}, \quad \sigma_{scaler} = \sigma_{train}$$

**Test transformation**:
$$X_{test}' = \frac{X_{test} - \mu_{train}}{\sigma_{train}}$$

Not contaminated by $\mu_{test}$ or $\sigma_{test}$

## Pipeline Structure

### Basic Syntax

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('step1_name', transformer1),
    ('step2_name', transformer2),
    ...
    ('final_estimator_name', model)
])
```

**Components**:
- **Steps**: List of (name, transformer/estimator) tuples
- **Transformers**: All steps except last (must have fit/transform)
- **Estimator**: Final step (must have fit/predict or fit/transform)

**Requirements**:
- All but last step must implement `fit` and `transform`
- Last step must implement `fit` (and `predict` or `transform`)

### make_pipeline Shortcut

**Simpler syntax** (auto-generates names):
```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)
```

**Generated names**: `standardscaler`, `pca`, `logisticregression`

**When to use**:
- Quick prototyping
- Don't need custom names
- Simple pipelines

**When NOT to use**:
- Need specific names for GridSearchCV
- Multiple instances of same transformer

## Pipeline Methods

### fit(X, y)

**Purpose**: Fit all transformers and final estimator sequentially

**Process**:
```
Step 1: transformer1.fit(X, y) then X = transformer1.transform(X)
Step 2: transformer2.fit(X, y) then X = transformer2.transform(X)
...
Final: estimator.fit(X, y)
```

**Mathematical representation**:
$$f_{pipeline}(X) = f_{estimator}(T_n(...T_2(T_1(X))))$$

Where $T_i$ are transformers, fitted sequentially

### predict(X) / predict_proba(X)

**Purpose**: Transform data through pipeline, predict with final estimator

**Process**:
```
Step 1: X = transformer1.transform(X)  # Use fitted transformer
Step 2: X = transformer2.transform(X)  # Use fitted transformer
...
Final: return estimator.predict(X)
```

**Key**: Only `transform`, never `fit` (parameters already learned)

### fit_predict(X, y)

**Purpose**: Fit on X, immediately predict on X

**Equivalent to**:
```python
pipeline.fit(X, y)
pipeline.predict(X)
```

**Use**: Training set predictions, some clustering algorithms

### transform(X)

**Purpose**: Apply all transformers, skip final estimator prediction

**Use when**: 
- Final step is a transformer (e.g., PCA)
- Want intermediate transformed data
- Feature engineering pipeline without model

### score(X, y)

**Purpose**: Transform X, evaluate final estimator performance

**Returns**: 
- Accuracy for classifiers
- R² for regressors
- Custom metric if defined

**Equivalent to**:
```python
X_transformed = pipeline[:-1].transform(X)
pipeline[-1].score(X_transformed, y)
```

## Accessing Pipeline Steps

### By Index

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('model', LogisticRegression())
])

pipeline[0]      # StandardScaler
pipeline[1]      # PCA
pipeline[-1]     # LogisticRegression (final estimator)
pipeline[:-1]    # All steps except final (returns Pipeline)
```

### By Name

```python
pipeline.named_steps['scaler']  # StandardScaler
pipeline.named_steps['pca']     # PCA
pipeline.named_steps['model']   # LogisticRegression
```

### Getting Parameters

```python
# After fitting
scaler_mean = pipeline.named_steps['scaler'].mean_
pca_components = pipeline.named_steps['pca'].components_
model_coef = pipeline.named_steps['model'].coef_
```

## Hyperparameter Tuning with Pipelines

### Parameter Naming Convention

**Format**: `step_name__parameter_name`

**Example**:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())
])

# Access PCA n_components:
'pca__n_components'

# Access SVM kernel:
'svm__kernel'

# Access SVM C:
'svm__C'
```

### GridSearchCV with Pipeline

**Full example**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'pca__n_components': [5, 10, 20],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
```

**Searches**: $3 \times 3 \times 2 = 18$ parameter combinations

**For each combination**:
- 5-fold cross-validation
- Proper fit/transform in each fold
- Total: $18 \times 5 = 90$ pipeline fits

### Cross-Validation Behavior

**With Pipeline**:
```
Fold 1:
  Train: pipeline.fit(X_fold1_train)     # Fit scaler on fold train
  Val:   pipeline.predict(X_fold1_val)   # Transform val with fold's scaler
  
Fold 2:
  Train: pipeline.fit(X_fold2_train)     # NEW scaler fit on fold 2 train
  Val:   pipeline.predict(X_fold2_val)   # Transform with fold 2's scaler
...
```

**Result**: Each fold has independent preprocessing (no leakage)

**Without Pipeline**: Must manually fit/transform each fold (error-prone)

## Caching Pipeline Steps

### Memory Parameter

**Purpose**: Cache transformer outputs to avoid recomputation

**Use when**:
- Expensive transformations (e.g., text vectorization)
- Hyperparameter tuning (reuse transformations)
- Same data, different model parameters

**Syntax**:
```python
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree

cachedir = mkdtemp()
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
], memory=cachedir)
```

**Behavior**:
- First fit: Compute and cache transformer outputs
- Subsequent fits (same data): Load from cache
- If transformer params change: Recompute

**Speedup Example**:
```
Without caching:
  10 GridSearch iterations × 5 CV folds × 60s vectorization = 3000s
  
With caching:
  1 × 60s (first vectorization) + 49 × 0s (cached) = 60s
```

**Cleanup**:
```python
rmtree(cachedir)  # Delete cache after use
```

## Practical Examples

### Example 1: Basic Classification Pipeline

**Task**: Classify iris species

**Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

**Process**:
1. Fit scaler on X_train (learn $\mu_{train}$, $\sigma_{train}$)
2. Transform X_train 
3. Fit SVM on scaled X_train
4. For prediction: scale X_test, then predict

### Example 2: Dimensionality Reduction Pipeline

**Task**: Reduce dimensions, then classify

**Pipeline**:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('classifier', RandomForestClassifier())
])
```

**Flow**:
$$X \xrightarrow{impute} X_1 \xrightarrow{scale} X_2 \xrightarrow{PCA} X_3 \xrightarrow{classify} \hat{y}$$

### Example 3: Text Classification Pipeline

**Task**: Classify documents

**Pipeline**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

pipeline.fit(documents_train, labels_train)
```

**Advantages**:
- Vectorizer vocabulary learned from train only
- Test documents use train vocabulary (no leakage)
- Single object for deployment

### Example 4: Complex Preprocessing Pipeline

**Task**: Handle mixed data types

**Pipeline**:
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

**Benefits**: 
- Appropriate transformation per feature type
- All in single pipeline
- Clean cross-validation

## Pipeline vs Manual Workflow

### Manual Workflow (Problematic)

```python
# Fit transformers
imputer = SimpleImputer().fit(X_train)
scaler = StandardScaler().fit(imputer.transform(X_train))

# Transform train
X_train_imputed = imputer.transform(X_train)
X_train_scaled = scaler.transform(X_train_imputed)

# Fit model
model = SVC().fit(X_train_scaled, y_train)

# Transform test (must remember order!)
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Predict
y_pred = model.predict(X_test_scaled)
```

**Issues**:
- Verbose (15+ lines)
- Error-prone (forget step, wrong order)
- Hard to maintain
- Difficult cross-validation
- Must save multiple objects for deployment

### Pipeline Workflow (Clean)

```python
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', SVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Advantages**:
- Concise (3 lines)
- Correct by construction
- Easy to maintain
- Cross-validation friendly
- Single object to save

## Common Patterns

### Pattern 1: Impute → Scale → Model

**Use**: Missing data + scaling

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('model', Ridge())
])
```

**Order matters**: Must impute before scaling

### Pattern 2: Feature Selection → Model

**Use**: Reduce features, then train

```python
Pipeline([
    ('selector', SelectKBest(f_classif, k=20)),
    ('classifier', LogisticRegression())
])
```

**Benefit**: Feature selection sees training data only

### Pattern 3: Multiple Transformations → Model

**Use**: Complex preprocessing

```python
Pipeline([
    ('outlier_removal', IsolationForest()),
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=30)),
    ('classifier', SVC())
])
```

**Key**: Each step sees output of previous step

### Pattern 4: Separate Preprocessing → Multiple Models

**Use**: Try different models with same preprocessing

```python
preprocessing = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

# Reuse preprocessing
pipeline1 = Pipeline([('prep', preprocessing), ('model', SVC())])
pipeline2 = Pipeline([('prep', preprocessing), ('model', RandomForest())])
```

## Debugging Pipelines

### Inspecting Intermediate Steps

**Get transformed data**:
```python
# After fitting pipeline
X_after_step2 = pipeline[:-1].transform(X)  # All but last step
X_after_step1 = pipeline[:1].transform(X)   # Only first step
```

**Check shapes**:
```python
for i, (name, transformer) in enumerate(pipeline.steps[:-1]):
    X_transformed = pipeline[:i+1].transform(X)
    print(f"{name}: {X_transformed.shape}")
```

### Common Issues

**1. Shape Mismatch**:
- Check each step's output shape
- Verify transformer expects correct input

**2. Wrong Order**:
- Imputation before scaling (correct)
- Scaling before imputation (incorrect - scales NaN!)

**3. Transformer/Estimator Confusion**:
- All but last must have `transform()`
- Last must have `predict()` or `transform()`

**4. Cross-Validation Issues**:
- Use pipeline with cross_val_score
- Don't manually preprocess before CV

### Verbose Output

**See what's happening**:
```python
pipeline = Pipeline([
    ('step1', Transformer1()),
    ('step2', Transformer2())
], verbose=True)

pipeline.fit(X, y)
```

**Output**:
```
[Pipeline] .... (step 1 of 2) Processing step1, total=   0.1s
[Pipeline] .... (step 2 of 2) Processing step2, total=   0.3s
```

## Saving and Loading Pipelines

### Using joblib

**Save**:
```python
from joblib import dump

pipeline.fit(X_train, y_train)
dump(pipeline, 'model_pipeline.joblib')
```

**Load**:
```python
from joblib import load

pipeline = load('model_pipeline.joblib')
predictions = pipeline.predict(X_new)
```

**What's saved**: 
- All transformer parameters (means, stds, vocabularies, etc.)
- Model weights
- Pipeline structure

**Benefits**:
- Exact reproduction of predictions
- Easy deployment
- Version control friendly

### Using pickle

**Save**:
```python
import pickle

with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
```

**Load**:
```python
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
```

**Note**: joblib more efficient for large numpy arrays

## Advanced Topics

### FeatureUnion in Pipeline

**Purpose**: Apply multiple transformers, concatenate results

```python
from sklearn.pipeline import FeatureUnion

feature_engineering = FeatureUnion([
    ('pca', PCA(n_components=10)),
    ('poly', PolynomialFeatures(degree=2))
])

pipeline = Pipeline([
    ('features', feature_engineering),
    ('classifier', LogisticRegression())
])
```

**Result**: Input gets both PCA and polynomial features

### Custom Transformers in Pipeline

**Create custom transformer**:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

pipeline = Pipeline([
    ('log', LogTransformer()),
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
```

**Requirements**: 
- Inherit from `BaseEstimator` and `TransformerMixin`
- Implement `fit()` and `transform()`
- `fit()` must return `self`

### Conditional Steps

**Skip steps based on parameter**:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Can be set to 'passthrough'
    ('model', SVC())
])

# Skip PCA:
pipeline.set_params(pca='passthrough')
```

**Use in GridSearch**:
```python
param_grid = {
    'pca': [PCA(5), PCA(10), 'passthrough'],
    'model__C': [0.1, 1, 10]
}
```

Searches with and without PCA

## Best Practices

### 1. Always Use Pipelines for Production

**Why**: 
- Prevents leakage
- Ensures consistency
- Simplifies deployment

### 2. Meaningful Step Names

**Good**:
```python
Pipeline([('imputer', ...), ('scaler', ...), ('model', ...)])
```

**Bad**:
```python
Pipeline([('step1', ...), ('step2', ...), ('step3', ...)])
```

### 3. Order Matters

**Correct**:
```python
Pipeline([('impute', ...), ('scale', ...), ('model', ...)])
```

**Incorrect**:
```python
Pipeline([('scale', ...), ('impute', ...), ('model', ...)])
# Scaling NaN values is meaningless!
```

### 4. Use Caching for Expensive Steps

**When**:
- Text vectorization
- Image feature extraction
- Large dataset transformations

### 5. Validate at Each Step

**Check**:
- No NaN after imputation
- Correct scale after normalization
- Expected dimensionality after PCA

### 6. Pipeline Everything

**Include**:
- Imputation
- Scaling
- Feature engineering
- Feature selection
- Model

**Don't**: Preprocess outside pipeline, then use pipeline

## Summary

Scikit-learn Pipelines are essential for building robust, reproducible machine learning workflows. They automate the sequential application of transformations and model fitting while preventing data leakage.

**Key Concepts**:

**Definition**:
- Sequential chain: transformers → estimator
- Single unified object
- Consistent train/test processing

**Core Methods**:
- `fit(X, y)`: Fit all steps sequentially
- `predict(X)`: Transform through steps, predict with final
- `score(X, y)`: Transform and evaluate

**Data Leakage Prevention**:
- Fit on train only: $\theta_{transform} = f(X_{train})$
- Transform test: $X_{test}' = T(X_{test}, \theta_{transform})$
- No test contamination

**Hyperparameter Tuning**:
- Syntax: `step_name__parameter`
- Works with GridSearchCV/RandomizedSearchCV
- Proper CV behavior (refit each fold)

**Best Practices**:
- Use for all production models
- Meaningful step names
- Correct transformation order
- Cache expensive steps
- Save complete pipeline

**Advantages**:
- Prevents leakage (automatic proper CV)
- Simplifies code (single object)
- Eases deployment (save one file)
- Enables end-to-end tuning
- Improves reproducibility

**Common Patterns**:
- Impute → Scale → Model
- Feature Selection → Model
- ColumnTransformer → Model
- FeatureUnion → Model

**Integration**:
- GridSearchCV: Tune all parameters
- Cross-validation: Proper fold handling
- ColumnTransformer: Heterogeneous data
- Custom transformers: Domain logic

Mastering pipelines is crucial for professional machine learning development, ensuring your models are reliable, maintainable, and ready for production deployment.

---

**Video Link**: https://youtu.be/xOccYkgRV4Q
