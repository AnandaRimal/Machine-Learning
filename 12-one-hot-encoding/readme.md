# One-Hot Encoding

## General Idea

One-hot encoding is a technique for representing categorical variables as binary vectors where each category becomes a separate binary (0/1) feature. For a variable with k unique categories, one-hot encoding creates k binary columns, with exactly one column having value 1 ("hot") and others having value 0 ("cold") for each observation. This transforms categorical data into a numerical format suitable for machine learning algorithms without imposing artificial ordering.

## Why Use One-Hot Encoding?

1. **No Ordinal Assumption**: Doesn't impose order on nominal categories
2. **Algorithm Compatibility**: Makes categorical data usable in algorithms requiring numerical input
3. **Linear Model Friendly**: Each category gets independent coefficient
4. **Interpretability**: Clear relationship between original categories and features
5. **Standard Practice**: Widely used and well-understood method
6. **Preserves Information**: No information loss from original categories
7. **Mathematical Correctness**: Proper representation for nominal data

## Role in Machine Learning

### Essential For

- **Nominal Categorical Variables**: Color, country, product type (no natural order)
- **Linear Models**: Linear/Logistic Regression, SVM with linear kernel
  - Each category gets separate coefficient
  - No assumption of equal spacing
  
- **Neural Networks**: Can learn separate weights per category
- **Distance-Based Models**: KNN, K-Means (with proper distance metric)

### Alternative For

- **Tree-Based Models**: Can handle categorical directly or use ordinal encoding
  - One-hot works but may not be optimal
  - Creates many binary splits
  
- **Ordinal Variables**: Use ordinal encoding instead to preserve order

### Problematic For

- **High-Cardinality Variables**: 100+ categories → 100+ features
  - Curse of dimensionality
  - Memory issues
  - Sparse matrices
  - Use alternatives: target encoding, embedding

## Mathematical Representation

### Encoding Function

For categorical variable $X$ with categories $\{c_1, c_2, ..., c_k\}$:

**One-hot vector** for category $c_i$:
$$\mathbf{x}_i = [0, 0, ..., 1, ..., 0] \in \{0,1\}^k$$

Where 1 appears at position $i$, rest are 0.

**Formal definition**:
$$\text{OneHot}(c_i) = \mathbf{e}_i$$

Where $\mathbf{e}_i$ is the $i$-th standard basis vector:
$$\mathbf{e}_i[j] = \begin{cases}
1 & \text{if } j = i \\
0 & \text{if } j \neq i
\end{cases}$$

### Properties

**Orthogonality**: Different categories have orthogonal vectors
$$\mathbf{e}_i \cdot \mathbf{e}_j = \begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}$$

**Euclidean Distance**: Distance between any two categories is constant
$$||\mathbf{e}_i - \mathbf{e}_j||_2 = \sqrt{2} \quad \forall i \neq j$$

**Sum Constraint**: Exactly one 1 per observation
$$\sum_{j=1}^{k} \mathbf{e}_i[j] = 1$$

### Example

**Color**: [Red, Blue, Green]

**One-Hot Encoding**:
```
Red   → [1, 0, 0]
Blue  → [0, 1, 0]
Green → [0, 0, 1]
```

**Data transformation**:
```
Original: [Red, Blue, Red, Green]

Encoded:
  Color_Red  Color_Blue  Color_Green
     1          0           0
     0          1           0
     1          0           0
     0          0           1
```

## Dummy Variable Encoding (k-1 Encoding)

### The Dummy Variable Trap

**Problem**: With k categories, using all k one-hot columns creates **perfect multicollinearity**

**Mathematical Explanation**:

For any observation:
$$\text{Color\_Red} + \text{Color\_Blue} + \text{Color\_Green} = 1$$

This creates linear dependence:
$$\mathbf{x}_k = 1 - \sum_{i=1}^{k-1}\mathbf{x}_i$$

**Impact on Linear Models**:
- Design matrix $X$ is not full rank
- $X^TX$ is singular (non-invertible)
- Cannot compute unique coefficients: $(X^TX)^{-1}X^Ty$ undefined
- Ordinary Least Squares fails

### Solution: Drop One Category

**Dummy Encoding**: Use k-1 binary features

**Example**: Color with 3 categories
```
Original encoding:     Dummy encoding:
Color_Red  Color_Blue  Color_Green    Color_Blue  Color_Green
   1          0           0       →      0           0        (Red = reference)
   0          1           0       →      1           0        (Blue)
   0          0           1       →      0           1        (Green)
```

**Interpretation**:
- Dropped category (Red) is **reference/baseline**
- Other coefficients are relative to reference
- Blue coefficient: Effect of Blue vs Red
- Green coefficient: Effect of Green vs Red

**Formula for Linear Regression**:
$$y = \beta_0 + \beta_1 \cdot \text{Color\_Blue} + \beta_2 \cdot \text{Color\_Green}$$

- $\beta_0$: Mean for Red (reference)
- $\beta_1$: Difference between Blue and Red
- $\beta_2$: Difference between Green and Red

### When to Use Full vs Dummy Encoding

**Use Full One-Hot (k categories)**:
- Tree-based models (no multicollinearity issue)
- Neural networks (can handle redundancy)
- Some regularized models (L1/L2 can handle)
- Non-linear models

**Use Dummy Encoding (k-1 categories)**:
- Linear/Logistic Regression (avoid singular matrix)
- Models requiring matrix inversion
- When interpretability relative to baseline desired

## Impact on ML Algorithms

### Linear Models

**Linear Regression**:
$$y = \beta_0 + \sum_{j=1}^{k-1}\beta_j x_j$$

Each category $j$ gets coefficient $\beta_j$:
- Represents difference from baseline category
- Independent effect of each category
- No assumption about order or spacing

**Advantages**:
- Proper treatment of nominal data
- Interpretable coefficients
- No artificial constraints

**Logistic Regression**: Similar interpretation for log-odds

### Tree-Based Models

**Decision Trees**:

**Splits**: Binary splits on each one-hot feature
- Is Color_Red = 1? → Yes/No branches

**Multiple Categories**:
- Need multiple splits to separate k categories
- Example: 4 categories require 3 binary splits

**Inefficiency**:
- More splits needed vs. using categorical directly
- Deeper trees
- Slower training

**Alternative**: Many implementations handle categorical natively
- XGBoost, LightGBM, CatBoost
- More efficient than one-hot

**Random Forest, Gradient Boosting**: Same considerations as decision trees

### Neural Networks

**Input Layer**: One-hot features as inputs

**Advantages**:
- Network learns separate weights per category
- Can learn complex interactions
- Standard approach

**Alternative**: Embedding layers
- More compact representation
- Learns dense vectors for categories
- Better for high-cardinality features

**Embedding**: Maps category to dense vector
$$\text{Category}_i \rightarrow \mathbf{v}_i \in \mathbb{R}^d$$

Where $d \ll k$ (dimensionality reduction)

### Distance-Based Models

**K-NN, K-Means**:

**Euclidean Distance** between one-hot vectors:
$$d(\mathbf{e}_i, \mathbf{e}_j) = \begin{cases}
0 & \text{if } i = j \\
\sqrt{2} & \text{if } i \neq j
\end{cases}$$

**Implication**: All different categories equally distant
- May not reflect true similarity
- Example: "Red" equally distant from "Blue" and "Green"

**Better Alternatives**:
- Domain-specific distance metrics
- Learned embeddings (capture similarity)
- Mixed distance (Gower's distance for mixed types)

## High-Cardinality Problem

### Definition

**High Cardinality**: Categorical variable with many unique values (>50, sometimes >10)

**Examples**:
- ZIP code: 40,000+ categories in US
- Product ID: Thousands to millions
- User ID: Millions
- City: Thousands
- Email domain: Thousands

### Problems

**1. Dimensionality Explosion**:
- 1000 categories → 1000 features
- Memory: $n \times k$ matrix (mostly sparse)
- Computation: Model training scales with feature count

**2. Sparsity**:
- Most one-hot features are 0
- Sparse matrix storage required
- Many zeros can cause numerical issues

**3. Overfitting**:
- Rare categories appear few times
- Model may overfit to noise
- Poor generalization

**4. Curse of Dimensionality**:
- Distance metrics become less meaningful
- More data needed to learn
- Model complexity increases

### Solutions

**1. Frequency-Based Grouping**:
- Keep top N frequent categories
- Group rare categories into "Other"

**2. Target Encoding**:
- Replace category with target mean
- Single numeric feature instead of k features
- Risk: target leakage, requires regularization

**3. Feature Hashing (Hash Encoding)**:
- Hash categories to fixed number of bins
- Reduces k categories to m features (m << k)
- Trade-off: hash collisions

**Hash Function**:
$$h(c_i) = \text{hash}(c_i) \mod m$$

**4. Embeddings**:
- Learn dense representation
- k categories → d-dimensional vectors (d << k)
- Captures semantic similarity

**5. Binary Encoding**:
- Encode categories as binary numbers
- k categories → $\lceil \log_2(k) \rceil$ features
- Example: 8 categories → 3 binary features

**6. Hierarchical Encoding**:
- Use hierarchy if available
- City → State → Country
- Multiple lower-dimensional encodings

## Sparse Matrix Representation

### Why Sparse?

One-hot encoded data is mostly zeros:
- For k categories, each row has k-1 zeros and 1 one
- Sparsity: $\frac{k-1}{k}$ (approaches 100% as k increases)

### Storage

**Dense Matrix**: Store all values
- Memory: $n \times k \times \text{sizeof}(\text{type})$
- Example: 1M rows, 1000 categories, float32 → 4GB

**Sparse Matrix** (CSR - Compressed Sparse Row):
- Store only non-zero values
- Memory: Roughly $n \times \text{avg\_nonzeros\_per\_row}$
- Example: Same data → ~4MB (1000x reduction)

**Formats**:
- **COO** (Coordinate): List of (row, col, value) triplets
- **CSR** (Compressed Sparse Row): Efficient for row operations
- **CSC** (Compressed Sparse Column): Efficient for column operations

### Computational Efficiency

**Sparse-aware algorithms**:
- Skip zero entries
- Time complexity: $O(\text{nnz})$ where nnz = number of non-zeros
- Much faster than dense $O(n \times k)$

**Libraries with sparse support**:
- scikit-learn (most estimators)
- XGBoost, LightGBM
- TensorFlow, PyTorch (with special handling)

## Handling Unseen Categories

### Problem

Test/production data contains category not in training data

### Solutions

**1. Ignore (Set all to 0)**:
```
Unknown category → [0, 0, 0, ..., 0]
```
- Simple
- May confuse model (no category matches)

**2. Add "Unknown" Category**:
- Create explicit "Unknown" column
- Unknown → [0, 0, ..., 1] (last column)
- Consistent representation

**3. Treat as Most Frequent**:
- Map to mode from training
- Conservative

**4. Raise Error**:
- Alert for data quality issue
- Requires manual intervention

**5. Ignore Feature** (if one-hot multi-feature):
- Use other features
- Graceful degradation

**Best Practice**: Add "Unknown" during training
- Include in initial encoding
- Consistent handling

## Inverse Transform

Convert one-hot vector back to category:

$$f^{-1}(\mathbf{x}) = c_i \quad \text{where } \mathbf{x}[i] = 1$$

**Algorithm**:
1. Find index where value = 1
2. Map index to category name

**Example**:
```
One-hot: [0, 1, 0]
Index with 1: 1
Category: Blue
```

**Use Cases**:
- Interpret predictions
- Display results
- Validate encoding

## Advantages and Disadvantages

### Advantages

1. **No Ordinal Assumption**: Treats categories equally
2. **Interpretable**: Clear mapping to original categories
3. **Widely Supported**: Most libraries handle seamlessly
4. **Mathematically Sound**: Proper representation for nominal data
5. **Independent Coefficients**: Each category gets separate parameter
6. **No Information Loss**: Reversible transformation

### Disadvantages

1. **Dimensionality**: k categories → k features (or k-1)
2. **Sparsity**: Mostly zeros (memory overhead if dense)
3. **High Cardinality**: Impractical for many categories
4. **Equal Distance**: All categories equidistant (may not reflect reality)
5. **Dummy Trap**: Need to drop one for linear models
6. **Computational Cost**: More features → slower training

## Best Practices

### 1. Check Cardinality

**Before encoding**:
- Count unique values
- If > 50-100, consider alternatives

### 2. Handle Missing Values First

**Options**:
- Treat as separate category
- Impute before encoding
- Drop rows (if few)

### 3. Use Sparse Matrices

**For high-dimensional data**:
- Significantly reduces memory
- Faster computation with sparse-aware algorithms

### 4. Drop One Category (Linear Models)

**Avoid multicollinearity**:
- Use `drop='first'` or `drop='if_binary'`
- Document which category is reference

### 5. Save Encoder

**For consistency**:
- Fit on training data
- Transform train/test with same encoder
- Save for production use

### 6. Monitor Feature Count

**Track dimensionality**:
- Original features vs. after encoding
- Ensure manageable for model
- Consider dimensionality reduction if needed

### 7. Validate Encoding

**Post-encoding checks**:
- Verify all categories mapped
- Check for unexpected columns
- Ensure exactly one 1 per row (for full one-hot)

## Comparison with Alternatives

| Method | Features Created | Use Case | Pros | Cons |
|--------|------------------|----------|------|------|
| **One-Hot** | k (or k-1) | Nominal, low cardinality | No assumptions, interpretable | High dimensionality |
| **Ordinal** | 1 | Ordinal data | Compact, preserves order | Only for ordered categories |
| **Target Encoding** | 1 | High cardinality | Data-driven, compact | Leakage risk, overfitting |
| **Embedding** | d (learned) | High cardinality, NNs | Learns similarity, compact | Requires training, black-box |
| **Hashing** | m (fixed) | Very high cardinality | Fixed size, fast | Hash collisions |
| **Binary** | ⌈log₂(k)⌉ | Moderate cardinality | Fewer features than one-hot | Less interpretable |

## Summary

One-hot encoding transforms categorical variables into binary vectors, creating k separate features for k categories. It's the standard method for nominal categorical data in machine learning, providing a mathematically sound representation without imposing artificial order.

**Key Points**:

**When to Use**:
- Nominal categorical variables (no order)
- Linear models (with k-1 encoding)
- Low to moderate cardinality (< 50 categories)
- When interpretability important

**Mathematical Foundation**:
- Standard basis vectors: $\mathbf{e}_i \in \{0,1\}^k$
- Exactly one 1, rest 0s
- Orthogonal representation
- Equal distance between categories: $\sqrt{2}$

**Dummy Variable Trap**:
- k categories create linear dependence
- Drop one category for linear models
- Reference category interpretation

**High Cardinality Challenge**:
- Many categories → many features
- Use alternatives: target encoding, embedding, hashing
- Sparse matrix storage essential

**Best Practices**:
- Check cardinality before encoding
- Use sparse matrices for efficiency
- Drop one category for linear models
- Save encoder for consistency
- Handle unknown categories explicitly
- Consider alternatives for >50 categories

**Advantages**:
- No ordinal assumptions
- Interpretable
- Standard and well-supported
- Works with linear models

**Limitations**:
- Dimensionality explosion
- Sparsity
- Equal distance assumption
- Dummy variable trap

One-hot encoding is fundamental to categorical data handling in machine learning. Understanding when to use it versus alternatives, managing the dummy variable trap, and handling high-cardinality scenarios are essential skills for effective feature engineering.

---
