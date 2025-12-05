# Ordinal Encoding

## General Idea

Ordinal encoding is a technique for converting categorical variables with an inherent order or ranking into numerical values. Unlike one-hot encoding which creates multiple binary columns, ordinal encoding assigns a single integer to each category based on its position in the defined order. This preserves the ordinal relationship while making the data compatible with machine learning algorithms that require numerical inputs.

## Why Use Ordinal Encoding?

1. **Preserves Order**: Maintains natural ranking relationships (e.g., Low < Medium < High)
2. **Dimensionality**: Creates only one feature (vs. k features for one-hot encoding)
3. **Memory Efficient**: Single integer column vs. multiple binary columns
4. **Tree-Based Models**: Works well with decision trees that can learn thresholds
5. **Interpretability**: Numeric values reflect actual ordering
6. **No Dummy Variable Trap**: Doesn't create linear dependencies
7. **Sparse Data**: Avoids creating sparse matrices

## Role in Machine Learning

### Essential For

- **Ordinal Categorical Variables**: Education level, size categories, ratings, rankings
- **Tree-Based Algorithms**: Random Forest, Gradient Boosting, XGBoost
  - Can learn optimal split points on encoded values
  - Order information helps create meaningful splits
  
- **Linear Models** (with caution): 
  - When true numeric relationship exists
  - Assumes equal spacing between categories
  
- **Neural Networks**: As input features when order matters

### Not Suitable For

- **Nominal Variables**: Categories without natural order (color, country)
  - Would impose artificial ranking
  - Use one-hot encoding instead
  
- **Linear Models** (generally):
  - Assumes equal distances between categories
  - Example: Encoding [Small=1, Medium=2, Large=3] assumes Medium-Small = Large-Medium
  - May not reflect reality

## Mathematical Representation

### Encoding Function

For a categorical variable $X$ with ordered categories $\{c_1, c_2, ..., c_k\}$:

$$f: X \rightarrow \mathbb{Z}$$
$$f(c_i) = i \quad \text{for } i = 1, 2, ..., k$$

Or with 0-based indexing:
$$f(c_i) = i - 1 \quad \text{for } i = 1, 2, ..., k$$

**Properties**:
- **Injective**: Each category maps to unique integer
- **Order-preserving**: If $c_i < c_j$ then $f(c_i) < f(c_j)$
- **Bijective**: Between category set and $\{0, 1, ..., k-1\}$ or $\{1, 2, ..., k\}$

### Example

**Education Level**: [High School, Bachelor's, Master's, PhD]

**Encoding**:
$$\begin{align}
\text{High School} &\rightarrow 0 \\
\text{Bachelor's} &\rightarrow 1 \\
\text{Master's} &\rightarrow 2 \\
\text{PhD} &\rightarrow 3
\end{align}$$

**Order preserved**: $0 < 1 < 2 < 3$ reflects education progression

## Ordinal vs Nominal Variables

### Ordinal Variables (Use Ordinal Encoding)

**Definition**: Categories have natural order/ranking

**Examples**:
- **Education**: Elementary < High School < Bachelor's < Master's < PhD
- **Size**: XS < S < M < L < XL < XXL
- **Rating**: Poor < Fair < Good < Excellent
- **Temperature**: Cold < Warm < Hot
- **Frequency**: Never < Rarely < Sometimes < Often < Always
- **Income Bracket**: 0-25k < 25-50k < 50-100k < 100k+
- **Priority**: Low < Medium < High < Critical
- **Satisfaction**: Very Unsatisfied < Unsatisfied < Neutral < Satisfied < Very Satisfied

**Characteristic**: Relationship $c_i < c_j$ is meaningful

### Nominal Variables (Do NOT Use Ordinal Encoding)

**Definition**: Categories without natural order

**Examples**:
- **Color**: Red, Blue, Green (no inherent order)
- **Country**: USA, France, Japan (alphabetical not meaningful)
- **Product Type**: Laptop, Phone, Tablet (no ranking)
- **Department**: Sales, Engineering, Marketing
- **Payment Method**: Cash, Credit, Debit
- **Gender**: Male, Female, Other

**Characteristic**: Relationship $c_i < c_j$ is arbitrary/meaningless

**Correct Encoding**: One-hot encoding

## Types of Ordinal Encoding

### 1. Label Encoding (Basic Ordinal)

**Method**: Assign integers 0, 1, 2, ..., k-1

**Advantages**:
- Simple
- Consistent
- Standard approach

**Disadvantages**:
- May not reflect actual distances
- Assumes equal spacing

### 2. Custom Mapping

**Method**: Manually specify mapping based on domain knowledge

**Example**: T-shirt sizes with non-uniform gaps
```
XS  → 0
S   → 2
M   → 5
L   → 8
XL  → 12
XXL → 18
```

**Rationale**: Actual size differences not uniform

**Advantages**:
- Reflects domain knowledge
- Better captures true relationships
- Can incorporate measurement data

**Disadvantages**:
- Requires expert knowledge
- Not automatic
- May overfit to specific dataset

### 3. Target-Based Encoding (for prediction tasks)

**Method**: Order categories by target mean, then encode

**Algorithm**:
1. Calculate mean of target for each category
2. Sort categories by target mean
3. Assign integers based on sorted order

**Example**: City vs House Price
```
Original categories: [CityA, CityB, CityC, CityD]
Mean prices: [150k, 300k, 200k, 400k]
Sorted by price: CityA < CityC < CityB < CityD
Encoding: CityA→0, CityC→1, CityB→2, CityD→3
```

**Advantages**:
- Data-driven ordering
- Often improves predictive performance
- Creates natural monotonic relationship with target

**Disadvantages**:
- Risk of **target leakage** (use only for truly ordinal variables)
- May not generalize to new data
- Can overfit

**Caution**: Only use for variables that have inherent ordinality; don't use to convert nominal to ordinal

## Impact on Different ML Algorithms

### Tree-Based Algorithms

**Decision Trees, Random Forest, Gradient Boosting**:

**Split Criterion**: $\text{feature} \leq \text{threshold}$

**With Ordinal Encoding**:
- Can split at any threshold: $x \leq 1.5$ (separates categories 0,1 from 2,3,...)
- Order information helps create meaningful groups
- Effective for variables with many categories

**Example**: Education [0=HS, 1=BS, 2=MS, 3=PhD]
- Split: $education \leq 1.5$ → High School/Bachelor's vs Master's/PhD
- Meaningful grouping based on graduate education

**Advantages**:
- Trees learn natural groupings
- Works well even with "wrong" spacing
- Can capture non-linear relationships

### Linear Models

**Linear/Logistic Regression**:

**Model**: $y = \beta_0 + \beta_1 x + ...$

**Assumption**: Linear relationship with encoded values

**Problem**: Assumes equal intervals
- Encoding [Low=1, Medium=2, High=3] assumes:
  - Medium - Low = High - Medium = 1
  - Effect of Low→Medium = Effect of Medium→High
- May not be true in reality

**When It Works**:
- True underlying numeric relationship
- Approximately equal spacing
- Example: Age groups with equal ranges

**When It Fails**:
- Highly unequal intervals
- Non-linear progression
- Example: Income brackets [0-25k, 25-50k, 50-100k, 100k+] → unequal widths

**Alternative for Linear Models**:
- **Polynomial features**: $x, x^2, x^3$ to capture non-linearity
- **Splines**: Piecewise linear/polynomial
- **One-hot encoding**: If order not truly linear

### Distance-Based Algorithms

**K-NN, K-Means, SVM**:

**Distance**: $d(x, y) = \sqrt{\sum(x_i - y_i)^2}$

**Issue**: Numeric differences treated as distances
- Distance from 0 to 1 = Distance from 1 to 2
- May not reflect actual category differences

**Mitigation**:
- Custom encoding with proper scaling
- One-hot encoding for nominal aspects
- Feature scaling after encoding

### Neural Networks

**Depends on Architecture**:

**Fully Connected**:
- Can learn arbitrary mappings
- Order helps initialization and optimization
- May benefit from ordinal over one-hot (fewer parameters)

**Embedding Layers**:
- Can learn optimal representation
- Ordinal encoding as input to embedding
- Network learns semantic similarity

## Handling Unknown Categories

### Problem

New category in test/production data not seen in training

### Solutions

**1. Assign Special Value**:
- Unknown → -1 (or max+1)
- Signals "other" category

**2. Map to Most Frequent**:
- Assign to mode from training
- Conservative approach

**3. Map to Middle**:
- Assign median encoded value
- Neutral position

**4. Raise Error**:
- Alert for data quality issue
- Manual investigation required

**5. Separate "Unknown" Category**:
- Treat as valid ordinal category
- Encode consistently

## Inverse Transform

Convert encoded integers back to original categories:

$$f^{-1}: \mathbb{Z} \rightarrow X$$
$$f^{-1}(i) = c_i$$

**Use Cases**:
- Interpret model predictions
- Display results to users
- Validate encoding correctness

**Example**:
```
Encoded prediction: 2
Inverse: Master's degree
```

## Practical Considerations

### 1. Defining Order

**Clear Cases**: Education, size, rating → obvious order

**Ambiguous Cases**: 
- **Month**: Jan=1, Feb=2, ..., Dec=12 (cyclical, not truly ordinal)
- **Day of Week**: Mon=0, ..., Sun=6 (cyclical)

**Solution for Cyclical**:
- Use sin/cos encoding instead
- Month: $\sin(\frac{2\pi \cdot month}{12}), \cos(\frac{2\pi \cdot month}{12})$

### 2. Number of Categories

**Few Categories** (2-5):
- Ordinal encoding straightforward
- Linear models may work

**Many Categories** (>10):
- Tree-based models excel
- Linear models struggle (too coarse approximation)
- Consider grouping similar categories

### 3. Missing Values

**Options**:
- **Encode as -1**: Separate category
- **Encode as mode**: Impute with most frequent
- **Encode as separate category**: If missingness meaningful
- **Drop rows**: If few missing

### 4. Multiple Ordinal Features

**Scale separately**:
- Each feature encoded independently
- Different ranges OK for trees
- May standardize for linear models

### 5. Documentation

**Always document**:
- Mapping dictionary (category → integer)
- Rationale for order
- Source of order (domain knowledge, data-driven, etc.)
- Handling of unknowns and missing values

## Common Mistakes

### 1. Encoding Nominal as Ordinal

**Wrong**: Encoding [Red=0, Blue=1, Green=2]
- Implies Green > Blue > Red (meaningless)
- Model learns false relationships

**Right**: One-hot encoding

### 2. Inconsistent Ordering

**Wrong**: Different order in train vs test
- Training: [Low=0, Medium=1, High=2]
- Test: [Low=0, High=1, Medium=2]

**Right**: Save and reuse mapping from training

### 3. Assuming Linear Spacing

**Wrong**: Using ordinal encoding for income brackets in linear regression
- Brackets: [0-20k, 20-40k, 40-80k, 80k+]
- Encoding: [0, 1, 2, 3]
- Assumes equal 20k increments (false for last two)

**Right**: 
- Use midpoints: [10k, 30k, 60k, 100k]
- Or one-hot encoding
- Or tree-based model

### 4. Ignoring New Categories

**Wrong**: Crash or assign random value when unknown category appears

**Right**: Have explicit strategy defined beforehand

## Example Workflow

### Step 1: Identify Ordinal Variables

Analyze each categorical variable:
- Is there natural order?
- Does X < Y have meaning?

### Step 2: Define Order

For each ordinal variable:
- List all categories
- Sort by inherent order
- Validate with domain expert

### Step 3: Create Mapping

```
mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
```

### Step 4: Encode Training Data

Apply mapping to training data:
- Store mapping for later use
- Handle missing values

### Step 5: Encode Test Data

Use same mapping:
- Check for unknown categories
- Apply predetermined handling strategy

### Step 6: Validate

- Check encoded values are integers
- Verify order preserved
- Spot-check examples
- Test inverse transform

## Comparison with Alternatives

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Ordinal Encoding** | Preserves order, 1 feature, memory efficient | Assumes spacing, only for ordinal | Tree models, true ordinal data |
| **One-Hot Encoding** | No assumptions, works for nominal | k features, sparse, dummy trap | Linear models, nominal data |
| **Target Encoding** | Data-driven, powerful | Leakage risk, overfitting | With regularization, many categories |
| **Binary Encoding** | Fewer features than one-hot | Not interpretable | High cardinality, need compression |

## Summary

Ordinal encoding converts ordered categorical variables into integers while preserving rank relationships. It's essential for efficiently representing ordinal data in machine learning models, particularly tree-based algorithms.

**Key Points**:

**When to Use**:
- Variables with natural order (education, size, rating)
- Tree-based models
- Memory/dimensionality constraints
- True ordinal relationships

**When NOT to Use**:
- Nominal variables (no order)
- Linear models (unless true linear relationship)
- Equal spacing assumption violated

**Mathematical Foundation**:
- Order-preserving mapping: $f(c_i) = i$
- Injective and bijective
- Preserves inequality: $c_i < c_j \Rightarrow f(c_i) < f(c_j)$

**Best Practices**:
- Verify true ordinality before using
- Document mapping and rationale
- Use same mapping for train/test
- Handle unknown categories explicitly
- Consider custom mappings for unequal spacing
- Validate with domain experts

**Advantages**:
- Single feature (vs k for one-hot)
- Preserves order information
- Memory efficient
- Works well with trees

**Limitations**:
- Assumes meaningful spacing (for linear models)
- Only for ordinal data
- May need custom mapping for true distances

Ordinal encoding is a simple yet powerful technique when applied correctly to truly ordinal variables. Understanding the distinction between ordinal and nominal data, and choosing the appropriate encoding method, is fundamental to successful feature engineering in machine learning.

---
