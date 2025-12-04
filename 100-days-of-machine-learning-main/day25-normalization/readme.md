# Normalization (Min-Max Scaling)

## General Idea

Normalization, specifically Min-Max scaling, is a feature scaling technique that transforms data to fit within a specific range, typically [0, 1]. It rescales features by linearly mapping the original range [min, max] to the desired range, preserving the original distribution shape while ensuring all values fall within the specified bounds.

## Why Use Normalization?

1. **Bounded Range**: Ensures features within specific limits [0, 1] or custom range
2. **Neural Networks**: Works well with sigmoid/tanh activations expecting bounded inputs
3. **Image Processing**: Pixel values naturally in [0, 255] or [0, 1]
4. **Interpretability**: Proportional values (0=minimum, 1=maximum) easy to understand
5. **Feature Comparison**: Makes features with different units directly comparable
6. **Algorithm Requirements**: Some algorithms expect or perform better with bounded inputs
7. **Gradient-Based Optimization**: Can improve convergence in certain cases

## Role in Machine Learning

### Essential For

- **Neural Networks**: Especially with sigmoid ($\sigma(z) = \frac{1}{1+e^{-z}}$) or tanh activations
  - Expected input range: [-1, 1] or [0, 1]
  - Prevents saturation in extreme regions
  
- **Image Processing**: CNNs, computer vision models
  - Pixel values: [0, 255] → [0, 1]
  - Consistent input scale
  
- **Distance-Based Algorithms** (like KNN, K-Means):
  - When features have known bounded ranges
  - Alternative to standardization

- **Genetic Algorithms**: Often expect genes in [0, 1]

### Alternative To Standardization

- When data is **not normally distributed**
- When **no outliers** present
- When **bounded range** is requirement
- When interpretability of **min-max proportion** is valuable

### Not Recommended For

- **Data with outliers**: Outliers compress main data into tiny range
- **Test data outside training range**: Values < min or > max become < 0 or > 1
- **Sparse data**: Like standardization, densifies sparse matrices

## Mathematical Formula

### Min-Max Normalization

For a feature $X$ with values $x_1, x_2, ..., x_n$:

$$x' = \frac{x - \min(X)}{\max(X) - \min(X)}$$

Where:
- $x'$: Normalized value (range [0, 1])
- $x$: Original value
- $\min(X)$: Minimum value in feature
- $\max(X)$: Maximum value in feature

**Result**: $x' \in [0, 1]$

### Custom Range [a, b]

To normalize to arbitrary range $[a, b]$:

$$x' = a + \frac{(x - \min(X)) \cdot (b - a)}{\max(X) - \min(X)}$$

**Result**: $x' \in [a, b]$

**Common ranges**:
- [0, 1]: Standard normalization
- [-1, 1]: Centered normalization (common for tanh activation)
- [0.1, 0.9]: Avoid exact boundary values

### Properties After Normalization

**Minimum**: $\min(X') = 0$ (or $a$ for custom range)

**Maximum**: $\max(X') = 1$ (or $b$ for custom range)

**Midpoint**: Value at original midpoint → 0.5

**Linearity**: Linear transformation (preserves relationships)

**Relative Position**: $\frac{x_i - \min}{\max - \min}$ represents position in range

## Example Calculation

**Original Data**: $X = [10, 20, 30, 40, 50]$

**Step 1**: Identify min and max
$$\min(X) = 10, \quad \max(X) = 50$$

**Step 2**: Calculate range
$$\text{Range} = \max(X) - \min(X) = 50 - 10 = 40$$

**Step 3**: Normalize each value

$$x'_1 = \frac{10 - 10}{40} = 0$$

$$x'_2 = \frac{20 - 10}{40} = 0.25$$

$$x'_3 = \frac{30 - 10}{40} = 0.5$$

$$x'_4 = \frac{40 - 10}{40} = 0.75$$

$$x'_5 = \frac{50 - 10}{40} = 1$$

**Normalized Data**: $X' = [0, 0.25, 0.5, 0.75, 1]$

**Verification**:
- Min = 0 ✓
- Max = 1 ✓
- Range = 1 ✓
- Linear spacing preserved ✓

### Example with Custom Range [-1, 1]

Using same data, normalize to [-1, 1]:

$$x' = -1 + \frac{(x - 10) \cdot (1 - (-1))}{50 - 10} = -1 + \frac{(x - 10) \cdot 2}{40}$$

$$x'_1 = -1 + \frac{(10-10) \cdot 2}{40} = -1$$

$$x'_2 = -1 + \frac{(20-10) \cdot 2}{40} = -0.5$$

$$x'_3 = -1 + \frac{(30-10) \cdot 2}{40} = 0$$

$$x'_4 = -1 + \frac{(40-10) \cdot 2}{40} = 0.5$$

$$x'_5 = -1 + \frac{(50-10) \cdot 2}{40} = 1$$

**Normalized Data**: $X' = [-1, -0.5, 0, 0.5, 1]$

## Normalization in Practice

### Training Phase

1. **Compute statistics** on training data:
   $$\min_{train} = \min(X_{train})$$
   $$\max_{train} = \max(X_{train})$$

2. **Transform training data**:
   $$x'_{train} = \frac{x_{train} - \min_{train}}{\max_{train} - \min_{train}}$$

3. **Save parameters**: $\min_{train}$, $\max_{train}$ for production

### Test/Production Phase

**Critical**: Use training statistics, not test statistics

$$x'_{test} = \frac{x_{test} - \min_{train}}{\max_{train} - \min_{train}}$$

**Consequence**: Test values may fall outside [0, 1]
- If $x_{test} < \min_{train}$: $x'_{test} < 0$
- If $x_{test} > \max_{train}$: $x'_{test} > 1$

**Handling out-of-range values**:

**Option 1**: Clip to [0, 1]
$$x' = \text{clip}(x', 0, 1) = \max(0, \min(1, x'))$$

**Option 2**: Allow extrapolation
- Keep values as is (< 0 or > 1)
- Model may handle gracefully

**Option 3**: Flag as anomaly
- Treat as outlier or novel data point

## Normalization vs Other Scaling Methods

### Normalization vs Standardization

| Aspect | Min-Max Normalization | Standardization |
|--------|----------------------|----------------|
| **Formula** | $\frac{x-\min}{\max-\min}$ | $\frac{x-\mu}{\sigma}$ |
| **Range** | [0, 1] or custom [a, b] | Unbounded (typically -3 to 3) |
| **Mean** | Depends on data | 0 |
| **Std** | Depends on data | 1 |
| **Outlier Sensitivity** | **Very high** | Moderate |
| **Distribution** | Preserves shape | Preserves shape |
| **Use** | Bounded data, no outliers | Normal-ish data, with outliers |
| **Interpretability** | Proportion of range | Std deviations from mean |
| **New data** | May fall outside [0,1] | Always valid |

**Example showing outlier impact**:

**Data without outlier**: [10, 20, 30, 40, 50]
- Normalized: [0, 0.25, 0.5, 0.75, 1]
- Good distribution

**Data with outlier**: [10, 20, 30, 40, 500]
- Normalized: [0, 0.02, 0.04, 0.06, 1]
- Main data compressed into [0, 0.06]!
- Outlier dominates

### Normalization vs Robust Scaling

**Robust Scaler**:
$$x' = \frac{x - \text{median}}{IQR}$$

**Comparison**:
| Aspect | Normalization | Robust Scaling |
|--------|--------------|----------------|
| **Outlier Impact** | Extreme (uses min/max) | Low (uses median/IQR) |
| **Range** | [0, 1] | Unbounded |
| **Best For** | Clean data | Outlier-heavy data |

## Impact on Machine Learning Algorithms

### Neural Networks

**Activation Functions**:

**Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Output range: (0, 1)
- Input: Preferably in reasonable range (e.g., [-5, 5])
- Normalization to [0, 1] works well

**Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- Output range: (-1, 1)
- Input: Normalization to [-1, 1] often used

**ReLU**: $\text{ReLU}(z) = \max(0, z)$
- Output range: [0, ∞)
- Less sensitive to input scale
- Standardization or normalization both work

**Why Normalization Helps**:
- Prevents saturation in sigmoid/tanh (gradients near 0)
- Balanced weight initialization more effective
- Faster convergence (similar to standardization)
- Numerical stability

### Gradient Descent

**Effect on optimization**:

**Cost function**: $J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$

**Without scaling**:
- Elliptical contours (if features on different scales)
- Slow convergence

**With normalization** (or standardization):
- More circular contours
- Faster, more direct path to minimum
- Better conditioning of Hessian matrix

**Learning rate selection**:
- Easier with scaled features
- Same learning rate works across features
- Less tuning required

### Distance-Based Algorithms

**K-Nearest Neighbors (KNN)**:

**Distance metric** (Euclidean):
$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

**Impact**:
- Features with larger ranges dominate distance
- Normalization ensures equal contribution

**Example**:
- Feature 1 (age): 20-80, normalized: [0, 1]
- Feature 2 (income): 20k-200k, normalized: [0, 1]
- Both contribute equally to distance calculation

**K-Means Clustering**: Same benefit (uses Euclidean distance)

### Image Processing

**Pixel values**: Original range [0, 255]

**Normalization**: 
$$\text{pixel}' = \frac{\text{pixel}}{255}$$

Result: [0, 1]

**Why beneficial**:
- Standard range across images
- Faster convergence in CNNs
- Better weight initialization
- Reduced memory (can use float16)

**Advanced**: Per-channel normalization using ImageNet statistics
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

## Problem: Outliers

### Impact of Outliers

**Original data**: [10, 15, 20, 25, 30, 35, 40, 100]

**Normalized**:
$$x' = \frac{x - 10}{100 - 10} = \frac{x - 10}{90}$$

**Results**:
- 10 → 0
- 15 → 0.056
- 20 → 0.111
- ...
- 40 → 0.333
- 100 → 1.0

**Problem**: Main data (10-40) compressed into [0, 0.33]

**Information loss**: 
- Fine distinctions in main range lost
- Outlier takes up most of [0, 1] range

### Solutions to Outlier Problem

**Option 1**: Remove outliers before normalization
- Detect using IQR, z-score, domain knowledge
- Normalize remaining data

**Option 2**: Clip outliers
- Cap values at percentiles (e.g., 1st and 99th)
- Then normalize

**Option 3**: Use robust scaling instead
- Uses median and IQR (not affected by outliers)

**Option 4**: Winsorization
- Replace outliers with nearest non-outlier value
- Then normalize

**Option 5**: Transform then normalize
- Log transform to reduce skew
- Then apply normalization

**Option 6**: Use standardization instead
- Less sensitive to outliers than normalization

## When to Use Normalization

**Choose Normalization when**:

1. **Bounded data** naturally (e.g., percentages, probabilities)
2. **No significant outliers** present
3. **Neural networks** with sigmoid/tanh activations
4. **Image data** (pixel values)
5. **Interpretability** of min-max proportion important
6. **Known range** for production data
7. **Sparse data** should remain sparse (use standardization with mean=False instead)

**Avoid Normalization when**:

1. **Outliers present** (use standardization or robust scaling)
2. **Unbounded test data** expected
3. **Normal distribution** (standardization more appropriate)
4. **Tree-based models** (scaling unnecessary)

## Normalization Variants

### Decimal Scaling

Normalize by moving decimal point:

$$x' = \frac{x}{10^d}$$

Where $d$ is smallest integer such that $\max(|x'|) < 1$

**Example**: Data [100, 500, 1000]
- $d = 4$ (move 4 decimal places)
- Result: [0.01, 0.05, 0.1]

**Use**: Quick approximation, retains magnitude information

### Vector Normalization (L2 Normalization)

Normalize each **sample** (row) to unit norm:

$$x'_i = \frac{x_i}{\sqrt{\sum_{j} x_{ij}^2}}$$

**Result**: Each sample has L2 norm = 1

**Use**: 
- Text classification (TF-IDF vectors)
- Cosine similarity comparisons
- Neural network embeddings

**Different from Min-Max**: Normalizes samples, not features

### Max Absolute Scaling

$$x' = \frac{x}{\max(|x|)}$$

**Result**: Range [-1, 1]

**Advantage**: Preserves sign, handles negative values simply

## Inverse Transformation

Convert normalized values back to original scale:

$$x = x' \cdot (\max - \min) + \min$$

**Use case**: Interpret predictions in original units

**Example**:
- Normalized prediction: 0.75
- Original min: 10, max: 50
- Original prediction: $0.75 \times (50-10) + 10 = 0.75 \times 40 + 10 = 40$

## Practical Implementation

### Multiple Features

Normalize each feature independently:

$$x'_{ij} = \frac{x_{ij} - \min_j}{\max_j - \min_j}$$

Where:
- $i$: Sample index
- $j$: Feature index
- $\min_j$, $\max_j$: Min and max of feature $j$

**Do NOT** normalize across samples (rows)

### Constant Features

If $\max_j = \min_j$ (all values identical):
- Division by zero issue
- Feature is uninformative
- **Solution**: Remove feature or set to arbitrary value (e.g., 0.5)

### Negative Values

Min-Max normalization handles negative values:

**Data**: [-10, -5, 0, 5, 10]
- Min = -10, Max = 10
- Normalized: [0, 0.25, 0.5, 0.75, 1]

**For [-1, 1] range**: Use custom range formula

### Time Series

**Caution**: 
- Training min/max may not apply to future
- Non-stationary data: Range changes over time
- **Alternatives**: 
  - Rolling window normalization
  - Normalize each window independently
  - Use differencing or percentage change

## Comparison Summary

| Method | Formula | Range | Outlier Sensitivity | Use Case |
|--------|---------|-------|--------------------|-----------
| **Min-Max** | $\frac{x-\min}{\max-\min}$ | [0, 1] | Very High | Bounded, no outliers |
| **Standardization** | $\frac{x-\mu}{\sigma}$ | Unbounded | Moderate | Normal, with outliers |
| **Robust** | $\frac{x-\text{med}}{IQR}$ | Unbounded | Low | Heavy outliers |
| **Max Abs** | $\frac{x}{\max(|x|)}$ | [-1, 1] | High | Sparse, signed data |

## Summary

Min-Max normalization is a simple, interpretable scaling technique that transforms features to a bounded range, typically [0, 1]. It's particularly effective for neural networks and image processing but is highly sensitive to outliers.

**Key Points**:

**Mathematical Foundation**:
- Formula: $x' = \frac{x-\min}{\max-\min}$
- Results in range [0, 1] or custom [a, b]
- Linear transformation (preserves shape)

**When to Use**:
- Neural networks (sigmoid/tanh activation)
- Image processing (pixel normalization)
- Data without outliers
- Bounded, known-range features

**Advantages**:
- Bounded range [0, 1]
- Intuitive interpretation (proportion of range)
- Works well with bounded activations
- Preserves zero values (if min=0)

**Disadvantages**:
- Extremely sensitive to outliers
- Test data may fall outside [0, 1]
- Not robust to range changes
- Densifies sparse data

**Best Practices**:
- Check for outliers before applying
- Use training min/max for test data
- Consider clipping test values to [0, 1]
- Use robust scaling if outliers present
- Include in pipeline for consistency
- Document scaling parameters

**Comparison to Standardization**:
- Normalization: Bounded range, outlier-sensitive
- Standardization: Unbounded, outlier-resistant
- Choose based on data distribution and algorithm requirements

Normalization is a fundamental preprocessing tool that, when applied appropriately to clean data, can significantly improve model training and performance, especially in deep learning contexts.

---

**Video Link**: https://youtu.be/eBrGyuA2MIg
