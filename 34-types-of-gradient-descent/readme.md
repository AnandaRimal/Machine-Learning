# Types of Gradient Descent

## Introduction

Gradient Descent has three main variants that differ in how much data they use to compute the gradient at each iteration. The choice between Batch, Stochastic, and Mini-Batch Gradient Descent significantly impacts training speed, memory usage, convergence behavior, and final model quality. Understanding these variants is crucial for efficiently training machine learning models, especially deep neural networks.

## Why Different Types Matter

The fundamental trade-off is:
- **More data per update** → More accurate gradient → Stable but slow
- **Less data per update** → Noisy gradient → Fast but erratic

Different scenarios require different approaches:
- Small dataset + convex problem → Batch GD
- Huge dataset + need speed → Stochastic GD
- Deep learning + GPUs → Mini-Batch GD

## The Three Types

### Overview Table

| Type | Data per Iteration | Speed | Stability | Memory | Use Case |
|------|-------------------|-------|-----------|---------|----------|
| **Batch** | All $n$ samples | Slow | Very Stable | High | Small datasets, convex problems |
| **Stochastic (SGD)** | 1 sample | Very Fast | Noisy | Low | Online learning, huge datasets |
| **Mini-Batch** | $b$ samples (32-512) | Fast | Balanced | Medium | Deep learning, most modern ML |

---

## 1. Batch Gradient Descent (BGD)

### Definition

Computes the gradient using **all training examples** in each iteration.

### Mathematical Formula

**Gradient calculation**:

$$\nabla J(\theta) = \frac{1}{n}\sum_{i=1}^{n}\nabla J_i(\theta)$$

**Parameter update**:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \frac{1}{n}\sum_{i=1}^{n}\nabla J_i(\theta^{(t)})$$

Where:
- $n$: Total number of training examples
- $\nabla J_i(\theta)$: Gradient for example $i$
- $\alpha$: Learning rate

### Detailed Example

**Dataset**: Linear regression with 5 samples

| $x$ | $y$ |
|-----|-----|
| 1   | 3   |
| 2   | 5   |
| 3   | 7   |
| 4   | 9   |
| 5   | 11  |

**Model**: $\hat{y} = \theta_0 + \theta_1 x$

**Initialize**: $\theta_0 = 0$, $\theta_1 = 0$, $\alpha = 0.01$

**Cost function**: $J(\theta) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$

**Iteration 1**:

Predictions with current parameters:
- All predictions: $\hat{y}_i = 0 + 0 \cdot x_i = 0$

Errors: $e_i = \hat{y}_i - y_i$
- $e_1 = 0 - 3 = -3$
- $e_2 = 0 - 5 = -5$
- $e_3 = 0 - 7 = -7$
- $e_4 = 0 - 9 = -9$
- $e_5 = 0 - 11 = -11$

Gradients:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{5}\sum_{i=1}^{5}e_i = \frac{-3-5-7-9-11}{5} = \frac{-35}{5} = -7$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{5}\sum_{i=1}^{5}e_i \cdot x_i = \frac{(-3)(1)+(-5)(2)+(-7)(3)+(-9)(4)+(-11)(5)}{5}$$
$$= \frac{-3-10-21-36-55}{5} = \frac{-125}{5} = -25$$

Update parameters:
$$\theta_0^{(1)} = 0 - 0.01 \times (-7) = 0.07$$
$$\theta_1^{(1)} = 0 - 0.01 \times (-25) = 0.25$$

**Iteration 2**:

New predictions: $\hat{y}_i = 0.07 + 0.25x_i$
- $\hat{y}_1 = 0.07 + 0.25(1) = 0.32$
- $\hat{y}_2 = 0.07 + 0.25(2) = 0.57$
- etc.

Continue until convergence...

**True solution**: $\theta_0 = 1, \theta_1 = 2$ (since $y = 1 + 2x$)

### Advantages

1. **Exact Gradient**: Uses all data, gradient points in true descent direction
2. **Smooth Convergence**: Monotonic decrease in cost (for convex functions)
3. **Guaranteed Convergence**: For convex problems with appropriate learning rate
4. **Theoretical Guarantees**: Well-studied convergence properties
5. **No Randomness**: Deterministic, reproducible results
6. **Optimal for Small Data**: Best choice when dataset fits in memory

### Disadvantages

1. **Slow for Large Datasets**: Must process all $n$ samples per update
2. **Memory Intensive**: Requires all data in memory simultaneously
3. **Redundant Computation**: Similar samples give similar gradients
4. **No Online Learning**: Can't update with new data incrementally
5. **Local Minima**: Gets stuck in local minima (non-convex problems)
6. **Slow Progress**: May take many epochs to converge

### When to Use

- Dataset size: **< 10,000 samples**
- Problem type: **Convex optimization**
- Memory: **Sufficient to hold all data**
- Need: **Precise convergence**
- Example: Small-scale linear regression, logistic regression

---

## 2. Stochastic Gradient Descent (SGD)

### Definition

Computes the gradient using **one randomly selected training example** per iteration.

### Mathematical Formula

**Random selection**: Pick index $i$ uniformly at random from $\{1, 2, ..., n\}$

**Gradient estimate**:

$$\nabla J(\theta) \approx \nabla J_i(\theta)$$

**Parameter update**:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \nabla J_i(\theta^{(t)})$$

### Detailed Example

Using the same dataset as before, but now we update after **each** sample.

**Initialize**: $\theta_0 = 0$, $\theta_1 = 0$, $\alpha = 0.01$

**Epoch 1, Sample 1** ($x_1=1, y_1=3$):

Prediction: $\hat{y}_1 = 0 + 0(1) = 0$

Error: $e_1 = 0 - 3 = -3$

Gradients:
$$\frac{\partial J_1}{\partial \theta_0} = e_1 = -3$$
$$\frac{\partial J_1}{\partial \theta_1} = e_1 \cdot x_1 = -3 \cdot 1 = -3$$

Update:
$$\theta_0 = 0 - 0.01(-3) = 0.03$$
$$\theta_1 = 0 - 0.01(-3) = 0.03$$

**Epoch 1, Sample 2** ($x_2=2, y_2=5$):

Prediction with new parameters: $\hat{y}_2 = 0.03 + 0.03(2) = 0.09$

Error: $e_2 = 0.09 - 5 = -4.91$

Gradients:
$$\frac{\partial J_2}{\partial \theta_0} = -4.91$$
$$\frac{\partial J_2}{\partial \theta_1} = -4.91 \cdot 2 = -9.82$$

Update:
$$\theta_0 = 0.03 - 0.01(-4.91) = 0.0791$$
$$\theta_1 = 0.03 - 0.01(-9.82) = 0.1282$$

**Continue for all 5 samples...**

**Key Difference**: Parameters updated **5 times per epoch** (vs. 1 time in Batch GD)

### Stochastic Nature

Due to random sampling, the cost function doesn't decrease smoothly:

```
Cost:
100 → 95 → 97 → 92 → 94 → 88 → 91 → 85 → ...
     (fluctuates but trends down)
```

### Advantages

1. **Very Fast**: Updates after each sample, quick progress
2. **Memory Efficient**: Processes one sample at a time
3. **Escapes Local Minima**: Noise helps jump out of poor solutions
4. **Online Learning**: Can update with streaming data
5. **Large Datasets**: Scalable to billions of samples
6. **Frequent Updates**: More parameter updates per epoch
7. **Implicit Regularization**: Noise acts as regularizer

### Disadvantages

1. **Noisy Updates**: High variance in gradient estimates
2. **Never Converges**: Oscillates around minimum, doesn't settle
3. **Requires Learning Rate Decay**: Need to reduce $\alpha$ over time
4. **Sensitive to Learning Rate**: Wrong $\alpha$ can diverge
5. **Slower to Exact Minimum**: Takes longer to reach precise minimum
6. **Difficult to Parallelize**: Sequential nature limits GPU speedup
7. **Hyperparameter Tuning**: More sensitive to initial settings

### When to Use

- Dataset size: **> 100,000 samples**
- Need: **Fast initial progress**
- Memory: **Limited**
- Online learning: **Yes**
- Example: Large-scale deep learning, online advertising

---

## 3. Mini-Batch Gradient Descent

### Definition

Computes the gradient using a **small random subset (batch)** of training examples per iteration.

### Mathematical Formula

**Batch selection**: Randomly sample $b$ indices from $\{1, 2, ..., n\}$ where $1 < b < n$

**Gradient estimate**:

$$\nabla J(\theta) \approx \frac{1}{b}\sum_{i \in \mathcal{B}}\nabla J_i(\theta)$$

where $\mathcal{B}$ is the mini-batch of size $b$.

**Parameter update**:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \frac{1}{b}\sum_{i \in \mathcal{B}}\nabla J_i(\theta^{(t)})$$

**Common batch sizes**: 32, 64, 128, 256, 512

### Detailed Example

Using the same dataset, with **batch size $b = 2$**.

**Initialize**: $\theta_0 = 0$, $\theta_1 = 0$, $\alpha = 0.01$

**Epoch 1, Batch 1** (samples 1 and 3):

Sample 1: $x_1=1, y_1=3$, prediction: $\hat{y}_1 = 0$, error: $e_1 = -3$

Sample 3: $x_3=3, y_3=7$, prediction: $\hat{y}_3 = 0$, error: $e_3 = -7$

Gradients (average over batch):
$$\frac{\partial J}{\partial \theta_0} = \frac{e_1 + e_3}{2} = \frac{-3 + (-7)}{2} = -5$$

$$\frac{\partial J}{\partial \theta_1} = \frac{e_1 \cdot x_1 + e_3 \cdot x_3}{2} = \frac{(-3)(1) + (-7)(3)}{2} = \frac{-24}{2} = -12$$

Update:
$$\theta_0 = 0 - 0.01(-5) = 0.05$$
$$\theta_1 = 0 - 0.01(-12) = 0.12$$

**Epoch 1, Batch 2** (samples 2 and 5):

Sample 2: $x_2=2, y_2=5$

Sample 5: $x_5=5, y_5=11$

With updated parameters: $\hat{y} = 0.05 + 0.12x$
- $\hat{y}_2 = 0.05 + 0.12(2) = 0.29$, $e_2 = -4.71$
- $\hat{y}_5 = 0.05 + 0.12(5) = 0.65$, $e_5 = -10.35$

Gradients:
$$\frac{\partial J}{\partial \theta_0} = \frac{-4.71 + (-10.35)}{2} = -7.53$$

$$\frac{\partial J}{\partial \theta_1} = \frac{(-4.71)(2) + (-10.35)(5)}{2} = \frac{-61.17}{2} = -30.585$$

Update:
$$\theta_0 = 0.05 - 0.01(-7.53) = 0.1253$$
$$\theta_1 = 0.12 - 0.01(-30.585) = 0.42585$$

**Continue...**

**Key**: With batch size 2, we get $\frac{5}{2} \approx 2-3$ updates per epoch (vs. 5 for SGD, 1 for Batch GD)

### Advantages

1. **Balanced Speed/Stability**: Faster than Batch, more stable than SGD
2. **GPU Efficient**: Vectorized operations on batches leverage parallelism
3. **Less Noisy Than SGD**: Averaging reduces gradient variance
4. **Memory Manageable**: Process batches, not entire dataset
5. **Faster Convergence**: Better than SGD due to reduced noise
6. **Flexible**: Tune batch size for hardware/problem
7. **Standard Practice**: Default choice in modern deep learning

### Disadvantages

1. **Hyperparameter**: Need to choose batch size
2. **Still Noisy**: More than Batch GD (though less than SGD)
3. **Not Exact Gradient**: Still an approximation
4. **Batch Size Tradeoffs**: Small→noisy, Large→slow, memory-intensive

### When to Use

- Dataset size: **Any size**
- Hardware: **GPU available**
- Problem: **Deep neural networks**
- Default choice: **Yes** (most versatile)
- Example: Image classification, NLP models

---

## Comparison with Visualizations

### Convergence Paths

Imagine a 2D parameter space with optimal point at center:

**Batch GD**:
```
Start → → → → → → → → → Optimum
(Smooth, straight path)
```

**SGD**:
```
Start → ↗ → ↘ → ↗ → ↘ → ↗ ~~ Optimum
(Zigzag, noisy, oscillates around optimum)
```

**Mini-Batch GD**:
```
Start → → ↗ → → ↘ → → ~ Optimum
(Mostly straight with some zigzag)
```

### Computational Complexity per Epoch

| Type | Updates per Epoch | Samples per Update | Total Computations |
|------|-------------------|--------------------|--------------------|
| Batch | 1 | $n$ | $n$ |
| SGD | $n$ | 1 | $n$ |
| Mini-Batch | $\frac{n}{b}$ | $b$ | $n$ |

**All process same data per epoch**, but:
- **Batch**: 1 accurate update
- **SGD**: $n$ noisy updates
- **Mini-Batch**: $\frac{n}{b}$ moderately accurate updates

### Learning Rate Considerations

**Batch GD**: Can use larger learning rate (stable gradient)

$$\alpha \in [0.001, 0.3]$$

**SGD**: Requires smaller learning rate (avoid divergence from noise)

$$\alpha \in [0.001, 0.01]$$

Often with **decay**: $\alpha_t = \frac{\alpha_0}{1 + kt}$

**Mini-Batch**: Medium learning rate

$$\alpha \in [0.001, 0.1]$$

---

## Code Examples

### 1. Batch Gradient Descent

```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    cost_history = []
    
    for epoch in range(epochs):
        # Compute predictions for ALL samples
        y_pred = X.dot(theta)
        
        # Compute error
        error = y_pred - y
        
        # Compute gradient using ALL samples
        gradient = (1/n_samples) * X.T.dot(error)
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Compute cost
        cost = (1/(2*n_samples)) * np.sum(error**2)
        cost_history.append(cost)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta, cost_history

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([3, 5, 7, 9, 11])

theta, costs = batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000)
print(f"Final parameters: {theta}")
```

### 2. Stochastic Gradient Descent

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=10):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    cost_history = []
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            # Select ONE sample
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            # Compute prediction for this sample
            y_pred = xi.dot(theta)
            
            # Compute error
            error = y_pred - yi
            
            # Compute gradient using THIS sample only
            gradient = xi.T.dot(error)
            
            # Update parameters
            theta = theta - learning_rate * gradient
        
        # Compute cost on full dataset for monitoring
        y_pred_all = X.dot(theta)
        cost = (1/(2*n_samples)) * np.sum((y_pred_all - y)**2)
        cost_history.append(cost)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta, cost_history

# Example usage
theta_sgd, costs_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=10)
print(f"Final parameters: {theta_sgd}")
```

### 3. Mini-Batch Gradient Descent

```python
def mini_batch_gradient_descent(X, y, learning_rate=0.01, epochs=50, batch_size=2):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    cost_history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute predictions for this batch
            y_pred = X_batch.dot(theta)
            
            # Compute error
            error = y_pred - y_batch
            
            # Compute gradient using this batch
            gradient = (1/len(X_batch)) * X_batch.T.dot(error)
            
            # Update parameters
            theta = theta - learning_rate * gradient
        
        # Compute cost
        y_pred_all = X.dot(theta)
        cost = (1/(2*n_samples)) * np.sum((y_pred_all - y)**2)
        cost_history.append(cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta, cost_history

# Example usage
theta_mb, costs_mb = mini_batch_gradient_descent(X, y, learning_rate=0.01, 
                                                   epochs=50, batch_size=2)
print(f"Final parameters: {theta_mb}")
```

### 4. Comparing All Three

```python
import matplotlib.pyplot as plt

# Run all three
theta_batch, costs_batch = batch_gradient_descent(X, y, epochs=1000)
theta_sgd, costs_sgd = stochastic_gradient_descent(X, y, epochs=10)
theta_mb, costs_mb = mini_batch_gradient_descent(X, y, epochs=50, batch_size=2)

# Plot convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(costs_batch)
plt.title('Batch GD')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(costs_sgd)
plt.title('Stochastic GD')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(costs_mb)
plt.title('Mini-Batch GD')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Batch GD parameters: {theta_batch}")
print(f"SGD parameters: {theta_sgd}")
print(f"Mini-Batch GD parameters: {theta_mb}")
```

### 5. With Learning Rate Decay (for SGD)

```python
def sgd_with_decay(X, y, initial_lr=0.1, epochs=50, decay_rate=0.01):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    cost_history = []
    
    for epoch in range(epochs):
        # Decay learning rate
        learning_rate = initial_lr / (1 + decay_rate * epoch)
        
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            y_pred = xi.dot(theta)
            error = y_pred - yi
            gradient = xi.T.dot(error)
            
            theta = theta - learning_rate * gradient
        
        y_pred_all = X.dot(theta)
        cost = (1/(2*n_samples)) * np.sum((y_pred_all - y)**2)
        cost_history.append(cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, LR: {learning_rate:.4f}, Cost: {cost:.4f}")
    
    return theta, cost_history
```

---

## What's in the Notebooks

The repository contains several notebooks demonstrating each variant:

### `batch-gradient-descent.ipynb`
- **Pure batch GD implementation** from scratch
- **Convergence visualization** showing smooth descent
- **Learning rate sensitivity** analysis
- **Comparison with analytical solution**
- **Computational time measurement**

### `stochastic-gradient-descent-from-scratch.ipynb`
- **SGD implementation** with random sampling
- **Noise visualization** in gradient estimates
- **Oscillation around optimum**
- **Learning rate decay strategies**
- **Online learning demonstration**

### `stochastic-gradient-descent-animation.ipynb`
- **Animated path** of SGD in 2D parameter space
- **Visual comparison** with batch GD
- **Cost function surface** visualization
- **Escape from local minima** demonstration

### `mini-batch-gradient-descent-from-scratch.ipynb`
- **Mini-batch implementation**
- **Batch size impact** (16, 32, 64, 128, 256)
- **Trade-off analysis**: speed vs. stability
- **GPU acceleration** with vectorization
- **Practical recommendations** for batch size selection

### GIF Animations

The repository includes animated visualizations:

- **`stochastic_animation_contour_plot.gif`**: SGD path on contour plot
- **`stochastic_animation_cost_plot.gif`**: Cost function evolution
- **`stochastic_animation_line_plot.gif`**: Parameter updates over time
- **`mini_batch_contour_plot.gif`**: Mini-batch path comparison

---

## Practical Guidelines

### Choosing Batch Size

**Rule of thumb**:
- Start with **32 or 64**
- If training slow → increase to 128 or 256
- If memory issues → decrease to 16 or 8
- If unstable → try smaller batch size first

**Powers of 2**: Use 32, 64, 128, 256 for efficient GPU memory usage

**Dataset-dependent**:
- Small dataset (< 1000): Use Batch GD or large mini-batches
- Medium (1000-100k): Mini-batch with size 32-128
- Large (> 100k): Mini-batch with size 64-256

### Learning Rate Guidelines

| Method | Typical Range | Starting Point |
|--------|---------------|----------------|
| Batch | 0.001 - 0.3 | 0.01 |
| SGD | 0.001 - 0.01 | 0.001 with decay |
| Mini-Batch | 0.001 - 0.1 | 0.01 |

**Learning rate scheduling**:
- **Step decay**: $\alpha_t = \alpha_0 \times 0.5^{floor(epoch/10)}$
- **Exponential**: $\alpha_t = \alpha_0 \times e^{-kt}$
- **1/t decay**: $\alpha_t = \frac{\alpha_0}{1 + kt}$

### Monitoring Convergence

**Batch GD**: Cost should decrease monotonically
```python
if cost[t+1] > cost[t]:
    print("Warning: Cost increased! Reduce learning rate")
```

**SGD**: Cost fluctuates, plot moving average
```python
moving_avg = np.convolve(costs, np.ones(10)/10, mode='valid')
plt.plot(moving_avg)
```

**Mini-Batch**: Some fluctuation, generally decreasing trend

---

## Summary Table

| Aspect | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|----------|---------------|---------------|
| **Data per update** | All $n$ | 1 | $b$ (32-512) |
| **Updates per epoch** | 1 | $n$ | $n/b$ |
| **Convergence** | Smooth | Noisy | Moderately smooth |
| **Speed (per epoch)** | Slow | Fast | Fast |
| **Memory** | High | Low | Medium |
| **Final accuracy** | High | Medium | High |
| **Hardware** | CPU | CPU/GPU | GPU optimal |
| **Best for** | Small data | Online learning | Deep learning |

## Key Takeaways

1. **Batch GD**: Most accurate, slowest, best for small datasets
2. **SGD**: Fastest initially, noisiest, best for huge datasets and online learning
3. **Mini-Batch**: Best of both worlds, standard choice for deep learning

**Modern practice**: Mini-Batch GD with:
- Batch size: 32-256 (power of 2)
- Learning rate: 0.001-0.01 with decay
- Optimizer: Adam, RMSprop (adaptive learning rates)

**Remember**: The "right" choice depends on:
- Dataset size
- Available memory
- Hardware (CPU vs GPU)
- Convergence requirements
- Training time constraints

---

*"Batch GD is like carefully planning each step of a journey. SGD is like taking quick random steps hoping to reach the destination. Mini-Batch GD is like taking deliberate but frequent steps—the sweet spot for most journeys."*
