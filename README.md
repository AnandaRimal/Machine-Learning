# 100 Days of Machine Learning 🚀

A comprehensive, hands-on journey through the complete machine learning pipeline - from data acquisition to advanced ensemble methods. This repository contains systematic daily learning modules covering data science fundamentals, preprocessing techniques, feature engineering, and machine learning algorithms with mathematical foundations and practical implementations.

## 📚 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Learning Path](#learning-path)
- [Module Categories](#module-categories)
- [Key Mathematical Concepts](#key-mathematical-concepts)
- [Usage](#usage)
- [Contributing](#contributing)

---

## 🎯 Overview

This repository represents a structured 100-day learning journey through machine learning and data science. Each module is designed to build upon previous concepts, providing:

- **Theoretical foundations** with mathematical formulas
- **Practical implementations** using Python and scikit-learn
- **Real-world examples** with actual datasets
- **Visual demonstrations** of concepts
- **Best practices** and when to use each technique

**Total Modules**: 45+ comprehensive tutorials  
**Skill Level**: Beginner to Advanced  
**Time Commitment**: ~2-3 hours per module

---

## 📂 Repository Structure

### 🔵 Phase 1: Data Acquisition & Manipulation (Days 15-18)

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [Day 15](day15%20-%20working%20with%20csv%20files/) | Working with CSV Files | Reading/writing CSV, handling delimiters, encodings |
| [Day 16](day16%20-%20working-with-json-and-sql/) | JSON & SQL Integration | JSON parsing, SQL queries, database connections |
| [Day 17](day17-api-to-dataframe/) | API to DataFrame | REST APIs, HTTP requests, pagination |
| [Day 18](day18-pandas-dataframe-using-web-scraping/) | Web Scraping | BeautifulSoup, HTML parsing, data extraction |

### 🟢 Phase 2: Exploratory Data Analysis (Days 19-22)

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [Day 19](day19-understanding-your-data-descriptive-stats/) | Descriptive Statistics | Mean, median, variance, skewness, kurtosis |
| [Day 20](day20-univariate-analysis/) | Univariate Analysis | Single variable distributions, histograms, box plots |
| [Day 21](day21-bivariate-analysis/) | Bivariate Analysis | Correlation, scatter plots, chi-square tests |
| [Day 22](day22-pandas-profiling/) | Automated EDA | Pandas profiling, comprehensive reports |

### 🟡 Phase 3: Feature Scaling (Days 24-25)

| Module | Topic | Mathematical Formula | Use Case |
|--------|-------|---------------------|----------|
| [Day 24](day24-standardization/) | Standardization | $z = \frac{x - \mu}{\sigma}$ | SVM, Neural Networks, PCA |
| [Day 25](day25-normalization/) | Normalization | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Gradient descent, bounded features |

### 🟠 Phase 4: Encoding Techniques (Days 26-28)

| Module | Topic | Best For |
|--------|-------|----------|
| [Day 26](day26-ordinal-encoding/) | Ordinal Encoding | Ordered categories (Low, Medium, High) |
| [Day 27](day27-one-hot-encoding/) | One-Hot Encoding | Nominal categories (Color, Country) |
| [Day 28](day28-column-transformer/) | Column Transformer | Multiple transformations simultaneously |

### 🔴 Phase 5: Advanced Preprocessing (Days 29-34)

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [Day 29](day29-sklearn-pipelines/) | Sklearn Pipelines | Automated workflows, reproducibility |
| [Day 30](day30-function-transformer/) | Function Transformer | Custom transformations |
| [Day 31](day31-power-transformer/) | Power Transformer | Box-Cox, Yeo-Johnson transformations |
| [Day 32](day32-binning-and-binarization/) | Binning & Binarization | Discretization, threshold-based conversion |
| [Day 33](day33-handling-mixed-variables/) | Mixed Variables | Combined numerical/categorical handling |
| [Day 34](day34-handling-date-and-time/) | Date & Time Features | Temporal feature extraction |

### 🟣 Phase 6: Missing Data Handling (Days 35-40)

| Module | Topic | Technique |
|--------|-------|-----------|
| [Day 35](day35-complete-case-analysis/) | Complete Case Analysis | Listwise deletion |
| [Day 36](day36-imputing-numerical-data/) | Numerical Imputation | Mean, median, mode imputation |
| [Day 37](day37-handling-missing-categorical-data/) | Categorical Imputation | Frequent category, missing indicator |
| [Day 38](day38-missing-indicator/) | Missing Indicator | Binary flag for missingness |
| [Day 39](day39-knn-imputer/) | KNN Imputer | K-nearest neighbors imputation |
| [Day 40](day40-iterative-imputer/) | Iterative Imputer | Multivariate imputation (MICE) |

### ⚫ Phase 7: Outlier Detection & Treatment (Days 42-44)

| Module | Topic | Method | Formula |
|--------|-------|--------|---------|
| [Day 42](day42-outlier-removal-using-zscore/) | Z-Score Method | Statistical | $z = \frac{x - \mu}{\sigma}$ |
| [Day 43](day43-outlier-removal-using-iqr-method/) | IQR Method | Robust | $IQR = Q_3 - Q_1$ |
| [Day 44](day44-outlier-detection-using-percentiles/) | Percentile Method | Quantile-based | Top/bottom percentiles |

### 🔵 Phase 8: Feature Engineering (Days 45, 47)

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [Day 45](day45-feature-construction-and-feature-splitting/) | Feature Construction & Splitting | Creating new features, parsing complex variables |
| [Day 47](day47-pca/) | Principal Component Analysis | Dimensionality reduction, eigenvalues |

### 🟢 Phase 9: Regression Algorithms (Days 48-57)

| Module | Topic | Algorithm | Complexity |
|--------|-------|-----------|------------|
| [Day 48](day48-simple-linear-regression/) | Simple Linear Regression | $y = mx + b$ | Basic |
| [Day 49](day49-regression-metrics/) | Regression Metrics | MSE, RMSE, MAE, R² | Evaluation |
| [Day 50](day50-multiple-linear-regression/) | Multiple Linear Regression | $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$ | Intermediate |
| [Day 51](day51-gradient-descent/) | Gradient Descent | $\theta := \theta - \alpha \nabla J(\theta)$ | Optimization |
| [Day 52](day52-types-of-gradient-descent/) | GD Types | Batch, Stochastic, Mini-batch | Advanced |
| [Day 53](day53-polynomial-regression/) | Polynomial Regression | $y = \beta_0 + \beta_1x + \beta_2x^2 + ...$ | Non-linear |
| [Day 55](day55-regularized-linear-models/) | Ridge Regression | $J(\theta) = MSE + \alpha\sum\theta^2$ | L2 Regularization |
| [Day 56](day56-lasso-regression/) | Lasso Regression | $J(\theta) = MSE + \alpha\sum|\theta|$ | L1 Regularization |
| [Day 57](day57-elasticnet-regression/) | ElasticNet | Combines L1 + L2 | Hybrid |

### 🟡 Phase 10: Classification Algorithms (Days 58-60)

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [Day 58](day58-logistic-regression/) | Logistic Regression | Sigmoid function, binary classification |
| [Day 59](day59-classification-metrics/) | Classification Metrics | Accuracy, Precision, Recall, F1-Score |
| [Day 60](day60-logistic-regression-contd/) | Advanced Logistic Regression | Softmax, multi-class classification |

### 🟠 Phase 11: Ensemble Methods (Days 65-68)

| Module | Topic | Technique | Type |
|--------|-------|-----------|------|
| [Day 65](day65-random-forest/) | Random Forest | Bagging + Feature randomness | Parallel |
| [Day 66](day66-adaboost/) | AdaBoost | Adaptive boosting | Sequential |
| [Day 68](day68-stacking-and-blending/) | Stacking & Blending | Meta-learning | Hybrid |
| [Gradient Boosting](gradient-boosting/) | Gradient Boosting | Residual correction | Sequential |

### 🔴 Phase 12: Unsupervised Learning

| Module | Topic | Algorithm |
|--------|-------|-----------|
| [K-Means](kmeans/) | K-Means Clustering | Centroid-based clustering |

---

## 🔧 Prerequisites

### Required Knowledge
- Python programming basics
- Basic mathematics (algebra, statistics)
- Understanding of NumPy and Pandas fundamentals

### Software Requirements
```
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
scipy >= 1.7.0
```

---

## 💻 Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd 100-days-of-machine-learning
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter scipy
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 🎓 Learning Path

### 🟢 Beginner Track (Weeks 1-3)
**Focus**: Data handling and basic analysis

1. **Week 1**: Data Acquisition
   - Days 15-18: CSV, JSON, SQL, Web Scraping
   
2. **Week 2**: Exploratory Data Analysis
   - Days 19-22: Statistics, Univariate, Bivariate Analysis
   
3. **Week 3**: Feature Scaling & Encoding
   - Days 24-28: Standardization, Normalization, Encoding

**Estimated Time**: 21-30 hours

### 🟡 Intermediate Track (Weeks 4-6)
**Focus**: Advanced preprocessing and feature engineering

4. **Week 4**: Pipelines & Transformations
   - Days 29-34: Pipelines, Custom transformers, Date handling
   
5. **Week 5**: Missing Data & Outliers
   - Days 35-40: Imputation techniques
   - Days 42-44: Outlier detection
   
6. **Week 6**: Feature Engineering
   - Day 45: Feature construction
   - Day 47: PCA

**Estimated Time**: 30-40 hours

### 🔴 Advanced Track (Weeks 7-10)
**Focus**: Machine learning algorithms

7. **Weeks 7-8**: Regression
   - Days 48-57: Linear, Polynomial, Regularized models
   
8. **Week 9**: Classification
   - Days 58-60: Logistic regression, Metrics
   
9. **Week 10**: Ensemble Methods
   - Days 65-68: Random Forest, AdaBoost, Stacking, Gradient Boosting

**Estimated Time**: 40-50 hours

---

## 📊 Key Mathematical Concepts

### Standardization (Z-Score)
$$z = \frac{x - \mu}{\sigma}$$

**When to use**: 
- Features on different scales
- Algorithms assuming normal distribution (SVM, Neural Networks)
- Before PCA

### Normalization (Min-Max Scaling)
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**When to use**:
- Need features in [0, 1] range
- Gradient descent optimization
- Image processing

### Box-Cox Transformation
$$x^{(\lambda)} = \begin{cases} 
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

**When to use**:
- Skewed distributions
- Making data more normal
- Stabilizing variance

### Ridge Regression (L2)
$$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$$

**When to use**:
- Multicollinearity present
- Want to keep all features
- Prevent overfitting

### Lasso Regression (L1)
$$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|$$

**When to use**:
- Feature selection needed
- Sparse solutions desired
- Many irrelevant features

### Logistic Regression (Sigmoid)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**When to use**:
- Binary classification
- Probability estimates needed
- Interpretable results required

### AdaBoost Weight Update
$$\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$$

$$w_{t+1}(i) = w_t(i) \cdot e^{-\alpha_t \cdot y_i \cdot h_t(x_i)}$$

**When to use**:
- Weak learners available
- Focus on hard-to-classify samples
- Reduce bias

### K-Means Objective
$$J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2$$

**When to use**:
- Customer segmentation
- Image compression
- Anomaly detection

---

## 🚀 Usage

### Running Individual Modules

```bash
# Navigate to specific day folder
cd day24-standardization

# Open Jupyter notebook
jupyter notebook day24.ipynb
```

### Using in Your Projects

```python
# Example: Creating a complete ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

---

## 📈 Progress Tracking

Use this checklist to track your learning:

### Data Manipulation
- [ ] Day 15: CSV Files
- [ ] Day 16: JSON & SQL
- [ ] Day 17: APIs
- [ ] Day 18: Web Scraping

### EDA
- [ ] Day 19: Descriptive Stats
- [ ] Day 20: Univariate Analysis
- [ ] Day 21: Bivariate Analysis
- [ ] Day 22: Pandas Profiling

### Preprocessing
- [ ] Day 24: Standardization
- [ ] Day 25: Normalization
- [ ] Day 26: Ordinal Encoding
- [ ] Day 27: One-Hot Encoding
- [ ] Day 28: Column Transformer
- [ ] Day 29: Pipelines
- [ ] Day 30: Function Transformer
- [ ] Day 31: Power Transformer
- [ ] Day 32: Binning
- [ ] Day 33: Mixed Variables
- [ ] Day 34: Date/Time

### Missing Data
- [ ] Day 35: Complete Case Analysis
- [ ] Day 36: Numerical Imputation
- [ ] Day 37: Categorical Imputation
- [ ] Day 38: Missing Indicator
- [ ] Day 39: KNN Imputer
- [ ] Day 40: Iterative Imputer

### Outliers
- [ ] Day 42: Z-Score
- [ ] Day 43: IQR Method
- [ ] Day 44: Percentiles

### Feature Engineering
- [ ] Day 45: Feature Construction
- [ ] Day 47: PCA

### Regression
- [ ] Day 48: Simple Linear
- [ ] Day 49: Metrics
- [ ] Day 50: Multiple Linear
- [ ] Day 51: Gradient Descent
- [ ] Day 52: GD Types
- [ ] Day 53: Polynomial
- [ ] Day 55: Ridge
- [ ] Day 56: Lasso
- [ ] Day 57: ElasticNet

### Classification
- [ ] Day 58: Logistic Regression
- [ ] Day 59: Metrics
- [ ] Day 60: Advanced Logistic

### Ensemble
- [ ] Day 65: Random Forest
- [ ] Day 66: AdaBoost
- [ ] Day 68: Stacking
- [ ] Gradient Boosting
- [ ] K-Means

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Scikit-learn documentation
- Andrew Ng's Machine Learning course
- Hands-On Machine Learning with Scikit-Learn
- Python Data Science Handbook

---

## 📧 Contact

For questions, suggestions, or discussions:
- Open an issue in the repository
- Submit a pull request
- Star ⭐ the repository if you find it helpful!

---

**Happy Learning! 🎉**

*Remember: Consistency is key. Even 30 minutes of focused learning daily compounds into expertise.*
