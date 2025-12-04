# 📑 Complete Module Index - 100 Days of Machine Learning

Quick reference index for all modules with direct links, formulas, and use cases.

---

## 📊 Data Acquisition & Manipulation

### Day 15: Working with CSV Files
- **Folder**: `day15 - working with csv files/`
- **Key Functions**: `pd.read_csv()`, `df.to_csv()`
- **Use Case**: Reading/writing tabular data
- **Best For**: Data import/export, Excel alternatives
- **Prerequisites**: Basic Python, Pandas

### Day 16: JSON & SQL
- **Folder**: `day16 - working-with-json-and-sql/`
- **Key Functions**: `pd.read_json()`, `pd.read_sql()`
- **Use Case**: API data, database queries
- **Best For**: Nested data structures, relational databases
- **Prerequisites**: Day 15

### Day 17: API to DataFrame
- **Folder**: `day17-api-to-dataframe/`
- **Key Libraries**: `requests`, `pandas`
- **Use Case**: Fetching real-time data from APIs
- **Best For**: Weather data, stock prices, social media
- **Prerequisites**: Day 16, HTTP basics

### Day 18: Web Scraping
- **Folder**: `day18-pandas-dataframe-using-web-scraping/`
- **Key Libraries**: `BeautifulSoup`, `requests`
- **Use Case**: Extracting data from websites
- **Best For**: When no API exists
- **Prerequisites**: HTML basics, Day 17

---

## 📈 Exploratory Data Analysis

### Day 19: Descriptive Statistics
- **Folder**: `day19-understanding-your-data-descriptive-stats/`
- **Key Functions**: `df.describe()`, `df.info()`, `df.corr()`
- **Metrics**: Mean, median, std, min, max, quartiles
- **Use Case**: Understanding dataset characteristics
- **Best For**: Initial data exploration
- **Prerequisites**: Day 15, basic statistics

### Day 20: Univariate Analysis
- **Folder**: `day20-univariate-analysis/`
- **Key Plots**: Histogram, box plot, density plot
- **Metrics**: Skewness, kurtosis
- **Use Case**: Analyzing single variables
- **Best For**: Distribution understanding, outlier detection
- **Prerequisites**: Day 19, Matplotlib

### Day 21: Bivariate Analysis
- **Folder**: `day21-bivariate-analysis/`
- **Key Plots**: Scatter plot, correlation heatmap
- **Metrics**: Pearson correlation, Spearman correlation
- **Use Case**: Relationship between two variables
- **Best For**: Feature selection, correlation analysis
- **Prerequisites**: Day 20

### Day 22: Pandas Profiling
- **Folder**: `day22-pandas-profiling/`
- **Key Function**: `ProfileReport()`
- **Use Case**: Automated comprehensive EDA
- **Best For**: Quick dataset overview, reporting
- **Prerequisites**: Day 19-21

---

## ⚖️ Feature Scaling

### Day 24: Standardization
- **Folder**: `day24-standardization/`
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Result**: Mean = 0, Std = 1
- **Use Case**: SVM, Neural Networks, PCA
- **Best For**: Gaussian-distributed data
- **Pros**: Handles outliers better than normalization
- **Cons**: Doesn't bound values
- **Prerequisites**: Day 19, statistics

### Day 25: Normalization
- **Folder**: `day25-normalization/`
- **Formula**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **Result**: Range [0, 1]
- **Use Case**: Neural Networks, K-NN, image processing
- **Best For**: Non-Gaussian data, bounded inputs needed
- **Pros**: Fixed range
- **Cons**: Sensitive to outliers
- **Prerequisites**: Day 24

---

## 🏷️ Encoding Techniques

### Day 26: Ordinal Encoding
- **Folder**: `day26-ordinal-encoding/`
- **Method**: Map categories to integers with order
- **Example**: Low=1, Medium=2, High=3
- **Use Case**: Education level, satisfaction rating
- **Best For**: Ordered categorical variables
- **Prerequisites**: Basic categorical data understanding

### Day 27: One-Hot Encoding
- **Folder**: `day27-one-hot-encoding/`
- **Method**: Create binary column per category
- **Example**: Color → Color_Red, Color_Blue, Color_Green
- **Use Case**: Country, color, brand names
- **Best For**: Nominal categorical variables
- **Caution**: High cardinality can create many columns
- **Prerequisites**: Day 26

### Day 28: Column Transformer
- **Folder**: `day28-column-transformer/`
- **Purpose**: Apply different transformations to different columns
- **Use Case**: Mixed data types (numerical + categorical)
- **Best For**: Complex preprocessing pipelines
- **Prerequisites**: Day 24-27

---

## 🔧 Advanced Preprocessing

### Day 29: Sklearn Pipelines
- **Folder**: `day29-sklearn-pipelines/`
- **Purpose**: Automate ML workflows
- **Benefits**: Prevent data leakage, reproducibility
- **Use Case**: Production ML models
- **Best For**: Complex preprocessing sequences
- **Prerequisites**: Day 28, sklearn basics

### Day 30: Function Transformer
- **Folder**: `day30-function-transformer/`
- **Purpose**: Custom transformations in pipelines
- **Use Case**: Domain-specific feature engineering
- **Best For**: Non-standard transformations
- **Prerequisites**: Day 29

### Day 31: Power Transformer
- **Folder**: `day31-power-transformer/`
- **Methods**: Box-Cox, Yeo-Johnson
- **Formula**: $x^{(\lambda)} = \frac{x^\lambda - 1}{\lambda}$ (Box-Cox)
- **Use Case**: Making skewed data more Gaussian
- **Best For**: Highly skewed distributions
- **Prerequisites**: Day 25, understanding of distributions

### Day 32: Binning & Binarization
- **Folder**: `day32-binning-and-binarization/`
- **Binning**: Continuous → Discrete (Age → Age Groups)
- **Binarization**: Threshold-based (>5 → 1, ≤5 → 0)
- **Use Case**: Simplifying continuous variables
- **Best For**: Non-linear relationships
- **Prerequisites**: Day 20

### Day 33: Handling Mixed Variables
- **Folder**: `day33-handling-mixed-variables/`
- **Definition**: Variables with both numerical and categorical values
- **Example**: "Height: 5.8, tall, short"
- **Use Case**: Messy real-world data
- **Best For**: Data cleaning
- **Prerequisites**: Day 26-27

### Day 34: Date & Time Features
- **Folder**: `day34-handling-date-and-time/`
- **Features**: Year, month, day, hour, day_of_week, is_weekend
- **Use Case**: Time series, temporal patterns
- **Best For**: Sales data, web traffic, events
- **Prerequisites**: Python datetime

---

## 🔍 Missing Data Handling

### Day 35: Complete Case Analysis
- **Folder**: `day35-complete-case-analysis/`
- **Method**: Remove rows with missing values
- **Function**: `df.dropna()`
- **Use Case**: <5% missing data
- **Pros**: Simple, preserves distribution
- **Cons**: Loses data, may introduce bias
- **Prerequisites**: Day 19

### Day 36: Numerical Imputation
- **Folder**: `day36-imputing-numerical-data/`
- **Methods**: Mean, median, mode, arbitrary value
- **Use Case**: Missing numerical values
- **Best For**: MCAR (Missing Completely At Random)
- **Prerequisites**: Day 35

### Day 37: Categorical Imputation
- **Folder**: `day37-handling-missing-categorical-data/`
- **Methods**: Most frequent, missing category
- **Use Case**: Missing categorical values
- **Best For**: Nominal categories
- **Prerequisites**: Day 36

### Day 38: Missing Indicator
- **Folder**: `day38-missing-indicator/`
- **Method**: Create binary flag for missingness
- **Use Case**: When missingness is informative
- **Example**: Income_missing → 1 (didn't want to share)
- **Prerequisites**: Day 36-37

### Day 39: KNN Imputer
- **Folder**: `day39-knn-imputer/`
- **Method**: Use K-nearest neighbors to impute
- **Use Case**: Missing values with patterns
- **Best For**: MAR (Missing At Random)
- **Pros**: Considers relationships
- **Cons**: Computationally expensive
- **Prerequisites**: Day 38, K-NN understanding

### Day 40: Iterative Imputer
- **Folder**: `day40-iterative-imputer/`
- **Method**: MICE (Multiple Imputation by Chained Equations)
- **Use Case**: Multiple missing features
- **Best For**: Complex missingness patterns
- **Pros**: Most sophisticated
- **Cons**: Slowest, most complex
- **Prerequisites**: Day 39, regression basics

---

## 🎯 Outlier Detection & Treatment

### Day 42: Z-Score Method
- **Folder**: `day42-outlier-removal-using-zscore/`
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Threshold**: |z| > 3 → outlier
- **Use Case**: Gaussian distributed data
- **Best For**: Symmetric distributions
- **Prerequisites**: Day 24, statistics

### Day 43: IQR Method
- **Folder**: `day43-outlier-removal-using-iqr-method/`
- **Formula**: $IQR = Q_3 - Q_1$
- **Threshold**: < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- **Use Case**: Skewed distributions
- **Best For**: Non-Gaussian data (robust method)
- **Prerequisites**: Day 20, quartiles

### Day 44: Percentile Method
- **Folder**: `day44-outlier-detection-using-percentiles/`
- **Method**: Remove top/bottom X percentile
- **Example**: Remove <1st and >99th percentile
- **Use Case**: When you know acceptable range
- **Best For**: Domain-specific outliers
- **Prerequisites**: Day 43

---

## 🛠️ Feature Engineering

### Day 45: Feature Construction & Splitting
- **Folder**: `day45-feature-construction-and-feature-splitting/`
- **Construction**: Combine features (BMI = weight/height²)
- **Splitting**: Parse complex features (Name → FirstName + LastName)
- **Use Case**: Creating domain-specific features
- **Best For**: Improving model performance
- **Prerequisites**: Day 34, domain knowledge

### Day 47: PCA (Principal Component Analysis)
- **Folder**: `day47-pca/`
- **Purpose**: Dimensionality reduction
- **Method**: Find principal components (directions of max variance)
- **Use Case**: High-dimensional data, visualization
- **Best For**: Removing multicollinearity
- **Pros**: Reduces features, removes correlation
- **Cons**: Loses interpretability
- **Prerequisites**: Day 24, linear algebra

---

## 📉 Regression Algorithms

### Day 48: Simple Linear Regression
- **Folder**: `day48-simple-linear-regression/`
- **Formula**: $y = mx + b$
- **Method**: Ordinary Least Squares (OLS)
- **Use Case**: Predicting continuous values
- **Best For**: Linear relationships
- **Prerequisites**: Basic algebra

### Day 49: Regression Metrics
- **Folder**: `day49-regression-metrics/`
- **Metrics**: MSE, RMSE, MAE, R², Adjusted R²
- **MSE**: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
- **R²**: $1 - \frac{SS_{res}}{SS_{tot}}$
- **Use Case**: Model evaluation
- **Best For**: Comparing regression models
- **Prerequisites**: Day 48

### Day 50: Multiple Linear Regression
- **Folder**: `day50-multiple-linear-regression/`
- **Formula**: $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
- **Use Case**: Multiple predictors
- **Best For**: Understanding feature importance
- **Prerequisites**: Day 48-49

### Day 51: Gradient Descent
- **Folder**: `day51-gradient-descent/`
- **Formula**: $\theta := \theta - \alpha \nabla J(\theta)$
- **Purpose**: Optimization algorithm
- **Use Case**: Training ML models
- **Best For**: Large datasets
- **Prerequisites**: Day 50, calculus basics

### Day 52: Types of Gradient Descent
- **Folder**: `day52-types-of-gradient-descent/`
- **Types**: Batch, Stochastic (SGD), Mini-batch
- **Batch**: Uses all data
- **SGD**: Uses one sample at a time
- **Mini-batch**: Uses small batches
- **Best For**: Different dataset sizes
- **Prerequisites**: Day 51

### Day 53: Polynomial Regression
- **Folder**: `day53-polynomial-regression/`
- **Formula**: $y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + ...$
- **Use Case**: Non-linear relationships
- **Best For**: Curved patterns
- **Caution**: Overfitting with high degrees
- **Prerequisites**: Day 50

### Day 55: Ridge Regression (L2)
- **Folder**: `day55-regularized-linear-models/`
- **Formula**: $J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$
- **Purpose**: Prevent overfitting
- **Use Case**: Multicollinearity
- **Best For**: Keeping all features
- **Prerequisites**: Day 53

### Day 56: Lasso Regression (L1)
- **Folder**: `day56-lasso-regression/`
- **Formula**: $J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|$
- **Purpose**: Feature selection + regularization
- **Use Case**: Many irrelevant features
- **Best For**: Sparse solutions
- **Prerequisites**: Day 55

### Day 57: ElasticNet Regression
- **Folder**: `day57-elasticnet-regression/`
- **Formula**: Combines L1 + L2 penalties
- **Purpose**: Best of Ridge + Lasso
- **Use Case**: Many correlated features
- **Best For**: Balanced approach
- **Prerequisites**: Day 55-56

---

## 📊 Classification Algorithms

### Day 58: Logistic Regression
- **Folder**: `day58-logistic-regression/`
- **Formula**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Purpose**: Binary classification
- **Use Case**: Yes/No, Pass/Fail predictions
- **Best For**: Probabilistic predictions
- **Prerequisites**: Day 48, sigmoid function

### Day 59: Classification Metrics
- **Folder**: `day59-classification-metrics/`
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Use Case**: Model evaluation
- **Best For**: Imbalanced datasets
- **Prerequisites**: Day 58, confusion matrix

### Day 60: Advanced Logistic Regression
- **Folder**: `day60-logistic-regression-contd/`
- **Topics**: Softmax, multi-class classification
- **Use Case**: More than 2 classes
- **Best For**: Multi-class problems
- **Prerequisites**: Day 58-59

---

## 🌲 Ensemble Methods

### Day 65: Random Forest
- **Folder**: `day65-random-forest/`
- **Method**: Bagging + Feature randomness
- **Components**: Multiple decision trees
- **Use Case**: General-purpose classification/regression
- **Best For**: Reducing overfitting
- **Pros**: Handles non-linearity, feature importance
- **Prerequisites**: Decision trees, bagging concept

### Day 66: AdaBoost
- **Folder**: `day66-adaboost/`
- **Method**: Adaptive Boosting (sequential)
- **Formula**: $\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
- **Use Case**: Weak learners → Strong learner
- **Best For**: Reducing bias
- **Prerequisites**: Day 65, boosting concept

### Day 68: Stacking & Blending
- **Folder**: `day68-stacking-and-blending/`
- **Method**: Meta-learning from base models
- **Stacking**: Uses cross-validation
- **Blending**: Uses holdout validation
- **Use Case**: Kaggle competitions, maximum performance
- **Best For**: Combining diverse models
- **Prerequisites**: Day 65-66

### Gradient Boosting
- **Folder**: `gradient-boosting/`
- **Method**: Sequential error correction
- **Use Case**: High-performance predictions
- **Best For**: Structured data competitions
- **Pros**: State-of-the-art performance
- **Prerequisites**: Day 66, gradients

---

## 🔄 Unsupervised Learning

### K-Means Clustering
- **Folder**: `kmeans/`
- **Method**: Centroid-based clustering
- **Formula**: $J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2$
- **Use Case**: Customer segmentation, image compression
- **Best For**: Spherical clusters
- **Prerequisites**: Day 24, distance metrics

---

## 📚 Supplementary Materials

### Notes
- **Folder**: `notes/`
- **Contents**: Study guides, quick references, formulas
- **Use Case**: Quick lookups, revision
- **Best For**: Interview prep, concept review

---

## 🎯 Quick Decision Tree

**Need to:**
- Load data? → Days 15-18
- Understand data? → Days 19-22
- Scale features? → Days 24-25
- Encode categories? → Days 26-28
- Build pipelines? → Days 29-31
- Handle missing data? → Days 35-40
- Detect outliers? → Days 42-44
- Engineer features? → Days 45, 47
- Predict continuous values? → Days 48-57
- Predict categories? → Days 58-60
- Boost performance? → Days 65-68
- Group similar data? → K-Means

---

**Total Modules**: 45+  
**Total Documentation Files**: 50  
**Coverage**: 100%

**Start your journey**: Open `README.md` or `QUICK_START.md`!
