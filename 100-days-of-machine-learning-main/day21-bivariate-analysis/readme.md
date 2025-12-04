# Bivariate Analysis

## General Idea

Bivariate analysis examines the relationship between two variables simultaneously to understand how they interact, correlate, or influence each other. Unlike univariate analysis which focuses on individual features, bivariate analysis reveals patterns, dependencies, and associations that are crucial for feature selection, understanding causality, and building effective machine learning models.

## Why Use Bivariate Analysis?

1. **Relationship Discovery**: Identify correlations and dependencies between variables
2. **Feature Selection**: Find features most related to target variable
3. **Multicollinearity Detection**: Identify redundant features (highly correlated predictors)
4. **Interaction Effects**: Discover how variables combine to influence outcomes
5. **Causality Investigation**: Explore potential causal relationships (though correlation ≠ causation)
6. **Model Building**: Understand which features to include, transform, or engineer
7. **Assumption Validation**: Check linearity, independence assumptions

## Role in Machine Learning

### Feature Engineering

- **Feature Selection**: Keep features correlated with target, remove redundant ones
- **Interaction Features**: Create products or combinations of correlated features
- **Dimensionality Reduction**: Identify features that provide similar information
- **Target Encoding**: Use target-feature relationships for categorical encoding

### Model Selection

- **Linear Models**: Require understanding of feature-target linearity
- **Tree-Based Models**: Benefit from knowing feature interactions
- **Regularization**: Multicollinearity affects Ridge/Lasso performance
- **Feature Importance**: Bivariate relationships guide interpretation

### Data Quality

- **Leakage Detection**: Features too correlated with target may indicate leakage
- **Confounding Variables**: Identify spurious correlations
- **Redundancy**: Remove duplicate or highly correlated features

## Types of Variable Pairs

### 1. Continuous vs Continuous

**Methods**:
- Scatter plots
- Correlation coefficients (Pearson, Spearman, Kendall)
- Regression analysis
- Hexbin plots (for large datasets)
- Contour plots

### 2. Categorical vs Categorical

**Methods**:
- Cross-tabulation (contingency tables)
- Chi-square test
- Cramér's V
- Stacked bar charts
- Mosaic plots
- Heat maps

### 3. Continuous vs Categorical

**Methods**:
- Box plots by category
- Violin plots
- Strip plots
- T-test / ANOVA
- Point-biserial correlation
- Mean/median comparison

## Continuous vs Continuous Analysis

### Scatter Plot

**Purpose**: Visualize relationship between two continuous variables

**Components**:
- **x-axis**: Independent variable (predictor)
- **y-axis**: Dependent variable (response)
- **Points**: Individual observations

**Patterns to Identify**:
- **Linear**: Straight-line relationship
- **Non-linear**: Curved relationship (polynomial, exponential, logarithmic)
- **No relationship**: Random scatter
- **Clusters**: Distinct groups
- **Outliers**: Points far from main pattern

**Enhancements**:
- **Color**: Add third variable (categorical)
- **Size**: Add fourth variable (continuous)
- **Trend line**: Show fitted relationship
- **Confidence interval**: Show uncertainty

### Correlation Coefficients

#### Pearson Correlation (r)

**Formula**:
$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Or equivalently:
$$r = \frac{Cov(X, Y)}{s_X \cdot s_Y}$$

**Properties**:
- **Range**: $-1 \leq r \leq 1$
- **Sign**: Positive (increase together) or negative (inverse relationship)
- **Magnitude**: Strength of linear relationship
- **Symmetric**: $r_{XY} = r_{YX}$
- **Units**: Dimensionless (scale-invariant)

**Interpretation**:
- $r = 1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship
- $r = 0$: No linear relationship (may still have non-linear)
- $0 < |r| < 0.3$: Weak correlation
- $0.3 \leq |r| < 0.7$: Moderate correlation
- $0.7 \leq |r| \leq 1$: Strong correlation

**Assumptions**:
- **Linearity**: Relationship is linear
- **Normality**: Both variables approximately normal (for significance testing)
- **Homoscedasticity**: Constant variance across range
- **No outliers**: Sensitive to extreme values

**Coefficient of Determination**: $R^2 = r^2$
- Proportion of variance in $Y$ explained by $X$
- Range: [0, 1]
- Example: $r = 0.8$ → $R^2 = 0.64$ (64% variance explained)

#### Spearman Rank Correlation (ρ)

**Formula**:
$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2 - 1)}$$

Where $d_i$ is the difference in ranks for observation $i$

**Alternatively**: Pearson correlation of rank-transformed data

**Properties**:
- **Range**: $-1 \leq \rho \leq 1$
- **Non-parametric**: No normality assumption
- **Monotonic**: Captures monotonic (not just linear) relationships
- **Robust**: Less sensitive to outliers than Pearson

**When to Use**:
- Non-linear but monotonic relationships
- Ordinal data
- Presence of outliers
- Non-normal distributions

**Example**: Hours studied vs exam score (non-linear but monotonic increase)

#### Kendall Tau Correlation (τ)

**Formula**:
$$\tau = \frac{n_c - n_d}{\frac{1}{2}n(n-1)}$$

Where:
- $n_c$: Number of concordant pairs
- $n_d$: Number of discordant pairs

**Concordant pair**: $(x_i, y_i)$ and $(x_j, y_j)$ where $(x_i - x_j)(y_i - y_j) > 0$

**Properties**:
- **Range**: $-1 \leq \tau \leq 1$
- **Interpretation**: Probability of concordance minus probability of discordance
- **More robust**: Than Spearman for small samples
- **Computationally expensive**: $O(n^2)$ vs $O(n \log n)$ for Spearman

**When to Use**:
- Small sample sizes
- Many tied ranks
- Prefer probabilistic interpretation

### Correlation Matrix

**Heatmap of pairwise correlations**:

$$\text{CorrMatrix} = \begin{bmatrix}
1 & r_{12} & r_{13} & \cdots & r_{1p}\\
r_{21} & 1 & r_{23} & \cdots & r_{2p}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
r_{p1} & r_{p2} & r_{p3} & \cdots & 1
\end{bmatrix}$$

**Diagonal**: Always 1 (perfect self-correlation)
**Symmetric**: $r_{ij} = r_{ji}$

**Visualization**:
- Color intensity: Strength of correlation
- Diverging colormap: Positive (red) vs negative (blue)
- Identify blocks of correlated features

**Uses**:
- **Feature selection**: Remove one from highly correlated pairs
- **Multicollinearity**: Identify problematic correlations for linear models
- **Feature grouping**: Cluster related features

**Multicollinearity Threshold**: $|r| > 0.8$ or $0.9$ is problematic

### Linear Regression

**Simple Linear Regression**: Model $Y$ as linear function of $X$

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
- $\beta_0$: Intercept
- $\beta_1$: Slope
- $\epsilon$: Error term

**Least Squares Estimation**:

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = r \cdot \frac{s_y}{s_x}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

**Interpretation**:
- $\beta_1$: Change in $Y$ for unit increase in $X$
- $\beta_0$: Expected $Y$ when $X = 0$

**Goodness of Fit**:
- $R^2$: Proportion of variance explained
- RMSE: Root mean squared error
- Residual plots: Check assumptions

**Assumptions**:
1. **Linearity**: Relationship is linear
2. **Independence**: Observations independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals normally distributed

## Categorical vs Categorical Analysis

### Cross-Tabulation (Contingency Table)

**Purpose**: Show frequency distribution of two categorical variables

**Example**: Gender vs Purchase Decision

|        | Purchase | No Purchase | Total |
|--------|----------|-------------|-------|
| Male   | 45       | 55          | 100   |
| Female | 60       | 40          | 100   |
| Total  | 105      | 95          | 200   |

**Row Percentages**: Proportion within row
**Column Percentages**: Proportion within column
**Total Percentages**: Proportion of grand total

**Marginal Distributions**: Row and column totals

### Chi-Square Test of Independence

**Null Hypothesis**: Variables are independent (no association)

**Test Statistic**:
$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where:
- $O_{ij}$: Observed frequency in cell $(i,j)$
- $E_{ij}$: Expected frequency under independence
- $r$: Number of rows
- $c$: Number of columns

**Expected Frequency**:
$$E_{ij} = \frac{(\text{Row}_i \text{ total}) \times (\text{Column}_j \text{ total})}{\text{Grand total}}$$

**Degrees of Freedom**:
$$df = (r - 1)(c - 1)$$

**Decision Rule**:
- If $p < 0.05$: Reject independence (variables are associated)
- If $p \geq 0.05$: Fail to reject (insufficient evidence of association)

**Assumptions**:
- Expected frequencies $\geq 5$ in each cell
- Independent observations
- Mutually exclusive categories

### Cramér's V

**Purpose**: Measure strength of association for categorical variables

**Formula**:
$$V = \sqrt{\frac{\chi^2}{n \times \min(r-1, c-1)}}$$

Where:
- $\chi^2$: Chi-square statistic
- $n$: Total sample size
- $r, c$: Number of rows, columns

**Range**: [0, 1]
- 0: No association
- 1: Perfect association

**Interpretation** (for 2×2 tables):
- $V < 0.1$: Weak association
- $0.1 \leq V < 0.3$: Moderate association
- $V \geq 0.3$: Strong association

**Advantage**: Scale-invariant, unlike $\chi^2$ which increases with sample size

### Visualization

#### Stacked Bar Chart

**Purpose**: Show distribution of one categorical variable across levels of another

**Components**:
- **Bars**: Each level of first variable
- **Segments**: Proportions of second variable
- **100% stacked**: Normalize to show percentages

**Use**: Compare proportions across categories

#### Mosaic Plot

**Purpose**: Visualize contingency table with area proportional to frequency

**Components**:
- **Tile area**: Proportional to cell frequency
- **Tile width**: Proportional to row marginal
- **Tile height**: Proportional to column conditional

**Use**: Identify patterns in high-dimensional contingency tables

#### Heat Map

**Purpose**: Color-coded matrix of frequencies or proportions

**Components**:
- **Rows**: Levels of first variable
- **Columns**: Levels of second variable
- **Color intensity**: Frequency or proportion

**Use**: Quickly spot patterns and strong associations

## Continuous vs Categorical Analysis

### Box Plot by Category

**Purpose**: Compare distribution of continuous variable across categorical groups

**Components**:
- **Multiple box plots**: One per category
- **Common y-axis**: Continuous variable
- **x-axis**: Categories

**Comparison Points**:
- **Median**: Central tendency per group
- **IQR**: Spread per group
- **Outliers**: Extreme values per group
- **Overlap**: How much distributions overlap

**Interpretation**:
- **Non-overlapping boxes**: Significant difference likely
- **Similar medians**: Groups similar on average
- **Different spread**: Groups vary in variability

### Violin Plot

**Purpose**: Combine box plot with kernel density estimation

**Components**:
- **Width**: Density at each value
- **Box plot**: Inside violin (optional)
- **Median/quartiles**: Marked

**Advantage**: Shows full distribution shape, not just summary statistics

### Statistical Tests

#### Independent Samples t-Test

**Purpose**: Compare means of two independent groups

**Null Hypothesis**: $\mu_1 = \mu_2$ (group means are equal)

**Test Statistic**:
$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

Where $s_p$ is pooled standard deviation:
$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

**Assumptions**:
- **Independence**: Samples independent
- **Normality**: Both groups approximately normal
- **Equal variance**: Homogeneity of variance (use Welch's t-test if violated)

**Interpretation**:
- $p < 0.05$: Means significantly different
- Effect size (Cohen's d): $d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$

#### Analysis of Variance (ANOVA)

**Purpose**: Compare means of three or more groups

**Null Hypothesis**: $\mu_1 = \mu_2 = ... = \mu_k$ (all group means equal)

**Test Statistic**:
$$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between}/(k-1)}{SS_{within}/(N-k)}$$

Where:
- $SS_{between}$: Sum of squares between groups
- $SS_{within}$: Sum of squares within groups
- $k$: Number of groups
- $N$: Total sample size

**Between-Group Variance**:
$$SS_{between} = \sum_{i=1}^{k}n_i(\bar{x}_i - \bar{x})^2$$

**Within-Group Variance**:
$$SS_{within} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(x_{ij} - \bar{x}_i)^2$$

**Interpretation**:
- $p < 0.05$: At least one group mean differs
- Post-hoc tests (Tukey HSD, Bonferroni): Identify which pairs differ

**Assumptions**:
- **Independence**: Samples independent
- **Normality**: Each group approximately normal
- **Equal variance**: Homogeneity across groups

**Effect Size** (Eta-squared):
$$\eta^2 = \frac{SS_{between}}{SS_{total}}$$

Proportion of variance explained by grouping

#### Point-Biserial Correlation

**Purpose**: Correlation between continuous and binary categorical variable

**Formula**: Special case of Pearson correlation where one variable is binary (0/1)

$$r_{pb} = \frac{\bar{x}_1 - \bar{x}_0}{s_n}\sqrt{\frac{n_0n_1}{n^2}}$$

Where:
- $\bar{x}_1, \bar{x}_0$: Means of continuous variable for groups 1 and 0
- $s_n$: Standard deviation of continuous variable
- $n_0, n_1$: Sample sizes of groups

**Range**: [-1, 1]

**Interpretation**: Same as Pearson correlation

## Advanced Visualizations

### Pair Plot (Scatter Plot Matrix)

**Purpose**: Visualize all pairwise relationships in dataset

**Components**:
- **Off-diagonal**: Scatter plots of variable pairs
- **Diagonal**: Histograms or KDE of individual variables
- **Color**: Can add categorical variable

**Use**: Quickly scan for correlations, patterns, outliers

**Limitation**: Becomes cluttered with many variables (> 10)

### Joint Plot

**Purpose**: Combine scatter plot with marginal distributions

**Components**:
- **Center**: Scatter plot of two variables
- **Top margin**: Histogram/KDE of x-variable
- **Right margin**: Histogram/KDE of y-variable
- **Optional**: Regression line, density contours

**Use**: See bivariate and univariate distributions simultaneously

### Hexbin Plot

**Purpose**: Alternative to scatter plot for large datasets

**Mechanism**: Divide plane into hexagons, color by density

**Advantage**: Addresses overplotting (overlapping points)

**Use**: Millions of points where scatter plot is saturated

## Handling Non-Linear Relationships

### Transformation

If relationship is non-linear, transform variables:

**Common transformations**:
- **Log**: $\log(x), \log(y)$, or both
- **Square root**: $\sqrt{x}$
- **Polynomial**: $x^2, x^3$
- **Reciprocal**: $1/x$

**Check linearity** after transformation (scatter plot, correlation)

### Polynomial Regression

**Model**: $y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \beta_px^p + \epsilon$

**Use**: Capture curves, U-shapes, S-shapes

**Trade-off**: Higher degree = more flexibility but risk of overfitting

### Spline Regression

**Piecewise polynomial**: Different polynomials in different regions

**Knots**: Points where pieces connect

**Use**: Flexible fitting without high-degree polynomials

## Practical Considerations

### Sample Size

**Small samples** (n < 30):
- Correlation may be unreliable
- Use exact tests (Fisher's exact for contingency tables)
- Visual inspection crucial

**Large samples** (n > 1000):
- Weak correlations may be "significant" but not meaningful
- Focus on effect size, not just p-value
- Computational efficiency matters

### Outliers

**Impact on correlation**:
- Pearson correlation very sensitive
- One outlier can change from strong to weak (or vice versa)

**Detection**:
- Visual: Scatter plot
- Statistical: Leverage, Cook's distance

**Handling**:
- Remove if error
- Transform if valid extreme value
- Use robust methods (Spearman)

### Causation vs Correlation

**Correlation does NOT imply causation**

**Spurious correlation**: Two variables correlated due to:
- **Confounding variable**: Third variable causes both
- **Reverse causation**: Y causes X (not X causes Y)
- **Chance**: Random coincidence

**Establishing causality requires**:
- Temporal precedence: Cause before effect
- Covariation: Correlation exists
- Elimination of confounds: Control for alternative explanations
- Mechanism: Plausible causal pathway

## Summary

Bivariate analysis reveals relationships between pairs of variables, essential for understanding data structure before building machine learning models.

**Key Techniques**:

**Continuous vs Continuous**:
- Scatter plots, correlation (Pearson, Spearman, Kendall)
- Linear regression for modeling
- Check linearity, transform if needed

**Categorical vs Categorical**:
- Contingency tables, chi-square test
- Cramér's V for association strength
- Stacked bar charts, mosaic plots

**Continuous vs Categorical**:
- Box plots, violin plots
- t-test (2 groups), ANOVA (3+ groups)
- Point-biserial correlation

**Important Principles**:
- Visualize relationships before quantifying
- Choose appropriate method based on variable types
- Consider non-linear relationships
- Be wary of outliers
- Remember: Correlation ≠ Causation
- Report effect sizes, not just significance

Bivariate analysis bridges univariate analysis and multivariate modeling, providing insights into feature relationships that guide feature engineering, selection, and model interpretation in machine learning workflows.

---

**Video link**: https://www.youtube.com/watch?v=4HyTlbHUKSw
