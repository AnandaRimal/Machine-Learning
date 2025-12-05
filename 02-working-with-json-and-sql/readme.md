# Working with JSON and SQL

## General Idea

JSON (JavaScript Object Notation) and SQL (Structured Query Language) databases represent two fundamental paradigms for data storage and retrieval in modern data science. JSON handles semi-structured, hierarchical data with flexibility, while SQL databases provide structured, relational data with strong consistency guarantees.

## Why Use JSON and SQL?

### JSON Benefits

1. **Hierarchical Structure**: Naturally represents nested and complex data
2. **Schema Flexibility**: No fixed schema required, adaptable to changes
3. **Human Readable**: Easy to read and debug
4. **Web Native**: Standard format for APIs and web services
5. **Language Agnostic**: Supported across all programming languages

### SQL Benefits

1. **Data Integrity**: ACID properties ensure consistency
2. **Relationships**: Efficiently model connections between entities
3. **Query Power**: Complex queries with joins, aggregations, filtering
4. **Standardization**: Consistent SQL syntax across databases
5. **Scalability**: Optimized for large-scale data operations
6. **Concurrent Access**: Multi-user access with transaction management

## Role in Machine Learning

### JSON in ML

- **API Data**: Most web APIs return data in JSON format
- **Configuration Files**: Store model hyperparameters and settings
- **Semi-structured Data**: Handle variable-length features and nested attributes
- **NoSQL Databases**: MongoDB, CouchDB store documents as JSON
- **Model Serialization**: Save model architectures (especially in deep learning)
- **Experiment Tracking**: Log training metrics and parameters

### SQL in ML

- **Feature Store**: Central repository for curated features
- **Data Warehousing**: Store and manage large training datasets
- **Data Versioning**: Track data changes over time
- **ETL Pipelines**: Extract, transform, load data for ML workflows
- **Production Data**: Real-time feature serving for predictions
- **Metadata Management**: Track model versions, experiments, datasets

## JSON Structure and Concepts

### Basic Data Types

1. **Object**: Key-value pairs enclosed in `{}`
2. **Array**: Ordered list of values enclosed in `[]`
3. **String**: Text in double quotes
4. **Number**: Integer or floating-point
5. **Boolean**: `true` or `false`
6. **Null**: Represents absence of value

### Hierarchical Structure Example

```json
{
  "customer": {
    "id": 12345,
    "name": "John Doe",
    "orders": [
      {"order_id": 1, "total": 150.50},
      {"order_id": 2, "total": 89.99}
    ]
  }
}
```

**Nesting Depth**: JSON supports arbitrary nesting, but deep nesting (>5 levels) impacts:
- Parsing performance: $O(d)$ where $d$ is depth
- Memory usage: Proportional to depth
- Query complexity: Navigation requires path traversal

### Schema vs. Schema-less

**Schema-less (JSON)**:
- Documents can have different structures
- Fields can be added/removed dynamically
- Trade-off: Less validation, potential inconsistency

**Schema-enforced (JSON Schema)**:
- Define expected structure and types
- Validation before processing
- Better data quality assurance

## SQL Database Concepts

### Relational Model

Based on mathematical set theory and relational algebra:

**Relation (Table)**: Set of tuples (rows) with named attributes (columns)

**Formal Definition**: $R \subseteq D_1 \times D_2 \times ... \times D_n$

Where $D_i$ is the domain of attribute $i$

### Key Concepts

#### 1. Primary Key

Unique identifier for each row:
- **Uniqueness**: No two rows share the same primary key
- **Non-null**: Primary key cannot be NULL
- **Immutability**: Should not change over time

**Mathematical Property**: Injective function $f: R \rightarrow K$ where $K$ is the key space

#### 2. Foreign Key

References primary key in another table:
- **Referential Integrity**: Foreign key must exist in referenced table
- **Cascading**: Updates/deletes can propagate

**Relationship**: $FK_A \subseteq PK_B$ (foreign key in A is subset of primary key in B)

#### 3. Normalization

Process of organizing data to reduce redundancy:

**First Normal Form (1NF)**:
- Atomic values (no multi-valued attributes)
- Each row is unique

**Second Normal Form (2NF)**:
- 1NF + No partial dependencies
- Non-key attributes fully dependent on primary key

**Third Normal Form (3NF)**:
- 2NF + No transitive dependencies
- Non-key attributes independent of each other

**Boyce-Codd Normal Form (BCNF)**:
- Stronger version of 3NF
- Every determinant is a candidate key

### ACID Properties

**Atomicity**: Transaction is all-or-nothing
- Either all operations complete or none do
- Mathematical: Indivisible operation unit

**Consistency**: Database remains in valid state
- All constraints satisfied before and after transaction
- Invariants preserved

**Isolation**: Concurrent transactions don't interfere
- Serializability: Result equivalent to serial execution
- Isolation levels: Read uncommitted, Read committed, Repeatable read, Serializable

**Durability**: Committed changes persist
- Survive system failures
- Write-ahead logging ensures recoverability

## SQL Query Language

### Relational Algebra Operations

**Selection (σ)**: Filter rows based on condition
$$\sigma_{condition}(R)$$

**Projection (π)**: Select specific columns
$$\pi_{columns}(R)$$

**Join (⋈)**: Combine tables based on condition
$$R \bowtie_{condition} S$$

**Union (∪)**: Combine result sets (removes duplicates)
$$R \cup S$$

**Intersection (∩)**: Common rows in both sets
$$R \cap S$$

**Difference (−)**: Rows in R but not in S
$$R - S$$

### Join Types

#### Inner Join

Returns only matching rows from both tables:
$$R \bowtie S = \{(r, s) | r \in R, s \in S, r.key = s.key\}$$

**Cardinality**: At most $|R| \times |S|$, typically much less

#### Left Outer Join

All rows from left table + matching from right (NULL for non-matches):
$$R ⟕ S$$

#### Right Outer Join

All rows from right table + matching from left:
$$R ⟖ S$$

#### Full Outer Join

All rows from both tables:
$$R ⟗ S$$

#### Cross Join

Cartesian product of all rows:
$$R \times S$$

Cardinality: Exactly $|R| \times |S|$

## Aggregation Functions

### Common Aggregates

**COUNT**: Number of rows
$$COUNT(*) = |R|$$

**SUM**: Total of numeric column
$$SUM(x) = \sum_{i=1}^{n} x_i$$

**AVG**: Mean value
$$AVG(x) = \frac{1}{n}\sum_{i=1}^{n} x_i = \mu$$

**MIN/MAX**: Minimum/Maximum value
$$MIN(x) = \min_{i} x_i, \quad MAX(x) = \max_{i} x_i$$

**Standard Deviation**:
$$STDDEV(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2} = \sigma$$

**Variance**:
$$VAR(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2 = \sigma^2$$

### Window Functions

Perform calculations across a set of rows related to the current row:

**ROW_NUMBER()**: Sequential number within partition

**RANK()**: Rank with gaps for ties

**DENSE_RANK()**: Rank without gaps

**LAG/LEAD**: Access previous/next row values

**Moving Average** (using window):
$$MA_k(x_i) = \frac{1}{k}\sum_{j=i-k+1}^{i} x_j$$

## JSON vs SQL Comparison

### Data Structure

| Aspect | JSON | SQL |
|--------|------|-----|
| Structure | Hierarchical, tree-like | Tabular, relational |
| Schema | Flexible, optional | Fixed, enforced |
| Relationships | Embedded or referenced | Foreign keys, joins |
| Nesting | Native support | Requires joins |
| Data Types | Limited (6 types) | Rich type system |
| Normalization | Often denormalized | Normalized by design |

### Query Performance

**JSON Traversal**: $O(d \times n)$ where $d$ is depth, $n$ is document count

**SQL Indexed Query**: $O(\log n)$ with B-tree index

**SQL Full Scan**: $O(n)$ without index

**Join Cost**: 
- Nested loop: $O(n \times m)$
- Hash join: $O(n + m)$
- Sort-merge: $O(n \log n + m \log m)$

### When to Choose

**Use JSON when**:
- Data structure varies between records
- Hierarchical relationships are primary
- Rapid development and schema evolution needed
- Working with web APIs
- Document-oriented data model fits naturally

**Use SQL when**:
- Data has consistent structure
- Complex queries with joins needed
- ACID guarantees required
- Multiple related entities
- Data integrity is critical
- Concurrent write access needed

## JSON in Machine Learning: Practical Examples

### Example 1: API Response with Variable Features

```json
{
  "customer_id": "C123",
  "demographics": {"age": 35, "income": 75000},
  "purchase_history": [
    {"date": "2024-01-01", "amount": 120.50, "items": ["laptop", "mouse"]},
    {"date": "2024-01-15", "amount": 45.99, "items": ["book"]}
  ],
  "preferences": {"newsletter": true, "category_interest": ["electronics", "books"]}
}
```

**ML Challenge**: 
- Variable-length arrays require special handling
- Nested structure needs flattening for traditional ML
- Different customers may have different fields

**Solutions**:
1. **Aggregation**: Count of purchases, total spend, average transaction
2. **Embedding**: Use sequence models for purchase history
3. **One-hot encoding**: For categorical preferences

### Example 2: Hierarchical Product Data

```json
{
  "product_id": "P456",
  "category": {
    "level1": "Electronics",
    "level2": "Computers",
    "level3": "Laptops"
  },
  "features": {
    "processor": {"brand": "Intel", "cores": 8, "speed_ghz": 2.6},
    "memory_gb": 16,
    "storage": [
      {"type": "SSD", "size_gb": 512},
      {"type": "HDD", "size_gb": 1000}
    ]
  }
}
```

**Feature Engineering**:
- Flatten hierarchy: `category_level1`, `category_level2`, `category_level3`
- Extract nested values: `processor_brand`, `processor_cores`, `processor_speed`
- Aggregate arrays: `total_storage_gb`, `has_ssd`, `storage_types_count`

## SQL in Machine Learning: Feature Engineering

### Aggregation-Based Features

From a transactions table, create customer-level features:

```sql
SELECT 
    customer_id,
    COUNT(*) as transaction_count,
    SUM(amount) as total_spend,
    AVG(amount) as avg_transaction,
    MAX(amount) as max_transaction,
    MIN(amount) as min_transaction,
    STDDEV(amount) as spend_volatility,
    DATEDIFF(NOW(), MAX(date)) as days_since_last_purchase
FROM transactions
GROUP BY customer_id
```

**Mathematical Interpretation**:
- Each aggregate is a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$
- Reduces multiple rows to single summary statistic per customer

### Window Functions for Sequential Features

```sql
SELECT 
    customer_id,
    transaction_date,
    amount,
    LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY transaction_date) as prev_amount,
    AVG(amount) OVER (PARTITION BY customer_id ORDER BY transaction_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as ma_3
FROM transactions
```

**Moving Average Formula**:
$$MA_3(i) = \frac{1}{3}(x_{i-2} + x_{i-1} + x_i)$$

### Time-Based Features

**Recency, Frequency, Monetary (RFM) Analysis**:

```sql
SELECT 
    customer_id,
    DATEDIFF(NOW(), MAX(date)) as recency,
    COUNT(*) as frequency,
    SUM(amount) as monetary
FROM transactions
WHERE date >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
GROUP BY customer_id
```

**Interpretation**:
- **Recency** (R): Lower is better (recent activity)
- **Frequency** (F): Higher is better (loyal customer)
- **Monetary** (M): Higher is better (high-value customer)

**RFM Score**: Combine into single metric
$$RFM_{score} = w_R \cdot R_{normalized} + w_F \cdot F_{normalized} + w_M \cdot M_{normalized}$$

### Join-Based Feature Enrichment

Combining multiple tables:

```sql
SELECT 
    c.customer_id,
    c.age,
    c.income,
    t.transaction_count,
    t.total_spend,
    p.favorite_category
FROM customers c
LEFT JOIN transaction_summary t ON c.customer_id = t.customer_id
LEFT JOIN preferences p ON c.customer_id = p.customer_id
```

**Result**: Wide feature matrix with customer attributes + transaction patterns + preferences

## Performance Optimization

### JSON Optimization

1. **Minimize Nesting**: Keep depth ≤ 3 levels
2. **Use JSON Indexes**: Database-specific JSON indexing
3. **Compress**: gzip reduces size by 70-90%
4. **Streaming**: Parse large files incrementally
5. **Schema Validation**: Prevent malformed data

**Parsing Time**: $T_{parse} = k \cdot size_{bytes}$ where $k$ depends on parser efficiency

### SQL Optimization

1. **Indexing Strategy**:
   - B-tree index: $O(\log n)$ lookup
   - Hash index: $O(1)$ for exact matches
   - Composite index: For multi-column queries

2. **Query Optimization**:
   - Filter early with WHERE clauses
   - Select only needed columns
   - Avoid SELECT DISTINCT if possible
   - Use EXPLAIN to analyze query plans

3. **Join Optimization**:
   - Ensure join columns are indexed
   - Join smaller table first
   - Use appropriate join type

4. **Partitioning**:
   - **Horizontal**: Split by rows (e.g., by date)
   - **Vertical**: Split by columns (e.g., hot vs cold data)

**Index Size**: For B-tree, height $h = \log_b(n)$ where $b$ is branching factor

## Data Extraction Patterns

### From JSON to Tabular Format

**Flattening Process**:
1. **Identify structure**: Analyze nesting levels
2. **Path notation**: `user.address.city` → `user_address_city`
3. **Array handling**: 
   - **Explosion**: One row per array element
   - **Aggregation**: Summary statistics
   - **Encoding**: Concatenate or one-hot

**Example Transformation**:
```
Input: 1 customer with 3 purchases (nested)
Output: 3 rows (one per purchase) with customer info repeated
```

### From SQL to Feature Matrix

**Pipeline**:
1. **Query**: SELECT with joins and aggregations
2. **Pivot**: Convert categorical values to columns
3. **Normalize**: Scale numeric features
4. **Encode**: Handle categorical variables

## Integration Patterns

### JSON to SQL ETL

1. **Schema Inference**: Analyze JSON structure
2. **Table Design**: Create normalized schema
3. **Transformation**: Flatten and split data
4. **Loading**: Bulk insert into tables
5. **Indexing**: Create indexes for performance

### SQL to JSON API

1. **Query Execution**: Retrieve data with joins
2. **Aggregation**: Group by entities
3. **Nesting**: Build hierarchical structure
4. **Serialization**: Convert to JSON format
5. **Transmission**: Send via HTTP/REST

## Summary

JSON and SQL represent complementary paradigms in the ML data ecosystem:

**JSON** excels at:
- Flexible, semi-structured data from modern web applications
- Hierarchical relationships and variable schemas
- API integration and configuration management
- Rapid prototyping without rigid schemas

**SQL** excels at:
- Structured data with complex relationships
- ACID transactions and data integrity
- Powerful querying and aggregation
- Feature engineering through joins and window functions

**Best Practice**: Use both strategically
- **Ingest**: JSON from APIs
- **Process**: SQL for feature engineering
- **Store**: SQL for structured features, JSON for configuration
- **Serve**: JSON for API responses

Understanding both technologies, their mathematical foundations, and optimization strategies enables building robust, efficient ML data pipelines.

---  
**Xampp download link**: https://www.apachefriends.org/index.html  
**World dataset**: https://www.kaggle.com/busielmorley/worldcities-pop-lang-rank-sql-create-tbls?select=world.sql  
**Pandas read_json documentation**: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html  
**Pandas read_sql_query documentation**: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html#pandas.read_sql_query
