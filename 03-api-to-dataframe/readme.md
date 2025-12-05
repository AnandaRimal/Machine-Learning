# API to DataFrame

## General Idea

APIs (Application Programming Interfaces) are structured interfaces that allow applications to communicate and exchange data over the internet. In machine learning, APIs serve as a critical bridge for accessing real-time data, third-party datasets, and cloud services. Converting API responses to DataFrames enables seamless integration of external data sources into ML workflows.

## Why Use APIs?

1. **Real-time Data Access**: Get up-to-date information (stock prices, weather, social media)
2. **Scalability**: Access massive datasets without local storage
3. **Standardization**: Consistent data format and structure (usually JSON or XML)
4. **Authentication & Security**: Controlled access with API keys and OAuth
5. **Rate Limiting**: Managed data access to prevent overload
6. **Version Control**: API versioning ensures backward compatibility
7. **Third-party Services**: Access specialized data (maps, translations, sentiment analysis)

## Role in Machine Learning

### Data Acquisition

- **Training Data**: Collect labeled datasets from specialized providers
- **Feature Enrichment**: Augment existing data with external features (demographics, geolocation)
- **Real-time Features**: Fetch current data for online learning and predictions
- **Data Validation**: Cross-reference with authoritative sources
- **Benchmark Datasets**: Access public ML datasets (UCI, Kaggle via API)

### Production ML Systems

- **Inference API**: Deploy models as web services
- **Feature Store API**: Retrieve pre-computed features
- **Model Serving**: REST/gRPC endpoints for predictions
- **Monitoring**: Track model performance via telemetry APIs
- **A/B Testing**: Serve different model versions

## API Fundamentals

### HTTP Methods

**GET**: Retrieve data (read-only, idempotent)
- Most common for data fetching
- Parameters in URL query string
- Safe to cache

**POST**: Submit data to create resources
- Send data in request body
- Not idempotent (repeated calls create multiple resources)
- Used for complex queries or large payloads

**PUT**: Update existing resource (idempotent)

**DELETE**: Remove resource

**PATCH**: Partial update of resource

### Request Components

1. **Base URL**: `https://api.example.com/v1/`
2. **Endpoint**: `/users` or `/data`
3. **Query Parameters**: `?limit=100&offset=0`
4. **Headers**: Metadata (authentication, content-type)
5. **Body**: Data payload (for POST/PUT)

**Complete URL Structure**:
```
https://api.example.com/v1/users?limit=100&page=2&sort=created_desc
|_____________________| |____| |_________________________________|
       Base URL        Endpoint        Query Parameters
```

### Response Components

1. **Status Code**: HTTP code indicating result
   - 200: Success
   - 201: Created
   - 400: Bad Request (client error)
   - 401: Unauthorized
   - 404: Not Found
   - 429: Too Many Requests (rate limit)
   - 500: Internal Server Error

2. **Headers**: Response metadata (content-type, rate limits)
3. **Body**: Actual data (usually JSON)

### Authentication Methods

**API Key**: Simple token passed in header or query string
```
Authorization: ApiKey YOUR_KEY_HERE
```

**OAuth 2.0**: Token-based authorization with refresh mechanism
- More secure than API keys
- Supports delegated access
- Token expiration and renewal

**Bearer Token**: JWT (JSON Web Token) in Authorization header
```
Authorization: Bearer eyJhbGc...
```

**Basic Authentication**: Username:password encoded in Base64
- Less secure, use with HTTPS only

## RESTful API Principles

REST (Representational State Transfer) is an architectural style:

### Key Constraints

1. **Client-Server**: Separation of concerns
2. **Stateless**: Each request contains all necessary information
3. **Cacheable**: Responses explicitly indicate if cacheable
4. **Uniform Interface**: Consistent resource identification
5. **Layered System**: Client doesn't know if connected to end server
6. **Code on Demand** (optional): Server can transfer executable code

### Resource-Based URLs

Resources identified by URLs:
- `/users` - Collection of users
- `/users/123` - Specific user
- `/users/123/orders` - User's orders

**Idempotency**: Multiple identical requests have same effect as single request
- GET, PUT, DELETE are idempotent
- POST is NOT idempotent

## Pagination

APIs limit response size using pagination:

### Offset-Based Pagination

```
/api/data?limit=100&offset=200
```

- **limit**: Number of records per page
- **offset**: Number of records to skip

**Mathematical Model**:
For page $p$ with size $s$:
$$offset = (p - 1) \times s$$

Total pages: $\lceil \frac{N}{s} \rceil$ where $N$ is total records

### Cursor-Based Pagination

```
/api/data?cursor=eyJpZCI6MTIz&limit=100
```

- **cursor**: Opaque token pointing to position
- More efficient for large datasets
- Handles real-time updates better

**Advantage**: $O(1)$ to find next page vs $O(n)$ for offset

### Page-Based Pagination

```
/api/data?page=3&per_page=50
```

Simpler but equivalent to offset-based

## Rate Limiting

APIs restrict request frequency to prevent abuse:

### Common Strategies

**Fixed Window**: X requests per time window
- Example: 1000 requests per hour
- Resets at fixed intervals

**Sliding Window**: Rolling time window
- More sophisticated, smoother distribution

**Token Bucket**: Tokens regenerate at fixed rate
- Allows bursts up to bucket capacity
- Formal model: Tokens $T(t) = \min(C, T(t-1) + r \cdot \Delta t)$
  - $C$: Bucket capacity
  - $r$: Refill rate
  - $\Delta t$: Time elapsed

**Leaky Bucket**: Requests processed at constant rate
- Smooths traffic, discards excess

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 456
X-RateLimit-Reset: 1640995200
```

**Exponential Backoff**: When rate limited, wait before retrying
$$wait\_time = base \times 2^{attempt} + jitter$$

Typical: $base = 1s$, max attempts = 5

## Data Formats

### JSON (Most Common)

```json
{
  "users": [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25}
  ],
  "total": 2,
  "page": 1
}
```

**Advantages**:
- Human-readable
- Native JavaScript support
- Compact for nested data

### XML

```xml
<response>
  <users>
    <user>
      <id>1</id>
      <name>Alice</name>
    </user>
  </users>
</response>
```

**Advantages**:
- Schema validation (XSD)
- Namespace support
- More verbose (larger size)

### CSV (Less Common)

Some APIs return CSV for tabular data:
- Smaller size than JSON for flat data
- Easier to import to spreadsheets
- Limited for nested structures

## API Response Structures

### Envelope Pattern

Response wrapped in metadata:

```json
{
  "status": "success",
  "data": [...],
  "pagination": {
    "total": 10000,
    "page": 1,
    "per_page": 100
  },
  "meta": {
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Direct Data

Response is the data itself:
```json
[
  {"id": 1, "value": 100},
  {"id": 2, "value": 200}
]
```

### Error Responses

```json
{
  "error": {
    "code": "INVALID_PARAM",
    "message": "Parameter 'date' is required",
    "details": {...}
  }
}
```

## Converting API to DataFrame

### Workflow

1. **Make Request**: Send HTTP request with parameters
2. **Validate Response**: Check status code
3. **Parse JSON**: Convert response to Python dict
4. **Extract Data**: Navigate nested structure
5. **Normalize**: Flatten hierarchical data
6. **Create DataFrame**: Convert to tabular format
7. **Data Types**: Convert and validate types
8. **Handle Missing**: Fill or drop missing values

### Flattening Nested JSON

For nested response:
```json
{
  "user": {
    "id": 123,
    "profile": {"name": "Alice", "age": 30},
    "scores": [85, 90, 88]
  }
}
```

**Flattened DataFrame**:
```
user_id | user_profile_name | user_profile_age | scores_mean
123     | Alice             | 30               | 87.67
```

### Handling Arrays

**Strategy 1**: Explode (one row per array element)
```
Before: 1 row with array [1, 2, 3]
After: 3 rows with values 1, 2, 3
```

**Strategy 2**: Aggregate
```
Array [1, 2, 3] → mean: 2, max: 3, count: 3
```

**Strategy 3**: JSON column
```
Store entire array as JSON string in cell
```

## Practical Considerations

### Error Handling

```python
try:
    response = make_api_request()
    if response.status_code == 200:
        data = response.json()
    elif response.status_code == 429:
        # Rate limited, wait and retry
        time.sleep(60)
    elif response.status_code >= 500:
        # Server error, retry with backoff
        retry_with_backoff()
    else:
        # Client error, log and skip
        log_error(response)
except ConnectionError:
    # Network issue, retry
    handle_network_error()
```

### Batch Requests

For large datasets, make multiple paginated requests:

**Sequential**: 
- Time: $T = n \times t_{request}$
- Simple but slow

**Parallel** (with rate limits):
- Time: $T \approx \lceil \frac{n}{k} \rceil \times t_{request}$
- $k$: concurrency limit
- Faster but complex

**Optimal Batch Size**:
Balance between:
- Fewer large requests (reduce overhead)
- More small requests (better fault tolerance)

Typical: 100-1000 records per request

### Caching Strategies

**Local Cache**: Store responses to avoid repeated requests
- Time-to-live (TTL): Cache expires after duration
- Cache key: Hash of request parameters

**Cache Hit Rate**: $\frac{hits}{hits + misses}$

**Conditional Requests**: Use ETag or Last-Modified headers
- Server returns 304 Not Modified if unchanged
- Saves bandwidth

### Data Validation

1. **Schema Validation**: Ensure response structure matches expected
2. **Type Checking**: Verify data types
3. **Range Validation**: Check values within expected bounds
4. **Completeness**: Verify all required fields present
5. **Consistency**: Cross-field validation

**Example Checks**:
- Date formats parseable
- Numeric fields are numbers
- Required fields not null
- Enum values in allowed set

## API-Specific Considerations

### Time Series Data

Stock prices, weather, IoT sensors:

**Challenges**:
- Large volumes
- Need proper time indexing
- Handle timezone conversions
- Deal with irregular sampling

**Best Practices**:
- Request data in chunks (daily, weekly)
- Use ISO 8601 datetime format
- Sort by timestamp immediately
- Handle daylight saving time

### Geospatial Data

Maps, geocoding, location services:

**Key Concepts**:
- Latitude/Longitude: $(-90° to 90°, -180° to 180°)$
- GeoJSON format for geometric data
- Coordinate reference systems (CRS)
- Distance calculations (Haversine formula)

**Haversine Distance**:
$$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$$

Where:
- $r$: Earth's radius (6371 km)
- $\phi$: Latitude
- $\lambda$: Longitude

### Social Media Data

Twitter, Reddit, Facebook APIs:

**Characteristics**:
- High volume, streaming data
- Nested user objects
- Variable content (text, images, video)
- Rate limits strict

**Preprocessing**:
- Text cleaning and tokenization
- Extract hashtags, mentions
- Parse timestamps
- Handle emoji and special characters

## Performance Optimization

### Request Optimization

1. **Minimize Requests**: Batch operations when possible
2. **Select Fields**: Request only needed fields (`?fields=id,name`)
3. **Compression**: Accept gzip encoding (`Accept-Encoding: gzip`)
4. **Connection Pooling**: Reuse HTTP connections
5. **Async Requests**: Non-blocking I/O for parallelism

### DataFrame Construction

1. **Pre-allocate**: If size known, pre-allocate DataFrame
2. **Batch Append**: Collect records, create DataFrame once
3. **Type Specification**: Specify dtypes upfront (avoid inference)
4. **Chunking**: For huge datasets, process in chunks

**Anti-pattern**: Appending rows one-by-one
- Time complexity: $O(n^2)$ due to copying

**Better**: Collect in list, then create DataFrame
- Time complexity: $O(n)$

### Memory Management

For large API responses:

**Streaming**: Process records as they arrive
**Sampling**: Fetch representative subset for exploration
**Aggregation**: Compute summaries instead of raw data
**Compression**: Use parquet or feather for storage

## Example Use Cases

### Use Case 1: Weather Data for ML

**Scenario**: Predict energy demand using weather features

**API**: OpenWeatherMap, Weather.gov

**Features to Extract**:
- Temperature (current, min, max)
- Humidity
- Precipitation
- Wind speed
- Cloud cover
- UV index

**Temporal Considerations**:
- Hourly forecasts
- Historical data for training
- Real-time for prediction

### Use Case 2: Financial Data

**Scenario**: Stock price prediction or portfolio optimization

**API**: Alpha Vantage, Yahoo Finance, IEX Cloud

**Features**:
- Open, High, Low, Close, Volume (OHLCV)
- Adjusted close (for splits/dividends)
- Technical indicators (Moving averages, RSI, MACD)
- Company fundamentals (P/E ratio, market cap)

**Mathematical Indicators**:

**Moving Average**:
$$MA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}$$

**Exponential Moving Average**:
$$EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}$$
Where $\alpha = \frac{2}{n+1}$

**Relative Strength Index**:
$$RSI = 100 - \frac{100}{1 + RS}$$
Where $RS = \frac{Average\ Gain}{Average\ Loss}$

### Use Case 3: Natural Language Data

**Scenario**: Sentiment analysis, text classification

**API**: News API, Twitter API, Reddit API

**Extraction**:
- Text content
- Metadata (author, timestamp, source)
- Engagement metrics (likes, shares, comments)

**Preprocessing Pipeline**:
1. Extract text field from JSON
2. Clean HTML/special characters
3. Tokenization
4. Remove stop words
5. Lemmatization/stemming
6. Create document-term matrix

## Security Best Practices

1. **Never hardcode API keys**: Use environment variables
2. **Use HTTPS**: Encrypt data in transit
3. **Validate SSL certificates**: Prevent man-in-the-middle attacks
4. **Rotate keys**: Periodically regenerate API keys
5. **Limit scope**: Use minimum necessary permissions
6. **Monitor usage**: Track API calls for anomalies
7. **Sanitize inputs**: Prevent injection attacks

## Summary

APIs are essential for modern ML workflows, providing access to diverse, real-time data sources. Understanding API fundamentals—HTTP methods, authentication, pagination, rate limiting—is crucial for efficient data acquisition. Converting API responses to DataFrames requires careful handling of nested JSON, proper error management, and consideration of performance optimization.

Key principles:
- Use appropriate authentication and respect rate limits
- Handle pagination for large datasets
- Flatten nested structures thoughtfully
- Implement robust error handling and retry logic
- Cache responses when appropriate
- Validate data quality throughout the pipeline

Mastering API integration enables ML practitioners to leverage external data sources, enrich features, and build production-ready systems that interface with the broader data ecosystem.

---  
**TMDB API**: https://developers.themoviedb.org/  
**RapidAPI**: https://rapidapi.com/collection/list-of-free-apis  
**JSON Viewer**: http://jsonviewer.stack.hu/
