# Web Scraping to DataFrame

## General Idea

Web scraping is the automated process of extracting data from websites by parsing HTML/XML content. Unlike structured APIs, web scraping deals with unstructured or semi-structured data embedded in web pages. For machine learning, web scraping enables access to vast amounts of publicly available data that isn't offered through formal APIs.

## Why Use Web Scraping?

1. **Data Availability**: Access information not available through APIs
2. **Public Data**: Collect publicly visible information at scale
3. **Price Monitoring**: Track product prices, availability
4. **News Aggregation**: Gather articles, headlines, content
5. **Social Media**: Extract public posts, reviews, ratings
6. **Research Data**: Collect academic papers, citations, datasets
7. **Real Estate**: Property listings, prices, locations
8. **Job Market**: Analyze job postings, salaries, skills

## Role in Machine Learning

### Data Collection

- **Training Datasets**: Build custom datasets for unique problems
- **Text Corpus**: Collect text for NLP models (sentiment, classification)
- **Image Datasets**: Download images for computer vision
- **Tabular Data**: Extract tables from research papers, statistics sites
- **Time Series**: Historical data from news, finance sites
- **Label Generation**: Collect ratings, reviews for supervised learning

### Practical Applications

- **Price Prediction**: E-commerce pricing models
- **Sentiment Analysis**: Customer reviews, social media
- **Recommendation Systems**: Product catalogs, user preferences
- **Trend Analysis**: News, search trends, social media
- **Competitive Analysis**: Monitor competitors' products, pricing

## HTML Structure Fundamentals

### HTML Document Object Model (DOM)

HTML represents documents as a tree structure:

```
html
├── head
│   ├── title
│   └── meta
└── body
    ├── div (class="container")
    │   ├── h1
    │   └── p
    └── table
        ├── tr
        │   ├── th
        │   └── th
        └── tr
            ├── td
            └── td
```

**Tree Depth**: Path from root to element
**Siblings**: Elements at same level with same parent
**Parent/Child**: Direct hierarchical relationship

### HTML Tags and Attributes

**Tags**: Define element type (`<div>`, `<table>`, `<p>`)

**Attributes**: Provide metadata
- `class`: CSS class for styling
- `id`: Unique identifier
- `href`: Link destination
- `src`: Resource source (images, scripts)
- `data-*`: Custom data attributes

**Example**:
```html
<a href="https://example.com" class="link" id="main-link">Click here</a>
```

### Common Elements for Data Extraction

**Tables**: `<table>`, `<tr>`, `<td>`, `<th>`
- Most structured, easiest to parse
- Directly maps to DataFrame

**Lists**: `<ul>`, `<ol>`, `<li>`
- Ordered or unordered collections

**Divs**: `<div>`, `<span>`
- Generic containers, identified by class/id
- Require inspection to understand structure

**Text**: `<p>`, `<h1>`-`<h6>`
- Paragraphs and headings

## Web Scraping Techniques

### 1. CSS Selectors

Navigate DOM using CSS patterns:

**Element Selectors**:
- `p`: All `<p>` tags
- `div.classname`: Divs with specific class
- `#idname`: Element with specific ID

**Relationship Selectors**:
- `div > p`: Direct child
- `div p`: Descendant
- `div + p`: Adjacent sibling

**Attribute Selectors**:
- `a[href]`: Links with href attribute
- `a[href^="https"]`: Links starting with "https"
- `input[type="text"]`: Text inputs

**Specificity**: More specific selectors are faster and more precise

### 2. XPath

XML Path Language for navigating XML/HTML:

**Syntax**:
- `//tag`: All tags in document
- `/html/body/div`: Absolute path
- `//div[@class="container"]`: Div with class
- `//table//tr[1]`: First row in any table
- `//td[text()="Value"]`: Cell containing "Value"

**Axes**:
- `//div/child::p`: Child paragraphs
- `//div/following-sibling::p`: Following siblings
- `//div/ancestor::body`: Ancestor elements

**Predicate Filtering**: `//tr[position()>1]` (skip header)

### 3. Beautiful Soup Methods

**Finding Elements**:
- `find()`: First matching element
- `find_all()`: All matching elements
- `select()`: CSS selector
- `select_one()`: First CSS match

**Navigation**:
- `.parent`: Parent element
- `.children`: Direct children
- `.next_sibling`: Next sibling
- `.find_next()`: Next matching element

**Extraction**:
- `.text`: Text content
- `.get_text()`: Text with separator
- `.get("attribute")`: Attribute value
- `.attrs`: All attributes dictionary

## Scraping Workflow

### 1. Request Web Page

**HTTP GET Request**:
```python
import requests
response = requests.get(url)
html_content = response.text
```

**Status Codes**:
- 200: Success
- 404: Not Found
- 403: Forbidden (access denied)
- 503: Service Unavailable

### 2. Parse HTML

**Parse HTML to DOM**:
```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')
```

**Parsers**:
- `html.parser`: Built-in, moderate speed
- `lxml`: Faster, requires installation
- `html5lib`: Most lenient, slowest

### 3. Navigate and Extract

**Find tables**:
```python
tables = soup.find_all('table')
```

**Extract data**:
```python
for row in table.find_all('tr'):
    cells = row.find_all('td')
    data = [cell.text.strip() for cell in cells]
```

### 4. Clean Data

- Strip whitespace
- Remove HTML entities (`&nbsp;`, `&amp;`)
- Handle encoding issues
- Parse dates, numbers
- Handle missing values

### 5. Convert to DataFrame

```python
import pandas as pd
df = pd.DataFrame(extracted_data, columns=column_names)
```

## Handling Dynamic Content

### JavaScript-Rendered Pages

Many modern websites use JavaScript frameworks (React, Vue, Angular):

**Problem**: Content not in initial HTML, loaded dynamically

**Solutions**:

**1. Selenium/Playwright**: Browser automation
- Executes JavaScript
- Waits for content to load
- Slower but comprehensive

**2. API Inspection**: Find underlying API
- Use browser DevTools Network tab
- Identify API endpoints
- Directly request API (faster)

**3. Requests-HTML**: Lightweight rendering
- Chromium headless browser
- Middle ground between requests and Selenium

### AJAX Requests

**Asynchronous JavaScript and XML**:
- Page loads, then fetches additional data
- Look for XHR (XMLHttpRequest) in Network tab
- Often JSON responses (easier than HTML parsing)

**Advantages of finding API**:
- Faster (no HTML parsing)
- Structured data (JSON)
- Less brittle (HTML changes frequently)

## Tables to DataFrame

### Simple Table Structure

```html
<table>
  <thead>
    <tr><th>Name</th><th>Age</th><th>City</th></tr>
  </thead>
  <tbody>
    <tr><td>Alice</td><td>30</td><td>NYC</td></tr>
    <tr><td>Bob</td><td>25</td><td>LA</td></tr>
  </tbody>
</table>
```

**Direct Conversion**:
- Headers: `<th>` tags
- Data: `<td>` tags in `<tr>` rows
- Pandas `read_html()` handles automatically

### Complex Table Structures

**Merged Cells** (`colspan`, `rowspan`):
```html
<td colspan="2">Merged</td>
```
- Single cell spans multiple columns
- Requires careful parsing to duplicate values

**Nested Tables**:
- Tables within table cells
- Flatten or extract separately

**Multi-level Headers**:
- Multiple `<tr>` in `<thead>`
- Create hierarchical column names

### Pandas read_html()

**Powerful built-in function**:
```python
tables = pd.read_html(url)  # Returns list of all tables
df = tables[0]  # First table
```

**Parameters**:
- `match`: Regex to filter tables
- `header`: Row(s) to use as column names
- `index_col`: Column to use as index
- `skiprows`: Rows to skip
- `attrs`: Dict of table attributes to match

**Advantages**:
- Automatic parsing
- Handles basic merging
- Type inference

**Limitations**:
- All tables extracted (may want specific one)
- Limited control over structure
- May misinterpret complex layouts

## Scraping Ethics and Legality

### Legal Considerations

1. **Terms of Service**: Check website's ToS
2. **robots.txt**: Respect crawling directives
3. **Copyright**: Don't violate content ownership
4. **CFAA**: Computer Fraud and Abuse Act (US law)
5. **GDPR**: Personal data protection (EU)

**robots.txt** location: `https://example.com/robots.txt`

```
User-agent: *
Disallow: /private/
Crawl-delay: 10
```

- `User-agent`: Which bots this applies to
- `Disallow`: Paths not to scrape
- `Crawl-delay`: Seconds between requests

### Ethical Best Practices

1. **Rate Limiting**: Don't overwhelm servers
   - Typical: 1-2 requests per second
   - Add delays: `time.sleep(1)`

2. **User Agent**: Identify your scraper
   ```python
   headers = {'User-Agent': 'MyBot/1.0 (myemail@example.com)'}
   ```

3. **Respect robots.txt**: Use libraries that honor it

4. **Cache Responses**: Avoid repeated requests

5. **Attribution**: Credit data sources

6. **Personal Data**: Don't scrape private information

### Rate Limiting Math

**Request Rate**: $r = \frac{1}{\Delta t}$ (requests per second)

Where $\Delta t$ is delay between requests

**Total Time**: For $n$ pages with delay $\Delta t$:
$$T = n \times \Delta t$$

**Exponential Backoff**: If blocked, increase delay
$$\Delta t_{new} = \Delta t \times 2^{attempt}$$

## Common Challenges

### 1. Dynamic HTML Structure

**Problem**: Sites redesign, changing element classes/IDs

**Solutions**:
- Use multiple selectors as fallback
- Select by semantic meaning (e.g., table headers)
- Regular expression patterns
- Monitor for breakage, update selectors

### 2. Pagination

**Scraping multiple pages**:

**URL Patterns**:
- `?page=1`, `?page=2`, ...
- `/page/1/`, `/page/2/`, ...

**Next Button**:
- Find "Next" link, extract href
- Follow until no more pages

**Infinite Scroll**:
- JavaScript loads content on scroll
- Use Selenium to scroll, or find AJAX endpoint

**Pagination Loop**:
```python
page = 1
while page <= max_pages:
    url = f"https://example.com/data?page={page}"
    scrape_page(url)
    page += 1
    time.sleep(delay)
```

### 3. Anti-Scraping Measures

**CAPTCHAs**: Human verification
- Difficult to automate
- Consider manual solving services (ethical concerns)

**IP Blocking**: Track and block IPs making many requests
- Use proxies, rotate IPs
- Respect rate limits to avoid

**Honeypot Links**: Hidden links to catch bots
- Invisible to humans (CSS: `display:none`)
- Bots follow all links, get blocked

**User-Agent Filtering**: Block known bot user agents
- Use realistic user agent strings
- Rotate user agents

**JavaScript Challenges**: Require JS execution
- Use headless browser (Selenium, Playwright)

### 4. Data Quality Issues

**Inconsistent Formatting**:
- Dates in multiple formats
- Numbers with commas, currency symbols
- Mixed case text

**Missing Data**:
- Empty cells
- Placeholder text ("N/A", "—")
- Non-existent elements

**Encoding Issues**:
- Special characters (é, ñ, 中)
- HTML entities (`&amp;`, `&quot;`)

**Solutions**:
- Robust parsing with try/except
- Regular expressions for pattern matching
- Standardization functions
- Explicit encoding specification

## Performance Optimization

### Sequential vs Parallel Scraping

**Sequential**: One page at a time
$$T_{seq} = n \times t_{avg}$$

**Parallel**: Multiple pages simultaneously
$$T_{par} = \frac{n}{k} \times t_{avg}$$

Where $k$ is concurrency (number of parallel requests)

**Considerations**:
- Server load (ethical)
- Rate limits
- Memory usage
- Connection pooling

### Caching

**Store scraped pages**:
- Avoid re-fetching unchanged content
- Faster development/testing
- Reduces server load

**Cache Strategies**:
- Disk cache: Save HTML to files
- Database cache: Store in SQL/NoSQL
- Memory cache: For current session

**Cache Invalidation**: 
- Time-based: Expire after duration
- Version-based: Check if page changed (ETag, Last-Modified)

### Asynchronous Scraping

**asyncio + aiohttp**: Non-blocking I/O
- Start many requests without waiting
- Process responses as they arrive
- Much faster for I/O-bound tasks

**Speedup**: Near-linear with concurrency (up to limits)

## Data Cleaning Post-Scraping

### Text Cleaning

1. **Whitespace**: Strip leading/trailing, normalize internal
2. **HTML Entities**: Convert `&nbsp;` → space, `&amp;` → &
3. **Special Characters**: Remove or normalize
4. **Case Normalization**: Uppercase, lowercase, title case

**Regular Expressions**:
- Extract patterns: emails, phone numbers, prices
- Remove noise: HTML tags, scripts, ads
- Standardize formats

### Type Conversion

**Strings to Numbers**:
- Remove currency symbols: `$1,234.56` → `1234.56`
- Parse percentages: `45.2%` → `0.452`
- Handle thousands separators

**Strings to Dates**:
- Parse multiple formats
- Handle timezone awareness
- Fill incomplete dates

**Validation**:
- Range checks (age between 0-120)
- Format validation (email regex)
- Cross-field consistency

## Example Use Cases

### Use Case 1: E-commerce Price Monitoring

**Target**: Amazon, eBay product listings

**Extracted Features**:
- Product title
- Current price
- Original price (if discounted)
- Rating (stars)
- Number of reviews
- Availability
- Seller information

**ML Application**: 
- Price prediction models
- Demand forecasting
- Discount pattern analysis

**Challenges**:
- Dynamic pricing (changes frequently)
- A/B testing (different users see different prices)
- JavaScript-heavy pages

### Use Case 2: News Aggregation

**Target**: News websites, blogs

**Extracted Features**:
- Headline
- Article text
- Author
- Publication date
- Category/tags
- Number of shares/comments

**ML Application**:
- Topic modeling (LDA, NMF)
- Sentiment analysis
- Trend detection
- Fake news detection

**Preprocessing**:
- Remove ads, navigation
- Extract main content (boilerplate removal)
- Handle paywalls

### Use Case 3: Job Market Analysis

**Target**: LinkedIn, Indeed, Glassdoor

**Extracted Features**:
- Job title
- Company
- Location
- Salary range
- Required skills
- Experience level
- Job description

**ML Application**:
- Salary prediction
- Skill demand analysis
- Career path recommendation
- Job matching

**Feature Engineering**:
- Extract skills from description (NER)
- Parse salary ranges
- Geocode locations
- Company size/industry

## Summary

Web scraping is a powerful technique for acquiring machine learning data from the vast information available on the internet. Understanding HTML structure, CSS/XPath selectors, and proper extraction techniques enables building custom datasets for unique ML problems.

**Key Principles**:
- Respect legal and ethical boundaries (ToS, robots.txt, rate limits)
- Handle dynamic content appropriately (JavaScript rendering, AJAX)
- Clean and validate scraped data thoroughly
- Optimize performance while being considerate of target servers
- Build robust scrapers that handle edge cases and site changes

**Best Practices**:
- Start with simple, static sites to learn
- Use browser DevTools to inspect structure
- Look for APIs before resorting to scraping
- Implement error handling and logging
- Cache responses to avoid redundant requests
- Monitor and maintain scrapers as sites evolve

Web scraping bridges the gap between unstructured web data and structured ML datasets, enabling data scientists to access information not available through traditional APIs. When combined with proper data cleaning and preprocessing, it becomes an invaluable skill for modern machine learning practitioners.

---

**Video Link**: https://www.youtube.com/watch?v=8NOdgjC1988
