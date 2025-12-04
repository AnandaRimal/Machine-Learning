# Pandas DataFrames via Web Scraping: HTML → Structured Data

Figure: Nano Banana – DOM to DataFrame schematic

**Overview**
- Extract tabular data from web pages: static HTML tables or structured tags.
- Use `pandas.read_html`, `requests`, and `BeautifulSoup` for parsing.

**Key Concepts**
- DOM tree: nodes (tags), attributes, text; CSS selectors target elements.
- Static vs dynamic pages: server-rendered HTML vs JS-rendered content.
- Respect robots and TOS: throttle requests; cache; handle pagination.

**Pros/Cons**
- Pros: Fast prototyping; direct table extraction; pandas integration.
- Cons: Fragile to layout changes; dynamic content needs browser automation.

**Code: Read HTML Tables**
```python
import pandas as pd
import requests

url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
tables = pd.read_html(url)
df = tables[0]
df.columns = [c.strip() for c in df.columns]
df.head()
```

**BeautifulSoup Parsing**
```python
from bs4 import BeautifulSoup
import pandas as pd
import requests

headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get('https://example.com/stats', headers=headers, timeout=20)
soup = BeautifulSoup(r.text, 'html.parser')
rows = []
for tr in soup.select('table.data tr'):
    cells = [td.get_text(strip=True) for td in tr.select('td,th')]
    rows.append(cells)
df = pd.DataFrame(rows[1:], columns=rows[0])
```

**Handling Pagination**
```python
def scrape_pages(base_url, pages=5):
    frames = []
    for p in range(1, pages+1):
        r = requests.get(f'{base_url}?page={p}', timeout=20)
        r.raise_for_status()
        page_df = pd.read_html(r.text)[0]
        frames.append(page_df)
    return pd.concat(frames, ignore_index=True)

df = scrape_pages('https://example.com/leaderboard')
```

**Dynamic Sites**
- Use `requests_html` or Selenium/Playwright when content loads via JS.
- Cache and throttle; reuse sessions; avoid brittle selectors.

**Summary**
- Start with `read_html` for straightforward tables; escalate to BeautifulSoup or browser automation for complex pages. Normalize and validate before analysis.
