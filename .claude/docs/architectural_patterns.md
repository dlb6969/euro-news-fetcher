# Architectural Patterns

## 1. External Configuration Pattern

Configuration is externalized to editable files rather than hardcoded:

| Config Type | File | Format | Reference |
|-------------|------|--------|-----------|
| Stock universe | `european_tickers.csv` | Single column CSV | `market_dashboard.py:47-52` |
| News keywords | `news_keywords.txt` | Line-delimited text | `market_dashboard.py:60-71` |
| API secrets | `.env` | KEY=VALUE | `market_dashboard.py:19-27` |
| Index mappings | Hardcoded dict | Python dict | `market_dashboard.py:29-44` |

**Convention**: Use `Path(__file__).parent` for file paths to ensure portability.

## 2. Caching Pattern

All expensive data fetches use Streamlit's cache decorator:

```python
@st.cache_data(ttl=300)  # 5-minute TTL
def fetch_market_data(tickers: List[str]) -> pd.DataFrame:
```

**Instances**:
- `fetch_market_data()` - `market_dashboard.py:145`
- `fetch_indices_data()` - `market_dashboard.py:119`
- `fetch_news_for_movers()` - `market_dashboard.py:252`
- `search_news_by_keywords()` - `market_dashboard.py:262`

**Cache clearing**: `st.cache_data.clear()` on refresh button (`market_dashboard.py:444-446`)

## 3. Query Params Navigation

SPA-style routing using `st.query_params`:

| Route | Param | Handler |
|-------|-------|---------|
| Home | (none) | `render_home_page()` |
| Stock news | `?ticker=XXX` | `render_news_page(ticker)` |
| Keyword search | `?page=keyword_search` | `render_keyword_search_page()` |

**Pattern** (`market_dashboard.py:616-632`):
```python
params = st.query_params
if params.get("ticker"):
    render_news_page(...)
elif params.get("page") == "keyword_search":
    render_keyword_search_page()
else:
    render_home_page()
```

**Navigation**: Set params then `st.rerun()` (`market_dashboard.py:432-434`)

## 4. API Client Pattern

External APIs wrapped in client classes with configuration checks:

```python
class FinnhubClient:
    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def get_quote(self, symbol: str) -> Optional[Dict]:
        if not self.is_configured():
            return None
        # ... API call
```

Reference: `market_dashboard.py:75-116`

**Convention**: Return `None` or empty list on failure, not exceptions.

## 5. Data Normalization Pattern

Handle API version differences with fallback parsing:

```python
if 'content' in item:  # New yfinance format
    news_item['title'] = content.get('title', 'No title')
else:  # Old format
    news_item['title'] = item.get('title', 'No title')
```

Reference: `market_dashboard.py:231-243`

**Convention**: Always provide default values with `.get(key, default)`.

## 6. Silent Error Handling

Wrap external calls in try/except, continue on failure:

```python
for ticker in tickers:
    try:
        # fetch data
    except Exception:
        continue  # Skip failed ticker
```

References:
- `market_dashboard.py:154-182` (market data)
- `market_dashboard.py:124-140` (indices)
- `market_dashboard.py:225-249` (news)

**Rationale**: Single ticker failures shouldn't break the entire dashboard.

## 7. Currency Normalization

UK stocks (.L suffix) quoted in GBP pence; apply conversion factor:

```python
mask = df["Ticker"].str.endswith(".L")
df.loc[mask, "Notional"] = df.loc[mask, "Notional"] * 1.2
```

Reference: `market_dashboard.py:189-198`

**Factor**: 1.2x approximates GBP-to-EUR conversion.

## 8. Render Function Pattern

Each page is a standalone render function:

| Function | Purpose | Reference |
|----------|---------|-----------|
| `render_home_page()` | Main dashboard | `market_dashboard.py:416` |
| `render_news_page(ticker)` | Stock detail | `market_dashboard.py:302` |
| `render_keyword_search_page()` | News search | `market_dashboard.py:339` |

**Convention**: Each render function handles its own "Back to Home" navigation.

## 9. Formatting Functions

Consistent value formatting via dedicated functions:

| Function | Output Example | Reference |
|----------|---------------|-----------|
| `format_notional()` | "1.2B", "25M", "500K" | `market_dashboard.py:201-210` |
| `format_volume()` | "1.5M", "500K" | `market_dashboard.py:213-220` |

**Convention**: Return strings ready for display, not raw numbers.

## 10. Batch Data Fetching

Use yfinance batch download instead of individual requests:

```python
df = yf.download(tickers, period="2d", group_by="ticker", threads=True)
```

Reference: `market_dashboard.py:151`

**Rationale**: Single network request for ~95 tickers vs 95 individual calls.
