# European Market Dashboard

## Project Overview

A Streamlit-based European stock market dashboard providing:
- Real-time stock screening (top gainers/losers) with notional volume filtering
- European market indices tracking (DAX, FTSE, CAC40, etc.)
- Keyword-based news search across ~95 European stocks
- Individual stock news and intraday charts

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit 1.28+ |
| Data Sources | yfinance (primary), Finnhub API (secondary) |
| Charting | Plotly |
| Data Processing | Pandas |

## Project Structure

```
euro_news_fetcher/
├── market_dashboard.py    # Main application (single-file architecture)
├── european_tickers.csv   # Stock universe (single column: ticker)
├── news_keywords.txt      # Keywords for news filtering (one per line)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (FINNHUB_API_KEY)
└── .claude/docs/          # Additional documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `market_dashboard.py` | All application logic: data fetching, UI rendering, routing |
| `european_tickers.csv` | Editable stock list in format `TICKER.EXCHANGE` (e.g., `ASML.AS`) |
| `news_keywords.txt` | Keywords for filtering news (guidance, earnings, etc.) |
| `.env` | Environment variables, notably `FINNHUB_API_KEY` |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run market_dashboard.py

# Run in headless mode (no browser prompt)
streamlit run market_dashboard.py --server.headless=true
```

Dashboard opens at `http://localhost:8501`

## Configuration

### Adding Stocks
Edit `european_tickers.csv` - one ticker per line with exchange suffix:
- `.DE` - Germany (Xetra)
- `.PA` - France (Euronext Paris)
- `.L` - UK (London)
- `.AS` - Netherlands (Euronext Amsterdam)

### Adding Keywords
Edit `news_keywords.txt` - one keyword per line (case-insensitive)

### API Keys
Add to `.env`:
```
FINNHUB_API_KEY=your_key_here
```

## Data Flow

1. `fetch_market_data()` batch downloads all tickers via yfinance
2. `filter_by_notional()` removes low-liquidity stocks (< €20M notional)
3. UK stocks (.L) get 1.2x GBP-to-EUR conversion factor
4. Results cached for 5 minutes via `@st.cache_data(ttl=300)`

## Navigation

Uses query params for SPA-style routing (`market_dashboard.py:616-632`):
- `?ticker=SAP.DE` → Stock news page
- `?page=keyword_search` → Keyword search page
- No params → Home page

## Additional Documentation

When working on specific features, consult:

| Topic | File |
|-------|------|
| Design patterns & conventions | `.claude/docs/architectural_patterns.md` |
