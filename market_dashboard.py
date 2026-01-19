"""
European Market Dashboard
A Streamlit-based dashboard for European stock screening and news search.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import re
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# File paths
TICKERS_FILE = Path(__file__).parent / "european_tickers.csv"
KEYWORDS_FILE = Path(__file__).parent / "news_keywords.txt"
BLOCKED_SOURCES_FILE = Path(__file__).parent / "blocked_sources.txt"

# Finnhub API Key from environment
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# European Index tickers (Yahoo Finance format) + S&P 500
EUROPEAN_INDICES = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "CAC 40": "^FCHI",
    "EURO STOXX 50": "^STOXX50E",
    "AEX": "^AEX",
    "IBEX 35": "^IBEX",
    "FTSE MIB": "FTSEMIB.MI",
    "SMI": "^SSMI",
    "PSI": "PSI20.LS",
    "BEL 20": "^BFX",
    "OMXS30": "^OMX",
    "OBX": "^OBX",
    "WIG20": "WIG20.WA",
}

# News Impact Scoring - Source Tiers (0-30 points)
SOURCE_TIERS = {
    # Tier 1 - Official/Primary Sources (30 points)
    "reuters": 30, "bloomberg": 30, "pr newswire": 30, "prnewswire": 30,
    "businesswire": 30, "business wire": 30, "globenewswire": 30,
    "globe newswire": 30, "accesswire": 30, "company press release": 30,
    # Tier 2 - Major Financial News (25 points)
    "cnbc": 25, "financial times": 25, "ft.com": 25, "wall street journal": 25,
    "wsj": 25, "marketwatch": 25, "barron's": 25, "barrons": 25,
    # Tier 3 - Quality Financial Sources (18 points)
    "yahoo finance": 18, "yahoo": 18, "investing.com": 18, "morningstar": 18,
    "thestreet": 18, "the street": 18,
    # Tier 4 - Aggregators (12 points)
    "benzinga": 12, "tipranks": 12, "zacks": 12, "finviz": 12,
    # Tier 5 - Blogs/Low Quality (5 points)
    "seekingalpha": 5, "seeking alpha": 5, "motley fool": 5, "fool.com": 5,
    "investorplace": 5, "investor place": 5, "24/7 wall st": 5,
}

# News Impact Scoring - Keywords
IMPACT_KEYWORDS_HIGH = [
    "guidance", "outlook", "forecast", "profit warning", "warning",
    "earnings", "quarterly results", "annual results", "results",
    "upgrade", "downgrade", "price target", "target raised", "target cut",
    "acquisition", "merger", "takeover", "bid", "deal",
    "ceo", "cfo", "management change", "appoints", "resigns",
    "dividend", "buyback", "share repurchase", "restructuring",
    "bankruptcy", "insolvency", "delisting",
]

IMPACT_KEYWORDS_NEUTRAL = [
    "beat", "miss", "revenue", "eps", "sales", "profit", "loss",
    "growth", "decline", "increase", "decrease",
]

IMPACT_KEYWORDS_NEGATIVE = [
    "preview", "what to expect", "ahead of", "looking ahead",
    "blog", "opinion", "commentary", "analysis",
    "review", "roundup", "weekly recap", "monthly recap",
    "top stocks", "stocks to watch", "best stocks", "worst stocks",
    "picks", "portfolio", "investment ideas",
]

# Blog/Non-Breaking News Title Patterns (for filtering)
BLOG_TITLE_PATTERNS = [
    # Listicles and roundups
    r"\d+\s+(stocks?|reasons?|things?|ways?)\s+to",
    r"top\s+\d+\s+",
    r"best\s+\d+\s+",
    r"\d+\s+best\s+",
    # Opinion/Analysis indicators
    r"^why\s+(i|you|we)\s+",
    r"^my\s+",
    r"here'?s\s+why",
    r"should\s+you\s+(buy|sell|invest)",
    r"is\s+it\s+time\s+to\s+(buy|sell)",
    r"is\s+.+\s+a\s+(buy|sell|hold)",
    # Previews and speculation
    r"what\s+to\s+(expect|watch|know)",
    r"could\s+(soar|surge|plunge|crash|rally)",
    r"(might|may|could)\s+be\s+(the\s+next|a\s+buy)",
    # Weekly/Monthly summaries
    r"(week|month)\s+in\s+review",
    r"(weekly|monthly|daily)\s+(recap|roundup|summary)",
    r"this\s+week\s+in\s+",
    # Stock picks and recommendations
    r"stock\s+pick",
    r"stocks?\s+to\s+(buy|sell|watch|avoid)",
    r"(buy|sell)\s+alert",
    # Question-based clickbait
    r"^is\s+.+\s+stock\s+",
    r"^can\s+.+\s+stock\s+",
    r"^will\s+.+\s+stock\s+",
]

# Pre-compile blog patterns for performance (compiled once at module load)
COMPILED_BLOG_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in BLOG_TITLE_PATTERNS]




def load_tickers_from_csv() -> List[str]:
    """Load tickers from CSV file (single column format)."""
    if TICKERS_FILE.exists():
        df = pd.read_csv(TICKERS_FILE)
        return df["ticker"].tolist()
    return []


def get_european_universe() -> List[str]:
    """Returns the list of European tickers from CSV."""
    return load_tickers_from_csv()


def load_keywords() -> List[str]:
    """Load news filter keywords from file."""
    if KEYWORDS_FILE.exists():
        with open(KEYWORDS_FILE, "r") as f:
            return [line.strip().lower() for line in f if line.strip()]
    return ["guidance", "earnings", "profit", "revenue", "forecast"]


def save_keywords(keywords: List[str]):
    """Save keywords to file."""
    with open(KEYWORDS_FILE, "w") as f:
        f.write("\n".join(keywords))


@lru_cache(maxsize=1)
def _load_blocked_sources_cached() -> tuple:
    """Load blocked news sources from file (cached as tuple for hashability)."""
    if BLOCKED_SOURCES_FILE.exists():
        with open(BLOCKED_SOURCES_FILE, "r") as f:
            sources = []
            for line in f:
                line = line.strip().lower()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    sources.append(line)
            return tuple(sources)
    return tuple()


def load_blocked_sources() -> List[str]:
    """Load blocked news sources from file (cached)."""
    return list(_load_blocked_sources_cached())


def is_blocked_source(publisher: str) -> bool:
    """Check if a publisher is in the blocked sources list."""
    if not publisher:
        return False
    publisher_lower = publisher.lower()
    blocked = _load_blocked_sources_cached()  # Use cached tuple directly
    return any(blocked_source in publisher_lower for blocked_source in blocked)


def is_blog_content(title: str) -> bool:
    """Check if a news title appears to be blog/opinion content rather than breaking news."""
    if not title:
        return False
    # Use pre-compiled patterns for performance (no need to lowercase - patterns are case-insensitive)
    for pattern in COMPILED_BLOG_PATTERNS:
        if pattern.search(title):
            return True
    return False


def filter_quality_news(news_items: List[Dict]) -> List[Dict]:
    """Filter out blocked sources and blog content from news items."""
    filtered = []
    for item in news_items:
        publisher = item.get("publisher", "")
        title = item.get("title", "")

        # Skip blocked sources
        if is_blocked_source(publisher):
            continue

        # Skip blog content
        if is_blog_content(title):
            continue

        filtered.append(item)
    return filtered


def format_news_timestamp(timestamp: int) -> str:
    """Format Unix timestamp to readable date/time string."""
    if not timestamp or timestamp <= 0:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        # If today, show time only
        if dt.date() == now.date():
            return dt.strftime("%H:%M")
        # If this year, show month/day and time
        elif dt.year == now.year:
            return dt.strftime("%b %d %H:%M")
        # Otherwise show full date
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


# News Impact Scoring Functions
def get_source_tier_score(publisher: str) -> int:
    """Calculate source tier score (0-30 points) based on publisher reputation."""
    if not publisher:
        return 5  # Default low score for unknown
    publisher_lower = publisher.lower()

    # Check for exact or partial matches in SOURCE_TIERS
    for source, score in SOURCE_TIERS.items():
        if source in publisher_lower:
            return score

    return 10  # Default score for unlisted sources


def calculate_keyword_score(title: str) -> int:
    """Calculate keyword-based score (0-40 points, can go negative before clamping)."""
    if not title:
        return 0

    title_lower = title.lower()
    score = 0

    # High-impact keywords: +10 each, max 40
    high_impact_count = 0
    for keyword in IMPACT_KEYWORDS_HIGH:
        if keyword in title_lower:
            high_impact_count += 1
    score += min(40, high_impact_count * 10)

    # Neutral keywords: +3 each
    for keyword in IMPACT_KEYWORDS_NEUTRAL:
        if keyword in title_lower:
            score += 3

    # Negative keywords: -15 each
    for keyword in IMPACT_KEYWORDS_NEGATIVE:
        if keyword in title_lower:
            score -= 15

    return score


def calculate_time_decay_score(timestamp: int, half_life_hours: int = 24) -> int:
    """Calculate time decay score (0-30 points) using half-life formula.

    Formula: score = 30 * (0.5 ^ (hours_old / half_life_hours))
    - Fresh news (< 1 hour): ~30 points
    - 1 day old: ~15 points
    - 2 days old: ~7.5 points
    - 7 days old: ~0.2 points
    """
    if not timestamp or timestamp <= 0:
        return 0  # No timestamp = no time score

    current_time = datetime.now().timestamp()
    hours_old = (current_time - timestamp) / 3600

    if hours_old < 0:
        hours_old = 0  # Future timestamps treated as fresh

    # Half-life decay formula
    score = 30 * (0.5 ** (hours_old / half_life_hours))
    return int(round(score))


def calculate_news_impact_score(news_item: dict) -> int:
    """Calculate overall impact score (0-100) for a news item.

    Components:
    - Source tier: 0-30 points
    - Keywords: 0-40 points
    - Time decay: 0-30 points
    """
    score = 0

    # 1. Source tier (0-30)
    publisher = news_item.get('publisher', '')
    score += get_source_tier_score(publisher)

    # 2. Keyword analysis (0-40, can go negative before clamping)
    title = news_item.get('title', '')
    keyword_score = calculate_keyword_score(title)
    score += max(0, min(40, keyword_score))  # Clamp to 0-40

    # 3. Time decay (0-30)
    timestamp = news_item.get('timestamp', 0)
    score += calculate_time_decay_score(timestamp)

    # Final clamp to 0-100
    return max(0, min(100, score))


def normalize_title(title: str) -> str:
    """Normalize title for deduplication - lowercase, strip, remove punctuation."""
    if not title:
        return ""
    # Lowercase and strip
    normalized = title.lower().strip()
    # Remove common punctuation and extra spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def deduplicate_news(news_items: List[Dict]) -> List[Dict]:
    """Group news by normalized title, consolidating tickers.

    Returns deduplicated list where same headline affecting multiple tickers
    shows as single item with 'tickers' list.
    """
    seen = {}  # key: normalized_title -> news_item with tickers list

    for item in news_items:
        key = normalize_title(item.get('title', ''))
        if not key:
            continue

        ticker = item.get('ticker', '')

        if key in seen:
            # Add ticker to existing item's ticker list
            if ticker and ticker not in seen[key]['tickers']:
                seen[key]['tickers'].append(ticker)
            # Keep the higher score version
            if item.get('impact_score', 0) > seen[key].get('impact_score', 0):
                seen[key]['impact_score'] = item.get('impact_score', 0)
        else:
            # First occurrence - create entry with tickers list
            seen[key] = {
                **item,
                'tickers': [ticker] if ticker else []
            }

    return list(seen.values())


def filter_recent_news(news_items: List[Dict], max_days: int = 7) -> List[Dict]:
    """Filter news to only include items from last N days."""
    if not news_items:
        return []

    cutoff = datetime.now().timestamp() - (max_days * 24 * 3600)

    filtered = []
    for item in news_items:
        timestamp = item.get('timestamp', 0)
        # If no timestamp, include it (we can't determine age)
        if timestamp == 0 or timestamp >= cutoff:
            filtered.append(item)

    return filtered


def process_news_for_display(news_items: List[Dict], max_days: int = 7, dedupe: bool = True, filter_quality: bool = True) -> List[Dict]:
    """Full news processing pipeline: filter → quality filter → score → dedupe → sort.

    Args:
        news_items: List of news items with ticker, title, publisher, link, timestamp
        max_days: Maximum age of news in days (default 7)
        dedupe: Whether to deduplicate by title (default True)
        filter_quality: Whether to filter out blocked sources and blog content (default True)

    Returns:
        Processed and sorted list of news items with impact_score added
    """
    if not news_items:
        return []

    # 1. Filter to recent news only
    recent_news = filter_recent_news(news_items, max_days)

    # 2. Filter out blocked sources and blog content
    if filter_quality:
        recent_news = filter_quality_news(recent_news)

    # 3. Calculate impact scores
    for item in recent_news:
        item['impact_score'] = calculate_news_impact_score(item)

    # 4. Deduplicate (optional)
    if dedupe:
        processed = deduplicate_news(recent_news)
    else:
        processed = recent_news

    # 5. Sort by impact score (highest first)
    processed.sort(key=lambda x: x.get('impact_score', 0), reverse=True)

    return processed


def get_score_badge(score: int) -> str:
    """Return HTML/markdown badge based on score."""
    if score >= 80:
        return ":green[HIGH]"
    elif score >= 50:
        return ":orange[MED]"
    else:
        return ""  # No badge for low scores


# Finnhub integration
class FinnhubClient:
    """Client for Finnhub API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a symbol."""
        if not self.is_configured():
            return None
        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "token": self.api_key}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Get company news."""
        if not self.is_configured():
            return []
        try:
            url = f"{self.base_url}/company-news"
            params = {
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": self.api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return []


@st.cache_data(ttl=300)
def fetch_indices_data() -> pd.DataFrame:
    """Fetch data for European indices with historical data for sparklines."""
    data = []

    for name, symbol in EUROPEAN_INDICES.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change_pct = ((current - previous) / previous) * 100
                # Store last 5 days of closing prices for sparkline
                sparkline_data = hist['Close'].tolist()[-5:]

                data.append({
                    "Index": name,
                    "Symbol": symbol,
                    "Price": current,
                    "Change%": change_pct,
                    "Sparkline": sparkline_data
                })
        except Exception:
            continue

    df = pd.DataFrame(data)
    # Sort by performance (best to worst)
    if not df.empty:
        df = df.sort_values("Change%", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def fetch_market_data(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches current price, previous close, volume for all tickers using batch download.
    """
    try:
        df = yf.download(tickers, period="2d", interval="1d", group_by="ticker", progress=False, threads=True)

        data = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = df
                else:
                    ticker_data = df[ticker]

                if ticker_data.empty or len(ticker_data) < 2:
                    continue

                current_price = ticker_data['Close'].iloc[-1]
                previous_close = ticker_data['Close'].iloc[-2]
                volume = ticker_data['Volume'].iloc[-1]

                if pd.isna(current_price) or pd.isna(previous_close):
                    continue

                # Convert GBX (pence) to GBP (pounds) for LSE stocks
                if ticker.upper().endswith(".L"):
                    current_price = current_price / 100
                    previous_close = previous_close / 100

                change_pct = ((current_price - previous_close) / previous_close) * 100
                notional = current_price * (volume if not pd.isna(volume) else 0)

                data.append({
                    "Ticker": ticker,
                    "Price": current_price,
                    "Change%": change_pct,
                    "Volume": volume if not pd.isna(volume) else 0,
                    "Notional": notional
                })
            except Exception:
                continue

        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def filter_by_notional(df: pd.DataFrame, min_notional: float = 20_000_000) -> pd.DataFrame:
    """Filter stocks by notional volume threshold."""
    if df.empty:
        return df

    df = df.copy()
    mask = df["Ticker"].str.endswith(".L")
    df.loc[mask, "Notional"] = df.loc[mask, "Notional"] * 1.2

    return df[df["Notional"] >= min_notional]


def get_currency_for_ticker(ticker: str) -> str:
    """Get currency symbol based on ticker suffix."""
    suffix_currency = {
        ".DE": "€",      # Germany - EUR
        ".PA": "€",      # France - EUR
        ".AS": "€",      # Netherlands - EUR
        ".BR": "€",      # Belgium - EUR
        ".MI": "€",      # Italy - EUR
        ".MA": "€",      # Spain - EUR
        ".LS": "€",      # Portugal - EUR
        ".HE": "€",      # Finland - EUR
        ".VI": "€",      # Austria - EUR
        ".L": "£",       # UK - GBP (converted from GBX)
        ".ST": "SEK",     # Sweden - SEK
        ".CO": "DKK",     # Denmark - DKK
        ".OL": "kr",     # Norway - NOK
        ".CH": "CHF",    # Switzerland - CHF
        ".PL": "zł",     # Poland - PLN
        ".WA": "zł",     # Poland - PLN
    }
    for suffix, currency in suffix_currency.items():
        if ticker.upper().endswith(suffix):
            return currency
    return "€"  # Default to EUR


def convert_to_tradingview_symbol(yahoo_symbol: str) -> str:
    """Convert Yahoo Finance symbol to TradingView format."""
    exchange_map = {
        ".DE": "XETR:", ".PA": "EURONEXT:", ".AS": "EURONEXT:",
        ".L": "LSE:", ".MI": "MIL:", ".ST": "STO:",
        ".CO": "CSE:", ".OL": "OSL:", ".CH": "SIX:", ".WA": "GPW:",
        ".BR": "EURONEXT:", ".MA": "BME:", ".LS": "EURONEXT:",
        ".HE": "OMXHEX:", ".VI": "VIE:", ".PL": "GPW:"
    }
    for suffix, prefix in exchange_map.items():
        if yahoo_symbol.upper().endswith(suffix):
            base = yahoo_symbol.upper().replace(suffix, "")
            return prefix + base
    return yahoo_symbol  # Fallback


def format_number(value: float) -> str:
    """Format number with K/M/B suffix."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.0f}"


def format_notional(value: float, ticker: str = "") -> str:
    """Formats notional value with currency."""
    currency = get_currency_for_ticker(ticker) if ticker else ""
    formatted = format_number(value)
    if currency and currency != "€":
        return f"{formatted} {currency}"
    return formatted


def format_volume(value: float) -> str:
    """Format volume for display."""
    return format_number(value)


def format_price(value: float, ticker: str = "") -> str:
    """Format price with currency symbol."""
    currency = get_currency_for_ticker(ticker) if ticker else ""
    if currency and currency != "€":
        return f"{value:.2f} {currency}"
    return f"{value:.2f}"


def convert_to_finnhub_symbol(symbol: str) -> str:
    """Convert Yahoo Finance symbol to Finnhub format."""
    # Finnhub uses different formats for European stocks
    # German stocks: SAP.DE -> SAP.XETR or just SAP
    # French stocks: MC.PA -> MC.PA
    # UK stocks: SHEL.L -> SHEL.LSE
    # Dutch stocks: ASML.AS -> ASML.AS

    if symbol.endswith('.DE'):
        return symbol  # Try as-is first
    elif symbol.endswith('.L'):
        return symbol.replace('.L', '.LSE')
    return symbol


def fetch_news_from_finnhub(symbol: str) -> List[Dict]:
    """Fetch news from Finnhub API."""
    if not FINNHUB_API_KEY:
        return []

    client = FinnhubClient(FINNHUB_API_KEY)
    finnhub_symbol = convert_to_finnhub_symbol(symbol)

    # Get news from last 7 days
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    news = client.get_company_news(finnhub_symbol, from_date, to_date)

    if not news:
        # Try without exchange suffix
        base_symbol = symbol.split('.')[0]
        news = client.get_company_news(base_symbol, from_date, to_date)

    parsed_news = []
    for item in news:
        parsed_news.append({
            "ticker": symbol,
            "title": item.get('headline', 'No title'),
            "publisher": item.get('source', 'Unknown'),
            "link": item.get('url', '#'),
            "source": "finnhub",
            "timestamp": item.get('datetime', 0)  # Unix timestamp from Finnhub
        })

    return parsed_news


def fetch_news_from_yfinance(symbol: str) -> List[Dict]:
    """Fetch news from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        parsed_news = []
        for item in news:
            news_item = {"ticker": symbol, "source": "yfinance", "timestamp": 0}

            if 'content' in item:
                content = item['content']
                news_item['title'] = content.get('title', 'No title')
                news_item['publisher'] = content.get('provider', {}).get('displayName', 'Unknown')
                news_item['link'] = content.get('canonicalUrl', {}).get('url', '#')
                # Extract timestamp from content.pubDate
                pub_date = content.get('pubDate', {})
                if isinstance(pub_date, dict):
                    news_item['timestamp'] = pub_date.get('utc', 0)
                elif isinstance(pub_date, (int, float)):
                    news_item['timestamp'] = int(pub_date)
            else:
                news_item['title'] = item.get('title', 'No title')
                news_item['publisher'] = item.get('publisher', 'Unknown')
                news_item['link'] = item.get('link', '#')
                # Extract timestamp from providerPublishTime
                news_item['timestamp'] = item.get('providerPublishTime', 0)

            parsed_news.append(news_item)

        return parsed_news
    except Exception:
        return []


def fetch_news_for_ticker(symbol: str) -> List[Dict]:
    """Fetch news for a single ticker. Uses Finnhub as primary, yfinance as fallback."""
    # Try Finnhub first (primary source)
    news = fetch_news_from_finnhub(symbol)

    if news:
        return news

    # Fallback to yfinance
    return fetch_news_from_yfinance(symbol)


@st.cache_data(ttl=300)
def fetch_news_for_movers(tickers: List[str]) -> List[Dict]:
    """Fetch news for stocks moving more than 1% - PARALLEL."""
    all_news = []
    tickers_to_fetch = tickers[:20]  # Limit to avoid too many requests

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(fetch_news_for_ticker, ticker): ticker
            for ticker in tickers_to_fetch
        }
        for future in as_completed(future_to_ticker):
            try:
                news = future.result()
                all_news.extend(news[:3])  # Top 3 news per ticker
            except Exception:
                continue
    return all_news


@st.cache_data(ttl=300)
def search_news_by_keywords(tickers: List[str], keywords: List[str]) -> List[Dict]:
    """Search news across all tickers for keyword matches - PARALLEL."""
    all_matching_news = []

    def fetch_and_filter(ticker: str) -> List[Dict]:
        """Fetch news for a ticker and filter by keywords."""
        matching = []
        news_items = fetch_news_for_ticker(ticker)
        for item in news_items:
            title = item.get('title', '').lower()
            if any(keyword in title for keyword in keywords):
                matching.append(item)
        return matching

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(fetch_and_filter, t) for t in tickers]
        for future in as_completed(futures):
            try:
                all_matching_news.extend(future.result())
            except Exception:
                continue

    return all_matching_news


def create_intraday_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Creates an intraday price chart."""
    fig = go.Figure()

    if not data.empty:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title=f"{symbol} - Intraday Price",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

    return fig


def render_news_page(ticker: str):
    """Render the news page for a specific ticker."""
    st.title(f"📰 {ticker}")
    st.divider()

    # Try to fetch basic stock info to validate ticker
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.fast_info
        full_info = yf_ticker.info

        # Get company name, sector, and market cap
        company_name = full_info.get("shortName", full_info.get("longName", ticker))
        sector = full_info.get("sector", "N/A")
        industry = full_info.get("industry", "")
        market_cap = full_info.get("marketCap", 0)

        # Display company name and sector
        st.markdown(f"**{company_name}**")
        if sector != "N/A" or industry:
            sector_text = f"{sector}" + (f" • {industry}" if industry else "")
            st.caption(sector_text)

        # Show current price info if available
        if hasattr(info, 'last_price') and info.last_price:
            # Convert GBX to GBP for LSE stocks
            display_price = info.last_price
            display_prev = info.previous_close if hasattr(info, 'previous_close') and info.previous_close else None
            if ticker.upper().endswith(".L"):
                display_price = display_price / 100
                if display_prev:
                    display_prev = display_prev / 100
            currency = get_currency_for_ticker(ticker)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Last Price", f"{currency}{display_price:.2f}")
            with col2:
                if display_prev:
                    change = display_price - display_prev
                    change_pct = (change / display_prev) * 100
                    st.metric("Change", f"{change:+.2f}", f"{change_pct:+.2f}%")
            with col3:
                if hasattr(info, 'last_volume') and info.last_volume:
                    st.metric("Volume", format_volume(info.last_volume))
            with col4:
                if market_cap and market_cap > 0:
                    st.metric("Market Cap", format_notional(market_cap))
            st.divider()
    except Exception:
        st.warning(f"Could not fetch price data for {ticker}. The ticker may be invalid.")

    # Fetch news
    news_items = fetch_news_for_ticker(ticker)
    news_source = news_items[0].get("source", "unknown") if news_items else "none"
    source_label = "Finnhub" if news_source == "finnhub" else "Yahoo Finance"

    # Process news with scoring (no deduplication for single ticker)
    processed_news = process_news_for_display(news_items, max_days=7, dedupe=False)

    st.subheader(f"📰 News (via {source_label}) - sorted by impact")

    if processed_news:
        # Header row with time column
        hcol_score, hcol_time, hcol_news = st.columns([0.5, 0.8, 5.5])
        with hcol_score:
            st.write("**Score**")
        with hcol_time:
            st.write("**Time**")
        with hcol_news:
            st.write("**Headline**")

        for item in processed_news[:15]:
            title = item.get("title", "No title")
            publisher = item.get("publisher", "Unknown")
            link = item.get("link", "#")
            impact_score = item.get("impact_score", 0)
            timestamp = item.get("timestamp", 0)
            source = item.get("source", "unknown")
            source_tag = "FH" if source == "finnhub" else "YF"

            # Format timestamp
            time_str = format_news_timestamp(timestamp)

            col_score, col_time, col_news = st.columns([0.5, 0.8, 5.5])
            with col_score:
                badge = get_score_badge(impact_score)
                st.write(f"{impact_score} {badge}")
            with col_time:
                st.caption(time_str)
            with col_news:
                # Clickable headline with source indicator
                st.markdown(f"[{title}]({link}) ({publisher}) [{source_tag}]")
    else:
        st.info("No news available for this ticker.")

    st.divider()

    # TradingView Mini Chart
    st.subheader("📊 Intraday Chart")
    tv_symbol = convert_to_tradingview_symbol(ticker)
    st.components.v1.html(f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
            {{
                "symbol": "{tv_symbol}",
                "width": "100%",
                "height": 220,
                "locale": "en",
                "dateRange": "1D",
                "colorTheme": "light",
                "isTransparent": false,
                "autosize": true,
                "largeChartUrl": ""
            }}
            </script>
        </div>
    ''', height=240)


def render_keyword_search_page():
    """Render the keyword news search page."""
    st.title("🔍 Keyword News Search")
    st.divider()

    # Keyword Settings in expandable section
    with st.expander("⚙️ Keyword Settings", expanded=True):
        col_saved, col_custom = st.columns(2)

        with col_saved:
            st.markdown("**Saved Keywords** (from news_keywords.txt)")
            saved_keywords = load_keywords()

            # Add new keyword
            col_add, col_btn = st.columns([3, 1])
            with col_add:
                new_keyword = st.text_input("Add new keyword", key="new_keyword", label_visibility="collapsed", placeholder="Add new keyword...")
            with col_btn:
                if st.button("➕ Add", use_container_width=True) and new_keyword:
                    if new_keyword.lower() not in saved_keywords:
                        saved_keywords.append(new_keyword.lower())
                        save_keywords(saved_keywords)
                        st.rerun()

            # Display saved keywords with delete buttons
            if saved_keywords:
                for i, kw in enumerate(saved_keywords):
                    col_kw, col_del = st.columns([4, 1])
                    with col_kw:
                        st.write(f"• {kw}")
                    with col_del:
                        if st.button("🗑️", key=f"del_{i}", help=f"Remove '{kw}'"):
                            saved_keywords.remove(kw)
                            save_keywords(saved_keywords)
                            st.rerun()
            else:
                st.info("No saved keywords. Add some above.")

        with col_custom:
            st.markdown("**Search Keywords** (for this search only)")
            st.caption("Edit to customize which keywords to search for")

            # Initialize with saved keywords
            default_search = ", ".join(saved_keywords) if saved_keywords else "guidance, earnings, profit"

            if "search_keywords_text" not in st.session_state:
                st.session_state["search_keywords_text"] = default_search

            search_keywords_input = st.text_area(
                "Keywords to search (comma-separated)",
                value=st.session_state["search_keywords_text"],
                height=150,
                key="search_keywords_area",
                label_visibility="collapsed",
                help="Only news containing these keywords will be shown"
            )
            st.session_state["search_keywords_text"] = search_keywords_input

            if st.button("🔄 Reset to Saved", use_container_width=True):
                st.session_state["search_keywords_text"] = ", ".join(saved_keywords)
                st.rerun()

    # Parse active search keywords
    active_keywords = [kw.strip().lower() for kw in search_keywords_input.split(",") if kw.strip()]

    st.divider()
    st.caption(f"**Active filters:** {', '.join(active_keywords) if active_keywords else 'None'}")

    if not active_keywords:
        st.warning("Please enter at least one keyword to search.")
        return

    if st.button("🔍 Search News", use_container_width=True, type="primary"):
        st.session_state["run_search"] = True

    if st.session_state.get("run_search", False):
        tickers = get_european_universe()

        with st.spinner(f"Searching news across {len(tickers)} stocks for keywords: {', '.join(active_keywords[:3])}..."):
            matching_news = search_news_by_keywords(tickers, active_keywords)
            # Process news: score, dedupe, sort by impact
            processed_news = process_news_for_display(matching_news, max_days=7, dedupe=True)
            # Fetch market data for all tickers to get change% and volume
            market_data = fetch_market_data(tickers)

        # Create a lookup dict for market data
        market_lookup = {}
        if not market_data.empty:
            for _, row in market_data.iterrows():
                market_lookup[row["Ticker"]] = {
                    "change": row["Change%"],
                    "volume": row["Volume"]
                }

        st.subheader(f"Found {len(processed_news)} news items (deduplicated)")

        if processed_news:
            # Filters and Sort row
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.5, 1, 1, 1])
            with filter_col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Impact Score (High to Low)", "Volume (High to Low)", "Change % (High to Low)", "Change % (Low to High)"],
                    key="keyword_sort"
                )
            with filter_col2:
                min_score = st.number_input("Min Score", min_value=0, max_value=100, value=0, step=10, key="keyword_min_score")
            with filter_col3:
                min_change = st.number_input("Min |Change%|", min_value=0.0, max_value=50.0, value=0.0, step=0.5, key="keyword_min_change")
            with filter_col4:
                max_results = st.number_input("Max Results", min_value=10, max_value=500, value=100, step=10, key="keyword_max_results")

            # Enrich news items with market data for sorting
            enriched_news = []
            for item in processed_news:
                tickers_list = item.get("tickers", [item.get("ticker", "")])
                first_ticker = tickers_list[0] if tickers_list else ""
                ticker_data = market_lookup.get(first_ticker, {})
                enriched_news.append({
                    **item,
                    "change": ticker_data.get("change", 0),
                    "volume": ticker_data.get("volume", 0)
                })

            # Apply filters
            filtered_news = [
                item for item in enriched_news
                if item.get("impact_score", 0) >= min_score
                and abs(item.get("change", 0)) >= min_change
            ]

            # Sort based on selection
            if "Impact Score" in sort_by:
                filtered_news.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
            elif "Volume" in sort_by:
                filtered_news.sort(key=lambda x: x["volume"], reverse=True)
            elif "High to Low" in sort_by:
                filtered_news.sort(key=lambda x: x["change"], reverse=True)
            else:  # Low to High
                filtered_news.sort(key=lambda x: x["change"], reverse=False)

            # Limit results
            filtered_news = filtered_news[:max_results]

            st.caption(f"Showing {len(filtered_news)} of {len(enriched_news)} results after filters")

            # Scrollable container with fixed height
            with st.container(height=600):
                # Header row with score column
                hcol_score, hcol_time, hcol_t, hcol_chg, hcol_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
                with hcol_score:
                    st.write("**Score**")
                with hcol_time:
                    st.write("**Time**")
                with hcol_t:
                    st.write("**Ticker(s)**")
                with hcol_chg:
                    st.write("**Chg%**")
                with hcol_news:
                    st.write("**Headline**")

                for item in filtered_news:
                    tickers_list = item.get("tickers", [item.get("ticker", "")])
                    first_ticker = tickers_list[0] if tickers_list else ""
                    title = item.get("title", "No title")
                    publisher = item.get("publisher", "Unknown")
                    link = item.get("link", "#")
                    change_pct = item.get("change", 0)
                    impact_score = item.get("impact_score", 0)
                    timestamp = item.get("timestamp", 0)
                    source = item.get("source", "unknown")
                    source_tag = "FH" if source == "finnhub" else "YF"

                    # Format timestamp
                    time_str = format_news_timestamp(timestamp)

                    # Highlight keywords in title
                    display_title = title
                    for kw in active_keywords:
                        if kw in title.lower():
                            pattern = re.compile(re.escape(kw), re.IGNORECASE)
                            display_title = pattern.sub(f"**{kw.upper()}**", display_title)

                    col_score, col_time, col_ticker, col_chg, col_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
                    with col_score:
                        badge = get_score_badge(impact_score)
                        st.write(f"{impact_score} {badge}")
                    with col_time:
                        st.caption(time_str)
                    with col_ticker:
                        ticker_display = ", ".join(tickers_list[:2])
                        if len(tickers_list) > 2:
                            ticker_display += f" +{len(tickers_list)-2}"
                        if st.button(ticker_display, key=f"news_{first_ticker}_{hash(title)}"):
                            st.query_params["ticker"] = first_ticker
                            st.rerun()
                    with col_chg:
                        color = "green" if change_pct >= 0 else "red"
                        st.write(f":{color}[{change_pct:+.1f}%]")
                    with col_news:
                        # Clickable headline with source indicator
                        st.markdown(f"[{display_title}]({link}) ({publisher}) [{source_tag}]")
        else:
            st.info("No news found matching the keywords.")


def render_home_page():
    """Render the main home page."""
    st.title("🇪🇺 European Market Dashboard")
    st.caption("Use the sidebar menu to navigate between pages")

    st.divider()

    # European Indices Table (sorted by performance)
    st.subheader("📊 Indices (Best → Worst)")
    with st.spinner("Loading indices..."):
        indices_data = fetch_indices_data()

    if not indices_data.empty:
        # Display as horizontal cards with sparklines
        cols = st.columns(5)
        for i, row in indices_data.iterrows():
            with cols[i % 5]:
                change = row["Change%"]
                # Create mini sparkline chart
                sparkline = row.get("Sparkline", [])
                if sparkline and len(sparkline) > 1:
                    fig = go.Figure()
                    line_color = "green" if change >= 0 else "red"
                    fig.add_trace(go.Scatter(
                        y=sparkline,
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        showlegend=False
                    ))
                    fig.update_layout(
                        height=40,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.metric(
                    label=row["Index"],
                    value=f"{row['Price']:,.0f}",
                    delta=f"{change:+.2f}%"
                )

    st.divider()

    # Fetch market data
    with st.spinner("Loading market data..."):
        tickers = get_european_universe()
        market_data = fetch_market_data(tickers)

    if market_data.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return

    # Filter by notional volume
    filtered_data = filter_by_notional(market_data)

    if filtered_data.empty:
        st.warning("No stocks met the notional volume filter. Showing all data.")
        filtered_data = market_data

    # Sort for gainers and losers
    gainers = filtered_data.nlargest(15, "Change%").copy()
    losers = filtered_data.nsmallest(15, "Change%").copy()

    # Get stocks moving more than 1%
    big_movers = filtered_data[abs(filtered_data["Change%"]) > 1.0]
    big_movers_tickers = big_movers["Ticker"].tolist()

    # Prepare display dataframes with formatted columns including currency
    def prepare_display_df(df):
        display_df = df[["Ticker", "Price", "Change%", "Volume", "Notional"]].copy()
        # Format price with currency
        display_df["Price"] = display_df.apply(lambda row: format_price(row["Price"], row["Ticker"]), axis=1)
        display_df["Change%"] = display_df["Change%"].apply(lambda x: f"{x:+.2f}%")
        display_df["Volume"] = display_df["Volume"].apply(format_volume)
        display_df["Notional"] = display_df.apply(lambda row: format_notional(row["Notional"] if isinstance(row["Notional"], (int, float)) else 0, row["Ticker"]), axis=1)
        display_df = display_df.reset_index(drop=True)
        return display_df

    # Keep raw data for selection
    gainers_raw = gainers.reset_index(drop=True)
    losers_raw = losers.reset_index(drop=True)
    gainers_display = prepare_display_df(gainers)
    losers_display = prepare_display_df(losers)

    # Display gainers and losers side by side with sortable dataframes
    col_gain, col_lose = st.columns(2)

    with col_gain:
        st.subheader("📈 Top Gainers")
        st.caption("Click row to view stock | Click headers to sort")

        selected_gainer = st.dataframe(
            gainers_display,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Change%": st.column_config.TextColumn("Change", width="small"),
                "Volume": st.column_config.TextColumn("Vol", width="small"),
                "Notional": st.column_config.TextColumn("Notional", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="gainers_table"
        )

        if selected_gainer and selected_gainer.selection.rows:
            selected_idx = selected_gainer.selection.rows[0]
            selected_ticker = gainers_raw.iloc[selected_idx]["Ticker"]
            st.query_params["ticker"] = selected_ticker
            st.rerun()

    with col_lose:
        st.subheader("📉 Top Losers")
        st.caption("Click row to view stock | Click headers to sort")

        selected_loser = st.dataframe(
            losers_display,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Change%": st.column_config.TextColumn("Change", width="small"),
                "Volume": st.column_config.TextColumn("Vol", width="small"),
                "Notional": st.column_config.TextColumn("Notional", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="losers_table"
        )

        if selected_loser and selected_loser.selection.rows:
            selected_idx = selected_loser.selection.rows[0]
            selected_ticker = losers_raw.iloc[selected_idx]["Ticker"]
            st.query_params["ticker"] = selected_ticker
            st.rerun()

    # News for big movers (>1% change) - filtered by keywords
    if big_movers_tickers:
        st.divider()
        keywords = load_keywords()
        st.subheader(f"📰 Keyword News for Stocks Moving >1%")
        st.caption(f"Filtering by: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''} | Sorted by impact score")

        # Create lookup for market data
        movers_lookup = {}
        for _, row in big_movers.iterrows():
            movers_lookup[row["Ticker"]] = {
                "change": row["Change%"],
                "volume": row["Volume"]
            }

        with st.spinner("Loading news for movers..."):
            movers_news = fetch_news_for_movers(big_movers_tickers)
            # Filter by keywords first
            keyword_filtered = []
            for item in movers_news:
                title = item.get("title", "").lower()
                if any(kw in title for kw in keywords):
                    keyword_filtered.append(item)
            # Process news: score, dedupe, sort by impact
            processed_news = process_news_for_display(keyword_filtered, max_days=7, dedupe=True)

        if processed_news:
            # Header row with score and time columns
            hcol_score, hcol_time, hcol_t, hcol_chg, hcol_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
            with hcol_score:
                st.write("**Score**")
            with hcol_time:
                st.write("**Time**")
            with hcol_t:
                st.write("**Ticker(s)**")
            with hcol_chg:
                st.write("**Chg%**")
            with hcol_news:
                st.write("**Headline**")

            for item in processed_news[:20]:
                # Handle deduplicated news (may have multiple tickers)
                tickers = item.get("tickers", [item.get("ticker", "")])
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                link = item.get("link", "#")
                impact_score = item.get("impact_score", 0)
                timestamp = item.get("timestamp", 0)
                source = item.get("source", "unknown")
                source_tag = "FH" if source == "finnhub" else "YF"

                # Format timestamp
                time_str = format_news_timestamp(timestamp)

                # Get market data for first ticker (for display)
                first_ticker = tickers[0] if tickers else ""
                ticker_data = movers_lookup.get(first_ticker, {})
                change_pct = ticker_data.get("change", 0)

                # Highlight keywords
                display_title = title
                for kw in keywords:
                    if kw in title.lower():
                        pattern = re.compile(re.escape(kw), re.IGNORECASE)
                        display_title = pattern.sub(f"**{kw.upper()}**", display_title)

                col_score, col_time, col_ticker, col_chg, col_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
                with col_score:
                    badge = get_score_badge(impact_score)
                    st.write(f"{impact_score} {badge}")
                with col_time:
                    st.caption(time_str)
                with col_ticker:
                    # Show tickers as buttons
                    ticker_display = ", ".join(tickers[:2])
                    if len(tickers) > 2:
                        ticker_display += f" +{len(tickers)-2}"
                    if st.button(ticker_display, key=f"mover_{first_ticker}_{hash(title)}"):
                        st.query_params["ticker"] = first_ticker
                        st.rerun()
                with col_chg:
                    color = "green" if change_pct >= 0 else "red"
                    st.write(f":{color}[{change_pct:+.1f}%]")
                with col_news:
                    # Clickable headline with source indicator
                    st.markdown(f"[{display_title}]({link}) ({publisher}) [{source_tag}]")
        else:
            st.info(f"No news matching keywords for stocks moving >1%.")

    # Finnhub API Status
    st.divider()
    with st.expander("⚙️ Finnhub API Status"):
        if FINNHUB_API_KEY:
            client = FinnhubClient(FINNHUB_API_KEY)
            st.success("Finnhub API key configured from .env file")
            if st.button("Test Connection"):
                quote = client.get_quote("AAPL")
                if quote and quote.get("c"):
                    st.success(f"Connected! AAPL price: ${quote['c']}")
                else:
                    st.error("Connection failed. Check your API key.")
        else:
            st.warning("No Finnhub API key found. Add FINNHUB_API_KEY to .env file.")

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data cached for 5 minutes")
    st.caption(f"Tickers: {TICKERS_FILE.name} | Keywords: {KEYWORDS_FILE.name}")


def render_scanner_page():
    """Render the Scanner page - finds big movers with guidance/earnings news."""
    st.title("🔎 Big Movers Scanner")
    st.divider()

    # Scanner Settings
    with st.expander("⚙️ Scanner Settings", expanded=True):
        col_thresh, col_keywords = st.columns([1, 2])

        with col_thresh:
            st.markdown("**% Change Threshold**")
            change_threshold = st.slider(
                "Minimum % move",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="scanner_threshold",
                label_visibility="collapsed"
            )

        with col_keywords:
            st.markdown("**Scanner Keywords**")
            # Default keywords
            default_keywords = "guidance, earnings, profit warning, outlook, forecast, quarterly, annual results, revenue, eps, beat, miss, upgrade, downgrade, target, analyst"

            # Initialize session state for scanner keywords
            if "scanner_keywords_text" not in st.session_state:
                st.session_state["scanner_keywords_text"] = default_keywords

            keywords_input = st.text_area(
                "Keywords (comma-separated)",
                value=st.session_state["scanner_keywords_text"],
                height=80,
                key="scanner_keywords_input",
                label_visibility="collapsed",
                help="Enter keywords separated by commas"
            )
            st.session_state["scanner_keywords_text"] = keywords_input

    # Parse keywords from input
    scanner_keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

    st.caption(f"Scanning for stocks moving >{change_threshold}% with keywords: {', '.join(scanner_keywords[:5])}{'...' if len(scanner_keywords) > 5 else ''}")

    if st.button("🔍 Run Scanner", type="primary", use_container_width=True):
        st.session_state["run_scanner"] = True

    if st.session_state.get("run_scanner", False):
        with st.spinner("Scanning European markets for big movers..."):
            # Fetch market data
            tickers = get_european_universe()
            market_data = fetch_market_data(tickers)

            if market_data.empty:
                st.error("Unable to fetch market data.")
                return

            # Filter for movers above threshold (up or down)
            big_movers = market_data[abs(market_data["Change%"]) >= change_threshold].copy()
            big_movers = big_movers.sort_values("Change%", key=abs, ascending=False)

        if big_movers.empty:
            st.info(f"No stocks currently moving more than {change_threshold}%.")
            return

        st.subheader(f"Found {len(big_movers)} stocks moving >{change_threshold}%")

        # For each big mover, fetch news and check for guidance/earnings
        results = []
        progress_bar = st.progress(0)

        for idx, (_, row) in enumerate(big_movers.iterrows()):
            ticker = row["Ticker"]
            progress_bar.progress((idx + 1) / len(big_movers))

            news = fetch_news_for_ticker(ticker)

            # Check if any news contains scanner keywords
            matching_news = []
            for item in news:
                title = item.get("title", "").lower()
                if any(kw in title for kw in scanner_keywords):
                    matching_news.append(item)

            if matching_news:
                # Process news items with scoring
                processed_news = process_news_for_display(matching_news, max_days=7, dedupe=False)
                # Get the highest scored news item
                best_news = processed_news[0] if processed_news else matching_news[0]
                best_score = best_news.get("impact_score", 0) if processed_news else 0

                results.append({
                    "ticker": ticker,
                    "price": row["Price"],
                    "change": row["Change%"],
                    "volume": row["Volume"],
                    "notional": row["Notional"],
                    "news": processed_news if processed_news else matching_news,
                    "best_score": best_score
                })

        progress_bar.empty()

        if results:
            # Sort results by best news score
            results.sort(key=lambda x: x["best_score"], reverse=True)

            st.success(f"Found {len(results)} stocks with relevant news! (sorted by impact score)")
            st.divider()

            # Header row with score and time columns
            hcol_score, hcol_time, hcol_t, hcol_c, hcol_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
            with hcol_score:
                st.write("**Score**")
            with hcol_time:
                st.write("**Time**")
            with hcol_t:
                st.write("**Ticker**")
            with hcol_c:
                st.write("**Chg%**")
            with hcol_news:
                st.write("**Headline**")

            for result in results:
                ticker = result["ticker"]
                change = result["change"]
                best_score = result["best_score"]

                # Color based on direction
                color = "green" if change > 0 else "red"
                arrow = "🟢" if change > 0 else "🔴"

                # Get best news item details
                if result["news"]:
                    item = result["news"][0]
                    title = item.get("title", "No title")
                    link = item.get("link", "#")
                    publisher = item.get("publisher", "Unknown")
                    timestamp = item.get("timestamp", 0)
                    source = item.get("source", "unknown")
                    source_tag = "FH" if source == "finnhub" else "YF"
                    time_str = format_news_timestamp(timestamp)

                    # Highlight keywords
                    display_title = title[:100] + "..." if len(title) > 100 else title
                    for kw in scanner_keywords:
                        if kw in display_title.lower():
                            pattern = re.compile(re.escape(kw), re.IGNORECASE)
                            display_title = pattern.sub(f"**{kw.upper()}**", display_title)
                else:
                    display_title = "No news"
                    link = "#"
                    publisher = ""
                    source_tag = ""
                    time_str = ""

                col_score, col_time, col_t, col_c, col_news = st.columns([0.4, 0.6, 1.0, 0.6, 4.5])
                with col_score:
                    badge = get_score_badge(best_score)
                    st.write(f"{best_score} {badge}")
                with col_time:
                    st.caption(time_str)
                with col_t:
                    if st.button(f"{arrow} {ticker}", key=f"scanner_{ticker}", use_container_width=True):
                        st.query_params["ticker"] = ticker
                        st.rerun()
                with col_c:
                    st.write(f":{color}[{change:+.1f}%]")
                with col_news:
                    # Clickable headline with source indicator
                    if publisher:
                        st.markdown(f"[{display_title}]({link}) ({publisher}) [{source_tag}]")
                    else:
                        st.write(display_title)
        else:
            st.warning("No stocks with guidance/earnings news found among big movers.")


@st.cache_data(ttl=3600)
def fetch_tradingeconomics_earnings(country: str) -> List[Dict]:
    """Fetch earnings calendar from TradingEconomics for a specific country."""
    url = f"https://tradingeconomics.com/{country}/earnings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        earnings_data = []

        # Find the earnings table
        table = soup.find('table', {'class': 'table'})
        if not table:
            return []

        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                try:
                    # Parse the row data
                    date_text = cols[0].get_text(strip=True)
                    company = cols[1].get_text(strip=True)
                    # Some tables have different structures
                    event = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                    actual = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                    forecast = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                    if company and date_text:
                        earnings_data.append({
                            "Date": date_text,
                            "Company": company,
                            "Event": event,
                            "Actual": actual,
                            "Forecast": forecast,
                            "Country": country.replace("-", " ").title()
                        })
                except Exception:
                    continue

        return earnings_data
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_all_european_earnings() -> pd.DataFrame:
    """Fetch earnings calendar for all major European countries - PARALLEL."""
    european_countries = [
        "germany", "united-kingdom", "france", "netherlands",
        "switzerland", "spain", "italy", "belgium", "sweden",
        "denmark", "norway", "finland", "austria", "poland"
    ]

    all_earnings = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_country = {
            executor.submit(fetch_tradingeconomics_earnings, country): country
            for country in european_countries
        }
        for future in as_completed(future_to_country):
            try:
                country_earnings = future.result()
                all_earnings.extend(country_earnings)
            except Exception:
                continue

    if not all_earnings:
        return pd.DataFrame()

    df = pd.DataFrame(all_earnings)
    return df


def render_earnings_calendar_page():
    """Render the European Earnings Calendar page."""
    st.title("📅 European Earnings Calendar")
    st.caption("Earnings reports from TradingEconomics.com - European stocks only")
    st.divider()

    # Country filter
    all_countries = [
        "All Countries", "Germany", "United Kingdom", "France", "Netherlands",
        "Switzerland", "Spain", "Italy", "Belgium", "Sweden",
        "Denmark", "Norway", "Finland", "Austria", "Poland"
    ]

    selected_country = st.selectbox("Filter by Country", all_countries, key="earnings_country")

    if st.button("🔄 Refresh Earnings Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Fetch earnings data
    with st.spinner("Fetching earnings calendar from TradingEconomics..."):
        earnings_df = fetch_all_european_earnings()

    if earnings_df.empty:
        st.warning("Could not fetch earnings data from TradingEconomics.")
        st.markdown("""
        **This may be due to:**
        - TradingEconomics website changes
        - Rate limiting
        - Network issues

        **Alternative sources:**
        - [TradingEconomics Earnings](https://tradingeconomics.com/earnings)
        - [Investing.com Earnings Calendar](https://www.investing.com/earnings-calendar/)
        - [TradingView Earnings](https://www.tradingview.com/markets/stocks-europe/earnings/)
        """)
    else:
        # Filter by country if selected
        if selected_country != "All Countries":
            earnings_df = earnings_df[earnings_df["Country"] == selected_country]

        if earnings_df.empty:
            st.info(f"No earnings data available for {selected_country}.")
        else:
            st.subheader(f"Earnings Calendar ({len(earnings_df)} entries)")

            # Display as sortable dataframe
            st.dataframe(
                earnings_df,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Event": st.column_config.TextColumn("Event", width="medium"),
                    "Actual": st.column_config.TextColumn("Actual", width="small"),
                    "Forecast": st.column_config.TextColumn("Forecast", width="small"),
                    "Country": st.column_config.TextColumn("Country", width="small"),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )

            # Summary by country
            st.divider()
            st.subheader("📊 Summary by Country")
            country_counts = earnings_df["Country"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                for country in country_counts.index[:len(country_counts)//2 + 1]:
                    st.write(f"**{country}:** {country_counts[country]} earnings")
            with col2:
                for country in country_counts.index[len(country_counts)//2 + 1:]:
                    st.write(f"**{country}:** {country_counts[country]} earnings")

    st.divider()
    st.caption("Data source: [TradingEconomics.com](https://tradingeconomics.com/earnings) | Cached for 1 hour")


def main():
    st.set_page_config(
        page_title="European Market Dashboard",
        page_icon="🇪🇺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation - always visible
    with st.sidebar:
        st.title("🇪🇺 Navigation")

        # Search bar in sidebar
        st.subheader("🔍 Quick Search")
        search_ticker = st.text_input(
            "Search Ticker",
            placeholder="e.g., SAP.DE, AAPL",
            key="sidebar_search",
            label_visibility="collapsed"
        )
        if st.button("Search", key="sidebar_search_btn", use_container_width=True):
            if search_ticker:
                st.query_params["ticker"] = search_ticker.upper().strip()
                st.query_params.pop("page", None)
                st.rerun()

        st.divider()

        # Navigation menu
        st.subheader("📋 Pages")

        if st.button("🏠 Home Dashboard", use_container_width=True, type="primary" if not st.query_params.get("page") and not st.query_params.get("ticker") else "secondary"):
            st.query_params.clear()
            st.rerun()

        if st.button("🔍 News Search", use_container_width=True, type="primary" if st.query_params.get("page") == "keyword_search" else "secondary"):
            st.query_params.clear()
            st.query_params["page"] = "keyword_search"
            st.rerun()

        if st.button("🔎 Big Movers Scanner", use_container_width=True, type="primary" if st.query_params.get("page") == "scanner" else "secondary"):
            st.query_params.clear()
            st.query_params["page"] = "scanner"
            st.rerun()

        if st.button("📅 Earnings Calendar", use_container_width=True, type="primary" if st.query_params.get("page") == "earnings" else "secondary"):
            st.query_params.clear()
            st.query_params["page"] = "earnings"
            st.rerun()

        st.divider()

        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # History / Back button
        if st.query_params.get("ticker") or st.query_params.get("page"):
            if st.button("⬅️ Back to Home", use_container_width=True):
                st.query_params.clear()
                st.rerun()

        # Footer
        st.caption("---")
        st.caption(f"📊 {len(get_european_universe())} stocks tracked")
        st.caption(f"⏱️ Data cached 5 min")

    # Main content area - route to appropriate page
    params = st.query_params
    selected_ticker = params.get("ticker", None)
    page = params.get("page", None)

    if selected_ticker:
        render_news_page(selected_ticker.upper())
    elif page == "keyword_search":
        render_keyword_search_page()
    elif page == "scanner":
        render_scanner_page()
    elif page == "earnings":
        render_earnings_calendar_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
