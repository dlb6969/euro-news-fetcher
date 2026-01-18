"""
European Market Dashboard
A Streamlit-based dashboard for European stock screening and news search.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import re


# File paths
TICKERS_FILE = Path(__file__).parent / "european_tickers.csv"
KEYWORDS_FILE = Path(__file__).parent / "news_keywords.txt"


def load_tickers_from_csv() -> pd.DataFrame:
    """Load tickers from CSV file."""
    if TICKERS_FILE.exists():
        return pd.read_csv(TICKERS_FILE)
    return pd.DataFrame(columns=["ticker", "country", "name"])


def get_european_universe() -> List[str]:
    """Returns the list of European tickers from CSV."""
    df = load_tickers_from_csv()
    return df["ticker"].tolist()


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


@st.cache_data(ttl=300)
def fetch_market_data(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches current price, previous close, volume for all tickers using batch download.
    Calculates: % change, notional volume (price * volume).
    Returns DataFrame with columns: Ticker, Price, Change%, Volume, Notional
    """
    try:
        # Batch download - much faster than individual requests
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
    """
    Filters stocks where Notional Volume > threshold.
    Applies 1.2x factor for .L (GBP) tickers.
    """
    if df.empty:
        return df

    df = df.copy()

    # Apply 1.2x factor for UK tickers (GBP to EUR approximation)
    mask = df["Ticker"].str.endswith(".L")
    df.loc[mask, "Notional"] = df.loc[mask, "Notional"] * 1.2

    return df[df["Notional"] >= min_notional]


def format_notional(value: float) -> str:
    """Formats notional value as '25M', '1.2B' etc."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.0f}K"
    else:
        return f"{value:.0f}"


def fetch_news_for_ticker(symbol: str) -> List[Dict]:
    """Fetch news for a single ticker."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        parsed_news = []
        for item in news:
            news_item = {"ticker": symbol}

            if 'content' in item:
                content = item['content']
                news_item['title'] = content.get('title', 'No title')
                news_item['publisher'] = content.get('provider', {}).get('displayName', 'Unknown')
                news_item['link'] = content.get('canonicalUrl', {}).get('url', '#')
            else:
                news_item['title'] = item.get('title', 'No title')
                news_item['publisher'] = item.get('publisher', 'Unknown')
                news_item['link'] = item.get('link', '#')

            parsed_news.append(news_item)

        return parsed_news
    except Exception:
        return []


@st.cache_data(ttl=300)
def search_news_by_keywords(tickers: List[str], keywords: List[str]) -> List[Dict]:
    """Search news across all tickers for keyword matches."""
    all_matching_news = []

    for ticker in tickers:
        news_items = fetch_news_for_ticker(ticker)
        for item in news_items:
            title = item.get('title', '').lower()
            if any(keyword in title for keyword in keywords):
                all_matching_news.append(item)

    return all_matching_news


def create_intraday_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Creates an intraday price chart using Plotly."""
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
    st.title(f"📰 News for {ticker}")

    # Home button
    if st.button("🏠 Back to Home", use_container_width=False):
        st.query_params.clear()
        st.rerun()

    st.divider()

    # Fetch and display all news (no filtering)
    news_items = fetch_news_for_ticker(ticker)

    if news_items:
        for item in news_items[:15]:
            title = item.get("title", "No title")
            publisher = item.get("publisher", "Unknown")
            link = item.get("link", "#")
            st.markdown(f"- **{title}** ({publisher}) [🔗]({link})")
    else:
        st.info("No news available for this ticker.")

    st.divider()

    # Intraday chart
    st.subheader("📊 Intraday Chart")
    try:
        yf_ticker = yf.Ticker(ticker)
        intraday_data = yf_ticker.history(period='1d', interval='5m')

        if not intraday_data.empty:
            chart = create_intraday_chart(intraday_data, ticker)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No intraday data available for this ticker.")
    except Exception:
        st.warning("No intraday data available for this ticker.")


def render_keyword_search_page():
    """Render the keyword news search page."""
    st.title("🔍 Keyword News Search")

    # Home button
    if st.button("🏠 Back to Home", use_container_width=False):
        st.query_params.clear()
        st.rerun()

    st.divider()

    # Keyword management
    keywords = load_keywords()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Current Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords set")

    with col2:
        with st.expander("Manage Keywords"):
            new_keyword = st.text_input("Add keyword", key="new_keyword")
            if st.button("Add") and new_keyword:
                if new_keyword.lower() not in keywords:
                    keywords.append(new_keyword.lower())
                    save_keywords(keywords)
                    st.cache_data.clear()
                    st.rerun()

            st.write("**Remove keywords:**")
            for i, kw in enumerate(keywords):
                col_kw, col_del = st.columns([3, 1])
                with col_kw:
                    st.write(f"• {kw}")
                with col_del:
                    if st.button("❌", key=f"del_{i}"):
                        keywords.remove(kw)
                        save_keywords(keywords)
                        st.cache_data.clear()
                        st.rerun()

    st.divider()

    # Search button
    if st.button("🔍 Search News", use_container_width=True, type="primary"):
        st.session_state["run_search"] = True

    if st.session_state.get("run_search", False):
        tickers = get_european_universe()

        with st.spinner(f"Searching news across {len(tickers)} stocks..."):
            matching_news = search_news_by_keywords(tickers, keywords)

        st.subheader(f"Found {len(matching_news)} matching news items")

        if matching_news:
            for item in matching_news:
                ticker = item.get("ticker", "")
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                link = item.get("link", "#")

                # Highlight matching keywords
                display_title = title
                for kw in keywords:
                    if kw in title.lower():
                        pattern = re.compile(re.escape(kw), re.IGNORECASE)
                        display_title = pattern.sub(f"**{kw.upper()}**", display_title)

                col_ticker, col_news = st.columns([1, 5])
                with col_ticker:
                    if st.button(ticker, key=f"news_{ticker}_{hash(title)}"):
                        st.query_params["ticker"] = ticker
                        st.rerun()
                with col_news:
                    st.markdown(f"{display_title} ({publisher}) [🔗]({link})")
        else:
            st.info("No news found matching the keywords.")


def render_home_page():
    """Render the main home page."""
    st.title("🇪🇺 European Market Dashboard")

    # Top row: Search bar and buttons
    col_search, col_news_btn, col_refresh = st.columns([3, 1, 1])

    with col_search:
        # Load tickers for autocomplete
        tickers_df = load_tickers_from_csv()
        ticker_options = tickers_df["ticker"].tolist()

        search_ticker = st.selectbox(
            "Search Stock",
            options=[""] + ticker_options,
            format_func=lambda x: "Type to search..." if x == "" else x,
            key="stock_search"
        )

        if search_ticker:
            st.query_params["ticker"] = search_ticker
            st.rerun()

    with col_news_btn:
        st.write("")  # Spacing
        if st.button("🔍 News Search", use_container_width=True):
            st.query_params["page"] = "keyword_search"
            st.rerun()

    with col_refresh:
        st.write("")  # Spacing
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

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
    gainers = filtered_data.nlargest(10, "Change%")
    losers = filtered_data.nsmallest(10, "Change%")

    # Display gainers and losers side by side
    col_gain, col_lose = st.columns(2)

    with col_gain:
        st.subheader("📈 Top Gainers")
        for _, row in gainers.iterrows():
            ticker = row["Ticker"]
            price = f"{row['Price']:.2f}"
            change = f"{row['Change%']:+.2f}%"
            notional = format_notional(row['Notional'])

            col_t, col_p, col_c, col_n = st.columns([2, 1.5, 1.5, 1.5])
            with col_t:
                if st.button(ticker, key=f"gain_{ticker}", use_container_width=True):
                    st.query_params["ticker"] = ticker
                    st.rerun()
            with col_p:
                st.write(price)
            with col_c:
                st.write(f":green[{change}]")
            with col_n:
                st.write(notional)

    with col_lose:
        st.subheader("📉 Top Losers")
        for _, row in losers.iterrows():
            ticker = row["Ticker"]
            price = f"{row['Price']:.2f}"
            change = f"{row['Change%']:+.2f}%"
            notional = format_notional(row['Notional'])

            col_t, col_p, col_c, col_n = st.columns([2, 1.5, 1.5, 1.5])
            with col_t:
                if st.button(ticker, key=f"lose_{ticker}", use_container_width=True):
                    st.query_params["ticker"] = ticker
                    st.rerun()
            with col_p:
                st.write(price)
            with col_c:
                st.write(f":red[{change}]")
            with col_n:
                st.write(notional)

    # Footer with last update time
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data cached for 5 minutes")
    st.caption(f"Tickers: {TICKERS_FILE.name} | Keywords: {KEYWORDS_FILE.name}")


def main():
    st.set_page_config(
        page_title="European Market Dashboard",
        page_icon="🇪🇺",
        layout="wide"
    )

    # Check query params for navigation
    params = st.query_params
    selected_ticker = params.get("ticker", None)
    page = params.get("page", None)

    if selected_ticker:
        render_news_page(selected_ticker.upper())
    elif page == "keyword_search":
        render_keyword_search_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
