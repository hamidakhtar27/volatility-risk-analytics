import yfinance as yf
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_RAW = Path("data/raw")


# -------------------------------------------------------------------
# Data Download
# -------------------------------------------------------------------
def download_price_data(
    ticker: str,
    start: str = "2005-01-01",
    end: str = None
) -> pd.DataFrame:
    """
    Download daily market price data and return a clean price series.

    Robust to:
    - MultiIndex columns (new yfinance behavior)
    - Missing 'Adj Close' (falls back to 'Close')

    Returns:
        pd.DataFrame with a single column: ['price']
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Handle MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Prefer Adjusted Close, fallback to Close
    if "Adj Close" in df.columns:
        price = df["Adj Close"]
    elif "Close" in df.columns:
        price = df["Close"]
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in data")

    price = price.to_frame(name="price")
    price.dropna(inplace=True)

    return price


# -------------------------------------------------------------------
# Save & Load Utilities
# -------------------------------------------------------------------
def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save raw data to data/raw directory.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RAW / filename)


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw data from data/raw directory.
    """
    filepath = DATA_RAW / filename
    if not filepath.exists():
        raise FileNotFoundError(f"{filename} not found in data/raw/")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
