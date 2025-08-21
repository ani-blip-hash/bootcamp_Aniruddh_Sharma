import os
import requests
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime
import yfinance as yf
from typing import Optional, List
import time
import json

def get_summary_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return basic summary statistics for numeric columns."""
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found for summary statistics")
        return pd.DataFrame()
    return dataframe[numeric_cols].describe()

# ---------- Enhanced Saving ----------
def save_with_timestamp(df: pd.DataFrame, prefix: str, source: str, ext: str = "csv") -> str:
    """Save DataFrame with timestamp and create directory structure."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.getenv("DATA_DIR", "./data/raw")
    
    # Create nested directory structure
    source_dir = os.path.join(data_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    
    filename = f"{prefix}_{ts}.{ext}"
    path = os.path.join(source_dir, filename)
    
    try:
        if ext.lower() == "csv":
            df.to_csv(path, index=False)
        elif ext.lower() == "json":
            df.to_json(path, orient="records", lines=True, date_format='iso')
        elif ext.lower() in ["xlsx", "excel"]:
            df.to_excel(path, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        print(f"💾 Saved {len(df)} rows to {path}")
        return path
    except Exception as e:
        print(f"❌ Failed to save file: {str(e)}")
        raise

# ---------- Alpha Vantage API Functions ----------
def fetch_alphavantage(symbol: str, function: str = "TIME_SERIES_DAILY", 
    outputsize: str = "compact", datatype: str = "json") -> pd.DataFrame:
    """
    Fetch data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., 'MSFT')
        function: API function (TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, etc.)
        outputsize: 'compact' (last 100 data points) or 'full' (full-length time series)
        datatype: 'json' or 'csv'
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables")
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": datatype
    }
    
    try:
        print(f"📊 Fetching {symbol} from Alpha Vantage ({function})")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        if datatype == "json":
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            
            if "Note" in data:
                print(f"⚠️ API Note: {data['Note']}")
                return pd.DataFrame()
            
            # Parse time series data
            df = _parse_alphavantage_json(data, symbol, function)
            
        else:  # CSV format
            df = pd.read_csv(response.text)
            df['symbol'] = symbol.upper()
            df['fetch_timestamp'] = datetime.now()
        
        if not df.empty:
            print(f"✅ Fetched {len(df)} records from Alpha Vantage for {symbol}")
        
        return df
        
    except requests.RequestException as e:
        print(f"❌ Network error fetching {symbol} from Alpha Vantage: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching {symbol} from Alpha Vantage: {str(e)}")
        return pd.DataFrame()

def _parse_alphavantage_json(data: dict, symbol: str, function: str) -> pd.DataFrame:
    """Parse Alpha Vantage JSON response into DataFrame."""
    
    # Map function types to their time series keys
    time_series_keys = {
        "TIME_SERIES_DAILY": "Time Series (Daily)",
        "TIME_SERIES_WEEKLY": "Weekly Time Series",
        "TIME_SERIES_MONTHLY": "Monthly Time Series",
        "TIME_SERIES_INTRADAY": "Time Series (5min)",  # default for intraday
    }
    
    # Find the time series data in response
    time_series_data = None
    for key in data.keys():
        if "Time Series" in key:
            time_series_data = data[key]
            break
    
    if not time_series_data:
        print(f"❌ No time series data found in Alpha Vantage response")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series_data, orient='index')
    
    # Reset index to get date column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    # Standardize column names (remove numbers and spaces, lowercase)
    column_mapping = {
        '1. open': 'open',
        '2. high': 'high', 
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume',
        '1. Open': 'open',
        '2. High': 'high',
        '3. Low': 'low', 
        '4. Close': 'close',
        '5. Volume': 'volume'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Convert date column and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True, ignore_index=True)
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add metadata
    df['symbol'] = symbol.upper()
    df['fetch_timestamp'] = datetime.now()
    df['data_source'] = 'alphavantage'
    
    return df

def fetch_stock_data(symbol: str, prefer_alphavantage: bool = True, 
    period: str = "6mo") -> pd.DataFrame:
    """
    Fetch stock data with Alpha Vantage as primary and yfinance as fallback.
    
    Args:
        symbol: Stock symbol
        prefer_alphavantage: If True, try Alpha Vantage first
        period: Period for yfinance fallback ('1d', '5d', '1mo', '3mo', '6mo', '1y', etc.)
    """
    
    if prefer_alphavantage:
        # Try Alpha Vantage first
        df = fetch_alphavantage(symbol, outputsize="compact")
        
        if not df.empty:
            return df
        
        print(f"⚠️ Alpha Vantage failed for {symbol}, falling back to yfinance...")
        time.sleep(1)  # Brief delay before fallback
    
    # Fallback to yfinance
    print(f"📈 Using yfinance fallback for {symbol}")
    df = fetch_yfinance(symbol, period=period)
    
    if not df.empty:
        df['data_source'] = 'yfinance'
    
    return df
def fetch_yfinance(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical stock data using yfinance with enhanced error handling."""
    try:
        print(f"📈 Fetching {symbol} data (period={period}, interval={interval})")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"❌ No data returned for {symbol}")
            return pd.DataFrame()

        # Reset index to get 'date' column
        df = df.reset_index()
        
        # Standardize column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # Add metadata
        df['symbol'] = symbol.upper()
        df['fetch_timestamp'] = datetime.now()
        df['data_source'] = 'yfinance'
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ Fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# ---------- Enhanced Web Scraping ----------
def scrape_sp500_table() -> pd.DataFrame:
    """Scrape S&P 500 companies table from Wikipedia with enhanced error handling."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        print(f"🌐 Scraping S&P500 data from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse tables
        dfs = pd.read_html(response.text)
        
        if len(dfs) == 0:
            raise ValueError("No tables found on the page")
        
        # The first table is typically the S&P 500 constituents
        df = dfs[0].copy()
        
        # Add metadata
        df['scrape_date'] = datetime.now().date()
        df['source_url'] = url
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        print(f"✅ Scraped {len(df)} S&P500 companies")
        return df
        
    except requests.RequestException as e:
        print(f"❌ Network error while scraping: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error scraping S&P500 data: {str(e)}")
        return pd.DataFrame()

# ---------- Enhanced Validation ----------
def validate_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None, 
    min_rows: int = 1) -> bool:
    """Enhanced DataFrame validation with detailed reporting."""
    print(f"\n🔍 Validating DataFrame...")
    
    if df is None:
        raise ValueError("DataFrame is None")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check required columns
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            available_cols = df.columns.tolist()
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {available_cols}"
            )
    
    # Report validation results
    print(f"✅ Validation passed!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check for missing data
    na_counts = df.isna().sum()
    if na_counts.sum() > 0:
        print(f"⚠️  Missing data found:")
        for col, count in na_counts[na_counts > 0].items():
            pct = (count / len(df)) * 100
            print(f"     {col}: {count} ({pct:.1f}%)")
    else:
        print("✅ No missing data found")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Memory usage: {memory_mb:.2f} MB")
    
    return True

# ---------- Additional Utility Functions ----------
def fetch_multiple_stocks(symbols: List[str], prefer_alphavantage: bool = True, 
    period: str = "6mo") -> pd.DataFrame:
    """Fetch data for multiple stock symbols using primary/fallback approach."""
    all_data = []
    alphavantage_requests = 0
    max_alphavantage_requests = 5  # Free tier limit per minute
    
    for i, symbol in enumerate(symbols):
        # Rate limiting for Alpha Vantage (free tier: 5 requests per minute)
        if prefer_alphavantage and alphavantage_requests >= max_alphavantage_requests:
            print(f"⚠️ Alpha Vantage rate limit reached, using yfinance for remaining symbols")
            prefer_alphavantage = False
        
        df = fetch_stock_data(symbol, prefer_alphavantage=prefer_alphavantage, period=period)
        
        if not df.empty:
            all_data.append(df)
            if prefer_alphavantage and df.get('data_source', '').iloc[0] == 'alphavantage':
                alphavantage_requests += 1
        
        # Rate limiting delay for Alpha Vantage
        if prefer_alphavantage and i < len(symbols) - 1:
            print("⏳ Rate limiting delay...")
            time.sleep(12)  # 12 seconds between requests for free tier
    
    if not all_data:
        print("❌ No data fetched for any symbols")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Report data sources used
    source_counts = combined_df['data_source'].value_counts()
    print(f"✅ Combined data for {len(symbols)} symbols: {combined_df.shape}")
    print(f"📊 Data sources used: {dict(source_counts)}")
    
    return combined_df

def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data quality report."""
    report = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_data': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        report['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return report
    
def get_alphavantage_functions() -> dict:
    """Return available Alpha Vantage functions and their descriptions."""
    return {
        "TIME_SERIES_INTRADAY": "Intraday time series (1min, 5min, 15min, 30min, 60min)",
        "TIME_SERIES_DAILY": "Daily time series (last 100 days compact, 20+ years full)",
        "TIME_SERIES_DAILY_ADJUSTED": "Daily time series with dividend/split adjustments",
        "TIME_SERIES_WEEKLY": "Weekly time series",
        "TIME_SERIES_MONTHLY": "Monthly time series",
        "GLOBAL_QUOTE": "Latest price and volume info",
        "SYMBOL_SEARCH": "Search for symbols",
        "OVERVIEW": "Company overview (fundamental data)"
    }

def fetch_company_overview(symbol: str) -> pd.DataFrame:
    """Fetch company fundamental data from Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables")
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        print(f"🏢 Fetching company overview for {symbol}")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "Error Message" in data or not data:
            print(f"❌ No company data found for {symbol}")
            return pd.DataFrame()
        
        # Convert single record to DataFrame
        df = pd.DataFrame([data])
        df['fetch_timestamp'] = datetime.now()
        df['data_source'] = 'alphavantage_overview'
        
        print(f"✅ Fetched company overview for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching company overview for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_global_quote(symbol: str) -> pd.DataFrame:
    """Fetch latest quote data from Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables")
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        print(f"💹 Fetching global quote for {symbol}")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "Error Message" in data:
            print(f"❌ Error: {data['Error Message']}")
            return pd.DataFrame()
        
        if "Global Quote" not in data:
            print(f"❌ No quote data found for {symbol}")
            return pd.DataFrame()
        
        quote_data = data["Global Quote"]
        
        # Standardize column names
        standardized_data = {}
        for key, value in quote_data.items():
            clean_key = key.split('. ')[-1].lower().replace(' ', '_')
            standardized_data[clean_key] = value
        
        df = pd.DataFrame([standardized_data])
        df['fetch_timestamp'] = datetime.now()
        df['data_source'] = 'alphavantage_quote'
        
        print(f"✅ Fetched quote for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching quote for {symbol}: {str(e)}")
        return pd.DataFrame()
