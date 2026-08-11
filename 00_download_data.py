"""
00_download_data.py
---------------------
Downloads S&P 500 and Nifty 50 historical data via yfinance.
Run this LOCALLY on your machine (not in a sandboxed environment).

If you hit YFRateLimitError, this script auto-retries with backoff.
If it keeps failing, wait 5-10 minutes and rerun -- Yahoo rate-limits
by IP, and it resets on its own.
"""
import time
import yfinance as yf

def download_with_retry(ticker, start="2000-01-01", max_retries=5, wait=30):
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                print(f"{ticker}: downloaded {df.shape[0]} rows")
                return df
            else:
                print(f"{ticker}: empty result on attempt {attempt}, retrying in {wait}s...")
        except Exception as e:
            print(f"{ticker}: attempt {attempt} failed ({e}), retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts. "
                        f"Try again in a few minutes -- Yahoo rate-limits by IP.")


if __name__ == "__main__":
    sp500 = download_with_retry("^GSPC", start="2000-01-01")
    time.sleep(5)  # small gap between requests to reduce rate-limit risk
    nifty = download_with_retry("^NSEI", start="2000-01-01")

    # Flatten column names if yfinance returns MultiIndex columns
    if isinstance(sp500.columns, __import__("pandas").MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    if isinstance(nifty.columns, __import__("pandas").MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    sp500 = sp500.reset_index()
    nifty = nifty.reset_index()

    sp500["Return"] = sp500["Close"].pct_change()
    nifty["Return"] = nifty["Close"].pct_change()

    sp500.to_csv("data/sp500_real.csv", index=False)
    nifty.to_csv("data/nifty_real.csv", index=False)

    print("\nSaved data/sp500_real.csv and data/nifty_real.csv")
    print(f"SP500: {sp500['Date'].min()} to {sp500['Date'].max()}, {len(sp500)} rows")
    print(f"NIFTY: {nifty['Date'].min()} to {nifty['Date'].max()}, {len(nifty)} rows")
