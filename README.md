# EVT Tail Risk Project — S&P 500 vs Nifty 50

Extreme Value Theory (Peak-Over-Threshold / GPD) tail risk model, benchmarked
against Normal, Student-t, and Historical VaR/CVaR, validated with Kupiec and
Christoffersen backtests.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
python3 00_download_data.py          # downloads real S&P 500 + Nifty 50 data via yfinance
python3 01_baseline_var.py           # Normal / Student-t / Historical VaR & CVaR
python3 02_evt_pot_model.py          # EVT / POT / GPD tail model
python3 03_backtest_and_compare.py   # rolling backtest + Kupiec/Christoffersen tests
python3 04_visualize.py              # diagnostic plots -> outputs/evt_diagnostic_plots.png
```

Each script reads/writes CSVs from `data/` and `outputs/` (relative paths,
run from the project root).

## If step 00 fails with YFRateLimitError

Yahoo rate-limits by IP. The script already retries with backoff, but if it
keeps failing:
- Wait 5-10 minutes and rerun `00_download_data.py`
- Or manually download CSVs from Yahoo Finance's Historical Data tab for
  `^GSPC` and `^NSEI`, and save them as `data/sp500_real.csv` and
  `data/nifty_real.csv` with at least `Date`, `Close` columns — then add a
  `Return` column (`Close.pct_change()`) before running steps 01-04.

## Files

| File | Purpose |
|---|---|
| `00_download_data.py` | Pulls historical index data via yfinance |
| `01_baseline_var.py` | Normal / Student-t / Historical VaR & CVaR (baseline comparison models) |
| `02_evt_pot_model.py` | Core EVT: Peak-Over-Threshold, GPD fit, tail index ξ, EVT VaR/CVaR |
| `03_backtest_and_compare.py` | Rolling walk-forward backtest, Kupiec POF test, Christoffersen independence test |
| `04_visualize.py` | Mean excess plot, tail distribution overlay, method comparison chart |

## Notes on parameters to tune once real data loads

- `threshold_pct` in `02_evt_pot_model.py` (default 90th percentile) — check
  the mean excess plot in step 04's output; pick a threshold where the plot
  is roughly linear.
- `window` in `03_backtest_and_compare.py` (default 500, comment suggests
  1000-1250 for real data) — larger window = more stable but slower-adapting
  VaR estimates.
- Expect the tail index ξ to come out **positive** on real equity data
  (heavy-tailed) — this is the expected and interesting finding, unlike the
  placeholder synthetic data used during development.
