"""
BASELINE VaR / CVaR MODELS
----------------------------
Implements the three standard approaches EVT will be benchmarked against:
1. Normal (parametric) VaR/CVaR
2. Student-t (parametric) VaR/CVaR
3. Historical Simulation (non-parametric, empirical quantile)

These are the models a typical risk desk uses day-to-day. EVT's whole
value proposition is that it outperforms these specifically in the tail
(99%, 99.5%), which is what we'll demonstrate in the backtest step.
"""
import numpy as np
import pandas as pd
from scipy import stats


def normal_var_cvar(returns, alpha=0.99):
    """Parametric VaR/CVaR assuming Normal distribution of returns."""
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    # CVaR (Expected Shortfall) under normality has closed form
    cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    return var, cvar


def student_t_var_cvar(returns, alpha=0.99):
    """Parametric VaR/CVaR assuming Student-t distribution (fits fat tails
    better than Normal, but still doesn't specifically model the tail)."""
    params = stats.t.fit(returns)
    df, loc, scale = params
    var = -stats.t.ppf(1 - alpha, df, loc=loc, scale=scale)
    # Monte Carlo CVaR since t-CVaR closed form is messier
    sim = stats.t.rvs(df, loc=loc, scale=scale, size=200_000, random_state=1)
    threshold = -var
    cvar = -sim[sim <= threshold].mean() if (sim <= threshold).sum() > 0 else var
    return var, cvar, df


def historical_var_cvar(returns, alpha=0.99):
    """Non-parametric: just take the empirical quantile of realized returns.
    Simple, robust, but needs a LOT of data to estimate 99%+ tails well —
    this is exactly the weakness EVT is designed to fix."""
    var = -np.percentile(returns, (1 - alpha) * 100)
    tail = returns[returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return var, cvar


def run_all_baselines(returns, name="INDEX"):
    results = []
    for alpha in [0.95, 0.99, 0.995]:
        n_var, n_cvar = normal_var_cvar(returns, alpha)
        t_var, t_cvar, t_df = student_t_var_cvar(returns, alpha)
        h_var, h_cvar = historical_var_cvar(returns, alpha)
        results.append({
            "index": name, "confidence": alpha,
            "normal_VaR": n_var, "normal_CVaR": n_cvar,
            "student_t_VaR": t_var, "student_t_CVaR": t_cvar, "t_df": t_df,
            "historical_VaR": h_var, "historical_CVaR": h_cvar,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    sp500 = pd.read_csv("data/sp500_real.csv")
    nifty = pd.read_csv("data/nifty_real.csv")

    sp_results = run_all_baselines(sp500["Return"].dropna().values, "SP500")
    nf_results = run_all_baselines(nifty["Return"].dropna().values, "NIFTY")

    combined = pd.concat([sp_results, nf_results], ignore_index=True)
    combined.to_csv("outputs/baseline_var_results.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4%}" if abs(x) < 1 else f"{x:.2f}")
    print(combined.to_string(index=False))
    print("\nSaved to outputs/baseline_var_results.csv")
