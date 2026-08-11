
import numpy as np
import pandas as pd
from scipy import stats
# import via exec since filenames start with digits (not valid module names)
import importlib.util

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

baseline = _load("01_baseline_var.py", "baseline")
evt = _load("02_evt_pot_model.py", "evt")


def kupiec_pof_test(breaches, n_total, alpha):
   
    n_breach = breaches.sum()
    p_hat = n_breach / n_total
    p_expected = 1 - alpha
    if n_breach == 0 or p_hat == 1:
        return np.nan, np.nan  # degenerate, can't compute LR cleanly
    lr = -2 * (
        (n_total - n_breach) * np.log(1 - p_expected) + n_breach * np.log(p_expected)
        - (n_total - n_breach) * np.log(1 - p_hat) - n_breach * np.log(p_hat)
    )
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value


def christoffersen_independence_test(breaches):
    
    breaches = breaches.astype(int)
    n00 = ((breaches[:-1] == 0) & (breaches[1:] == 0)).sum()
    n01 = ((breaches[:-1] == 0) & (breaches[1:] == 1)).sum()
    n10 = ((breaches[:-1] == 1) & (breaches[1:] == 0)).sum()
    n11 = ((breaches[:-1] == 1) & (breaches[1:] == 1)).sum()

    if (n01 + n00) == 0 or (n11 + n10) == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0

    ll_restricted = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    ll_unrestricted = (
        n00 * safe_log(1 - pi01) + n01 * safe_log(pi01)
        + n10 * safe_log(1 - pi11) + n11 * safe_log(pi11)
    )
    lr = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value


def rolling_backtest(returns, alpha=0.99, window=1000, method="evt"):
   
    n = len(returns)
    breaches = []
    for t in range(window, n):
        train = returns[t - window:t]
        actual = returns[t]

        if method == "evt":
            df, _ = evt.run_evt_pot(train, alphas=(alpha,))
            var = df["EVT_VaR"].iloc[0]
        elif method == "normal":
            var, _ = baseline.normal_var_cvar(train, alpha)
        elif method == "student_t":
            var, _, _ = baseline.student_t_var_cvar(train, alpha)
        elif method == "historical":
            var, _ = baseline.historical_var_cvar(train, alpha)
        else:
            raise ValueError(method)

        breach = 1 if -actual > var else 0  # loss exceeds VaR estimate
        breaches.append(breach)

    return np.array(breaches)


def summarize_backtest(name, method, alpha, breaches):
    n = len(breaches)
    n_breach = breaches.sum()
    expected_rate = 1 - alpha
    observed_rate = n_breach / n
    kupiec_lr, kupiec_p = kupiec_pof_test(breaches, n, alpha)
    christ_lr, christ_p = christoffersen_independence_test(breaches)
    return {
        "index": name, "method": method, "confidence": alpha,
        "n_obs": n, "n_breaches": n_breach,
        "expected_rate": expected_rate, "observed_rate": observed_rate,
        "kupiec_p_value": kupiec_p, "kupiec_pass": (kupiec_p > 0.05) if not np.isnan(kupiec_p) else None,
        "christoffersen_p_value": christ_p,
        "christoffersen_pass": (christ_p > 0.05) if not np.isnan(christ_p) else None,
    }


if __name__ == "__main__":
    sp500 = pd.read_csv("data/sp500_real.csv")
    returns = sp500["Return"].dropna().values

    print("Running rolling backtests (this walks forward day-by-day, may take a moment)...")
    print("NOTE: using a reduced window/sample here for speed on placeholder data.")

    results = []
    alpha = 0.99
    window = 500  # reduced for speed; use 1000-1250 (~4-5yrs) on real data
    test_returns = returns[-1500:]  # last chunk only, for speed in this demo

    for method in ["normal", "student_t", "historical", "evt"]:
        breaches = rolling_backtest(test_returns, alpha=alpha, window=window, method=method)
        summary = summarize_backtest("SP500", method, alpha, breaches)
        results.append(summary)
        print(f"  {method:12s} done -> {summary['n_breaches']} breaches / {summary['n_obs']} obs "
              f"(observed rate {summary['observed_rate']:.3%} vs expected {summary['expected_rate']:.3%})")

    df = pd.DataFrame(results)
    df.to_csv("outputs/backtest_results.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nSaved to outputs/backtest_results.csv")
