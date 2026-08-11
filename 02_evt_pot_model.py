"""
EXTREME VALUE THEORY: PEAK-OVER-THRESHOLD (POT) MODEL
--------------------------------------------------------
Core methodology. Instead of assuming a global distribution (Normal, t) for
ALL returns, EVT only models the TAIL — the losses that exceed some high
threshold u. By the Pickands-Balkema-de Haan theorem, exceedances over a
sufficiently high threshold asymptotically follow a Generalized Pareto
Distribution (GPD), regardless of the parent distribution. This is why EVT
is the standard for tail risk in Basel/FRTB-style capital frameworks.

Steps:
1. Convert returns to LOSSES (positive = bad, so we work with -returns)
2. Choose threshold u (typically top 5-10% of losses) via a mean excess plot
3. Fit GPD(xi, sigma) to exceedances (loss - u) via MLE
4. Derive VaR_alpha and CVaR_alpha (Expected Shortfall) in closed form
5. Report the tail index xi -- xi > 0 means heavy-tailed (fatter than
   exponential), which is the expected/interesting finding for equities
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def mean_excess_plot_data(losses, thresholds=None):
    """Compute mean excess function e(u) = E[L - u | L > u] across a range
    of thresholds. Used to visually select a threshold where e(u) becomes
    roughly linear (the GPD regime)."""
    if thresholds is None:
        thresholds = np.percentile(losses, np.arange(80, 99, 1))
    me = []
    for u in thresholds:
        exceed = losses[losses > u] - u
        me.append(exceed.mean() if len(exceed) > 10 else np.nan)
    return pd.DataFrame({"threshold": thresholds, "mean_excess": me})


def fit_gpd_mle(exceedances):
    """MLE fit of Generalized Pareto Distribution to exceedances.
    GPD pdf: f(x; xi, sigma) = (1/sigma) * (1 + xi*x/sigma)^(-1/xi - 1)
    Uses scipy's genpareto (loc fixed at 0 since exceedances already shifted)."""
    xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
    return xi, sigma


def gpd_var_cvar(xi, sigma, u, n_total, n_exceed, alpha):
    """Closed-form EVT VaR and CVaR (Expected Shortfall) via the POT method.

    VaR_alpha = u + (sigma/xi) * [ ((n/n_u)*(1-alpha))^(-xi) - 1 ]
    CVaR_alpha = VaR_alpha / (1 - xi) + (sigma - xi*u) / (1 - xi)
    """
    phi_u = n_exceed / n_total  # proportion of exceedances (P(L > u))
    if xi != 0:
        var = u + (sigma / xi) * ((phi_u / (1 - alpha)) ** xi - 1)
    else:
        var = u + sigma * np.log(phi_u / (1 - alpha))

    if xi < 1:
        cvar = var / (1 - xi) + (sigma - xi * u) / (1 - xi)
    else:
        cvar = np.nan  # undefined / infinite mean for xi >= 1
    return var, cvar


def run_evt_pot(returns, name="INDEX", threshold_pct=90, alphas=(0.95, 0.99, 0.995)):
    losses = -returns  # work in loss space: positive = bad day
    u = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > u] - u
    n_total, n_exceed = len(losses), len(exceedances)

    xi, sigma = fit_gpd_mle(exceedances)

    rows = []
    for alpha in alphas:
        var, cvar = gpd_var_cvar(xi, sigma, u, n_total, n_exceed, alpha)
        rows.append({
            "index": name, "confidence": alpha,
            "EVT_VaR": var, "EVT_CVaR": cvar,
            "tail_index_xi": xi, "gpd_scale_sigma": sigma,
            "threshold_u": u, "n_exceedances": n_exceed, "pct_data_in_tail": n_exceed / n_total,
        })
    return pd.DataFrame(rows), mean_excess_plot_data(losses)


if __name__ == "__main__":
    sp500 = pd.read_csv("data/sp500_real.csv")
    nifty = pd.read_csv("data/nifty_real.csv")

    sp_evt, sp_me = run_evt_pot(sp500["Return"].dropna().values, "SP500")
    nf_evt, nf_me = run_evt_pot(nifty["Return"].dropna().values, "NIFTY")

    combined = pd.concat([sp_evt, nf_evt], ignore_index=True)
    combined.to_csv("outputs/evt_pot_results.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(combined.to_string(index=False))
    print(f"\nSP500 tail index (xi): {sp_evt.tail_index_xi.iloc[0]:.4f}  "
          f"({'heavy-tailed' if sp_evt.tail_index_xi.iloc[0] > 0 else 'light-tailed'})")
    print(f"NIFTY  tail index (xi): {nf_evt.tail_index_xi.iloc[0]:.4f}  "
          f"({'heavy-tailed' if nf_evt.tail_index_xi.iloc[0] > 0 else 'light-tailed'})")
    print("\nSaved to outputs/evt_pot_results.csv")
