
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evt = _load("02_evt_pot_model.py", "evt")
baseline = _load("01_baseline_var.py", "baseline")


def plot_index_row(axes_row, returns, name):
    losses = -returns
    evt_df, me_df = evt.run_evt_pot(returns, name)

    # 1. Mean excess plot
    axes_row[0].plot(me_df["threshold"], me_df["mean_excess"], marker="o", color="#1f77b4")
    axes_row[0].set_title(f"{name}: Mean Excess Plot")
    axes_row[0].set_xlabel("Threshold u (loss level)")
    axes_row[0].set_ylabel("Mean Excess e(u)")
    axes_row[0].grid(alpha=0.3)

    # 2. Loss distribution with tail highlighted
    axes_row[1].hist(losses, bins=100, density=True, alpha=0.6, color="#888888", label="All losses")
    u = evt_df["threshold_u"].iloc[0]
    axes_row[1].axvline(u, color="red", linestyle="--", label=f"u={u:.4f}")
    axes_row[1].hist(losses[losses > u], bins=30, density=False, alpha=0.7, color="#d62728", label="Exceedances")
    axes_row[1].set_title(f"{name}: Loss Distribution & Threshold")
    axes_row[1].set_xlabel("Daily loss")
    axes_row[1].legend(fontsize=8)
    axes_row[1].grid(alpha=0.3)

    # 3. VaR comparison across methods at 99%
    n_var, _ = baseline.normal_var_cvar(returns, 0.99)
    t_var, _, _ = baseline.student_t_var_cvar(returns, 0.99)
    h_var, _ = baseline.historical_var_cvar(returns, 0.99)
    e_var = evt_df[evt_df.confidence == 0.99]["EVT_VaR"].iloc[0]

    methods = ["Normal", "Student-t", "Historical", "EVT (POT)"]
    values = [n_var, t_var, h_var, e_var]
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728"]
    axes_row[2].bar(methods, values, color=colors)
    axes_row[2].set_title(f"{name}: 99% VaR Comparison")
    axes_row[2].set_ylabel("VaR (daily loss)")
    axes_row[2].tick_params(axis="x", rotation=15)
    axes_row[2].grid(alpha=0.3, axis="y")

    return evt_df


if __name__ == "__main__":
    sp500 = pd.read_csv("data/sp500_real.csv")
    nifty = pd.read_csv("data/nifty_real.csv")
    sp_returns = sp500["Return"].dropna().values
    nf_returns = nifty["Return"].dropna().values

    # --- Panel 1: 2x3 diagnostic grid ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sp_evt_df = plot_index_row(axes[0], sp_returns, "SP500")
    nf_evt_df = plot_index_row(axes[1], nf_returns, "NIFTY")
    plt.tight_layout()
    plt.savefig("outputs/evt_diagnostic_plots.png", dpi=150)
    print("Saved outputs/evt_diagnostic_plots.png")

    # --- Panel 2: tail index comparison (the headline cross-market chart) ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    xi_sp = sp_evt_df["tail_index_xi"].iloc[0]
    xi_nf = nf_evt_df["tail_index_xi"].iloc[0]
    bars = ax2.bar(["S&P 500", "Nifty 50"], [xi_sp, xi_nf], color=["#1f77b4", "#ff7f0e"])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Tail Index (\u03be)")
    ax2.set_title("GPD Tail Index: Developed vs Emerging Market")
    for bar, val in zip(bars, [xi_sp, xi_nf]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("outputs/tail_index_comparison.png", dpi=150)
    print("Saved outputs/tail_index_comparison.png")

    print(f"\nS&P 500 xi = {xi_sp:.4f}  |  Nifty 50 xi = {xi_nf:.4f}")
    print(f"Nifty tail is {'fatter' if xi_nf > xi_sp else 'thinner'} than S&P "
          f"by {abs(xi_nf - xi_sp):.4f} in xi units")
