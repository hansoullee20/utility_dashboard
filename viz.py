import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from data import to_numeric_series


def plot_hist_with_tails(
    s: pd.Series,
    bins: int,
    lo: float,
    hi: float,
    title: str,
    show_median: bool = True,
):
    vals = to_numeric_series(s).dropna()
    if vals.empty:
        st.info(f"No numeric values for {title}")
        return None  # no stats

    x = vals.values.astype(float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    med = float(np.median(x))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=bins, edgecolor="black", alpha=0.7)

    # tail shading (robust to float equality)
    eps = 1e-12
    if lo > xmin + eps:
        ax.axvspan(xmin, lo, alpha=0.25, zorder=3)
    if hi < xmax - eps:
        ax.axvspan(hi, xmax, alpha=0.25, zorder=3)

    # quantile cutoff lines
    ax.axvline(lo, linestyle="--", linewidth=2, zorder=4)
    ax.axvline(hi, linestyle="--", linewidth=2, zorder=4)

    # median line
    if show_median:
        ax.axvline(med, linestyle="-", linewidth=2, zorder=5, color = 'red')

    ax.set_title(title)
    ax.grid(
    True,
    which="both",
    axis="both",
    linestyle="--",
    linewidth=0.8,
    alpha=0.8,
    )
    


    st.pyplot(fig, clear_figure=True)

    stats = {
        "n": int(vals.notna().sum()),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(xmin),
        "p20": float(np.quantile(x, 0.20)),
        "median": float(med),
        "p80": float(np.quantile(x, 0.80)),
        "max": float(xmax),
    }
    return stats
