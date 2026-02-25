import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        return None

    x = vals.values.astype(float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    med = float(np.median(x))

    counts, edges = np.histogram(x, bins=bins)
    midpoints = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    # Color each bar: tail = orange, normal = steelblue
    colors = [
        "orange" if (m <= lo or m >= hi) else "steelblue"
        for m in midpoints
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=midpoints,
        y=counts,
        width=widths,
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=0.5,
        opacity=0.8,
        hovertemplate="Range: %{customdata[0]:.4g} – %{customdata[1]:.4g}<br>Count: %{y}<extra></extra>",
        customdata=np.stack([edges[:-1], edges[1:]], axis=1),
    ))

    eps = 1e-12

    # Tail shading
    if lo > xmin + eps:
        fig.add_vrect(x0=xmin, x1=lo, fillcolor="orange", opacity=0.12, line_width=0)
    if hi < xmax - eps:
        fig.add_vrect(x0=hi, x1=xmax, fillcolor="orange", opacity=0.12, line_width=0)

    # Cutoff lines
    fig.add_vline(x=lo, line_dash="dash", line_color="darkorange", line_width=2,
                  annotation_text=f"lo={lo:.4g}", annotation_position="top left")
    fig.add_vline(x=hi, line_dash="dash", line_color="darkorange", line_width=2,
                  annotation_text=f"hi={hi:.4g}", annotation_position="top right")

    # Median line
    if show_median:
        fig.add_vline(x=med, line_color="red", line_width=2,
                      annotation_text=f"median={med:.4g}", annotation_position="top left")

    fig.update_layout(
        title=title,
        height=400,
        bargap=0,
        xaxis=dict(
            showgrid=True, gridcolor="lightgrey", gridwidth=1,
            minor=dict(showgrid=True, gridcolor="whitesmoke", gridwidth=0.5),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="lightgrey", gridwidth=1,
            minor=dict(showgrid=True, gridcolor="whitesmoke", gridwidth=0.5),
        ),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    stats = {
        "n":      int(vals.notna().sum()),
        "mean":   float(np.mean(x)),
        "std":    float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min":    float(xmin),
        "p20":    float(np.quantile(x, 0.20)),
        "median": float(med),
        "p80":    float(np.quantile(x, 0.80)),
        "max":    float(xmax),
    }
    return stats
