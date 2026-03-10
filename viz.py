import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data import to_numeric_series

_BAR_NORMAL = "#4C72B0"
_BAR_TAIL   = "#DD8A00"
_LINE       = "#111111"
_MEDIAN     = "#C44E52"


def plot_hist_with_tails(
    s: pd.Series,
    bins: int,
    lo: float,
    hi: float,
    title: str,
    show_median: bool = True,
    source_df: pd.DataFrame = None,
    val_col: str = None,
    key: str = "hist",
    display_cols: list = None,
    tail_pct: float = None,
    val_scale: float = 1.0,
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
    tail_mask = np.array([(m <= lo or m >= hi) for m in midpoints])
    normal_mask = ~tail_mask

    fig = go.Figure()

    _bar_kwargs = dict(
        marker_line_color="white",
        marker_line_width=0.8,
        opacity=0.9,
        unselected=dict(marker=dict(opacity=0.9)),
        selected=dict(marker=dict(opacity=1.0)),
        textposition="outside",
        textfont=dict(size=9, color="#666666"),
        hovertemplate="<b>%{customdata[0]:.4g} – %{customdata[1]:.4g}</b><br>Count: %{y}<extra></extra>",
    )

    def _cd(mask):
        e0, e1 = edges[:-1][mask], edges[1:][mask]
        return np.stack([e0, e1], axis=1) if e0.size > 0 else np.empty((0, 2))

    fig.add_trace(go.Bar(
        x=midpoints[normal_mask],
        y=counts[normal_mask],
        width=widths[normal_mask],
        name="Normal",
        marker_color=_BAR_NORMAL,
        text=[str(c) if c > 0 else "" for c in counts[normal_mask]],
        customdata=_cd(normal_mask),
        **_bar_kwargs,
    ))

    fig.add_trace(go.Bar(
        x=midpoints[tail_mask],
        y=counts[tail_mask],
        width=widths[tail_mask],
        name="Tails",
        marker_color=_BAR_TAIL,
        text=[str(c) if c > 0 else "" for c in counts[tail_mask]],
        customdata=_cd(tail_mask),
        **_bar_kwargs,
    ))

    # Dummy traces for legend (lines only)
    fig.add_trace(go.Scatter(x=[None], y=[None], name="Median", mode="lines",
                             line=dict(color=_MEDIAN, width=2, dash="dot"), showlegend=True))
    tail_pct_val = float(tail_pct) if tail_pct is not None else float((x <= lo).mean() * 100)
    fig.add_trace(go.Scatter(x=[None], y=[None], name=f"Bottom/Top {tail_pct_val:.0f}%", mode="lines",
                             line=dict(color=_LINE, width=2, dash="dash"), showlegend=True))

    eps = 1e-12
    if lo > xmin + eps:
        fig.add_vrect(x0=xmin, x1=lo, fillcolor=_BAR_TAIL, opacity=0.1, line_width=0)
    if hi < xmax - eps:
        fig.add_vrect(x0=hi, x1=xmax, fillcolor=_BAR_TAIL, opacity=0.1, line_width=0)

    fig.add_vline(x=lo, line_dash="dash", line_color=_LINE, line_width=1.5)
    fig.add_vline(x=hi, line_dash="dash", line_color=_LINE, line_width=1.5)

    if show_median:
        fig.add_vline(x=med, line_dash="dot", line_color=_MEDIAN, line_width=1.5)

    box_lines = [f"Bottom {tail_pct_val:.0f}%  {lo:.4g}", f"Top {tail_pct_val:.0f}%     {hi:.4g}"]
    if show_median:
        box_lines.append(f"Median       {med:.4g}")

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.55,
        xanchor="right", yanchor="top",
        text="<br>".join(box_lines),
        showarrow=False,
        font=dict(size=11, color="#333333", family="monospace"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#AAAAAA",
        borderwidth=1,
        borderpad=6,
        align="left",
    )

    n_tail = int(np.sum(counts[midpoints <= lo]) + np.sum(counts[midpoints >= hi]))
    n_total = int(counts.sum())
    tail_pct = 100 * n_tail / n_total if n_total > 0 else 0

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>   <span style='font-size:12px;color:#888'>n={n_total} · tail={n_tail} ({tail_pct:.1f}%)</span>",
            font=dict(size=13, color="#222222"), x=0,
        ),
        height=380,
        bargap=0,
        margin=dict(l=50, r=20, t=55, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
            zeroline=False, showline=True, linecolor="#AAAAAA", linewidth=1,
            tickfont=dict(size=11, color="#222222"),
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(size=11, color="#222222")),
            showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
            zeroline=True, zerolinecolor="#AAAAAA", zerolinewidth=1,
            showline=True, linecolor="#AAAAAA", linewidth=1,
            rangemode="tozero",
            tickfont=dict(size=11, color="#222222"),
        ),
        font=dict(family="Arial, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.99, xanchor="right",
            y=0.97, yanchor="top",
            font=dict(size=11, color="#333333", family="monospace"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#AAAAAA",
            borderwidth=1,
        ),
    )

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

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=key)

    st.dataframe(pd.DataFrame([{
        "n": stats["n"], "min": round(stats["min"], 4), "p20": round(stats["p20"], 4),
        "median": round(stats["median"], 4), "mean": round(stats["mean"], 4),
        "std": round(stats["std"], 4), "p80": round(stats["p80"], 4), "max": round(stats["max"], 4),
    }]), hide_index=True, use_container_width=True)

    if source_df is not None and val_col is not None:
        _scaled = source_df[val_col] / val_scale

        # ── Bin-click table (directly below stats) ────────────────────────────
        pts = (event.selection.points if event and hasattr(event, "selection") else [])
        if pts:
            cd = pts[0].get("customdata", [])
            if isinstance(cd, dict):
                cd = list(cd.values())
            if len(cd) >= 2:
                x0, x1 = float(cd[0]), float(cd[1])
                mask = (_scaled >= x0) & (_scaled <= x1)
                bin_df = source_df[mask].copy()
                cols = [c for c in (display_cols or []) if c in bin_df.columns]
                if not cols:
                    cols = [c for c in ["brand", "building"] if c in bin_df.columns] + [val_col]
                st.markdown(f"**Bin {x0:.4g} – {x1:.4g}** — {len(bin_df)} business(es)")
                st.dataframe(bin_df[cols].reset_index(drop=True),
                             hide_index=True, use_container_width=True)

        # ── Outlier tables (top / bottom separated) ───────────────────────────
        _top_mask = _scaled > hi
        _bot_mask = _scaled < lo
        _n_top = int(_top_mask.sum())
        _n_bot = int(_bot_mask.sum())
        _n_out = _n_top + _n_bot

        _id_back  = [c for c in ["building", "floor"] if c in source_df.columns]
        _id_front = [c for c in (display_cols or []) if c in source_df.columns
                     and c not in _id_back and c != val_col]
        if not _id_front:
            _id_front = [c for c in ["brand"] if c in source_df.columns]
        _usage_cols = [val_col] if val_col in source_df.columns else []
        _out_cols = _id_front + _usage_cols + _id_back

        with st.expander(f"🔶 이상치 목록 — {_n_out}건  (상위 {_n_top} · 하위 {_n_bot})",
                         expanded=_n_out > 0):
            if _n_out == 0:
                st.caption("이상치 없음")
            else:
                if _n_top > 0:
                    _top_df = source_df[_top_mask].copy().sort_values(val_col, ascending=False)
                    st.markdown(f"**🔺 상위 이상치** — {hi:.4g} 초과 ({_n_top}건)")
                    st.dataframe(_top_df[_out_cols].reset_index(drop=True),
                                 hide_index=True, use_container_width=True)
                if _n_bot > 0:
                    if _n_top > 0:
                        st.divider()
                    _bot_df = source_df[_bot_mask].copy().sort_values(val_col, ascending=True)
                    st.markdown(f"**🔻 하위 이상치** — {lo:.4g} 미만 ({_n_bot}건)")
                    st.dataframe(_bot_df[_out_cols].reset_index(drop=True),
                                 hide_index=True, use_container_width=True)

    return stats
