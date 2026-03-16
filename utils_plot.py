import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_BLDG_COLOR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}


def handle_chart_click(ev, df: pd.DataFrame, brand_col: str = "brand",
                       field: str = "x", trunc: int = 0) -> None:
    """Generic click handler for plotly charts — show selected brand detail."""
    pts = ev.selection.points if ev and hasattr(ev, "selection") else []
    if not pts:
        return
    val = pts[0].get(field) or ""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if not val:
        return
    if trunc:
        fdf = df[df[brand_col].astype(str).str[:trunc] == str(val)[:trunc]]
    else:
        fdf = df[df[brand_col] == val]
    if not fdf.empty:
        st.caption(f"선택됨: **{val}**")
        st.dataframe(fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_label: str,
    color_col: str | None = "building",
    key: str | None = None,
    height: int = 420,
    show_logy: bool = True,
) -> None:
    """Plotly bar chart with building-colour support and click-to-show row."""
    _key = key or f"bar_{title[:30].replace(' ', '_')}"
    _logy = st.checkbox("Log 스케일", key=f"{_key}_logy") if show_logy else False
    fig = px.bar(
        df, x=x, y=y,
        color=color_col if color_col and color_col in df.columns else None,
        title=title,
        labels={y: y_label, x: "Brand"},
        color_discrete_map=_BLDG_COLOR,
        log_y=_logy,
    )
    fig.update_layout(
        height=height, xaxis_tickangle=-45, showlegend=True,
        margin=dict(t=50, b=80),
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    _ev = st.plotly_chart(fig, use_container_width=True, key=_key, on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _pt = _pts[0]
        _brand = _pt.get("x") or _pt.get("customdata") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = df[df[x] == _brand] if _brand and x in df.columns else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

def render_sheet_mom_tab(
    curr: pd.DataFrame,
    prev: pd.DataFrame | None,
    compare_cols: list[str],
    col_labels: dict[str, str],
    col_units: dict[str, str],
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    key_prefix: str = "mom",
    no_prev_msg: str = "이전 달 파일이 없습니다.",
    mode: str = "mom",
) -> None:
    """Generic comparison tab for sheet views (water, hotwater, electricity).

    mode="mom" → 월별 변화 labels; mode="yoy" → 전년 대비 labels.
    """
    import plotly.graph_objects as go
    from utils import fmt_won

    if mode == "yoy":
        _heading = "📅 전년 대비"
        _prev_label = "전년"
        _curr_label = "올해"
        _chart_label = "전년 동월 대비 변화"
    else:
        _heading = "📈 월별 변화"
        _prev_label = "전월"
        _curr_label = "이번달"
        _chart_label = "전월 대비 변화"

    period_str = (
        f"{prev_billing_period} → {billing_period}"
        if billing_period and prev_billing_period
        else billing_period or "이번 달"
    )
    st.subheader(f"{_heading}  ({period_str})")

    if prev is None or prev.empty:
        st.info(no_prev_msg)
        return

    id_cols = [c for c in ["brand", "building"] if c in curr.columns and c in prev.columns]
    valid_cols = [c for c in compare_cols if c in curr.columns and c in prev.columns]
    if not valid_cols:
        st.info("비교할 열이 없습니다.")
        return

    curr_agg = curr.groupby(id_cols)[valid_cols].sum().reset_index()
    prev_agg = prev.groupby(id_cols)[valid_cols].sum().reset_index()
    merged = curr_agg.merge(prev_agg, on=id_cols, how="outer", suffixes=("_c", "_p"))
    for c in valid_cols:
        merged[f"{c}_c"] = merged[f"{c}_c"].fillna(0)
        merged[f"{c}_p"] = merged[f"{c}_p"].fillna(0)
        merged[f"{c}_chg"] = merged[f"{c}_c"] - merged[f"{c}_p"]
        merged[f"{c}_pct"] = (
            merged[f"{c}_chg"] / merged[f"{c}_p"].replace(0, float("nan")) * 100
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi_cols = valid_cols[:5]
    kc = st.columns(len(kpi_cols))
    for ci, c in enumerate(kpi_cols):
        _cv = merged[f"{c}_c"].sum()
        _pv = merged[f"{c}_p"].sum()
        _dv = _cv - _pv
        _pct = _dv / _pv * 100 if _pv else 0
        _unit = col_units.get(c, "")
        if _unit == "원":
            _val_str   = fmt_won(_cv)
            _delta_str = f"{fmt_won(_dv, signed=True)} ({_pct:+.1f}%)"
        else:
            _val_str   = f"{_cv:,.1f} {_unit}"
            _delta_str = f"{_dv:+,.1f} {_unit} ({_pct:+.1f}%)"
        kc[ci].metric(
            col_labels.get(c, c),
            _val_str,
            delta=_delta_str,
            delta_color="inverse",
        )

    st.divider()

    # ── Column selector + bar chart ───────────────────────────────────────────
    sel_col = st.selectbox(
        "항목",
        valid_cols,
        format_func=lambda c: col_labels.get(c, c),
        key=f"{key_prefix}_sel",
    )
    _unit = col_units.get(sel_col, "")
    _chg_col = f"{sel_col}_chg"
    _mom_h_logy = st.checkbox("Log 스케일", key=f"{key_prefix}_bar_logy")
    plot_df = merged[["brand", "building", f"{sel_col}_c", f"{sel_col}_p", _chg_col]].copy()
    plot_df = plot_df.sort_values(_chg_col, ascending=True).reset_index(drop=True)

    _colors = plot_df[_chg_col].apply(lambda v: "#C44E52" if v > 0 else "#2ca02c").tolist()
    fig = go.Figure(go.Bar(
        x=plot_df[_chg_col],
        y=plot_df["brand"],
        orientation="h",
        marker_color=_colors,
        text=plot_df[_chg_col].apply(
            lambda v: fmt_won(v, signed=True) if _unit == "원" else f"{v:+,.1f}"
        ),
        textposition="outside",
        textfont=dict(size=9, color="#222222"),
        hovertemplate=f"<b>%{{y}}</b><br>변화: %{{x:+,.1f}} {_unit}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#888888", line_width=1)
    _x_cfg = dict(title=f"변화 ({_unit})")
    if _mom_h_logy:
        _x_cfg["type"] = "log"
    fig.update_layout(
        title=f"{col_labels.get(sel_col, sel_col)} {_chart_label} ({_unit})",
        height=max(430, len(plot_df) * 22 + 80),
        xaxis=_x_cfg,
        margin=dict(t=55, b=40, l=10, r=130),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_bar_{sel_col}", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = _pts[0].get("y", "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = plot_df[plot_df["brand"] == _brand]
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    st.divider()

    # ── Top / bottom tables ───────────────────────────────────────────────────
    _val_fmt = (lambda v: fmt_won(v) if pd.notna(v) else "—") if _unit == "원" else (lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
    _chg_fmt = (lambda v: fmt_won(v, signed=True) if pd.notna(v) else "—") if _unit == "원" else (lambda v: f"{v:+,.1f}" if pd.notna(v) else "—")

    def _fmt_table(df):
        d = df[["brand", "building"]].copy()
        d[_prev_label] = df[f"{sel_col}_p"].apply(_val_fmt)
        d[_curr_label] = df[f"{sel_col}_c"].apply(_val_fmt)
        d["변화"] = df[_chg_col].apply(_chg_fmt)
        return d

    _n = min(10, len(plot_df))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🔴 증가 상위 {_n}개**")
        st.dataframe(_fmt_table(plot_df.nlargest(_n, _chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)
    with c2:
        st.markdown(f"**🟢 감소 상위 {_n}개**")
        st.dataframe(_fmt_table(plot_df.nsmallest(_n, _chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)

    st.divider()

    # ── Full change table ─────────────────────────────────────────────────────
    with st.expander("📋 전체 변화 목록", expanded=False):
        _all_cols = id_cols + [f"{c}_p" for c in valid_cols] + [f"{c}_c" for c in valid_cols] + [f"{c}_chg" for c in valid_cols]
        _all_cols = [c for c in _all_cols if c in merged.columns]
        st.dataframe(
            merged[_all_cols].sort_values(f"{valid_cols[-1]}_chg", ascending=False).reset_index(drop=True),
            hide_index=True, use_container_width=True,
        )


def plot_hist_full_or_tails(s: pd.Series, bins: int, lo: float, hi: float, title: str, mode: str = "tails", show_median: bool = True):
    vals = s.dropna().astype(float)
    if vals.size == 0:
        st.info(f"No numeric values for {title}")
        return None
    x = vals.values
    xmin, xmax = float(np.min(x)), float(np.max(x))
    med = float(np.median(x))
    fig, ax = plt.subplots(figsize=(7,4))
    if mode == "full":
        ax.hist(x, bins=bins, edgecolor="black", alpha=0.7)
        # reference lines
        ax.axvline(lo, linestyle="--", linewidth=2, zorder=4)
        ax.axvline(hi, linestyle="--", linewidth=2, zorder=4)
    else:
        ax.hist(x, bins=bins, edgecolor="black", alpha=0.7)
        eps = 1e-12
        if lo > xmin + eps:
            ax.axvspan(xmin, lo, alpha=0.25, zorder=3)
        if hi < xmax - eps:
            ax.axvspan(hi, xmax, alpha=0.25, zorder=3)
        ax.axvline(lo, linestyle="--", linewidth=2, zorder=4)
        ax.axvline(hi, linestyle="--", linewidth=2, zorder=4)
    if show_median:
        ax.axvline(med, linestyle="-", linewidth=2, zorder=5, color='red')
    ax.set_title(title)
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.8, alpha=0.8)
    st.pyplot(fig, clear_figure=True)
    stats = {
        "n": int(vals.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(xmin),
        "p20": float(np.quantile(x, 0.20)),
        "median": float(med),
        "p80": float(np.quantile(x, 0.80)),
        "max": float(xmax),
    }
    return stats
