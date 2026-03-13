import io
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data import (
    read_ehp_oac_sheet, group_raw_slice_by_year,
    read_billing_sheet, BILLING_SHEET_NAME,
)
from viz import plot_hist_with_tails
from utils_plot import handle_chart_click

_PALETTE = [
    "#4C72B0", "#DD8A00", "#C44E52", "#55A868",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial Black, Arial, sans-serif", color="#111111"),
)


@st.cache_data(show_spinner="Parsing EHP sheet...", max_entries=1)
def _cached_read(name: str, data: bytes, sheet: str):
    return read_ehp_oac_sheet(name, data, sheet)


def _build_pivot_from_oac(oac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (pivot month×year, per-meter usage) from read_ehp_oac_sheet output.

    Returns:
        pivot   – index=월 label, columns=year ints, values=total kWh
        usage   – per-meter monthly usage with '계량기 번호' + 'YYYY_Mwol' columns
    """
    cum_cols = sorted([c for c in oac.columns if _YM_PAT.match(c)],
                      key=lambda c: _YM_PAT.match(c).groups())
    if not cum_cols:
        return pd.DataFrame(), pd.DataFrame()

    readings = oac[cum_cols].apply(pd.to_numeric, errors="coerce")
    usage_m = readings.diff(axis=1)

    # Build pivot (sum all meters per month)
    records = []
    for c in cum_cols:
        m = _YM_PAT.match(c)
        yr, mo = int(m.group(1)), int(m.group(2))
        total = usage_m[c].sum(min_count=1)
        records.append({"year": yr, "month": mo, "total": total})

    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None

    # Build per-meter usage with column names matching _tab_meter's yr_pat
    rename_map = {}
    for c in cum_cols:
        mm = _YM_PAT.match(c)
        yr, mo = int(mm.group(1)), int(mm.group(2))
        rename_map[c] = f"{yr}_{mo}월"
    usage_df = usage_m.rename(columns=rename_map).copy()
    meter_col = oac["meter_no"].values if "meter_no" in oac.columns else [""] * len(oac)
    usage_df.insert(0, "계량기 번호", meter_col)

    return pivot, usage_df


# ─── Tab 4: Per-Meter Analysis ────────────────────────────────────────────────

def _tab_meter(usage: pd.DataFrame) -> None:
    yr_pat = re.compile(r'^(20\d{2})_(\d{1,2})월$')
    if "계량기 번호" not in usage.columns:
        st.warning("계량기 번호 column not found.")
        return
    meters = sorted(usage["계량기 번호"].dropna().unique(), key=str)
    sel = st.selectbox("계량기 번호 선택", meters, key="ehp_meter_sel")
    row = usage[usage["계량기 번호"] == sel]
    records = []
    for col in usage.columns:
        m = yr_pat.match(str(col))
        if not m:
            continue
        yr, mo = int(m.group(1)), int(m.group(2))
        val = pd.to_numeric(row[col], errors="coerce").sum(min_count=1)
        records.append({"year": yr, "month": mo, "total": val})
    if not records:
        st.info("No data for this meter.")
        return
    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None
    sub1, sub2 = st.tabs(["Yearly Usage", "Month Comparison"])
    with sub1:
        _tab_yearly(pivot, key="ehp_yearly_meter")
    with sub2:
        _tab_month(pivot, key="ehp_month_sel_meter")


# ─── Tab 1: Yearly Usage ──────────────────────────────────────────────────────

def _tab_yearly(pivot: pd.DataFrame, key: str = "ehp_yearly") -> None:
    year_totals = pivot.sum(min_count=1)
    years  = [str(y) for y in year_totals.index]
    values = year_totals.values

    fig = go.Figure(go.Bar(
        x=years, y=values,
        marker_color="#4C72B0",
        text=[f"{v:,.0f}" for v in values], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>연간 총 사용량 (kWh)</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(type="category", title=dict(text="연도", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero", tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")

    summary = pd.DataFrame({"Year": years, "Total kWh": [f"{v:,.0f}" for v in values]})
    st.dataframe(summary.iloc[::-1].reset_index(drop=True), hide_index=True, use_container_width=True)


# ─── Tab: Monthly Usage ───────────────────────────────────────────────────────

def _tab_monthly_total(pivot: pd.DataFrame, key: str = "ehp_monthly_total") -> None:
    all_labels = [f"{m}월" for m in range(1, 13)]
    available  = [m for m in all_labels if m in pivot.index]

    year_options = ["전체"] + [str(y) for y in sorted(pivot.columns, reverse=True)]
    sel_year = st.selectbox("연도 선택", year_options, key=f"{key}_year")

    if sel_year == "전체":
        totals = pivot.loc[available].sum(axis=1, min_count=1)
    else:
        yr = int(sel_year)
        totals = pivot.loc[available, yr] if yr in pivot.columns else pd.Series(dtype=float)

    values = totals.values

    fig = go.Figure(go.Bar(
        x=available, y=values,
        marker_color="#4C72B0",
        text=[f"{v:,.0f}" if pd.notna(v) else "" for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>월별 총 사용량 (kWh) — {sel_year}</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(type="category", title=dict(text="월", font=dict(color="#111111")),
                   showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")),
                   showgrid=True, gridcolor="#AAAAAA", zeroline=True, zerolinecolor="#AAAAAA",
                   rangemode="tozero", tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")

    summary = pd.DataFrame({
        "Month": available,
        "Total kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in values],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ─── Tab 2: Year Comparison ───────────────────────────────────────────────────

def _tab_compare(pivot: pd.DataFrame) -> None:
    available = list(pivot.columns)
    if len(available) < 2:
        st.warning("Need at least 2 years of data.")
        return

    c1, c2 = st.columns(2)
    with c1:
        yr_a = st.selectbox("Year A", available, index=len(available) - 2, key="ehp_cmp_a")
    with c2:
        yr_b = st.selectbox("Year B", available, index=len(available) - 1, key="ehp_cmp_b")

    if yr_a == yr_b:
        st.info("Select two different years.")
        return

    months = list(pivot.index)
    ta = pivot[yr_a].tolist()
    tb = pivot[yr_b].tolist()

    fig = go.Figure([
        go.Bar(name=str(yr_a), x=months, y=ta, marker_color=_PALETTE[0],
               hovertemplate=f"<b>{yr_a}</b> %{{x}}: %{{y:,.0f}} kWh<extra></extra>"),
        go.Bar(name=str(yr_b), x=months, y=tb, marker_color=_PALETTE[1],
               hovertemplate=f"<b>{yr_b}</b> %{{x}}: %{{y:,.0f}} kWh<extra></extra>"),
    ])
    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="group",
        title=dict(text=f"<b>월별 사용량 — {yr_a} vs {yr_b}</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(title=dict(text="월", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero", tickfont=dict(color="#111111")),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle",
                    font=dict(size=13, color="#111111")),
        margin=dict(l=60, r=90, t=65, b=50),
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key="ehp_compare_chart", on_select="rerun")

    tbl = pd.DataFrame({
        "Month": months,
        f"{yr_a} kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in ta],
        f"{yr_b} kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in tb],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ─── Tab 3: Month Comparison ──────────────────────────────────────────────────

def _tab_month(pivot: pd.DataFrame, key: str = "ehp_month_sel") -> None:
    all_labels = [f"{m}월" for m in range(1, 13)]
    available  = [m for m in all_labels if m in pivot.index]

    sel_month = st.selectbox("Month", available, key=key)
    row    = pivot.loc[sel_month]
    years  = [str(y) for y in row.index]
    values = row.values

    fig = go.Figure(go.Bar(
        x=years, y=values,
        marker_color="#4C72B0",
        text=[f"{v:,.0f}" if pd.notna(v) else "" for v in values], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>{sel_month} 연도별 사용량</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(type="category", title=dict(text="연도", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero", tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=f"ehp_month_chart_{key}", on_select="rerun")

    tbl = pd.DataFrame({"Year": years, "kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in values]})
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ─── Tab 5: Anomaly Detection ─────────────────────────────────────────────────

def _tab_anomaly(pivot: pd.DataFrame) -> None:
    """Flag month-year cells that deviate significantly from the same month's historical median."""
    if pivot.empty or len(pivot.columns) < 2:
        st.warning("At least 2 years of data needed for anomaly detection.")
        return

    st.caption("IQR fence per calendar month across all years. Red = high spike, green = unusual drop.")

    # Compute per-month statistics across years
    records = []
    for mo_label in pivot.index:
        row = pivot.loc[mo_label].dropna()
        if row.empty:
            continue
        q25, q75 = row.quantile(0.25), row.quantile(0.75)
        iqr = q75 - q25
        hi_fence = q75 + 1.5 * iqr
        lo_fence = max(q25 - 1.5 * iqr, 0.0)
        med = row.median()
        for yr, val in row.items():
            is_hi = val > hi_fence
            is_lo = (lo_fence > 0) and (val < lo_fence)
            if is_hi or is_lo:
                records.append({
                    "Month": mo_label, "Year": yr, "Usage (kWh)": round(val, 0),
                    "Median": round(med, 0), "Δ vs Median": round(val - med, 0),
                    "Fence (Hi)": round(hi_fence, 0), "Fence (Lo)": round(lo_fence, 0),
                    "Flag": "🔴 High" if is_hi else "🟢 Low",
                })

    # Heatmap — colour by Z-score per month
    z_data, z_text = [], []
    month_labels = list(pivot.index)
    year_labels  = [str(y) for y in pivot.columns]
    for mo_label in month_labels:
        row = pivot.loc[mo_label]
        row_vals = pd.to_numeric(row, errors="coerce")
        mu, sd = row_vals.mean(), row_vals.std()
        z_row, t_row = [], []
        for yr in pivot.columns:
            v = row_vals.get(yr, np.nan)
            z = (v - mu) / sd if (pd.notna(v) and sd > 0) else 0.0
            z_row.append(round(z, 2) if pd.notna(v) else None)
            t_row.append(f"{int(v):,}" if pd.notna(v) else "")
        z_data.append(z_row)
        z_text.append(t_row)

    fig_heat = go.Figure(go.Heatmap(
        z=z_data,
        x=year_labels,
        y=month_labels,
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="RdYlGn_r",
        zmid=0,
        colorbar=dict(title="Z-score"),
        hovertemplate="<b>%{y} %{x}</b><br>Usage: %{text} kWh<br>Z: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>월별 사용량 이상 탐지 — Z-score (red = high spike)</b>",
                   font=dict(size=13, color="#111111"), x=0),
        height=max(380, len(month_labels) * 28 + 100),
        xaxis=dict(title="Year", tickfont=dict(color="#111111")),
        yaxis=dict(title="Month", tickfont=dict(color="#111111")),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    _ev = st.plotly_chart(fig_heat, use_container_width=True, key="ehp_anomaly_heatmap", on_select="rerun")

    if records:
        anom_df = pd.DataFrame(records).sort_values(["Month", "Year"])
        st.markdown(f"**{len(records)} anomalous month-year(s) detected**")
        st.dataframe(anom_df, hide_index=True, use_container_width=True)
    else:
        st.success("No anomalies detected (all months within IQR × 1.5 fences).")


# ─── Tab 6: Panel Trend ───────────────────────────────────────────────────────

_YM_PAT = re.compile(r'^cum_(20\d{2})_(\d{2})$')


def _tab_panel_trend(name: str, data: bytes, sheet: str) -> None:
    """Line chart of monthly usage per panel over time (from OAC cumulative readings)."""
    oac = _cached_read(name, data, sheet)
    if oac.empty:
        st.warning("Could not load OAC meter data.")
        return

    cum_cols = [c for c in oac.columns if _YM_PAT.match(c)]
    if not cum_cols:
        st.warning("No cumulative reading columns found.")
        return

    cum_sorted = sorted(cum_cols, key=lambda c: _YM_PAT.match(c).groups())
    readings = oac[cum_sorted].apply(pd.to_numeric, errors="coerce")
    usage_m  = readings.diff(axis=1)
    usage_m.columns = cum_sorted

    group_col = "panel_name" if "panel_name" in oac.columns else "meter_no"
    labels = oac[group_col].astype(str).str.strip().fillna("(unknown)")
    all_labels = sorted(labels.unique())

    sel_panels = st.multiselect("판넬 선택", all_labels, default=all_labels[:min(5, len(all_labels))],
                                key="ehp_panel_sel")
    if not sel_panels:
        st.info("Select at least one panel.")
        return

    time_axis = []
    for c in cum_sorted:
        m = _YM_PAT.match(c)
        yr, mo = int(m.group(1)), int(m.group(2))
        time_axis.append(f"{yr}-{mo:02d}")

    fig = go.Figure()
    for i, panel in enumerate(sel_panels):
        mask = labels == panel
        if not mask.any():
            continue
        vals = usage_m[mask].sum(min_count=1)
        fig.add_trace(go.Scatter(
            x=time_axis, y=vals.values,
            mode="lines+markers",
            name=panel,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{panel}</b><br>%{{x}}: %{{y:,.0f}} kWh<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>판넬별 월간 사용량 추세 (kWh)</b>", font=dict(size=13, color="#111111"), x=0),
        height=420,
        xaxis=dict(title=dict(text="Year-Month", font=dict(color="#111111")),
                   tickangle=-45, tickfont=dict(size=9, color="#111111"),
                   showgrid=True, gridcolor="#DDDDDD",
                   type="category", categoryorder="array", categoryarray=time_axis),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")),
                   showgrid=True, gridcolor="#DDDDDD", zeroline=False,
                   tickfont=dict(color="#111111")),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                    font=dict(size=10, color="#111111")),
        margin=dict(l=60, r=160, t=60, b=80),
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key="ehp_panel_trend_chart", on_select="rerun")

    yearly: dict = {}
    for panel in sel_panels:
        mask = labels == panel
        if not mask.any():
            continue
        row_totals: dict = {}
        for c in cum_sorted:
            m = _YM_PAT.match(c)
            yr = int(m.group(1))
            v = usage_m[mask][c].sum(min_count=1)
            row_totals[yr] = row_totals.get(yr, 0) + (v if pd.notna(v) else 0)
        yearly[panel] = row_totals

    if yearly:
        tbl = pd.DataFrame(yearly).T.fillna(0).astype(int)
        tbl.index.name = group_col
        tbl["합계"] = tbl.sum(axis=1)
        tbl = tbl.reset_index()
        st.markdown("**연간 합계 (kWh)**")
        st.dataframe(tbl, hide_index=True, use_container_width=True)


# ─── Public entry point ───────────────────────────────────────────────────────

# ─── Tab: 냉난방 고지서 ───────────────────────────────────────────────────────

def _tab_hvac_billing(name: str, data: bytes) -> pd.DataFrame | None:
    """Show 냉난방 (hvac_excl / hvac_comm) analysis from 관리비 고지서.
    Returns the billing DataFrame for use in PDF generation, or None if unavailable."""
    try:
        bill = read_billing_sheet(name, data, BILLING_SHEET_NAME)
    except Exception:
        st.info("관리비 고지서 시트를 찾을 수 없습니다. (수도광열비 부과 내역)")
        return None

    needed = {"hvac_excl", "hvac_comm", "building", "brand"}
    if not needed.issubset(bill.columns):
        st.info("냉난방 컬럼을 찾을 수 없습니다.")
        return None

    bill = bill.copy()
    bill["hvac_total"] = bill["hvac_excl"] + bill["hvac_comm"]

    # ── Summary metrics ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("냉난방 전용 합계",  f"{bill['hvac_excl'].sum():,.2f} 만원")
    c2.metric("냉난방 공용 합계",  f"{bill['hvac_comm'].sum():,.2f} 만원")
    c3.metric("냉난방 합계",       f"{bill['hvac_total'].sum():,.2f} 만원")

    # ── Building-level grouped bar ────────────────────────────────────────
    bldg = (bill.groupby("building")[["hvac_excl", "hvac_comm"]]
                .sum().reset_index()
                .sort_values("hvac_excl", ascending=False))

    fig_bldg = go.Figure([
        go.Bar(name="전용", x=bldg["building"], y=bldg["hvac_excl"],
               marker_color=_PALETTE[0],
               text=[f"{v:,.2f}" for v in bldg["hvac_excl"]], textposition="outside",
               hovertemplate="<b>%{x}</b> 전용: %{y:,.2f} 만원<extra></extra>"),
        go.Bar(name="공용", x=bldg["building"], y=bldg["hvac_comm"],
               marker_color=_PALETTE[1],
               text=[f"{v:,.2f}" for v in bldg["hvac_comm"]], textposition="outside",
               hovertemplate="<b>%{x}</b> 공용: %{y:,.2f} 만원<extra></extra>"),
    ])
    fig_bldg.update_layout(
        **_BASE_LAYOUT,
        barmode="group",
        title=dict(text="<b>건물별 냉난방 비용 (만원)</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(title="건물", tickfont=dict(color="#111111"), showgrid=False),
        yaxis=dict(title="만원", tickfont=dict(color="#111111"), showgrid=True,
                   gridcolor="#AAAAAA", rangemode="tozero"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        margin=dict(l=60, r=20, t=80, b=50),
    )
    _ev_bldg = st.plotly_chart(fig_bldg, use_container_width=True, key="ehp_hvac_bldg_chart", on_select="rerun")
    handle_chart_click(_ev_bldg, bldg, brand_col="building", field="x")

    # ── Top brands by hvac_excl ───────────────────────────────────────────
    top_n = 20
    brand_grp = (bill.groupby("brand")["hvac_excl"]
                     .sum().reset_index()
                     .sort_values("hvac_excl", ascending=False)
                     .head(top_n))

    fig_brand = go.Figure(go.Bar(
        x=brand_grp["brand"], y=brand_grp["hvac_excl"],
        marker_color=_PALETTE[0],
        text=[f"{v:,.2f}" for v in brand_grp["hvac_excl"]], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.2f} 만원<extra></extra>",
    ))
    fig_brand.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>상호별 냉난방 전용 Top {top_n} (만원)</b>",
                   font=dict(size=14, color="#111111"), x=0),
        height=400,
        xaxis=dict(tickangle=-40, tickfont=dict(color="#111111", size=9), showgrid=False),
        yaxis=dict(title="만원", tickfont=dict(color="#111111"), showgrid=True,
                   gridcolor="#AAAAAA", rangemode="tozero"),
        margin=dict(l=60, r=20, t=70, b=100),
        showlegend=False,
    )
    _ev_brand = st.plotly_chart(fig_brand, use_container_width=True, key="ehp_hvac_brand_chart", on_select="rerun")
    handle_chart_click(_ev_brand, brand_grp, brand_col="brand", field="x")

    # ── Building summary table ────────────────────────────────────────────
    tbl = bill.groupby("building").agg(
        전용=("hvac_excl",  "sum"),
        공용=("hvac_comm",  "sum"),
        합계=("hvac_total", "sum"),
        건수=("brand",      "count"),
    ).reset_index().rename(columns={"building": "건물"})
    tbl[["전용", "공용", "합계"]] = tbl[["전용", "공용", "합계"]].round(2)
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    return bill


# ─── Public entry point ───────────────────────────────────────────────────────

def render_ehp_view(name: str, data: bytes, sheet: str) -> None:
    st.subheader("EHP 전기 사용량 분석")

    analysis_type = st.radio("분석 유형", ["OAC", "전용 EHP"],
                             horizontal=True, key="ehp_analysis_type")

    if analysis_type == "OAC":
        _render_oac(name, data, sheet)
    else:
        _render_ehp_dedicated(name, data, sheet)


def _render_oac(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ OAC 전기 사용량 — 단위: kWh")

    oac = _cached_read(name, data, sheet)
    if oac.empty:
        st.error("No monthly usage data found.")
        return

    pivot, usage = _build_pivot_from_oac(oac)
    if pivot.empty:
        st.error("No monthly usage data found.")
        return

    with st.expander("Download Report (PDF)", expanded=False):
        st.caption("OAC 사용량 분석 결과를 업무용 PDF로 생성합니다.")
        lang_oac = st.radio("Language", ["한국어 (ko)", "English (en)"],
                            horizontal=True, key="ehp_oac_report_lang")
        if st.button("Generate PDF Report", key="ehp_oac_gen_report"):
            from ehp_report import generate_ehp_oac_pdf
            with st.spinner("Generating PDF…"):
                _ded_df, _ded_col0 = _parse_dedicated_ehp_df(data, sheet)
                try:
                    _bill_df = read_billing_sheet(name, data, BILLING_SHEET_NAME)
                except Exception:
                    _bill_df = None
                pdf_bytes = generate_ehp_oac_pdf(
                    pivot, usage,
                    context={"date": date.today(), "sheet_name": sheet},
                    lang="ko" if lang_oac.startswith("한") else "en",
                    dedicated_df=_ded_df,
                    dedicated_col0=_ded_col0,
                    bill_df=_bill_df,
                )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"ehp_oac_report_{date.today()}.pdf",
                mime="application/pdf",
                key="ehp_oac_dl_report",
            )

    tab1, tab2, tab_mo, tab3, tab4, tab5, tab6 = st.tabs([
        "Yearly Usage", "Year Comparison", "Monthly Usage", "Month Comparison", "계량기별",
        "이상 탐지", "판넬별 추세",
    ])
    with tab1:
        _tab_yearly(pivot)
    with tab2:
        _tab_compare(pivot)
    with tab_mo:
        _tab_monthly_total(pivot)
    with tab3:
        _tab_month(pivot)
    with tab4:
        _tab_meter(usage)
    with tab5:
        _tab_anomaly(pivot)
    with tab6:
        _tab_panel_trend(name, data, sheet)

    with st.expander("Cumulative readings — grouped by year"):
        st.dataframe(oac, use_container_width=True)

    with st.expander("Monthly usage — grouped by year"):
        usage_groups = group_raw_slice_by_year(usage)
        u_tabs = st.tabs([str(y) for y in usage_groups])
        for tab, (yr, ydf) in zip(u_tabs, usage_groups.items()):
            with tab:
                st.dataframe(ydf.rename(columns=lambda c: c.replace(f"{yr}_", "")), use_container_width=True)


def _parse_dedicated_ehp_df(data: bytes, sheet: str) -> tuple:
    """Parse 전용 EHP section from the Excel file. Returns (sliced_df, col0) or (None, None)."""
    full = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")
    for ri in range(len(full)):
        if full.iloc[ri].astype(str).str.contains("전용 EHP", na=False).any():
            end = len(full)
            for rj in range(ri + 1, len(full)):
                if full.iloc[rj].astype(str).str.contains("▣", na=False).any():
                    end = rj
                    break
            sliced = full.iloc[ri+1:end, :11].reset_index(drop=True)
            sliced.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in sliced.iloc[0]]
            sliced = sliced.iloc[1:].reset_index(drop=True)
            col0 = sliced.columns[0]
            sliced[col0] = sliced[col0].ffill()
            sliced = sliced[sliced[col0].astype(str).str.endswith("동")].reset_index(drop=True)
            data_cols = [c for c in sliced.columns if c != col0]
            sliced = sliced[sliced[data_cols].notna().any(axis=1)].reset_index(drop=True)
            if "판넬명" in sliced.columns:
                sliced["판넬명"] = sliced.groupby(col0)["판넬명"].ffill().astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            if "장비번호" in sliced.columns:
                sliced["장비번호"] = sliced.groupby(col0)["장비번호"].ffill()
            if "전기 사용량" in sliced.columns and "매장별 가동시간" in sliced.columns:
                e = pd.to_numeric(sliced["전기 사용량"], errors="coerce")
                t = pd.to_numeric(sliced["매장별 가동시간"], errors="coerce")
                sliced["효율 (kWh/hr)"] = (e / t.replace(0, float("nan"))).round(3)
            return sliced, col0
    return None, None


def _render_ehp_dedicated(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ 전용 EHP 검침 자료")

    _sliced, _col0 = _parse_dedicated_ehp_df(data, sheet)
    if _sliced is None:
        st.error("전용 EHP 검침 자료 섹션을 찾을 수 없습니다.")
        return

    # ── Metric selection ─────────────────────────────────────────────────────
    _has_elec = "전기 사용량" in _sliced.columns
    _has_time = "매장별 가동시간" in _sliced.columns
    _metric_options = []
    if _has_elec:
        _metric_options.append("전기 사용량")
    if _has_time:
        _metric_options.append("매장별 가동시간")
    if "효율 (kWh/hr)" in _sliced.columns:
        _metric_options.append("효율 (kWh/hr)")

    if not _metric_options:
        st.warning("전기 사용량 및 매장별 가동시간 column not found.")
    else:
        _metric_label_map = {
            "전기 사용량": ("전기 사용량 (kWh)", "kWh"),
            "매장별 가동시간": ("가동시간 (hr)", "hr"),
            "효율 (kWh/hr)": ("효율 (kWh/hr)", "kWh/hr"),
        }
        _metric_tab_names = {
            "전기 사용량": "전기 사용량",
            "매장별 가동시간": "가동시간",
            "효율 (kWh/hr)": "효율",
        }
        all_dong  = sorted(_sliced[_col0].dropna().unique(), key=str)

        # Build chart type options based on available columns
        _chart_opts = ["건물별"]
        if "장비번호" in _sliced.columns:
            _chart_opts.append("장비별")
        if "판넬명" in _sliced.columns:
            _chart_opts.append("판넬별")
        if "상호" in _sliced.columns:
            _chart_opts.append("상호별")

        # ── Shared controls (apply to all metric tabs) ────────────────
        _graph_sel = st.radio("그래프", ["히스토그램", "바 차트"],
                              horizontal=True, key="ehp_ded_graph")
        _c1, _c2 = st.columns(2)
        with _c1:
            sel_dong = st.selectbox("동 선택", ["전체"] + all_dong, key="ehp_ded_dong")
        with _c2:
            chart_type = st.selectbox("차트 유형", _chart_opts, key="ehp_ded_chart",
                                      disabled=(_graph_sel == "히스토그램"))

        if sel_dong != "전체":
            _sliced = _sliced[_sliced[_col0] == sel_dong]

        with st.expander("검색"):
            _sc1, _sc2, _sc3 = st.columns(3)
            with _sc1:
                _q_panel = st.text_input("판넬명", key="ehp_search_panel")
            with _sc2:
                _q_jangbi = st.text_input("장비번호", key="ehp_search_jangbi")
            with _sc3:
                _q_sangho = st.text_input("상호", key="ehp_search_sangho")
        if _q_panel and "판넬명" in _sliced.columns:
            _sliced = _sliced[_sliced["판넬명"].astype(str).str.contains(_q_panel, case=False, na=False)]
        if _q_jangbi and "장비번호" in _sliced.columns:
            _sliced = _sliced[_sliced["장비번호"].astype(str).str.contains(_q_jangbi, case=False, na=False)]
        if _q_sangho and "상호" in _sliced.columns:
            _sliced = _sliced[_sliced["상호"].astype(str).str.contains(_q_sangho, case=False, na=False)]

        # ── Metric tabs ───────────────────────────────────────────────
        _tab_labels = [_metric_tab_names[m] for m in _metric_options]
        if _has_elec and _has_time:
            _tab_labels.append("산점도")
        _metric_tabs = st.tabs(_tab_labels)
        for _mtab, metric_sel in zip(_metric_tabs, _metric_options):
            with _mtab:
                val_col_label, y_unit = _metric_label_map[metric_sel]
                _ts = _sliced.copy()
                _ts[metric_sel] = pd.to_numeric(_ts[metric_sel], errors="coerce")

                def _bar_chart(grouped, title, x_title, _vcl=val_col_label, _yu=y_unit, _key=metric_sel):
                    grouped = grouped.sort_values(_vcl, ascending=False).reset_index(drop=True)
                    x_labels = grouped[grouped.columns[0]].tolist()
                    fig = go.Figure(go.Bar(
                        x=x_labels, y=grouped[_vcl],
                        marker_color="#4C72B0",
                        text=[f"{v:,.0f}" for v in grouped[_vcl]],
                        textposition="outside", cliponaxis=False,
                        hovertemplate=f"<b>%{{x}}</b>: %{{y:,.0f}} {_yu}<extra></extra>",
                    ))
                    fig.update_layout(
                        **_BASE_LAYOUT,
                        title=dict(text=title, font=dict(size=14, color="#111111"), x=0),
                        height=420,
                        xaxis=dict(title=dict(text=x_title, font=dict(color="#111111")), tickfont=dict(color="#111111"), showgrid=False, zeroline=False),
                        yaxis=dict(title=dict(text=_yu, font=dict(color="#111111")), tickfont=dict(color="#111111"), showgrid=True, gridcolor="#AAAAAA", zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero"),
                        margin=dict(l=60, r=20, t=70, b=80),
                        showlegend=False,
                    )
                    _ev = st.plotly_chart(fig, use_container_width=True, key=f"ehp_ded_bar_{_key}", on_select="rerun")
                    handle_chart_click(_ev, grouped, brand_col=grouped.columns[0], field="x")
                    st.dataframe(grouped, hide_index=True, use_container_width=True)

                total_val = _ts[metric_sel].sum(min_count=1)
                st.metric(f"합계 {val_col_label}", f"{total_val:,.0f} {y_unit}" if pd.notna(total_val) else "N/A")

                if _graph_sel == "히스토그램":
                    _bins = st.session_state.get("bins", 50)
                    _ehp_iqr_k = st.slider("IQR 배수 (k)", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                                           key=f"ehp_iqr_k_{metric_sel}",
                                           help="이상치 기준: Q1 − k×IQR  /  Q3 + k×IQR")
                    _s    = _ts[metric_sel]
                    _v    = _s.dropna()
                    if not _v.empty:
                        _eq1 = float(_v.quantile(0.25))
                        _eq3 = float(_v.quantile(0.75))
                        _eiqr = _eq3 - _eq1
                        _lo = _eq1 - _ehp_iqr_k * _eiqr
                        _hi = _eq3 + _ehp_iqr_k * _eiqr
                        st.markdown(
                            f"$$Q_1 = {_eq1:,.0f},\\quad Q_3 = {_eq3:,.0f},\\quad IQR = {_eiqr:,.0f}$$\n\n"
                            f"$$\\text{{Lower}} = {_lo:,.0f},\\quad \\text{{Upper}} = {_hi:,.0f}$$"
                        )
                        _display_cols = [c for c in [_col0, "판넬명", "장비번호", "상호", metric_sel] if c in _ts.columns]
                        plot_hist_with_tails(
                            _s, _bins, _lo, _hi,
                            title=metric_sel,
                            key=f"ehp_hist_{metric_sel}",
                            source_df=_ts,
                            val_col=metric_sel,
                            display_cols=_display_cols,
                        )
                else:
                    _chart_col_map = {
                        "건물별": (_col0,     "동",      f"<b>건물별 {metric_sel} 합계</b>"),
                        "장비별": ("장비번호", "장비번호", f"<b>장비별 {metric_sel}</b>"),
                        "판넬별": ("판넬명",   "판넬명",   f"<b>판넬별 {metric_sel} 합계</b>"),
                        "상호별": ("상호",     "상호",     f"<b>상호별 {metric_sel}</b>"),
                    }
                    grp_col, x_title, title = _chart_col_map[chart_type]
                    grouped = _ts.groupby(grp_col)[metric_sel].sum().reset_index()
                    grouped.columns = [grp_col, val_col_label]
                    _bar_chart(grouped, title, x_title)

    # ── Scatter tab ───────────────────────────────────────────────────
    if _has_elec and _has_time:
        with _metric_tabs[-1]:
            _sc_df = _sliced.copy()
            _sc_df["전기 사용량"] = pd.to_numeric(_sc_df["전기 사용량"], errors="coerce")
            _sc_df["매장별 가동시간"] = pd.to_numeric(_sc_df["매장별 가동시간"], errors="coerce")
            _sc_df = _sc_df.dropna(subset=["전기 사용량", "매장별 가동시간"])

            _color_col = _col0
            _color_vals = _sc_df[_color_col].astype(str)
            _uniq_colors = sorted(_color_vals.unique())
            _cmap = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(_uniq_colors)}

            _sc_fig = go.Figure()
            for _grp in _uniq_colors:
                _gdf = _sc_df[_color_vals == _grp]
                _hover_cols = [c for c in ["판넬명", "장비번호", "상호"] if c in _gdf.columns]
                _custom = _gdf[_hover_cols].values if _hover_cols else None
                _hover_extra = "".join(
                    f"<br>{c}: %{{customdata[{i}]}}" for i, c in enumerate(_hover_cols)
                )
                _sc_fig.add_trace(go.Scatter(
                    x=_gdf["매장별 가동시간"],
                    y=_gdf["전기 사용량"],
                    mode="markers",
                    name=_grp,
                    marker=dict(color=_cmap[_grp], size=8, opacity=1.0,
                                line=dict(color="white", width=0.5)),
                    customdata=_custom,
                    hovertemplate=(
                        f"<b>{_grp}</b><br>"
                        "가동시간: %{x:,.1f} hr<br>"
                        "전기 사용량: %{y:,.1f} kWh"
                        + _hover_extra + "<extra></extra>"
                    ),
                ))

            # Trend line (OLS)
            _sc_x = _sc_df["매장별 가동시간"].values.astype(float)
            _sc_y = _sc_df["전기 사용량"].values.astype(float)
            _trend_stats = None
            if len(_sc_x) >= 3:
                _coeffs = np.polyfit(_sc_x, _sc_y, 1)
                _xr = np.linspace(_sc_x.min(), _sc_x.max(), 100)
                _yr = np.polyval(_coeffs, _xr)
                _corr = float(np.corrcoef(_sc_x, _sc_y)[0, 1])
                _r2 = _corr ** 2
                _slope, _intercept = float(_coeffs[0]), float(_coeffs[1])
                _trend_stats = {
                    "slope": _slope, "intercept": _intercept,
                    "r": _corr, "r2": _r2, "n": len(_sc_x),
                    "x_mean": float(_sc_x.mean()), "y_mean": float(_sc_y.mean()),
                }
                _sc_fig.add_trace(go.Scatter(
                    x=_xr, y=_yr, mode="lines", name=f"Trend (r={_corr:.2f})",
                    line=dict(color="#C44E52", width=1.5, dash="dash"), showlegend=True,
                ))

            _sc_fig.update_layout(
                **_BASE_LAYOUT,
                title=dict(text="<b>가동시간 (hr) vs 전기 사용량 (kWh)</b>",
                           font=dict(size=14, color="#000000"), x=0),
                height=440,
                xaxis=dict(title=dict(text="가동시간 (hr)", font=dict(color="#000000")),
                           showgrid=True, gridcolor="#DDDDDD",
                           zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero", tickfont=dict(color="#111111")),
                yaxis=dict(title=dict(text="전기 사용량 (kWh)", font=dict(color="#000000")),
                           showgrid=True, gridcolor="#DDDDDD",
                           zeroline=True, zerolinecolor="#AAAAAA", rangemode="tozero", tickfont=dict(color="#111111")),
                legend=dict(orientation="v", x=1.01, xanchor="left", y=1,
                            font=dict(size=10, color="#333333")),
                margin=dict(l=60, r=160, t=60, b=50),
            )
            _ev_sc = st.plotly_chart(_sc_fig, use_container_width=True, key="ehp_ded_scatter", on_select="rerun")

            if _trend_stats:
                _sl  = _trend_stats["slope"]
                _ic  = _trend_stats["intercept"]
                _r   = _trend_stats["r"]
                _r2  = _trend_stats["r2"]
                _n   = _trend_stats["n"]
                _xm  = _trend_stats["x_mean"]
                _ym  = _trend_stats["y_mean"]

                _ic_str = f"+ {_ic:,.1f}" if _ic >= 0 else f"- {abs(_ic):,.1f}"
                st.markdown(
                    f"**추세선 방정식:** `y = {_sl:,.2f}x {_ic_str}`  &nbsp;|&nbsp;  "
                    f"**r = {_r:.3f}**  &nbsp;|&nbsp;  "
                    f"**R² = {_r2:.3f}**  &nbsp;|&nbsp;  "
                    f"**n = {_n}**"
                )
                st.markdown(
                    f"**평균 가동시간:** {_xm:,.1f} hr  &nbsp;|&nbsp;  "
                    f"**평균 전기 사용량:** {_ym:,.1f} kWh"
                )

                # Interpretation
                _abs_r = abs(_r)
                if _abs_r >= 0.8:
                    _corr_desc = "강한 양의 상관관계" if _r > 0 else "강한 음의 상관관계"
                elif _abs_r >= 0.5:
                    _corr_desc = "중간 정도의 양의 상관관계" if _r > 0 else "중간 정도의 음의 상관관계"
                elif _abs_r >= 0.3:
                    _corr_desc = "약한 양의 상관관계" if _r > 0 else "약한 음의 상관관계"
                else:
                    _corr_desc = "거의 상관관계 없음"

                if _r2 >= 0.7:
                    _r2_desc = f"가동시간이 전기 사용량 분산의 {_r2*100:.0f}%를 설명합니다 (설명력 높음)."
                elif _r2 >= 0.4:
                    _r2_desc = f"가동시간이 전기 사용량 분산의 {_r2*100:.0f}%를 설명합니다 (설명력 중간)."
                else:
                    _r2_desc = f"가동시간이 전기 사용량 분산의 {_r2*100:.0f}%를 설명합니다 (설명력 낮음 — 다른 요인 영향 가능)."

                _slope_desc = (
                    f"가동시간이 1시간 증가할 때 전기 사용량은 평균 **{_sl:,.2f} kWh** "
                    + ("증가하는" if _sl > 0 else "감소하는") + " 경향이 있습니다."
                )

                st.info(
                    f"**해석:** 가동시간과 전기 사용량 사이에 **{_corr_desc}** (r = {_r:.3f})가 있습니다. "
                    f"{_r2_desc} {_slope_desc}"
                )


    with st.expander("Download Report (PDF)", expanded=False):
        st.caption("전용 EHP 분석 결과를 업무용 PDF로 생성합니다.")
        lang_ded = st.radio("Language", ["한국어 (ko)", "English (en)"],
                            horizontal=True, key="ehp_ded_report_lang")
        if st.button("Generate PDF Report", key="ehp_ded_gen_report"):
            from ehp_report import generate_ehp_dedicated_pdf
            with st.spinner("Generating PDF…"):
                _pdf_bytes = generate_ehp_dedicated_pdf(
                    _sliced, _col0,
                    context={"date": date.today()},
                    lang="ko" if lang_ded.startswith("한") else "en",
                )
            st.download_button(
                label="Download PDF Report",
                data=_pdf_bytes,
                file_name=f"ehp_dedicated_report_{date.today()}.pdf",
                mime="application/pdf",
                key="ehp_ded_dl_report",
            )

    with st.expander("Raw data"):
        st.dataframe(_sliced, use_container_width=True)
