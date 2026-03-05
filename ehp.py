import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import numpy as np
from data import (
    read_ehp_oac_sheet, read_ehp_raw_slice, compute_monthly_usage,
    group_raw_slice_by_year, read_billing_sheet, BILLING_SHEET_NAME,
)
from viz import plot_hist_with_tails

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


def _build_pivot(raw_slice: pd.DataFrame) -> pd.DataFrame:
    """Sum all meter rows per month → pivot: index=월, columns=year."""
    import re as _re
    yr_pat = _re.compile(r'^(20\d{2})_(\d{1,2})월$')
    usage = compute_monthly_usage(raw_slice)
    records = []
    for col in usage.columns:
        m = yr_pat.match(str(col))
        if not m:
            continue
        yr, mo = int(m.group(1)), int(m.group(2))
        total = pd.to_numeric(usage[col], errors="coerce").sum(min_count=1)
        records.append({"year": yr, "month": mo, "total": total})
    if not records:
        return pd.DataFrame()
    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None
    return pivot


# ─── Tab 4: Per-Meter Analysis ────────────────────────────────────────────────

def _tab_meter(usage: pd.DataFrame) -> None:
    import re as _re
    yr_pat = _re.compile(r'^(20\d{2})_(\d{1,2})월$')
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
    _tab_yearly(pivot)
    st.divider()
    _tab_month(pivot, key="ehp_month_sel_meter")


# ─── Tab 1: Yearly Usage ──────────────────────────────────────────────────────

def _tab_yearly(pivot: pd.DataFrame) -> None:
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
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame({"Year": years, "Total kWh": [f"{v:,.0f}" for v in values]})
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
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        legend=dict(orientation="h", x=0, y=1.05, yanchor="bottom"),
        margin=dict(l=60, r=20, t=65, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

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
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

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
    flags = {}
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
            flags[(mo_label, yr)] = ("high" if is_hi else "low" if is_lo else "normal")
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
        yaxis=dict(title="Month", tickfont=dict(color="#111111"), autorange="reversed"),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

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

    # Compute monthly usage (diff across sorted cum columns)
    cum_sorted = sorted(cum_cols, key=lambda c: _YM_PAT.match(c).groups())
    readings = oac[cum_sorted].apply(pd.to_numeric, errors="coerce")
    usage_m  = readings.diff(axis=1)
    usage_m.columns = cum_sorted

    # Build a group label column: panel_name or meter_no
    group_col = "panel_name" if "panel_name" in oac.columns else "meter_no"
    labels = oac[group_col].astype(str).str.strip().fillna("(unknown)")
    all_labels = sorted(labels.unique())

    sel_panels = st.multiselect("판넬 선택", all_labels, default=all_labels[:min(5, len(all_labels))],
                                key="ehp_panel_sel")
    if not sel_panels:
        st.info("Select at least one panel.")
        return

    # Build time-series: aggregate rows per label → sum usage per month
    time_axis = []
    for c in cum_sorted:
        m = _YM_PAT.match(c)
        yr, mo = int(m.group(1)), int(m.group(2))
        time_axis.append(f"{yr}-{mo:02d}")

    fig = go.Figure()
    colors = _PALETTE
    for i, panel in enumerate(sel_panels):
        mask = labels == panel
        if not mask.any():
            continue
        vals = usage_m[mask].sum(min_count=1)
        fig.add_trace(go.Scatter(
            x=time_axis, y=vals.values,
            mode="lines+markers",
            name=panel,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{panel}</b><br>%{{x}}: %{{y:,.0f}} kWh<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>판넬별 월간 사용량 추세 (kWh)</b>", font=dict(size=13, color="#111111"), x=0),
        height=420,
        xaxis=dict(title=dict(text="Year-Month", font=dict(color="#111111")),
                   tickangle=-45, tickfont=dict(size=9, color="#111111"),
                   showgrid=True, gridcolor="#DDDDDD"),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")),
                   showgrid=True, gridcolor="#DDDDDD", zeroline=False,
                   tickfont=dict(color="#111111")),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                    font=dict(size=10, color="#111111")),
        margin=dict(l=60, r=160, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table: yearly totals per selected panel
    yearly: dict = {}
    for panel in sel_panels:
        mask = labels == panel
        if not mask.any():
            continue
        row_totals = {}
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

def render_ehp_view(name: str, data: bytes, sheet: str) -> None:
    st.subheader("EHP 전기 사용량 분석")

    analysis_type = st.radio("분석 유형", ["OAC", "전용 EHP"], horizontal=True, key="ehp_analysis_type")

    if analysis_type == "OAC":
        _render_oac(name, data, sheet)
    else:
        _render_ehp_dedicated(name, data, sheet)


def _render_oac(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ OAC 전기 사용량 — 단위: kWh")

    raw_slice = read_ehp_raw_slice(name, data, sheet)

    pivot = _build_pivot(raw_slice)
    if pivot.empty:
        st.error("No monthly usage data found.")
        return

    usage = compute_monthly_usage(raw_slice)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Yearly Usage", "Year Comparison", "Month Comparison", "계량기별",
        "이상 탐지", "판넬별 추세",
    ])
    with tab1:
        _tab_yearly(pivot)
    with tab2:
        _tab_compare(pivot)
    with tab3:
        _tab_month(pivot)
    with tab4:
        _tab_meter(usage)
    with tab5:
        _tab_anomaly(pivot)
    with tab6:
        _tab_panel_trend(name, data, sheet)

    with st.expander("Cumulative readings — grouped by year"):
        year_groups = group_raw_slice_by_year(raw_slice)
        yr_tabs = st.tabs([str(y) for y in year_groups])
        for tab, (yr, ydf) in zip(yr_tabs, year_groups.items()):
            with tab:
                st.dataframe(ydf.rename(columns=lambda c: c.replace(f"{yr}_", "")), use_container_width=True)

    with st.expander("Monthly usage — grouped by year"):
        usage_groups = group_raw_slice_by_year(usage)
        u_tabs = st.tabs([str(y) for y in usage_groups])
        for tab, (yr, ydf) in zip(u_tabs, usage_groups.items()):
            with tab:
                st.dataframe(ydf.rename(columns=lambda c: c.replace(f"{yr}_", "")), use_container_width=True)


def _render_ehp_dedicated(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ 전용 EHP 검침 자료")

    import io as _io
    _full = __import__('pandas').read_excel(_io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")
    for ri in range(len(_full)):
        if _full.iloc[ri].astype(str).str.contains("전용 EHP", na=False).any():
            _end = len(_full)
            for _rj in range(ri + 1, len(_full)):
                if _full.iloc[_rj].astype(str).str.contains("▣", na=False).any():
                    _end = _rj
                    break
            _sliced = _full.iloc[ri+1:_end, :11].reset_index(drop=True)
            _sliced.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in _sliced.iloc[0]]
            _sliced = _sliced.iloc[1:].reset_index(drop=True)
            _col0 = _sliced.columns[0]
            _sliced[_col0] = _sliced[_col0].ffill()
            _sliced = _sliced[_sliced[_col0].astype(str).str.endswith("동")].reset_index(drop=True)
            # Drop pure building-header rows (only _col0 has a value, all other columns are NaN)
            _data_cols = [c for c in _sliced.columns if c != _col0]
            _sliced = _sliced[_sliced[_data_cols].notna().any(axis=1)].reset_index(drop=True)
            if "판넬명" in _sliced.columns:
                _sliced["판넬명"] = _sliced.groupby(_col0)["판넬명"].ffill().astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            if "장비번호" in _sliced.columns:
                _sliced["장비번호"] = _sliced.groupby(_col0)["장비번호"].ffill()

            # ── Metric selection ─────────────────────────────────────────────
            _metric_options = []
            if "전기 사용량" in _sliced.columns:
                _metric_options.append("전기 사용량")
            if "매장별 가동시간" in _sliced.columns:
                _metric_options.append("매장별 가동시간")

            # Compute efficiency column when both source metrics are present
            _has_elec = "전기 사용량" in _sliced.columns
            _has_time = "매장별 가동시간" in _sliced.columns
            if _has_elec and _has_time:
                _e = pd.to_numeric(_sliced["전기 사용량"], errors="coerce")
                _t = pd.to_numeric(_sliced["매장별 가동시간"], errors="coerce")
                _sliced["효율 (kWh/hr)"] = (_e / _t.replace(0, float("nan"))).round(3)
                _metric_options.append("효율 (kWh/hr)")

            if not _metric_options:
                st.warning("전기 사용량 및 매장별 가동시간 column not found.")
            else:
                _metric_label_map = {
                    "전기 사용량": ("전기 사용량 (kWh)", "kWh"),
                    "매장별 가동시간": ("가동시간 (hr)", "hr"),
                    "효율 (kWh/hr)": ("효율 (kWh/hr)", "kWh/hr"),
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

                chart_type = st.selectbox("차트 유형", _chart_opts, key="ehp_ded_chart")

                sel_dong = st.selectbox("동 선택", ["전체"] + all_dong, key="ehp_ded_dong")
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

                _metric_tabs = st.tabs(_metric_options)
                for _mtab, metric_sel in zip(_metric_tabs, _metric_options):
                    with _mtab:
                        val_col_label, y_unit = _metric_label_map[metric_sel]
                        _ts = _sliced.copy()
                        _ts[metric_sel] = pd.to_numeric(_ts[metric_sel], errors="coerce")

                        def _bar_chart(grouped, x_labels, title, x_title, _vcl=val_col_label, _yu=y_unit):
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
                                yaxis=dict(title=dict(text=_yu, font=dict(color="#111111")), tickfont=dict(color="#111111"), showgrid=True, gridcolor="#AAAAAA", zeroline=False),
                                margin=dict(l=60, r=20, t=70, b=80),
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(grouped, hide_index=True, use_container_width=True)

                        total_val = _ts[metric_sel].sum(min_count=1)
                        st.metric(f"합계 {val_col_label}", f"{total_val:,.0f} {y_unit}" if pd.notna(total_val) else "N/A")

                        _graph_opts = ["바 차트", "히스토그램"]
                        if metric_sel == "효율 (kWh/hr)":
                            _graph_opts = ["바 차트", "히스토그램"]
                        _graph_sel = st.radio("그래프", _graph_opts, horizontal=True, key=f"ehp_graph_{metric_sel}")

                        if _graph_sel == "히스토그램":
                            import numpy as _np
                            import numpy as _np
                            _bins = st.session_state.get("bins", 50)
                            _tail = st.session_state.get("tail", 20)
                            _s    = _ts[metric_sel]
                            _v    = _s.dropna()
                            if not _v.empty:
                                _lo = float(_np.percentile(_v, _tail))
                                _hi = float(_np.percentile(_v, 100 - _tail))
                                _display_cols = [c for c in [_col0, "판넬명", "장비번호", "상호", metric_sel] if c in _ts.columns]
                                plot_hist_with_tails(
                                    _s, _bins, _lo, _hi,
                                    title=metric_sel,
                                    tail_pct=_tail,
                                    key=f"ehp_hist_{metric_sel}",
                                    source_df=_ts,
                                    val_col=metric_sel,
                                    display_cols=_display_cols,
                                )

                        else:
                            if chart_type == "건물별":
                                grouped = _ts.groupby(_col0)[metric_sel].sum().reset_index()
                                grouped.columns = [_col0, val_col_label]
                                _bar_chart(grouped, grouped[_col0].tolist(), f"<b>건물별 {metric_sel} 합계</b>", "동")
                            elif chart_type == "장비별":
                                grouped = _ts.groupby("장비번호")[metric_sel].sum().reset_index()
                                grouped.columns = ["장비번호", val_col_label]
                                _bar_chart(grouped, grouped["장비번호"].tolist(), f"<b>장비별 {metric_sel}</b>", "장비번호")
                            elif chart_type == "판넬별":
                                grouped = _ts.groupby("판넬명")[metric_sel].sum().reset_index()
                                grouped.columns = ["판넬명", val_col_label]
                                _bar_chart(grouped, grouped["판넬명"].tolist(), f"<b>판넬별 {metric_sel} 합계</b>", "판넬명")
                            elif chart_type == "상호별":
                                grouped = _ts.groupby("상호")[metric_sel].sum().reset_index()
                                grouped.columns = ["상호", val_col_label]
                                _bar_chart(grouped, grouped["상호"].tolist(), f"<b>상호별 {metric_sel}</b>", "상호")

            # ── Scatter: 가동시간 vs 전기 사용량 ────────────────────────────
            if _has_elec and _has_time:
                st.divider()
                st.subheader("가동시간 vs 전기 사용량 산점도")
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
                        marker=dict(color=_cmap[_grp], size=8, opacity=0.8,
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
                if len(_sc_x) >= 3:
                    _coeffs = np.polyfit(_sc_x, _sc_y, 1)
                    _xr = np.linspace(_sc_x.min(), _sc_x.max(), 100)
                    _yr = np.polyval(_coeffs, _xr)
                    _corr = float(np.corrcoef(_sc_x, _sc_y)[0, 1])
                    _sc_fig.add_trace(go.Scatter(
                        x=_xr, y=_yr, mode="lines", name=f"Trend (r={_corr:.2f})",
                        line=dict(color="#C44E52", width=1.5, dash="dash"), showlegend=True,
                    ))

                _sc_fig.update_layout(
                    **_BASE_LAYOUT,
                    title=dict(text="<b>가동시간 (hr) vs 전기 사용량 (kWh)</b>",
                               font=dict(size=13, color="#111111"), x=0),
                    height=440,
                    xaxis=dict(title="가동시간 (hr)", showgrid=True, gridcolor="#DDDDDD",
                               zeroline=False, tickfont=dict(color="#111111")),
                    yaxis=dict(title="전기 사용량 (kWh)", showgrid=True, gridcolor="#DDDDDD",
                               zeroline=False, tickfont=dict(color="#111111")),
                    legend=dict(orientation="v", x=1.01, xanchor="left", y=1,
                                font=dict(size=10, color="#333333")),
                    margin=dict(l=60, r=160, t=60, b=50),
                )
                st.plotly_chart(_sc_fig, use_container_width=True)

            # ── 전기요금 연계 (Billing cross-reference) ─────────────────────
            with st.expander("전기요금 연계 — EHP 사용량 vs 부과 요금"):
                try:
                    _bill = read_billing_sheet(name, data, BILLING_SHEET_NAME)
                    _bill_brand = _bill[["brand", "elect_total", "elect_excl", "elect_comm", "size_m2"]].copy()
                    _bill_brand = _bill_brand[_bill_brand["elect_total"] > 0]

                    # Join on brand name (상호 in EHP ↔ brand in billing)
                    if "상호" in _sliced.columns:
                        _ehp_brand = _sliced.copy()
                        _ehp_brand["전기 사용량"] = pd.to_numeric(_ehp_brand["전기 사용량"], errors="coerce")
                        _ehp_agg = _ehp_brand.groupby("상호")["전기 사용량"].sum(min_count=1).reset_index()
                        _ehp_agg.columns = ["brand", "ehp_kwh"]
                        _joined = _bill_brand.merge(_ehp_agg, on="brand", how="inner")
                        if _joined.empty:
                            st.info("부과 내역과 상호명이 일치하는 항목이 없습니다.")
                        else:
                            # Cost per kWh: elect_total (만원) × 10,000 / kWh
                            _joined["원/kWh"] = (
                                (_joined["elect_total"] * 10_000) / _joined["ehp_kwh"].replace(0, float("nan"))
                            ).round(1)
                            _joined = _joined.sort_values("원/kWh", ascending=False)

                            # Bar chart
                            _bc_fig = go.Figure(go.Bar(
                                y=_joined["brand"],
                                x=_joined["원/kWh"].fillna(0),
                                orientation="h",
                                marker_color="#DD8A00",
                                marker_line_color="white",
                                marker_line_width=0.5,
                                text=[f"{v:,.0f}" for v in _joined["원/kWh"].fillna(0)],
                                textposition="outside",
                                textfont=dict(size=9, color="#666666"),
                                hovertemplate="<b>%{y}</b><br>%{x:,.0f} 원/kWh<extra></extra>",
                            ))
                            _med_eff = float(_joined["원/kWh"].median())
                            _bc_fig.add_vline(x=_med_eff, line_dash="dot", line_color="#C44E52",
                                              line_width=1.5,
                                              annotation_text=f"Median {_med_eff:,.0f}",
                                              annotation_position="top right",
                                              annotation_font=dict(size=10, color="#C44E52"))
                            _max_lbl = _joined["brand"].astype(str).str.len().max() if len(_joined) else 20
                            _bc_fig.update_layout(
                                **_BASE_LAYOUT,
                                title=dict(text="<b>전기 요금 효율 (원/kWh)</b>",
                                           font=dict(size=13, color="#111111"), x=0),
                                height=max(380, len(_joined) * 22 + 100),
                                xaxis=dict(title="원/kWh", showgrid=True, gridcolor="#DDDDDD",
                                           zeroline=False, tickfont=dict(size=10, color="#555555")),
                                yaxis=dict(showgrid=False, zeroline=False, automargin=True,
                                           tickfont=dict(size=10, color="#555555")),
                                showlegend=False,
                                margin=dict(l=min(max(_max_lbl * 7, 120), 300), r=60, t=60, b=40),
                            )
                            st.plotly_chart(_bc_fig, use_container_width=True)

                            _show_cols = ["brand", "ehp_kwh", "elect_total", "원/kWh", "size_m2"]
                            _show_cols = [c for c in _show_cols if c in _joined.columns]
                            st.dataframe(_joined[_show_cols].reset_index(drop=True),
                                         hide_index=True, use_container_width=True)
                    else:
                        st.info("상호 column not found in 전용 EHP data — cannot join with billing.")
                except Exception as _be:
                    st.info(f"부과 내역 시트를 불러올 수 없습니다: {_be}")

            with st.expander("Raw data"):
                st.dataframe(_sliced, use_container_width=True)
            break
    else:
        st.error("전용 EHP 검침 자료 섹션을 찾을 수 없습니다.")
