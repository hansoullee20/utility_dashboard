import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import read_ehp_oac_sheet, read_ehp_raw_slice, compute_monthly_usage, group_raw_slice_by_year

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

    tab1, tab2, tab3, tab4 = st.tabs(["Yearly Usage", "Year Comparison", "Month Comparison", "계량기별"])
    with tab1:
        _tab_yearly(pivot)
    with tab2:
        _tab_compare(pivot)
    with tab3:
        _tab_month(pivot)
    with tab4:
        _tab_meter(usage)

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
            _sliced.columns = _sliced.iloc[0]
            _sliced = _sliced.iloc[1:].reset_index(drop=True)
            _col0 = _sliced.columns[0]
            _sliced[_col0] = _sliced[_col0].ffill()
            if "판넬명" in _sliced.columns:
                _sliced["판넬명"] = _sliced["판넬명"].ffill()
            if "장비번호" in _sliced.columns:
                _sliced["장비번호"] = _sliced["장비번호"].ffill()
            _sliced = _sliced[_sliced[_col0].astype(str).str.endswith("동")].reset_index(drop=True)
            st.dataframe(_sliced, use_container_width=True)
            break
    else:
        st.error("전용 EHP 검침 자료 섹션을 찾을 수 없습니다.")
