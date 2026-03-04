import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import read_ehp_oac_sheet, read_ehp_raw_slice

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


def _build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return pivot: index=month label (1월..12월), columns=year (int), values=kWh total."""
    cum_cols = sorted(c for c in df.columns if c.startswith("cum_"))
    records = []
    for i in range(1, len(cum_cols)):
        cur, prev = cum_cols[i], cum_cols[i - 1]
        _, yr, mo = cur.split("_")
        usage = (df[cur] - df[prev]).clip(lower=0).sum(min_count=1)
        records.append({"year": int(yr), "month": int(mo), "total": usage})

    if not records:
        return pd.DataFrame()

    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None
    return pivot


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

def _tab_month(pivot: pd.DataFrame) -> None:
    all_labels = [f"{m}월" for m in range(1, 13)]
    available  = [m for m in all_labels if m in pivot.index]

    sel_month = st.selectbox("Month", available, key="ehp_month_sel")
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
    st.subheader("EHP(OAC) 전기 사용량 분석")
    st.caption("단위: kWh — 월별 누계 검침에서 월 사용량 산출")

    df = _cached_read(name, data, sheet)
    if df.empty:
        st.error("Parser returned no usable data.")
        return

    pivot = _build_pivot(df)
    if pivot.empty:
        st.error("No monthly usage data found.")
        return

    tab1, tab2, tab3 = st.tabs(["Yearly Usage", "Year Comparison", "Month Comparison"])
    with tab1:
        _tab_yearly(pivot)
    with tab2:
        _tab_compare(pivot)
    with tab3:
        _tab_month(pivot)

    with st.expander("Raw slice — col M to DG, unprocessed"):
        raw_slice = read_ehp_raw_slice(name, data, sheet)
        st.dataframe(raw_slice, use_container_width=True)

    with st.expander("Parsed df — cumulative readings"):
        st.dataframe(df, use_container_width=True)

    with st.expander("Computed monthly usage (pivot)"):
        st.dataframe(pivot, use_container_width=True)
