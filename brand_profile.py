"""brand_profile.py — Comprehensive single-brand profile across all data sources."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import to_numeric_series, st_safe
from utils_plot import handle_chart_click

_C_PREV  = "#A8C4E0"   # light blue  — previous month
_C_CURR  = "#4C72B0"   # blue        — this month
_C_AVG   = "#DD8A00"   # amber       — fleet average
_C_POS   = "#E07070"   # red-ish     — positive change (increase)
_C_NEG   = "#6AAB6A"   # green       — negative change (decrease)

_CAT_META = [
    ("water",  "수도",  "m³"),
    ("hwater", "온수",  "m³"),
    ("elect",  "전기",  "kWh"),
    ("heat",   "열",    "m³/MWh"),
]
_LABEL = {p: lbl for p, lbl, _ in _CAT_META}
_UNIT  = {p: u   for p, _,   u in _CAT_META}

_QUAD_LABEL = {
    "HH": "🔴 HH — 변화·비율 모두 상위",
    "HL": "🟠 HL — 변화 상위, 비율 하위",
    "LH": "🟡 LH — 변화 하위, 비율 상위",
    "LL": "🟢 LL — 변화·비율 모두 하위",
    "—":  "— 정상",
}


def _quad(cv, pv, lo_c, hi_c, lo_p, hi_p) -> str:
    if pd.isna(cv) or pd.isna(pv):
        return "—"
    if cv >= hi_c and pv >= hi_p: return "HH"
    if cv >= hi_c and pv <= lo_p: return "HL"
    if cv <= lo_c and pv >= hi_p: return "LH"
    if cv <= lo_c and pv <= lo_p: return "LL"
    return "—"


def _v(row, col):
    val = row.get(col)
    return to_numeric_series(pd.Series([val]))[0]


def _fmt(v, decimals=2, sign=False):
    if pd.isna(v): return "—"
    fmt = f"{{:+,.{decimals}f}}" if sign else f"{{:,.{decimals}f}}"
    return fmt.format(v)


def _filter_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Filter df to rows matching brand, stripping whitespace on both sides."""
    if "brand" not in df.columns:
        return pd.DataFrame()
    return df[df["brand"].astype(str).str.strip() == brand.strip()].copy()


def _peer_chart(
    df: pd.DataFrame,
    selected: str,
    cols: list[tuple[str, str]],
    title: str,
    unit: str = "",
) -> None:
    """One subplot per metric, each with its own y-axis, brand vs peer average.

    cols: list of (col_name, display_label).
    """
    num_cols = [c for c, _ in cols if c in df.columns]
    if not num_cols or "brand" not in df.columns:
        return

    agg = df.copy()
    for c in num_cols:
        agg[c] = to_numeric_series(agg[c])
    brand_agg = agg.groupby(agg["brand"].astype(str).str.strip())[num_cols].sum(min_count=1)

    sel = selected.strip()
    if sel not in brand_agg.index:
        return
    peers = brand_agg.drop(index=sel, errors="ignore")

    # Build per-metric data
    metrics = []
    for col, lbl in cols:
        if col not in brand_agg.columns:
            continue
        bv = brand_agg.loc[sel, col]
        pv = peers[col].mean() if not peers.empty else np.nan
        metrics.append((lbl, float(bv) if not pd.isna(bv) else 0.0,
                              float(pv) if not pd.isna(pv) else 0.0))
    if not metrics:
        return

    n = len(metrics)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[m[0] for m in metrics])

    for i, (lbl, bv, pv) in enumerate(metrics, start=1):
        show_legend = i == 1
        fig.add_trace(go.Bar(
            name="이번 브랜드", x=["이번 브랜드"], y=[bv],
            marker_color=_C_CURR,
            text=[f"{bv:,.2f}"], textposition="outside", cliponaxis=False,
            legendgroup="brand", showlegend=show_legend,
        ), row=1, col=i)
        fig.add_trace(go.Bar(
            name="동종 평균", x=["동종 평균"], y=[pv],
            marker_color=_C_AVG, opacity=0.75,
            text=[f"{pv:,.2f}"], textposition="outside", cliponaxis=False,
            legendgroup="peer", showlegend=show_legend,
        ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font_size=13),
        barmode="group",
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=70, b=20, l=30, r=10),
        height=300,
    )
    fig.update_yaxes(zeroline=True, rangemode="tozero")
    _ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                          key=f"peer_{title[:8]}")
    handle_chart_click(_ev, df, brand_col="brand", field="x")


def _bar(labels, values, colors, title, unit):
    """Single grouped bar chart."""
    text = [f"{v:,.2f}" if not pd.isna(v) else "" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=text, textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13),
        yaxis=dict(title=unit, zeroline=True),
        margin=dict(t=40, b=20, l=40, r=10),
        height=300,
        showlegend=False,
    )
    return fig


def _render_comparison_charts(row, cur_df, present, selected_brand):
    """Two charts per category:
    1. This month: selected brand vs peer average.
    2. MoM change: selected brand's delta vs peer average delta.
    """
    if not present:
        return

    peers = cur_df[cur_df["brand"].astype(str) != selected_brand]

    # ── Chart 1: this month vs peer average ──────────────────────────────────
    st.subheader("📊 이번달 사용량 — 동종 업체 평균 대비")
    cols = st.columns(min(len(present), 2))
    for i, p in enumerate(present):
        lbl  = _LABEL.get(p, p)
        unit = _UNIT.get(p, "")
        curr     = _v(row, f"{p}_current")
        avg_curr = to_numeric_series(peers[f"{p}_current"]).mean() if f"{p}_current" in peers.columns else np.nan

        with cols[i % 2]:
            _ev = st.plotly_chart(
                _bar(
                    ["이번달", "동종 평균"],
                    [curr, avg_curr],
                    [_C_CURR, _C_AVG],
                    lbl, unit,
                ),
                use_container_width=True,
                on_select="rerun",
                key=f"comp_curr_{p}",
            )
            handle_chart_click(_ev, cur_df, brand_col="brand", field="x")

    # ── Chart 2: MoM change — brand vs peer average ───────────────────────────
    st.subheader("📉 전월 대비 변화량 — 동종 업체 평균 대비")
    chg_cats, brand_chgs, avg_chgs = [], [], []
    for p in present:
        cv  = _v(row, f"{p}_change")
        avg = to_numeric_series(peers[f"{p}_change"]).mean() if f"{p}_change" in peers.columns else np.nan
        if pd.isna(cv) and pd.isna(avg):
            continue
        chg_cats.append(_LABEL.get(p, p))
        brand_chgs.append(cv   if not pd.isna(cv)  else 0)
        avg_chgs.append(avg    if not pd.isna(avg)  else 0)

    if chg_cats:
        brand_colors = [_C_POS if v > 0 else _C_NEG for v in brand_chgs]
        avg_colors   = [_C_POS if v > 0 else _C_NEG for v in avg_chgs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="이번 브랜드",
            x=chg_cats, y=brand_chgs,
            marker_color=brand_colors,
            text=[f"{v:+,.2f}" for v in brand_chgs],
            textposition="outside", cliponaxis=False,
        ))
        fig.add_trace(go.Bar(
            name="동종 평균",
            x=chg_cats, y=avg_chgs,
            marker_color=avg_colors,
            opacity=0.55,
            text=[f"{v:+,.2f}" for v in avg_chgs],
            textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(zeroline=True, zerolinewidth=1.5),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(t=40, b=20, l=40, r=10),
            height=320,
        )
        _ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                              key="comp_mom_chg")
        handle_chart_click(_ev, cur_df, brand_col="brand", field="x")


def render_brand_profile_tab(
    cur_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    present: list[str],
    tail: int,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    billing_df: pd.DataFrame | None = None,
    water_df: pd.DataFrame | None = None,
    hotwater_df: pd.DataFrame | None = None,
    electricity_df: pd.DataFrame | None = None,
) -> None:
    """Render brand profile with sub-tabs: single profile + comparison."""
    brands = sorted(cur_df["brand"].dropna().astype(str).unique().tolist())
    if not brands:
        st.info("표시할 브랜드가 없습니다.")
        return

    tab_single, tab_compare = st.tabs(["🔍 브랜드 프로필", "⚖️ 브랜드 비교"])

    with tab_compare:
        _render_brand_comparison(
            cur_df, present, billing_period, prev_billing_period,
            billing_df, water_df, hotwater_df, electricity_df, brands,
        )

    with tab_single:
        _render_single_brand_profile(
            cur_df, ref_df, present, tail, billing_period,
            prev_billing_period, billing_df, water_df, hotwater_df,
            electricity_df, brands,
        )


def _render_single_brand_profile(
    cur_df, ref_df, present, tail, billing_period, prev_billing_period,
    billing_df, water_df, hotwater_df, electricity_df, brands,
):
    """Single-brand deep-dive (original profile view)."""
    _search = st.session_state.get("profile_sel_search", "").strip().lower()
    brands_shown = [b for b in brands if _search in b.lower()] if _search else brands
    if not brands_shown:
        brands_shown = brands

    st.text_input("🔍 브랜드 검색", placeholder="브랜드명 입력...",
                  key="profile_sel_search")
    selected = st.selectbox("브랜드 선택", brands_shown, index=0,
                            key="profile_brand_select")

    row = cur_df[cur_df["brand"].astype(str) == selected]
    if row.empty:
        st.warning("해당 브랜드 데이터를 찾을 수 없습니다.")
        return
    row = row.iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    bldg  = str(row.get("building", "—"))
    floor = str(row.get("floor",    "—"))
    m2    = _v(row, "size_m2")
    py    = _v(row, "size_py")
    size_str = (
        f"{m2:,.1f} m²  /  {py:,.1f} 평" if not pd.isna(m2) and not pd.isna(py)
        else f"{m2:,.1f} m²" if not pd.isna(m2) else "—"
    )
    period_str = (
        f"{prev_billing_period} → {billing_period}"
        if prev_billing_period and billing_period else billing_period or "—"
    )

    # Brand header card
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#4C72B010,#4C72B005);'
        f'border:1px solid #4C72B025;border-radius:12px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="font-size:1.5rem;font-weight:800;color:#333;margin-bottom:8px">'
        f'{selected}</div>'
        f'<div style="display:flex;gap:24px;font-size:0.88rem;color:#555">'
        f'<span>🏢 <b>{bldg}</b></span>'
        f'<span>📍 <b>{floor}</b></span>'
        f'<span>📐 <b>{size_str}</b></span>'
        f'<span>📅 <b>{period_str}</b></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Debug: brand name matching across sheets
    _sheet_dfs = {
        "수도광열비": billing_df,
        "수도": water_df,
        "온수": hotwater_df,
        "전기": electricity_df,
    }
    _missing = [
        name for name, df in _sheet_dfs.items()
        if df is not None and not df.empty and _filter_brand(df, selected).empty
    ]
    if _missing:
        with st.expander("⚠️ 일부 시트에서 브랜드를 찾을 수 없음"):
            st.caption(f"선택된 브랜드: **{selected}**")
            for name, df in _sheet_dfs.items():
                if df is not None and not df.empty and name in _missing:
                    similar = [b for b in df["brand"].astype(str).str.strip().unique()
                               if selected.strip().lower()[:3] in b.lower()]
                    st.caption(f"**{name}** 시트 유사 브랜드: {similar[:10] or '없음'}")

    # ── Compute quadrant thresholds ───────────────────────────────────────────
    thresholds: dict[str, tuple] = {}
    for p in present:
        cc, pc = f"{p}_change", f"{p}_pct"
        ca = to_numeric_series(cur_df[cc]).dropna()
        pa = to_numeric_series(cur_df[pc]).dropna()
        if not ca.empty and not pa.empty:
            thresholds[p] = (
                ca.quantile(tail / 100), ca.quantile(1 - tail / 100),
                pa.quantile(tail / 100), pa.quantile(1 - tail / 100),
            )

    # ── 검침내역: usage summary ───────────────────────────────────────────────
    st.subheader("📊 사용량 요약 (검침내역)")
    usage_rows = []
    for p in present:
        cv_curr = _v(row, f"{p}_current")
        cv_prev = _v(row, f"{p}_previous")
        cv_chg  = _v(row, f"{p}_change")
        cv_pct  = _v(row, f"{p}_pct")
        cv_pm2  = _v(row, f"{p}_usage_per_m2")
        cv_ppy  = _v(row, f"{p}_usage_per_py")

        lo_c, hi_c, lo_p, hi_p = thresholds.get(p, (np.nan,) * 4)
        q_label = _QUAD_LABEL.get(_quad(cv_chg, cv_pct, lo_c, hi_c, lo_p, hi_p), "—")
        u = _UNIT.get(p, "")

        usage_rows.append({
            "항목":             _LABEL.get(p, p),
            f"이번달 ({u})":    _fmt(cv_curr),
            f"전달 ({u})":      _fmt(cv_prev),
            "변화량":           _fmt(cv_chg, sign=True),
            "변화율 (%)":       _fmt(cv_pct, sign=True),
            "m²당":             _fmt(cv_pm2, 4),
            "평당":             _fmt(cv_ppy, 4),
            "분류":             q_label,
        })
    if usage_rows:
        st.dataframe(pd.DataFrame(usage_rows), hide_index=True, use_container_width=True)

    # ── Radar chart: brand vs peers (percentile) ─────────────────────────────
    _radar_cats, _radar_brand, _radar_peer = [], [], []
    for p in present:
        col_c = f"{p}_current"
        if col_c not in cur_df.columns:
            continue
        s = to_numeric_series(cur_df[col_c]).dropna()
        bv = _v(row, col_c)
        if s.empty or pd.isna(bv):
            continue
        pctile = float((s < bv).sum()) / len(s) * 100
        _radar_cats.append(_LABEL.get(p, p))
        _radar_brand.append(round(pctile, 1))
        _radar_peer.append(50.0)
    if len(_radar_cats) >= 3:
        _rc1, _rc2 = st.columns([2, 1])
        with _rc1:
            _radar_cats_loop = _radar_cats + [_radar_cats[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=_radar_brand + [_radar_brand[0]],
                theta=_radar_cats_loop,
                name=selected[:20],
                fill="toself",
                fillcolor="rgba(76,114,176,0.15)",
                line=dict(color=_C_CURR, width=2.5),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=_radar_peer + [_radar_peer[0]],
                theta=_radar_cats_loop,
                name="중위 (50%)",
                line=dict(color="#999", dash="dot", width=1.5),
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100],
                                    ticktext=["25%", "50%", "75%", "100%"],
                                    gridcolor="rgba(0,0,0,0.08)"),
                    angularaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
                ),
                title=dict(text="사용량 백분위 — 동종 대비 위치", font_size=13),
                height=320, margin=dict(t=50, b=20, l=60, r=60),
                legend=dict(orientation="h", y=-0.05),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="profile_radar")
        with _rc2:
            for cat, pctl in zip(_radar_cats, _radar_brand):
                _clr = "#C44E52" if pctl >= 80 else "#DD8A00" if pctl >= 60 else "#2ca02c"
                st.markdown(
                    f'<div style="margin:6px 0;padding:6px 10px;border-left:3px solid {_clr};'
                    f'border-radius:4px;font-size:0.85rem">'
                    f'<b>{cat}</b> <span style="color:{_clr};font-weight:700">'
                    f'상위 {100-pctl:.0f}%</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Peer ranking ──────────────────────────────────────────────────────────
    peer_lines = []
    for p in present:
        s = to_numeric_series(cur_df[f"{p}_change"]).dropna().sort_values()
        cv = _v(row, f"{p}_change")
        if s.empty or pd.isna(cv): continue
        rank = int((s < cv).sum()) + 1
        pctile = round(rank / len(s) * 100, 1)
        peer_lines.append(
            f"**{_LABEL.get(p,p)}** 변화량 {cv:+,.2f} {_UNIT.get(p,'')} — "
            f"전체 {len(s)}개 중 **{rank}위** (상위 {100-pctile:.1f}%)"
        )
    if peer_lines:
        st.caption("  \n".join(peer_lines))

    st.divider()
    _render_comparison_charts(row, cur_df, present, selected)
    st.divider()

    # ── 수도광열비: billing summary ───────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        st.subheader("💰 수도광열비 부과 내역")
        bdf = _filter_brand(billing_df, selected)
        if bdf.empty:
            st.caption("해당 브랜드의 수도광열비 내역을 찾을 수 없습니다.")
        else:
            _id_cols     = ["building", "floor", "unit", "size_m2"]
            _bill_detail = [
                "total", "total_excl", "total_comm",
                "water_excl", "water_comm", "water_total",
                "elect_excl", "elect_comm", "elect_total",
                "hotwater_excl", "hotwater_comm",
                "hvac_excl", "hvac_comm",
                "heat_total",
            ]
            _bill_cols = [c for c in _id_cols + _bill_detail if c in bdf.columns]
            _peer_chart(billing_df, selected, [
                ("total",       "총합계"),
                ("water_total", "상하수도"),
                ("elect_total", "전기"),
                ("heat_total",  "열"),
            ], "수도광열비 — 동종 평균 대비 (만원)", "만원")
            st.dataframe(st_safe(bdf[_bill_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 수도: detailed water ──────────────────────────────────────────────────
    if water_df is not None and not water_df.empty:
        st.subheader("💧 수도 사용 내역")
        wdf = _filter_brand(water_df, selected)
        if wdf.empty:
            st.caption("해당 브랜드의 수도 내역을 찾을 수 없습니다.")
        else:
            _w_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _w_data = [
                "usage_m3", "pipe_fee_comm",
                "water_excl", "water_comm",
                "sewage_excl", "sewage_comm",
                "levy_excl", "levy_comm",
                "total_excl", "total_comm", "total",
                "avg_unit_price",
            ]
            _w_cols = [c for c in _w_id + _w_data if c in wdf.columns]
            _peer_chart(water_df, selected, [
                ("usage_m3", "사용량 (m³)"),
                ("total",    "총액"),
            ], "수도 — 동종 평균 대비")
            st.dataframe(st_safe(wdf[_w_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 온수: detailed hot water ──────────────────────────────────────────────
    if hotwater_df is not None and not hotwater_df.empty:
        st.subheader("🌡️ 온수 사용 내역")
        hwdf = _filter_brand(hotwater_df, selected)
        if hwdf.empty:
            st.caption("해당 브랜드의 온수 내역을 찾을 수 없습니다.")
        else:
            _hw_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _hw_data = ["usage_m3", "fee_excl", "fee_comm", "total"]
            _hw_cols = [c for c in _hw_id + _hw_data if c in hwdf.columns]
            _peer_chart(hotwater_df, selected, [
                ("usage_m3", "사용량 (m³)"),
                ("total",    "총액"),
            ], "온수 — 동종 평균 대비")
            st.dataframe(st_safe(hwdf[_hw_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 전기: detailed electricity ────────────────────────────────────────────
    if electricity_df is not None and not electricity_df.empty:
        st.subheader("⚡ 전기 사용 내역")
        edf = _filter_brand(electricity_df, selected)
        if edf.empty:
            st.caption("해당 브랜드의 전기 내역을 찾을 수 없습니다.")
        else:
            _e_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _e_kwh  = ["kwh_elec01", "kwh_elec02", "kwh_fcu", "kwh_ehp", "kwh_pump", "kwh_total"]
            _e_fee  = ["excl_total", "ehp_total", "comm_total", "grand_total"]
            _e_cols = [c for c in _e_id + _e_kwh + _e_fee if c in edf.columns]
            _peer_chart(electricity_df, selected, [
                ("kwh_total",   "총 사용량 (kWh)"),
                ("grand_total", "총 요금"),
            ], "전기 — 동종 평균 대비")
            st.dataframe(st_safe(edf[_e_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── Raw meter readings (pre-aggregation) ─────────────────────────────────
    st.subheader("📋 원본 검침 데이터 (층/호별)")
    raw_brand = _filter_brand(ref_df, selected) if ref_df is not None else pd.DataFrame()
    if raw_brand.empty:
        st.info("원본 데이터를 찾을 수 없습니다.")
    else:
        _id   = [c for c in ["building", "floor", "unit", "size_m2", "size_py"] if c in raw_brand.columns]
        _mtr  = [c for c in raw_brand.columns if "_meter_" in c]
        _usg  = [c for c in raw_brand.columns if "_current" in c or "_previous" in c]
        _chg  = [c for c in raw_brand.columns if "_change" in c or "_pct" in c]
        _rest = [c for c in raw_brand.columns
                 if c not in _id + _mtr + _usg + _chg + ["brand"]]
        ordered = [c for c in _id + _mtr + _usg + _chg + _rest if c in raw_brand.columns]
        raw_brand = raw_brand[ordered].reset_index(drop=True)
        raw_brand.insert(0, "No", range(1, len(raw_brand) + 1))
        st.dataframe(st_safe(raw_brand), hide_index=True, use_container_width=True)


# ── Brand Comparison ─────────────────────────────────────────────────────────

_COMPARE_COLORS = [
    "#4C72B0", "#DD8A00", "#C44E52", "#2CA02C", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _render_brand_comparison(
    cur_df, present, billing_period, prev_billing_period,
    billing_df, water_df, hotwater_df, electricity_df, brands,
):
    """Side-by-side comparison of 2+ brands."""
    selected = st.multiselect(
        "비교할 브랜드 선택 (2개 이상)", brands,
        default=brands[:2] if len(brands) >= 2 else brands,
        key="compare_brands",
    )
    if len(selected) < 2:
        st.info("비교하려면 2개 이상의 브랜드를 선택하세요.")
        return

    rows = cur_df[cur_df["brand"].astype(str).isin(selected)].copy()
    if rows.empty:
        st.warning("선택한 브랜드의 데이터를 찾을 수 없습니다.")
        return

    # ── 1. Overview table ─────────────────────────────────────────────────────
    st.subheader("📋 기본 정보")
    info_rows = []
    for b in selected:
        r = rows[rows["brand"].astype(str) == b]
        if r.empty:
            continue
        r = r.iloc[0]
        m2 = _v(r, "size_m2")
        info_rows.append({
            "브랜드": b,
            "건물": str(r.get("building", "—")),
            "층": str(r.get("floor", "—")),
            "면적(㎡)": round(m2, 1) if not pd.isna(m2) else None,
        })
    if info_rows:
        st.dataframe(pd.DataFrame(info_rows), hide_index=True, use_container_width=True)

    # ── 2. Usage comparison (검침내역) ────────────────────────────────────────
    st.subheader("📊 사용량 비교 (검침내역)")
    for p in present:
        lbl = _LABEL.get(p, p)
        unit = _UNIT.get(p, "")
        curr_col = f"{p}_current"
        chg_col = f"{p}_change"
        pct_col = f"{p}_pct"
        pm2_col = f"{p}_usage_per_m2"

        comp_rows = []
        for b in selected:
            r = rows[rows["brand"].astype(str) == b]
            if r.empty:
                continue
            r = r.iloc[0]
            comp_rows.append({
                "브랜드": b,
                f"이번달 ({unit})": _v(r, curr_col),
                "변화량": _v(r, chg_col),
                "변화율(%)": _v(r, pct_col),
                "m²당": _v(r, pm2_col),
            })
        if not comp_rows:
            continue
        cdf = pd.DataFrame(comp_rows)
        st.markdown(f"**{lbl}**")
        st.dataframe(
            cdf, hide_index=True, use_container_width=True,
            column_config={
                f"이번달 ({unit})": st.column_config.NumberColumn(format="%,.2f"),
                "변화량": st.column_config.NumberColumn(format="%+,.2f"),
                "변화율(%)": st.column_config.NumberColumn(format="%+,.2f"),
                "m²당": st.column_config.NumberColumn(format="%,.4f"),
            },
        )

    # ── 3. Grouped bar charts ─────────────────────────────────────────────────
    st.subheader("📊 이번달 사용량 비교")
    _usage_cols = [(p, f"{p}_current", _LABEL.get(p, p), _UNIT.get(p, ""))
                   for p in present if f"{p}_current" in rows.columns]
    if _usage_cols:
        n_cats = len(_usage_cols)
        fig_u = make_subplots(rows=1, cols=n_cats,
                              subplot_titles=[lbl for _, _, lbl, _ in _usage_cols])
        for ci, (_, col, lbl, unit) in enumerate(_usage_cols, 1):
            for bi, b in enumerate(selected):
                r = rows[rows["brand"].astype(str) == b]
                val = float(_v(r.iloc[0], col)) if not r.empty else 0.0
                clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
                fig_u.add_trace(go.Bar(
                    name=b, x=[b], y=[val if not pd.isna(val) else 0],
                    marker_color=clr,
                    text=[f"{val:,.1f}" if not pd.isna(val) else ""],
                    textposition="outside", cliponaxis=False,
                    legendgroup=b, showlegend=(ci == 1),
                ), row=1, col=ci)
            fig_u.update_yaxes(title_text=unit, row=1, col=ci)
        fig_u.update_layout(
            barmode="group", height=350,
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(t=70, b=20, l=30, r=10),
        )
        fig_u.update_yaxes(zeroline=True, rangemode="tozero")
        _ev = st.plotly_chart(fig_u, use_container_width=True, on_select="rerun",
                              key="cmp_usage_bar")
        handle_chart_click(_ev, rows, brand_col="brand", field="x")

    # ── 4. MoM change comparison ──────────────────────────────────────────────
    st.subheader("📉 전월 대비 변화량 비교")
    _chg_cols = [(p, f"{p}_change", _LABEL.get(p, p))
                 for p in present if f"{p}_change" in rows.columns]
    if _chg_cols:
        cat_labels = [lbl for _, _, lbl in _chg_cols]
        fig_c = go.Figure()
        for bi, b in enumerate(selected):
            r = rows[rows["brand"].astype(str) == b]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = [float(_v(r, col)) if not pd.isna(_v(r, col)) else 0
                    for _, col, _ in _chg_cols]
            clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
            fig_c.add_trace(go.Bar(
                name=b, x=cat_labels, y=vals,
                marker_color=clr,
                text=[f"{v:+,.2f}" for v in vals],
                textposition="outside", cliponaxis=False,
            ))
        fig_c.update_layout(
            barmode="group", height=350,
            yaxis=dict(zeroline=True, zerolinewidth=1.5),
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(t=40, b=20, l=40, r=10),
        )
        _ev = st.plotly_chart(fig_c, use_container_width=True, on_select="rerun",
                              key="cmp_mom_bar")
        handle_chart_click(_ev, rows, brand_col="brand", field="x")

    # ── 5. Billing comparison ─────────────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        st.subheader("💰 수도광열비 비교")
        _bill_metrics = [
            ("total", "총합계"), ("water_total", "상하수도"),
            ("elect_total", "전기"), ("heat_total", "열"),
        ]
        _avail = [(c, l) for c, l in _bill_metrics if c in billing_df.columns]
        if _avail:
            bill_rows = []
            for b in selected:
                bdf = _filter_brand(billing_df, b)
                if bdf.empty:
                    continue
                row_data = {"브랜드": b}
                for col, lbl in _avail:
                    row_data[lbl] = to_numeric_series(bdf[col]).sum()
                bill_rows.append(row_data)
            if bill_rows:
                bdf_cmp = pd.DataFrame(bill_rows)
                st.dataframe(bdf_cmp, hide_index=True, use_container_width=True,
                             column_config={l: st.column_config.NumberColumn(f"{l} (만원)", format="%,.2f")
                                            for _, l in _avail})

                fig_b = go.Figure()
                for bi, b in enumerate(selected):
                    br = next((r for r in bill_rows if r["브랜드"] == b), None)
                    if not br:
                        continue
                    vals = [br.get(l, 0) for _, l in _avail]
                    clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
                    fig_b.add_trace(go.Bar(
                        name=b, x=[l for _, l in _avail], y=vals,
                        marker_color=clr,
                        text=[f"{v:,.1f}" for v in vals],
                        textposition="outside", cliponaxis=False,
                    ))
                fig_b.update_layout(
                    barmode="group", height=350,
                    yaxis=dict(title="만원", zeroline=True, rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, x=0),
                    margin=dict(t=40, b=20, l=40, r=10),
                )
                _ev = st.plotly_chart(fig_b, use_container_width=True, on_select="rerun",
                                      key="cmp_bill_bar")
                handle_chart_click(_ev, billing_df, brand_col="brand", field="x")

    # ── 6. Radar chart (normalized) ───────────────────────────────────────────
    st.subheader("🕸️ 레이더 비교")
    radar_cols = [(f"{p}_current", _LABEL.get(p, p)) for p in present
                  if f"{p}_current" in rows.columns]
    if len(radar_cols) >= 3:
        fig_r = go.Figure()
        cats = [lbl for _, lbl in radar_cols]
        # Normalize each metric 0–1 across selected brands
        raw_vals = {}
        for b in selected:
            r = rows[rows["brand"].astype(str) == b]
            if r.empty:
                continue
            raw_vals[b] = [float(_v(r.iloc[0], c)) if not pd.isna(_v(r.iloc[0], c)) else 0
                           for c, _ in radar_cols]
        if raw_vals:
            all_v = np.array(list(raw_vals.values()))
            maxes = all_v.max(axis=0)
            maxes[maxes == 0] = 1  # avoid div-by-zero
            for bi, (b, vals) in enumerate(raw_vals.items()):
                normed = [v / m for v, m in zip(vals, maxes)]
                clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
                fig_r.add_trace(go.Scatterpolar(
                    r=normed + [normed[0]],  # close the polygon
                    theta=cats + [cats[0]],
                    name=b, fill="toself", opacity=0.3,
                    line=dict(color=clr, width=2),
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
                height=420,
                legend=dict(orientation="h", y=-0.1, x=0),
                margin=dict(t=30, b=60, l=40, r=40),
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.caption("각 항목을 브랜드 간 최댓값 기준으로 0–1 정규화한 레이더 차트입니다.")
    elif radar_cols:
        st.caption("레이더 차트는 3개 이상의 유틸리티 항목이 필요합니다.")
