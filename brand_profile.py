"""brand_profile.py — Comprehensive single-brand profile across all data sources."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import to_numeric_series, st_safe

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
    st.plotly_chart(fig, use_container_width=True)


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
            st.plotly_chart(
                _bar(
                    ["이번달", "동종 평균"],
                    [curr, avg_curr],
                    [_C_CURR, _C_AVG],
                    lbl, unit,
                ),
                use_container_width=True,
            )

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
        st.plotly_chart(fig, use_container_width=True)


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
    """Render a comprehensive profile for a user-selected brand."""
    brands = sorted(cur_df["brand"].dropna().astype(str).unique().tolist())
    if not brands:
        st.info("표시할 브랜드가 없습니다.")
        return

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

    st.markdown(f"## {selected}")

    # ── Debug: brand name matching across sheets ──────────────────────────────
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
        with st.expander("⚠️ 일부 시트에서 브랜드를 찾을 수 없음 (클릭하여 확인)"):
            st.caption(f"선택된 브랜드: **{selected}**")
            for name, df in _sheet_dfs.items():
                if df is not None and not df.empty and name in _missing:
                    similar = [b for b in df["brand"].astype(str).str.strip().unique()
                               if selected.strip().lower()[:3] in b.lower()]
                    st.caption(f"**{name}** 시트 유사 브랜드: {similar[:10] or '없음'}")

    hc = st.columns(4)
    hc[0].metric("건물", bldg)
    hc[1].metric("층", floor)
    hc[2].metric("면적", size_str if size_str != "—" else "정보 없음")
    hc[3].metric("기준월", period_str)
    st.divider()

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
