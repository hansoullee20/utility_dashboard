"""summary.py — Cross-sheet utility summary: water + hotwater + electricity + heating."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from utils import BLD_COLOR as _BLD_COLOR, iqr_upper as _iqr_upper
from viz import plot_hist_with_tails as _plot_hist


from utils import fmt_won as _fmt_won
from utils_plot import handle_chart_click as _handle_chart_click
from cross_features import build_elec_breakdown, build_water_breakdown
from tab_cross import _render_elec_breakdown, _render_water_breakdown


def _sum_cols(df: pd.DataFrame, cols: list) -> pd.Series:
    """Sum only the columns that exist in df; return zeros if none present."""
    present = [c for c in cols if c in df.columns]
    return df[present].sum(axis=1) if present else pd.Series(0, index=df.index)


def _iqr_whiskers(s: pd.Series) -> tuple[float, float]:
    """Return (lower_whisker, upper_whisker) based on 1.5×IQR."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _synced_slider_input(prefix: str, label: str, mn: int, mx: int, default: int, step: int) -> int:
    """Render a slider + number_input pair that stay in sync via session_state.

    Returns the current integer value. Keys used: ``{prefix}`` (slider) and
    ``{prefix}_i`` (number input). Both must be pre-initialised in session_state
    before calling this function.
    """
    key_s, key_i = prefix, f"{prefix}_i"

    def _sync_from_slider(): st.session_state[key_i] = st.session_state[key_s]
    def _sync_from_input():  st.session_state[key_s] = st.session_state[key_i]

    col_s, col_i = st.columns([3, 1])
    with col_s:
        st.slider(label, mn, mx, value=st.session_state[key_s], step=step,
                  key=key_s, on_change=_sync_from_slider)
    with col_i:
        st.number_input(label, mn, mx, value=st.session_state[key_i], step=step,
                        key=key_i, label_visibility="hidden", on_change=_sync_from_input)
    return int(st.session_state.get(key_s, default))


def _init_session_keys(pairs: list[tuple[str, int]]) -> None:
    """Set session_state defaults only when keys are not yet present."""
    for key, default in pairs:
        if key not in st.session_state:
            st.session_state[key] = default


def _leakage_for(source_df, usage_col, fee_col):
    met = source_df[source_df[usage_col] > 0]
    if len(met) < 2: return {}, 0.0
    med_rate = (met[fee_col] / met["size_m2"].replace(0, np.nan)).median()
    if not pd.notna(med_rate): return {}, 0.0
    unmet = source_df[source_df[usage_col] == 0]
    per_brand = {}
    for _, r in unmet.iterrows():
        per_brand[r["brand"]] = per_brand.get(r["brand"], 0) + float(r["size_m2"]) * med_rate
    total = sum(per_brand.values())
    return per_brand, total


def render_summary_view(
    water_df: pd.DataFrame | None,
    hotwater_df: pd.DataFrame | None,
    elec_df: pd.DataFrame | None,
    split_by_building: bool = True,
    prev_water_df: pd.DataFrame | None = None,
    prev_hotwater_df: pd.DataFrame | None = None,
    prev_elec_df: pd.DataFrame | None = None,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    yoy_water_df: pd.DataFrame | None = None,
    yoy_hotwater_df: pd.DataFrame | None = None,
    yoy_elec_df: pd.DataFrame | None = None,
    yoy_billing_period: str | None = None,
    billing_df: pd.DataFrame | None = None,
    prev_billing_df: pd.DataFrame | None = None,
    yoy_billing_df: pd.DataFrame | None = None,
    meter_df: pd.DataFrame | None = None,
) -> None:
    _available      = [n for n, d in [("수도", water_df), ("온수", hotwater_df), ("전기", elec_df), ("난방", billing_df)] if d is not None]
    _prev_available = [n for n, d in [("수도", prev_water_df), ("온수", prev_hotwater_df), ("전기", prev_elec_df), ("난방", prev_billing_df)] if d is not None]
    st.header("📊 통합 유틸리티 분석")
    _cap = f"{'·'.join(_available)} 데이터를 브랜드 기준으로 통합한 종합 분석입니다."
    if _prev_available:
        _cap += f"  |  📈 전월 비교 가능: {'·'.join(_prev_available)}"
    else:
        _cap += "  |  전월 파일 없음 — 단일 월 분석"
    st.caption(_cap)

    # ── Aggregate each available sheet by (brand, building) ─────────────────────
    def _gk(df):
        return ["brand", "building"] if "building" in df.columns else ["brand"]

    def _agg_sheet(df, agg_dict):
        if df is None or df.empty:
            return None
        gk = _gk(df)
        return df.groupby(gk).agg(**agg_dict).reset_index()

    _JOIN = ["brand", "building"]

    _w_agg = dict(floor=("floor", "first"), size_m2=("size_m2", "sum"),
                  water_total=("total", "sum"))
    if water_df is not None and "usage_m3" in (water_df.columns if water_df is not None else []):
        _w_agg["water_m3"] = ("usage_m3", "sum")
    _w = _agg_sheet(water_df, _w_agg)
    if _w is None:
        _w = pd.DataFrame(columns=["brand", "building", "floor", "size_m2", "water_total"])

    _hw_agg = dict(hw_total=("total", "sum"))
    if hotwater_df is not None and "usage_m3" in (hotwater_df.columns if hotwater_df is not None else []):
        _hw_agg["hw_m3"] = ("usage_m3", "sum")
    _hw = _agg_sheet(hotwater_df, _hw_agg)
    if _hw is None:
        _hw = pd.DataFrame(columns=_JOIN + ["hw_total"])

    _el_agg = dict(elec_total=("grand_total", "sum"), kwh_total=("kwh_total", "sum"))
    if elec_df is not None:
        if "kwh_ehp" in elec_df.columns:
            _el_agg["kwh_ehp"] = ("kwh_ehp", "sum")
        if "kwh_fcu" in elec_df.columns:
            _el_agg["kwh_fcu"] = ("kwh_fcu", "sum")
    _el = _agg_sheet(elec_df, _el_agg)
    if _el is not None:
        _el["kwh_hvac"] = _el.get("kwh_ehp", 0) + _el.get("kwh_fcu", 0)
        _el = _el.drop(columns=["kwh_ehp", "kwh_fcu"], errors="ignore")
    else:
        _el = pd.DataFrame(columns=_JOIN + ["elec_total", "kwh_total", "kwh_hvac"])

    if billing_df is not None and not billing_df.empty and "heat_total" in billing_df.columns:
        _ht = _agg_sheet(billing_df, dict(heat_total=("heat_total", "sum")))
        _ht["heat_total"] = _ht["heat_total"] * 10000
    else:
        _ht = pd.DataFrame(columns=_JOIN + ["heat_total"])

    if meter_df is not None and not meter_df.empty and "heat_current" in meter_df.columns:
        from data import to_numeric_series as _tns
        _hm = meter_df.copy()
        _hm["heat_m3"] = _tns(_hm["heat_current"])
        _hm = _agg_sheet(_hm, dict(heat_m3=("heat_m3", "sum")))
    else:
        _hm = pd.DataFrame(columns=_JOIN + ["heat_m3"])

    # Ensure all frames have building column for join
    for _df in [_w, _hw, _el, _ht, _hm]:
        if "building" not in _df.columns:
            _df["building"] = ""

    merged = (_w.merge(_hw, on=_JOIN, how="outer")
                .merge(_el, on=_JOIN, how="outer")
                .merge(_ht, on=_JOIN, how="outer")
                .merge(_hm, on=_JOIN, how="outer"))

    for col in ["water_total", "hw_total", "elec_total", "heat_total", "kwh_total", "kwh_hvac", "size_m2",
                "water_m3", "hw_m3", "heat_m3"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0

    # Recover brand_raw from source data for display
    _raw_parts = [
        df.groupby(_gk(df))["brand_raw"].first()
        for df in [water_df, hotwater_df, elec_df, billing_df, meter_df]
        if df is not None and not df.empty and "brand_raw" in df.columns
    ]
    if _raw_parts:
        _raw_map = pd.concat(_raw_parts)
        if isinstance(_raw_map.index, pd.MultiIndex):
            _raw_map = _raw_map.groupby(level=list(range(_raw_map.index.nlevels))).first()
            _raw_lookup = _raw_map.reset_index()
            _raw_lookup.columns = list(_raw_lookup.columns[:-1]) + ["brand_raw"]
            merged = merged.merge(_raw_lookup, on=_JOIN, how="left", suffixes=("", "_raw_dup"))
            merged.drop(columns=[c for c in merged.columns if c.endswith("_raw_dup")], inplace=True)
        else:
            _raw_map = _raw_map.groupby(level=0).first()
            merged["brand_raw"] = merged["brand"].map(_raw_map)

    # Fill missing floor/size from other sheets
    _meta_parts = [
        df.groupby(_gk(df))[["floor", "size_m2"]].first()
        for df in [water_df, hotwater_df, elec_df, billing_df]
        if df is not None and not df.empty
        and all(c in df.columns for c in ["floor", "size_m2"])
    ]
    if _meta_parts and "floor" in merged.columns:
        _meta = pd.concat(_meta_parts)
        if isinstance(_meta.index, pd.MultiIndex):
            _meta = _meta.groupby(level=list(range(_meta.index.nlevels))).first()
            _meta_r = _meta.reset_index()
            _meta_r.columns = _JOIN + ["_fill_floor", "_fill_size"]
            merged = merged.merge(_meta_r, on=_JOIN, how="left")
            _no_floor = merged["floor"].isna() | (merged["floor"].astype(str).str.strip() == "")
            merged["floor"] = merged["floor"].where(~_no_floor, merged.get("_fill_floor"))
            merged["size_m2"] = merged["size_m2"].where(merged["size_m2"] > 0, merged.get("_fill_size", 0))
            merged.drop(columns=["_fill_floor", "_fill_size"], errors="ignore", inplace=True)
        else:
            _meta = _meta.groupby(level=0).first()
            merged = merged.set_index("brand")
            _no_bld = merged["building"].isna() | (merged["building"].astype(str).str.strip() == "")
            merged["floor"] = merged["floor"].where(~_no_bld, _meta["floor"].reindex(merged.index))
            merged["size_m2"] = merged["size_m2"].where(
                merged["size_m2"] > 0, _meta["size_m2"].reindex(merged.index, fill_value=0))
            merged = merged.reset_index()

    _PY_FACTOR = 3.3058
    merged["size_py"] = (merged["size_m2"] / _PY_FACTOR).round(2)

    merged["util_total"] = merged["water_total"] + merged["hw_total"] + merged["elec_total"] + merged["heat_total"]
    merged = merged[merged["util_total"] > 0].sort_values("util_total", ascending=False).reset_index(drop=True)

    from utils import display_brand as _display_brand
    merged = _display_brand(merged)

    # ── Previous month aggregation & MoM change ────────────────────────────────
    _has_prev = any(d is not None and not d.empty
                    for d in [prev_water_df, prev_hotwater_df, prev_elec_df, prev_billing_df])

    if _has_prev:
        _prev_aggs = []
        if prev_water_df is not None and not prev_water_df.empty:
            _prev_aggs.append(("water_prev", prev_water_df, "total"))
        if prev_hotwater_df is not None and not prev_hotwater_df.empty:
            _prev_aggs.append(("hw_prev", prev_hotwater_df, "total"))
        if prev_elec_df is not None and not prev_elec_df.empty:
            _prev_aggs.append(("elec_prev", prev_elec_df, "grand_total"))
        if prev_billing_df is not None and not prev_billing_df.empty and "heat_total" in prev_billing_df.columns:
            _prev_aggs.append(("heat_prev", prev_billing_df, "heat_total"))

        for col_name, pdf, val_col in _prev_aggs:
            _pk = _gk(pdf)
            _prev_s = pdf.groupby(_pk)[val_col].sum()
            if col_name == "heat_prev":
                _prev_s = _prev_s * 10000
            if isinstance(_prev_s.index, pd.MultiIndex):
                _prev_r = _prev_s.reset_index()
                _prev_r.columns = _pk + [col_name]
                merged = merged.merge(_prev_r, on=_pk, how="left")
                merged[col_name] = merged[col_name].fillna(0)
            else:
                merged[col_name] = merged["brand"].map(_prev_s).fillna(0)

        for col in ["water_prev", "hw_prev", "elec_prev", "heat_prev"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged["util_prev"]    = merged["water_prev"] + merged["hw_prev"] + merged["elec_prev"] + merged["heat_prev"]
        merged["water_change"] = merged["water_total"] - merged["water_prev"]
        merged["hw_change"]    = merged["hw_total"]    - merged["hw_prev"]
        merged["elec_change"]  = merged["elec_total"]  - merged["elec_prev"]
        merged["heat_change"]  = merged["heat_total"]  - merged["heat_prev"]
        merged["util_change"]  = merged["util_total"]  - merged["util_prev"]

    # ── Top metrics ────────────────────────────────────────────────────────────
    _util_sum  = merged["util_total"].sum()
    _period_lbl = billing_period or ""
    _prev_lbl   = prev_billing_period or ""
    _period_cap = f"{_prev_lbl} → {_period_lbl}" if _period_lbl and _prev_lbl else _period_lbl

    if _period_cap:
        st.caption(f"기간: {_period_cap}")

    with st.container(border=True):
        mc = st.columns(6)
        mc[0].metric("통합 브랜드", f"{len(merged)}개")

        if _has_prev:
            _util_prev_sum = merged["util_prev"].sum()
            _util_delta    = _util_sum - _util_prev_sum
            _util_pct      = _util_delta / _util_prev_sum * 100 if _util_prev_sum else 0
            mc[1].metric("총 유틸리티 비용", _fmt_won(_util_sum),
                         delta=f"{_fmt_won(_util_delta)} ({_util_pct:+.1f}%)",
                         delta_color="inverse")
            for i, (col_curr, col_prev, label) in enumerate([
                ("water_total", "water_prev", "수도"),
                ("hw_total",    "hw_prev",    "온수"),
                ("elec_total",  "elec_prev",  "전기"),
                ("heat_total",  "heat_prev",  "난방"),
            ], start=2):
                _curr = merged[col_curr].sum()
                _prev = merged[col_prev].sum()
                _d = _curr - _prev
                _p = _d / _prev * 100 if _prev else 0
                mc[i].metric(
                    label,
                    f"{_fmt_won(_curr)} ({_curr/_util_sum*100:.0f}%)",
                    delta=f"{_fmt_won(_d)} ({_p:+.1f}%)",
                    delta_color="inverse",
                )
        else:
            mc[1].metric("총 유틸리티 비용", _fmt_won(_util_sum))
            mc[2].metric("수도", f"{_fmt_won(merged['water_total'].sum())} ({merged['water_total'].sum()/_util_sum*100:.0f}%)")
            mc[3].metric("온수", f"{_fmt_won(merged['hw_total'].sum())} ({merged['hw_total'].sum()/_util_sum*100:.0f}%)")
            mc[4].metric("전기", f"{_fmt_won(merged['elec_total'].sum())} ({merged['elec_total'].sum()/_util_sum*100:.0f}%)")
            mc[5].metric("난방", f"{_fmt_won(merged['heat_total'].sum())} ({merged['heat_total'].sum()/_util_sum*100:.0f}%)")

    # Mini composition bar
    _w_p = merged["water_total"].sum() / _util_sum * 100 if _util_sum else 0
    _hw_p = merged["hw_total"].sum() / _util_sum * 100 if _util_sum else 0
    _el_p = merged["elec_total"].sum() / _util_sum * 100 if _util_sum else 0
    _ht_p = merged["heat_total"].sum() / _util_sum * 100 if _util_sum else 0
    st.markdown(
        f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:4px 0 8px">'
        f'<div style="width:{_w_p}%;background:#4C72B0" title="수도 {_w_p:.0f}%"></div>'
        f'<div style="width:{_hw_p}%;background:#C44E52" title="온수 {_hw_p:.0f}%"></div>'
        f'<div style="width:{_el_p}%;background:#DD8A00" title="전기 {_el_p:.0f}%"></div>'
        f'<div style="width:{_ht_p}%;background:#E377C2" title="난방 {_ht_p:.0f}%"></div>'
        f'</div>'
        f'<div style="display:flex;gap:20px;font-size:0.92rem;font-weight:600;color:inherit;opacity:0.75;margin-bottom:6px">'
        f'<span style="color:#4C72B0">■ 수도 {_w_p:.0f}%</span>'
        f'<span style="color:#C44E52">■ 온수 {_hw_p:.0f}%</span>'
        f'<span style="color:#DD8A00">■ 전기 {_el_p:.0f}%</span>'
        f'<span style="color:#E377C2">■ 난방 {_ht_p:.0f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Business Insight Summary ────────────────────────────────────────────
    _insights: list[str] = []
    # Dominant utility
    _dom_name, _dom_pct = max(
        [("수도", _w_p), ("온수", _hw_p), ("전기", _el_p), ("난방", _ht_p)],
        key=lambda x: x[1],
    )
    _insights.append(
        f"전체 비용의 **{_dom_pct:.0f}%**가 {_dom_name}에 집중 → "
        f"{_dom_name} 절감이 전체 비용 절감에 가장 효과적입니다"
    )
    # Top spender
    if not merged.empty:
        _top_brand = merged.iloc[0]
        _top_pct = _top_brand["util_total"] / _util_sum * 100 if _util_sum else 0
        _insights.append(
            f"**{_top_brand['brand']}**가 전체의 {_top_pct:.1f}% 차지 "
            f"({_fmt_won(_top_brand['util_total'])}) → "
            f"이 업체의 사용 패턴을 우선 점검하면 비용 관리 효과가 큽니다"
        )
    # Concentration
    _n10 = max(1, len(merged) // 10)
    _top10_sum = merged.head(_n10)["util_total"].sum()
    _top10_pct = _top10_sum / _util_sum * 100 if _util_sum else 0
    if _top10_pct >= 50:
        _insights.append(
            f"상위 {_n10}개 업체가 전체 비용의 **{_top10_pct:.0f}%** 차지 → "
            f"소수 업체에 비용이 편중되어 있어, 해당 업체 집중 관리가 필요합니다"
        )
    else:
        _insights.append(
            f"상위 {_n10}개 업체가 전체의 **{_top10_pct:.0f}%** → "
            f"비용이 비교적 균등하게 분산되어 있어, 전체적인 절감 정책이 효과적입니다"
        )
    # MoM change
    if _has_prev:
        _util_prev_s = merged["util_prev"].sum()
        _chg_pct = (_util_sum - _util_prev_s) / _util_prev_s * 100 if _util_prev_s else 0
        if abs(_chg_pct) >= 5:
            _dir = "증가" if _chg_pct > 0 else "감소"
            _reason = "계절 요인 또는 특정 업체의 사용량 변동을 확인하세요" if _chg_pct > 0 else "절감 노력의 성과일 수 있으나, 공실 증가 여부도 확인 필요"
            _insights.append(
                f"전월 대비 **{abs(_chg_pct):.1f}% {_dir}** → {_reason}"
            )
        else:
            _insights.append(f"전월 대비 변동 {abs(_chg_pct):.1f}% → 안정적인 사용 패턴을 유지하고 있습니다")

    with st.container(border=True):
        st.markdown(
            '<p style="margin:0 0 6px;font-size:0.9rem;font-weight:700;color:#4C72B0">'
            '비즈니스 인사이트</p>',
            unsafe_allow_html=True,
        )
        st.markdown("  \n".join(f"- {i}" for i in _insights))

    _has_yoy = any(d is not None for d in [yoy_water_df, yoy_hotwater_df, yoy_elec_df, yoy_billing_df])

    _has_compare = _has_prev or _has_yoy
    _tab_labels = ["사용 분석", "유틸리티 구성"]
    if _has_compare:
        _tab_labels.insert(1, "📈 기간 비교")
    _tabs = st.tabs(_tab_labels)

    # Unpack tabs dynamically
    _ti = iter(_tabs)
    tab_rank = next(_ti)
    tab_compare = next(_ti) if _has_compare else None
    tab_mix = next(_ti)

    def _boxplot_with_labels(s: pd.Series, label_s: pd.Series,
                             x_title: str, key: str,
                             source_df: pd.DataFrame = None,
                             disp_cols: list = None):
        """Horizontal box plot — dots for outliers, labels only on extremes.
        Returns the plotly chart selection event."""
        _lo_w, _hi_w = _iqr_whiskers(s)

        _hi_mask = s >= _hi_w
        _lo_mask = s <= _lo_w

        fig = go.Figure()
        fig.add_trace(go.Box(
            x=s, name="", orientation="h", boxpoints=False,
            marker_color="#4C72B0",
            fillcolor="rgba(76,114,176,0.35)",
            line=dict(color="#4C72B0", width=2.5),
            width=0.4,
            hovertemplate="%{x:,.0f}<extra></extra>",
        ))

        for _mask, _clr, _name, _extreme_fn, _dot_y, _ay in [
            (_hi_mask, "#DD8A00", "상위 이상치", "idxmax",  0.70, -30),
            (_lo_mask, "#C44E52", "하위 이상치", "idxmin", -0.70,  30),
        ]:
            if not _mask.any():
                continue
            _sx, _sl = s[_mask], label_s[_mask].astype(str)
            fig.add_trace(go.Scatter(
                x=_sx, y=[_dot_y] * int(_mask.sum()),
                mode="markers", name=_name,
                showlegend=True, visible=True,
                marker=dict(color=_clr, size=9, opacity=0.9,
                            line=dict(color=_clr, width=1.5)),
                hovertemplate="<b>%{customdata}</b><br>%{x:,.0f}<extra></extra>",
                customdata=_sl.values,
            ))
            _extreme_idx = getattr(_sx, _extreme_fn)()
            fig.add_annotation(
                x=float(s[_extreme_idx]), y=_dot_y,
                text=str(label_s[_extreme_idx])[:16],
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowcolor=_clr, ax=0, ay=_ay,
                font=dict(size=10, color="white"),
                bgcolor=_clr, bordercolor=_clr, borderwidth=1,
            )

        _med = float(s.median())
        fig.add_vline(x=_med, line_color="#C44E52", line_dash="dash", line_width=2)
        fig.add_annotation(
            x=_med, y=0.75, yref="paper",
            text=f"중앙값 {_med:,.0f}",
            showarrow=False,
            font=dict(size=10, color="#C44E52"),
            xanchor="left", xshift=6,
        )

        fig.update_layout(
            xaxis_title=x_title, height=260,
            xaxis=dict(griddash="dot", linewidth=1),
            yaxis=dict(showticklabels=False, range=[-1.3, 1.3]),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=50),
        )
        ev = st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")
        if source_df is not None and ev and hasattr(ev, "selection") and ev.selection.points:
            _bname = ev.selection.points[0].get("customdata")
            if _bname:
                _bdf = source_df[source_df["brand"] == _bname]
                if not _bdf.empty:
                    st.caption(f"선택됨: **{_bname}**")
                    _dcols = [c for c in (disp_cols or []) if c in _bdf.columns]
                    if not _dcols:
                        _dcols = [c for c in ["brand", "building", "floor"] if c in _bdf.columns]
                    st.dataframe(_bdf[_dcols].reset_index(drop=True),
                                 hide_index=True, use_container_width=True)
        return ev

    # ═══════════════════════════ 기간 비교 (MoM + YoY) ═══════════════════════
    if tab_compare is not None:
        with tab_compare:
            # ── Mode selector ─────────────────────────────────────────────────
            _cmp_modes = []
            if _has_prev:
                _cmp_modes.append("📈 전월 대비")
            if _has_yoy:
                _cmp_modes.append("📅 전년 대비")
            _cmp_mode = st.radio("비교 기준", _cmp_modes, horizontal=True,
                                 key="summary_cmp_mode") if len(_cmp_modes) > 1 else _cmp_modes[0]

            if _cmp_mode == "📈 전월 대비":
                _period_str = f"{_prev_lbl} → {_period_lbl}" if _period_lbl and _prev_lbl else "전월 대비"
                st.subheader(f"📈 월별 유틸리티 비용 변화  ({_period_str})")

                _cmp_specs = [
                    ("전체",    "util_total",  "util_prev",  "util_change"),
                    ("💧 수도", "water_total", "water_prev", "water_change"),
                    ("🌡 온수", "hw_total",    "hw_prev",    "hw_change"),
                    ("⚡ 전기", "elec_total",  "elec_prev",  "elec_change"),
                    ("🔥 난방", "heat_total",  "heat_prev",  "heat_change"),
                ]
                _cmp_specs = [(lbl, cur, prv, chg) for lbl, cur, prv, chg in _cmp_specs
                              if cur in merged.columns and prv in merged.columns
                              and merged[prv].sum() > 0]
                _cmp_data = merged
                _key_pfx = "mom"
                _prev_col_label = "전월(원)"
                _cur_col_label = "이번달(원)"
                _chart_suffix = "전월 대비 변화 (만원)"

            else:  # 📅 전년 대비
                _yoy_lbl = yoy_billing_period or "전년"
                _yoy_period_str = f"{_yoy_lbl} → {_period_lbl}" if _period_lbl and _yoy_lbl else "전년 대비"
                st.subheader(f"📅 전년 동월 유틸리티 비용 변화  ({_yoy_period_str})")

                # Build YoY merged data
                _yoy_parts = []
                if water_df is not None and yoy_water_df is not None:
                    _yc = water_df.groupby("brand")[["total"]].sum().rename(columns={"total": "water_total"})
                    _yp = yoy_water_df.groupby("brand")[["total"]].sum().rename(columns={"total": "water_yoy"})
                    _yoy_parts.append((_yc, _yp, "water_total", "water_yoy", "water_yoy_chg", "💧 수도"))
                if hotwater_df is not None and yoy_hotwater_df is not None:
                    _yc = hotwater_df.groupby("brand")[["total"]].sum().rename(columns={"total": "hw_total"})
                    _yp = yoy_hotwater_df.groupby("brand")[["total"]].sum().rename(columns={"total": "hw_yoy"})
                    _yoy_parts.append((_yc, _yp, "hw_total", "hw_yoy", "hw_yoy_chg", "🌡 온수"))
                if elec_df is not None and yoy_elec_df is not None:
                    _yc = elec_df.groupby("brand")[["grand_total"]].sum().rename(columns={"grand_total": "elec_total"})
                    _yp = yoy_elec_df.groupby("brand")[["grand_total"]].sum().rename(columns={"grand_total": "elec_yoy"})
                    _yoy_parts.append((_yc, _yp, "elec_total", "elec_yoy", "elec_yoy_chg", "⚡ 전기"))
                if billing_df is not None and yoy_billing_df is not None and "heat_total" in billing_df.columns and "heat_total" in yoy_billing_df.columns:
                    _yc = (billing_df.groupby("brand")[["heat_total"]].sum() * 10000).rename(columns={"heat_total": "heat_total"})
                    _yp = (yoy_billing_df.groupby("brand")[["heat_total"]].sum() * 10000).rename(columns={"heat_total": "heat_yoy"})
                    _yoy_parts.append((_yc, _yp, "heat_total", "heat_yoy", "heat_yoy_chg", "🔥 난방"))

                if not _yoy_parts:
                    st.info("전년 동월 데이터가 없습니다.")
                    _cmp_specs = []
                    _cmp_data = pd.DataFrame()
                else:
                    _yoy_merged = None
                    for _yc, _yp, _cur_col, _prv_col, _chg_col, _ in _yoy_parts:
                        _m = _yc.merge(_yp, left_index=True, right_index=True, how="outer").fillna(0)
                        _m[_chg_col] = _m[_cur_col] - _m[_prv_col]
                        if _yoy_merged is None:
                            _yoy_merged = _m
                        else:
                            _yoy_merged = _yoy_merged.merge(_m, left_index=True, right_index=True, how="outer").fillna(0)
                    _cmp_data = _yoy_merged.reset_index()
                    # Add "전체" total columns
                    _yoy_cur_cols = [c for _, _, c, _, _, _ in _yoy_parts]
                    _yoy_prv_cols = [c for _, _, _, c, _, _ in _yoy_parts]
                    _cmp_data["util_yoy_total"] = _cmp_data[[c for c in _yoy_cur_cols if c in _cmp_data.columns]].sum(axis=1)
                    _cmp_data["util_yoy_prev"]  = _cmp_data[[c for c in _yoy_prv_cols if c in _cmp_data.columns]].sum(axis=1)
                    _cmp_data["util_yoy_chg"]   = _cmp_data["util_yoy_total"] - _cmp_data["util_yoy_prev"]
                    _cmp_specs = [("전체", "util_yoy_total", "util_yoy_prev", "util_yoy_chg")]
                    _cmp_specs += [(_lbl, _cur, _prv, _chg)
                                   for _, _, _cur, _prv, _chg, _lbl in _yoy_parts]
                _key_pfx = "yoy_s"
                _prev_col_label = "전년(원)"
                _cur_col_label = "올해(원)"
                _chart_suffix = "전년 동월 대비 변화 (만원)"

            # ── Shared rendering: KPIs + chart + tables ───────────────────────
            if _cmp_specs and not _cmp_data.empty:
                # Total current sum for composition %
                _cmp_total = sum(_cmp_data[cur].sum() for lbl, cur, _, _ in _cmp_specs if lbl == "전체") or \
                             sum(_cmp_data[cur].sum() for _, cur, _, _ in _cmp_specs)
                _kc = st.columns(len(_cmp_specs))
                for _ci, (lbl, cur, prv, chg) in enumerate(_cmp_specs):
                    _c_sum = _cmp_data[cur].sum()
                    _p_sum = _cmp_data[prv].sum()
                    _d_sum = _c_sum - _p_sum
                    _pct = _d_sum / _p_sum * 100 if _p_sum else 0
                    if lbl == "전체":
                        _val = _fmt_won(_c_sum)
                    else:
                        _share = _c_sum / _cmp_total * 100 if _cmp_total else 0
                        _val = f"{_fmt_won(_c_sum)} ({_share:.0f}%)"
                    _kc[_ci].metric(
                        lbl, _val,
                        delta=f"{_fmt_won(_d_sum)} ({_pct:+.1f}%)",
                        delta_color="inverse",
                    )

                st.divider()

                _util_sel = st.selectbox(
                    "유틸리티", [s[0] for s in _cmp_specs],
                    key=f"{_key_pfx}_util_sel",
                )
                _spec = next(s for s in _cmp_specs if s[0] == _util_sel)
                _, _cur_col, _prv_col, _chg_col = _spec

                # Bar chart
                _cmp_logy = st.checkbox("Log 스케일", key=f"{_key_pfx}_change_logy")
                _plot_df = _cmp_data[["brand", _chg_col]].copy()
                if "building" in _cmp_data.columns:
                    _plot_df["building"] = _cmp_data["building"]
                _plot_df = _plot_df.sort_values(_chg_col, ascending=False).reset_index(drop=True)
                _plot_df["color"] = _plot_df[_chg_col].apply(
                    lambda v: "#C44E52" if v > 0 else "#2ca02c"
                )

                _fig_cmp = go.Figure(go.Bar(
                    x=_plot_df["brand"],
                    y=_plot_df[_chg_col] / 1e4,
                    marker_color=_plot_df["color"],
                    text=(_plot_df[_chg_col] / 1e4).apply(lambda v: f"{v:+,.0f}"),
                    textposition="outside",
                    textfont=dict(size=9, color="#333333"),
                    hovertemplate="<b>%{x}</b><br>변화: %{y:+,.0f} 만원<extra></extra>",
                ))
                _fig_cmp.add_hline(y=0, line_color="#888888", line_width=1)
                _fig_cmp.update_layout(
                    title=f"{_util_sel} {_chart_suffix}",
                    height=420,
                    xaxis_tickangle=-45,
                    yaxis_title="변화 (만원)",
                    yaxis_type="log" if _cmp_logy else None,
                    margin=dict(t=55, b=80),
                    showlegend=False,
                )
                _ev_cmp = st.plotly_chart(_fig_cmp, use_container_width=True,
                                          key=f"{_key_pfx}_change_bar", on_select="rerun")
                _sel_cmp = _ev_cmp.selection.points if _ev_cmp and hasattr(_ev_cmp, "selection") else []
                if _sel_cmp:
                    _brand = _sel_cmp[0].get("x", "")
                    if isinstance(_brand, (list, tuple)):
                        _brand = _brand[0]
                    _fdf = _cmp_data[_cmp_data["brand"] == _brand] if _brand else pd.DataFrame()
                    if not _fdf.empty:
                        st.caption(f"선택됨: **{_brand}**")
                        _show_cols = [c for c in ["brand", "building", "floor",
                                                   _cur_col, _prv_col, _chg_col] if c in _cmp_data.columns]
                        st.dataframe(_fdf[_show_cols].reset_index(drop=True),
                                     hide_index=True, use_container_width=True)

                st.divider()

                # Full change table
                _tbl_df = _cmp_data[
                    ["brand"] + [c for c in ["building", "floor", "size_m2"] if c in _cmp_data.columns]
                    + [_cur_col, _prv_col, _chg_col]
                ].copy()
                _tbl_df["변화율(%)"] = np.where(
                    _tbl_df[_prv_col] > 0,
                    (_tbl_df[_chg_col] / _tbl_df[_prv_col] * 100).round(1),
                    np.nan,
                )
                _tbl_df = _tbl_df.rename(columns={
                    _cur_col: _cur_col_label, _prv_col: _prev_col_label, _chg_col: "변화(원)",
                })
                _tbl_df = _tbl_df.sort_values("변화(원)", ascending=False).reset_index(drop=True)
                _tbl_df[_cur_col_label] = _tbl_df[_cur_col_label].map(_fmt_won)
                _tbl_df[_prev_col_label] = _tbl_df[_prev_col_label].map(_fmt_won)
                _tbl_df["변화(원)"] = _tbl_df["변화(원)"].map(_fmt_won)
                with st.expander("📋 전체 변화 상세", expanded=False):
                    st.dataframe(_tbl_df, hide_index=True, use_container_width=True)

                # Top / bottom
                _chg_cols = ["brand"] + (["building"] if "building" in _plot_df.columns else []) + [_chg_col]
                _raw_chg = _plot_df[_chg_cols].copy()
                _n_show = min(10, len(_raw_chg))
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.markdown(f"**🔴 상승 상위 {_n_show}개**")
                    _top = _raw_chg.nlargest(_n_show, _chg_col).copy()
                    _top[_chg_col] = _top[_chg_col].map(_fmt_won)
                    st.dataframe(_top, hide_index=True, use_container_width=True)
                with _c2:
                    st.markdown(f"**🟢 감소 상위 {_n_show}개**")
                    _bot = _raw_chg.nsmallest(_n_show, _chg_col).copy()
                    _bot[_chg_col] = _bot[_chg_col].map(_fmt_won)
                    st.dataframe(_bot, hide_index=True, use_container_width=True)

    # ═══════════════════════════ 사용 분석 ═════════════════════════════════════
    with tab_rank:
        # ── Category selector ────────────────────────────────────────────────
        _CAT_META = [
            ("💧 수도",  "water_m3",  "#4C72B0", "수도 (m³)"),
            ("🌡️ 온수", "hw_m3",     "#C44E52", "온수 (m³)"),
            ("⚡ 전기",  "kwh_total", "#DD8A00", "전기 (kWh)"),
        ]
        # Per-m² columns for area normalization toggle
        _sz = merged["size_m2"].replace(0, float("nan"))
        for _uc, _pm2c in [("water_m3", "water_m3_pm2"),
                            ("hw_m3", "hw_m3_pm2"),
                            ("kwh_total", "kwh_total_pm2")]:
            if _uc in merged.columns:
                merged[_pm2c] = (merged[_uc] / _sz).round(4)

        _avail_cats = [label for label, col, _, _ in _CAT_META if merged[col].sum() > 0]
        if not _avail_cats:
            st.info("사용량 데이터가 없습니다.")
        else:
            _fc1, _fc2 = st.columns([3, 1])
            with _fc1:
                _cat_sel = st.radio("분석 기준", _avail_cats, horizontal=True, key="sum_rank_cat")
            with _fc2:
                _per_m2 = st.checkbox("m²당", key="sum_rank_per_m2",
                                      help="면적당 사용량으로 정규화하여 대형 매장 편향 제거")

            _raw_col  = next(col for label, col, _, _ in _CAT_META if label == _cat_sel)
            _sel_clr  = next(clr for label, col, clr, _ in _CAT_META if label == _cat_sel)
            _raw_label = next(lbl for label, col, clr, lbl in _CAT_META if label == _cat_sel)
            if _per_m2 and f"{_raw_col}_pm2" in merged.columns:
                _sel_col = f"{_raw_col}_pm2"
                _sel_label = _raw_label.replace("(", "(/m² ").replace(")", ")")
            else:
                _sel_col = _raw_col
                _sel_label = _raw_label
        _sel_series = merged[_sel_col].dropna()
        _UNIT_MAP = {"water_m3": "m³", "hw_m3": "m³", "kwh_total": "kWh",
                     "water_m3_pm2": "m³/m²", "hw_m3_pm2": "m³/m²", "kwh_total_pm2": "kWh/m²"}
        _sel_unit = _UNIT_MAP.get(_sel_col, "")

        # Derived stats for selected series
        _sel_pos  = _sel_series[_sel_series > 0]
        _r_up_sel = _iqr_upper(_sel_pos) if len(_sel_pos) >= 4 else float("inf")
        _r_lo_w, _r_hi_w = _iqr_whiskers(_sel_series)
        _avg_sel  = _sel_series.mean()
        _med_sel  = _sel_series.median()
        _total_sel = _sel_series.sum()
        _top1_sel  = merged.loc[_sel_series.idxmax()]

        merged = merged.copy()
        merged["이상치"] = _sel_series.apply(
            lambda v: "▲ 상위" if v >= _r_hi_w else ("▼ 하위" if v <= _r_lo_w else "")
        )

        _n_brands = len(merged)

        # ── Metrics row ──────────────────────────────────────────────────────
        with st.container(border=True):
            sc = st.columns(4)
            sc[0].metric("합계",   f"{_total_sel:,.1f} {_sel_unit}")
            sc[1].metric("평균",   f"{_avg_sel:,.1f} {_sel_unit}")
            sc[2].metric("중앙값", f"{_med_sel:,.1f} {_sel_unit}")
            sc[3].metric("1위",    _top1_sel["brand"])

        # ── Shared table helpers ─────────────────────────────────────────────
        _RANK_DISP_COLS = [c for c in ["brand", "이상치", "water_m3", "hw_m3", "kwh_total", "building", "floor"]
                           if c in merged.columns]
        _RANK_COL_CFG = {
            "brand":       st.column_config.TextColumn("브랜드"),
            "이상치":      st.column_config.TextColumn("이상치", width="small"),
            "building":    st.column_config.TextColumn("건물", width="small"),
            "floor":       st.column_config.TextColumn("층",   width="small"),
            "water_m3":    st.column_config.NumberColumn("수도 (m³)",   format="%.1f"),
            "hw_m3":       st.column_config.NumberColumn("온수 (m³)",   format="%.1f"),
            "kwh_total":   st.column_config.NumberColumn("전기 (kWh)",  format="%.0f"),
        }

        def _rank_tables(df_sorted: pd.DataFrame, top_mask, bot_mask, mid_mask):
            _top_df = df_sorted[top_mask].sort_values(_sel_col, ascending=False)
            _bot_df = df_sorted[bot_mask].sort_values(_sel_col, ascending=False)
            _mid_df = df_sorted[~top_mask & ~bot_mask].sort_values(_sel_col, ascending=False)
            st.markdown(f"**▲ 상위 이상치** ({len(_top_df)}개)")
            if not _top_df.empty:
                st.dataframe(_top_df[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**▼ 하위 이상치** ({len(_bot_df)}개)")
            if not _bot_df.empty:
                st.dataframe(_bot_df[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**정상 범위** ({len(_mid_df)}개)")
            if not _mid_df.empty:
                st.dataframe(_mid_df[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)

        st.divider()
        _init_session_keys([
            ("sum_rank_hist_bins", 50), ("sum_rank_hist_bins_i", 50),
        ])
        _rank_view = st.radio(
            "그래프 보기", ["히스토그램", "순위 차트", "박스플롯"],
            horizontal=True, key="sum_rank_view",
        )

        if _rank_view == "순위 차트":
            _rc1, _rc2 = st.columns([3, 1])
            with _rc1:
                _n = st.slider("상위 N개", 1, len(merged), min(20, len(merged)), key="sum_rank_n")
            with _rc2:
                _rank_logy = st.checkbox("Log 스케일", key="sum_rank_logy")
            _top = merged.nlargest(_n, _sel_col).sort_values(_sel_col, ascending=True)
            _xv = _top[_sel_col].values
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=_xv, y=[str(b)[:26] for b in _top["brand"]],
                name=_cat_sel, orientation="h", marker_color=_sel_clr,
                hovertemplate="<b>%{y}</b><br>" + _cat_sel + f": %{{x:,.1f}} {_sel_unit}<extra></extra>",
                text=[f"{v:,.1f}" if v >= 0.5 else ("" if v == 0 else f"{v:,.2f}") for v in _xv],
                textposition="inside", textfont=dict(size=9, color="white"),
            ))
            _r_up = _r_up_sel
            if _r_up < float("inf"):
                fig_r.add_vline(x=_r_up, line_dash="dash", line_color="#8B2BE2", line_width=2)
                fig_r.add_annotation(
                    x=_r_up, y=1, yref="paper",
                    text=f"⚠ IQR 상한 {_r_up:,.1f} {_sel_unit}",
                    showarrow=False, xanchor="left", xshift=6,
                    font=dict(size=11, color="#8B2BE2"),
                    bgcolor="rgba(255,255,255,0.6)", bordercolor="#8B2BE2", borderwidth=1,
                )
            _xaxis_cfg = dict(griddash="dot")
            if _rank_logy:
                _xaxis_cfg["type"] = "log"
            fig_r.update_layout(
                barmode="relative", height=max(480, _n * 22 + 80),
                xaxis_title=f"{_sel_label} ({_sel_unit})",
                xaxis=_xaxis_cfg,
                legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle"),
                margin=dict(l=10, r=100, t=30, b=40),
            )
            _rev = st.plotly_chart(fig_r, use_container_width=True, key="sum_rank_chart", on_select="rerun")
            if _rev and hasattr(_rev, "selection") and _rev.selection.points:
                _ry = _rev.selection.points[0].get("y") or _rev.selection.points[0].get("label", "")
                if _ry:
                    _rdf = merged[[str(b)[:26] == _ry for b in merged["brand"]]]
                    if not _rdf.empty:
                        st.caption(f"선택됨: **{_ry}**")
                        st.dataframe(_rdf[_RANK_DISP_COLS].reset_index(drop=True),
                                     column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)
            with st.expander("📋 순위 테이블", expanded=False):
                _rank_tbl = (
                    merged.nlargest(_n, _sel_col)
                    .sort_values(_sel_col, ascending=False)
                    .reset_index(drop=True)
                )
                _rank_tbl.index = _rank_tbl.index + 1
                st.dataframe(_rank_tbl[_RANK_DISP_COLS], column_config=_RANK_COL_CFG,
                             use_container_width=True)

        elif _rank_view == "박스플롯":
            _boxplot_with_labels(_sel_series, merged["brand"],
                                 f"{_sel_label} ({_sel_unit})", "sum_rank_box",
                                 source_df=merged,
                                 disp_cols=_RANK_DISP_COLS)
            _top_mask = _sel_series >= _r_hi_w
            _bot_mask = _sel_series <= _r_lo_w
            with st.expander("📋 이상치 상세 테이블", expanded=False):
                _rank_tables(merged, _top_mask, _bot_mask, ~_top_mask & ~_bot_mask)

        else:  # 히스토그램
            _h_bins = _synced_slider_input("sum_rank_hist_bins", "Bins", 5, 200, 50, 5)
            _iqr_k = st.slider("IQR 배수 (k)", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                               key="sum_rank_iqr_k",
                               help="이상치 기준: Q1 − k×IQR  /  Q3 + k×IQR")
            _hq1  = float(_sel_series.quantile(0.25))
            _hq3  = float(_sel_series.quantile(0.75))
            _hiqr = _hq3 - _hq1
            _lo_u = _hq1 - _iqr_k * _hiqr
            _hi_u = _hq3 + _iqr_k * _hiqr
            st.markdown(
                f"$$Q_1 = {_hq1:,.1f},\\quad Q_3 = {_hq3:,.1f},\\quad IQR = {_hiqr:,.1f}$$\n\n"
                f"$$\\text{{Lower}} = Q_1 - {_iqr_k}\\times IQR = {_lo_u:,.1f}\\text{{ {_sel_unit}}}"
                f",\\quad \\text{{Upper}} = Q_3 + {_iqr_k}\\times IQR = {_hi_u:,.1f}\\text{{ {_sel_unit}}}$$"
            )
            _plot_hist(_sel_series, _h_bins, float(_lo_u), float(_hi_u),
                       f"{_sel_label} 분포 ({_sel_unit})", key="sum_rank_hist",
                       source_df=merged, val_col=_sel_col, val_scale=1.0,
                       display_cols=["brand", "building", "floor",
                                     "water_m3", "hw_m3", "kwh_total"],
                       show_bins_slider=False)
            _top_iqr_m = _sel_series > _hi_u
            _bot_iqr_m = _sel_series < _lo_u
            with st.expander("📋 이상치 상세 테이블", expanded=False):
                _rank_tables(merged, _top_iqr_m, _bot_iqr_m, ~_top_iqr_m & ~_bot_iqr_m)

        # ── Category breakdowns (when specific utility selected) ─────────
        if _cat_sel == "⚡ 전기" and elec_df is not None and not elec_df.empty:
            st.divider()
            try:
                elec_br = build_elec_breakdown(elec_df, meter_df=meter_df)
                if not elec_br.empty:
                    _render_elec_breakdown(elec_br, split_by_building=split_by_building)
            except Exception as _e:
                st.warning(f"전기 분류 로드 실패: {_e}")

        if _cat_sel == "💧 수도" and water_df is not None and not water_df.empty:
            st.divider()
            try:
                water_br = build_water_breakdown(water_df, meter_df=meter_df)
                if not water_br.empty:
                    _render_water_breakdown(water_br, split_by_building=split_by_building)
            except Exception as _e:
                st.warning(f"수도 분류 로드 실패: {_e}")

        # ── Building comparison dashboard ─────────────────────────────────
        if "building" in merged.columns and merged["building"].nunique() > 1:
            st.divider()
            with st.expander("🏢 건물별 비교", expanded=False):
                _bld_agg_cols = dict(
                    brands=("brand", "count"),
                    area=("size_m2", "sum"),
                    water_m3=("water_m3", "sum"),
                    hw_m3=("hw_m3", "sum"),
                    kwh_total=("kwh_total", "sum"),
                )
                if "heat_m3" in merged.columns and merged["heat_m3"].sum() > 0:
                    _bld_agg_cols["heat_m3"] = ("heat_m3", "sum")
                _bld_agg = merged.groupby("building").agg(**_bld_agg_cols
                ).reindex(["A", "B", "C", "D"]).dropna(how="all")
                _bld_area = _bld_agg["area"].replace(0, float("nan"))
                _bld_agg["water_pm2"] = (_bld_agg["water_m3"] / _bld_area).round(3)
                _bld_agg["hw_pm2"] = (_bld_agg["hw_m3"] / _bld_area).round(3)
                _bld_agg["kwh_pm2"] = (_bld_agg["kwh_total"] / _bld_area).round(2)
                _has_heat_vol = "heat_m3" in _bld_agg.columns and _bld_agg["heat_m3"].sum() > 0
                if _has_heat_vol:
                    _bld_agg["heat_pm2"] = (_bld_agg["heat_m3"] / _bld_area).round(3)

                from plotly.subplots import make_subplots as _make_sub

                _bld_logy = st.checkbox("Log 스케일", key="sum_bld_logy")
                _bc1, _bc2 = st.columns(2)
                with _bc1:
                    # Dual-axis bar: m³ (left) + kWh (right) per building
                    _fig_bld = _make_sub(specs=[[{"secondary_y": True}]])
                    _buildings = _bld_agg.index.tolist()
                    _fig_bld.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["water_m3"], name="수도 (m³)",
                        marker_color="#4C72B0", offsetgroup=0,
                    ), secondary_y=False)
                    _fig_bld.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["hw_m3"], name="온수 (m³)",
                        marker_color="#C44E52", offsetgroup=1,
                    ), secondary_y=False)
                    if _has_heat_vol:
                        _fig_bld.add_trace(go.Bar(
                            x=_buildings, y=_bld_agg["heat_m3"], name="난방 (m³)",
                            marker_color="#E377C2", offsetgroup=2,
                        ), secondary_y=False)
                    _fig_bld.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["kwh_total"], name="전기 (kWh)",
                        marker_color="#DD8A00", offsetgroup=3,
                    ), secondary_y=True)
                    _fig_bld.update_layout(
                        title="건물별 총 사용량", barmode="group", height=380,
                        margin=dict(t=45, b=30),
                        legend=dict(orientation="h", yanchor="top", y=-0.12,
                                    xanchor="center", x=0.5, font=dict(size=10)),
                    )
                    _fig_bld.update_yaxes(title_text="m³", secondary_y=False,
                                          type="log" if _bld_logy else None)
                    _fig_bld.update_yaxes(title_text="kWh", secondary_y=True,
                                          type="log" if _bld_logy else None)
                    st.plotly_chart(_fig_bld, use_container_width=True, key="sum_bld_abs")

                with _bc2:
                    # Dual-axis bar: per-m² usage per building
                    _fig_pm2 = _make_sub(specs=[[{"secondary_y": True}]])
                    _fig_pm2.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["water_pm2"], name="수도 (m³/m²)",
                        marker_color="#4C72B0", offsetgroup=0,
                    ), secondary_y=False)
                    _fig_pm2.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["hw_pm2"], name="온수 (m³/m²)",
                        marker_color="#C44E52", offsetgroup=1,
                    ), secondary_y=False)
                    if _has_heat_vol:
                        _fig_pm2.add_trace(go.Bar(
                            x=_buildings, y=_bld_agg["heat_pm2"], name="난방 (m³/m²)",
                            marker_color="#E377C2", offsetgroup=2,
                        ), secondary_y=False)
                    _fig_pm2.add_trace(go.Bar(
                        x=_buildings, y=_bld_agg["kwh_pm2"], name="전기 (kWh/m²)",
                        marker_color="#DD8A00", offsetgroup=3,
                    ), secondary_y=True)
                    _fig_pm2.update_layout(
                        title="건물별 면적당 사용량", barmode="group", height=380,
                        margin=dict(t=45, b=30),
                        legend=dict(orientation="h", yanchor="top", y=-0.12,
                                    xanchor="center", x=0.5, font=dict(size=10)),
                    )
                    _fig_pm2.update_yaxes(title_text="m³/m²", secondary_y=False,
                                          type="log" if _bld_logy else None)
                    _fig_pm2.update_yaxes(title_text="kWh/m²", secondary_y=True,
                                          type="log" if _bld_logy else None)
                    st.plotly_chart(_fig_pm2, use_container_width=True, key="sum_bld_pm2")

                # Summary table
                _bld_disp = _bld_agg.copy()
                _bld_disp.index.name = "건물"
                _rename_map = {
                    "brands": "브랜드수", "area": "면적(m²)",
                    "water_m3": "수도 (m³)", "hw_m3": "온수 (m³)", "kwh_total": "전기 (kWh)",
                    "water_pm2": "수도/m²", "hw_pm2": "온수/m²", "kwh_pm2": "전기/m²",
                }
                if _has_heat_vol:
                    _rename_map["heat_m3"] = "난방 (m³)"
                    _rename_map["heat_pm2"] = "난방/m²"
                _bld_disp = _bld_disp.rename(columns=_rename_map)
                st.dataframe(_bld_disp.reset_index(), hide_index=True, use_container_width=True)

    # ═══════════════════════════ 유틸리티 구성 ════════════════════════════════
    with tab_mix:
        _mix_total = merged["util_total"].sum()
        _mix_w_pct = merged["water_total"].sum() / _mix_total * 100
        _mix_hw_pct = merged["hw_total"].sum() / _mix_total * 100
        _mix_el_pct = merged["elec_total"].sum() / _mix_total * 100
        _mix_ht_pct = merged["heat_total"].sum() / _mix_total * 100
        _mix_dom = max([("수도", _mix_w_pct), ("온수", _mix_hw_pct), ("전기", _mix_el_pct), ("난방", _mix_ht_pct)], key=lambda x: x[1])
        _elec_heavy = merged[merged["util_total"] > 0].copy()
        _elec_heavy["_ep"] = _elec_heavy["elec_total"] / _elec_heavy["util_total"]
        _top_elec = _elec_heavy.loc[_elec_heavy["_ep"].idxmax()]
        st.caption(
            f"비중: 수도 {_mix_w_pct:.0f}% · 온수 {_mix_hw_pct:.0f}% · "
            f"전기 {_mix_el_pct:.0f}% · 난방 {_mix_ht_pct:.0f}% — "
            f"지배 항목: **{_mix_dom[0]}**"
        )
        _mv1, _mv2 = st.columns(2)
        with _mv1:
            # Overall donut
            _dvals = {"수도": merged["water_total"].sum(),
                      "온수": merged["hw_total"].sum(),
                      "전기": merged["elec_total"].sum(),
                      "난방": merged["heat_total"].sum()}
            fig_d = go.Figure(go.Pie(
                labels=list(_dvals.keys()), values=list(_dvals.values()), hole=0.45,
                marker=dict(colors=["#4C72B0","#C44E52","#DD8A00","#E377C2"]),
                textinfo="label+percent+value", textfont=dict(size=13),
            ))
            fig_d.update_layout(title="전체 유틸리티 비중", height=380,
                                margin=dict(l=20,r=20,t=50,b=20))
            _ev_donut = st.plotly_chart(fig_d, use_container_width=True, key="sum_mix_donut", on_select="rerun")
            _handle_chart_click(_ev_donut, merged, brand_col="brand", field="x")

        with _mv2:
            # Per-brand mix: who is dominated by electricity vs water?
            merged_mix = merged[merged["util_total"]>0].copy()
            merged_mix["elec_pct"] = merged_mix["elec_total"] / merged_mix["util_total"] * 100
            merged_mix["water_pct"] = merged_mix["water_total"] / merged_mix["util_total"] * 100
            merged_mix["hw_pct"]   = merged_mix["hw_total"] / merged_mix["util_total"] * 100
            merged_mix["heat_pct"] = merged_mix["heat_total"] / merged_mix["util_total"] * 100

            # Scatter: water_pct vs elec_pct, sized by util_total
            _top5_idx = merged_mix.nlargest(5, "util_total").index
            fig_mix = go.Figure()
            for bld in sorted(merged_mix["building"].dropna().unique()):
                sub = merged_mix[merged_mix["building"]==bld]
                fig_mix.add_trace(go.Scatter(
                    x=sub["water_pct"], y=sub["elec_pct"],
                    mode="markers", name=f"{bld}동",
                    marker=dict(color=_BLD_COLOR.get(str(bld),"#888"),
                                size=(sub["util_total"]/merged_mix["util_total"].max()*40+6).round(0),
                                opacity=0.8,
                                line=dict(width=1, color="rgba(0,0,0,0.25)")),
                    customdata=sub[["brand","util_total","hw_pct","heat_pct"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "수도 %{x:.1f}%  전기 %{y:.1f}%  온수 %{customdata[2]:.1f}%  난방 %{customdata[3]:.1f}%<br>"
                        "합계 %{customdata[1]:,.0f} 원<extra></extra>"
                    ),
                ))
            for _, row in merged_mix.loc[_top5_idx].iterrows():
                fig_mix.add_annotation(
                    x=row["water_pct"], y=row["elec_pct"],
                    text=str(row["brand"])[:14],
                    showarrow=False, yshift=12,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.15)",
                    bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
                )
            fig_mix.update_layout(
                title="수도 비중 vs 전기 비중 (버블=총비용)",
                height=380, xaxis_title="수도 비중 (%)", yaxis_title="전기 비중 (%)",
                xaxis=dict(griddash="dot", range=[-5,105]),
                yaxis=dict(griddash="dot", range=[-5,105]),
                margin=dict(l=20,r=20,t=50,b=40),
            )
            _mev = st.plotly_chart(fig_mix, use_container_width=True, key="sum_mix_scatter", on_select="rerun")
            if _mev and hasattr(_mev, "selection") and _mev.selection.points:
                _mpt = _mev.selection.points[0]
                _cd = _mpt.get("customdata", [])
                if isinstance(_cd, dict):
                    _cd = list(_cd.values())
                if _cd:
                    _mbrand = _cd[0] if isinstance(_cd, (list, tuple)) else str(_cd)
                    _mdf = merged_mix[merged_mix["brand"] == _mbrand]
                    if not _mdf.empty:
                        st.caption(f"선택됨: **{_mbrand}**")
                        _mix_show_cols = ["brand", "building", "floor", "util_total",
                                          "water_total", "hw_total", "elec_total", "heat_total",
                                          "water_pct", "elec_pct", "hw_pct", "heat_pct"]
                        _mix_show = [c for c in _mix_show_cols if c in _mdf.columns]
                        st.dataframe(_mdf[_mix_show].reset_index(drop=True),
                                     hide_index=True, use_container_width=True)

        # Summary table: top 20 by util_total
        with st.expander("📋 브랜드별 비용 상세", expanded=False):
            _tbl_cols = ["brand","building","floor","water_total","hw_total","elec_total","heat_total","util_total"]
            _tbl = merged[_tbl_cols].head(20).copy()
            _tbl["water_total"] = _tbl["water_total"].apply(lambda v: f"{v:,.0f}")
            _tbl["hw_total"]    = _tbl["hw_total"].apply(lambda v: f"{v:,.0f}")
            _tbl["elec_total"]  = _tbl["elec_total"].apply(lambda v: f"{v:,.0f}")
            _tbl["heat_total"]  = _tbl["heat_total"].apply(lambda v: f"{v:,.0f}")
            _tbl["util_total"]  = _tbl["util_total"].apply(lambda v: f"{v:,.0f}")
            _tbl = _tbl.rename(columns={"brand":"브랜드","building":"건물","floor":"층",
                                         "water_total":"수도 (원)","hw_total":"온수 (원)",
                                         "elec_total":"전기 (원)","heat_total":"난방 (원)","util_total":"합계 (원)"})
            st.dataframe(
                _tbl,
                column_config={
                    "브랜드": st.column_config.TextColumn("브랜드"),
                    "건물":   st.column_config.TextColumn("건물", width="small"),
                    "층":     st.column_config.TextColumn("층", width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )


    # 경영 보고 moved to tab_mgmt.py (rendered in 점검대상 tab)
