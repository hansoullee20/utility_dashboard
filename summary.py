"""summary.py — Cross-sheet utility summary: water + hotwater + electricity."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import BLD_COLOR as _BLD_COLOR, iqr_upper as _iqr_upper
from viz import plot_hist_with_tails as _plot_hist

_UTIL_COLOR = {"수도": "#4C72B0", "온수": "#C44E52", "전기": "#DD8A00"}


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
) -> None:
    _available = [n for n, d in [("수도", water_df), ("온수", hotwater_df), ("전기", elec_df)] if d is not None]
    st.header("📊 통합 유틸리티 분석")
    st.caption(f"{'·'.join(_available)} 데이터를 브랜드 기준으로 통합한 종합 분석입니다.")

    # ── Aggregate each available sheet by brand ────────────────────────────────
    if water_df is not None and not water_df.empty:
        _w = water_df.groupby("brand").agg(
            building=("building","first"), floor=("floor","first"),
            size_m2=("size_m2","sum"), water_total=("total","sum"),
        ).reset_index()
    else:
        _w = pd.DataFrame(columns=["brand","building","floor","size_m2","water_total"])

    if hotwater_df is not None and not hotwater_df.empty:
        _hw = hotwater_df.groupby("brand").agg(hw_total=("total","sum")).reset_index()
    else:
        _hw = pd.DataFrame(columns=["brand","hw_total"])

    if elec_df is not None and not elec_df.empty:
        _el = elec_df.groupby("brand").agg(
            elec_total=("grand_total","sum"), kwh_total=("kwh_total","sum")
        ).reset_index()
    else:
        _el = pd.DataFrame(columns=["brand","elec_total","kwh_total"])

    merged = _w.merge(_hw, on="brand", how="outer").merge(_el, on="brand", how="outer")
    # Fill missing utilities with 0
    for col in ["water_total","hw_total","elec_total","kwh_total","size_m2"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0
    # Fill building/floor/size_m2 from all available sources (vectorized)
    _meta_parts = [
        df.groupby("brand")[["building", "floor", "size_m2"]].first()
        for df in [water_df, hotwater_df, elec_df]
        if df is not None and not df.empty
    ]
    if _meta_parts:
        _meta = pd.concat(_meta_parts).groupby(level=0).first()
        merged = merged.set_index("brand")
        _no_bld = merged["building"].isna() | (merged["building"].astype(str).str.strip() == "")
        merged["building"] = merged["building"].where(~_no_bld, _meta["building"].reindex(merged.index))
        merged["floor"]    = merged["floor"].where(~_no_bld, _meta["floor"].reindex(merged.index))
        merged["size_m2"]  = merged["size_m2"].where(
            merged["size_m2"] > 0,
            _meta["size_m2"].reindex(merged.index, fill_value=0),
        )
        merged = merged.reset_index()

    merged["util_total"] = merged["water_total"] + merged["hw_total"] + merged["elec_total"]
    merged = merged[merged["util_total"] > 0].sort_values("util_total", ascending=False).reset_index(drop=True)

    def _fmt_won(v: float) -> str:
        """Auto-scale: 억원 / 만원 / 원"""
        if abs(v) >= 1e8:
            return f"{v/1e8:.2f} 억원"
        elif abs(v) >= 1e4:
            return f"{v/1e4:,.0f} 만원"
        else:
            return f"{v:,.0f} 원"

    # ── Top metrics ────────────────────────────────────────────────────────────
    _util_sum = merged["util_total"].sum()
    mc = st.columns(5)
    mc[0].metric("통합 브랜드",    f"{len(merged)}개")
    mc[1].metric("총 유틸리티 비용", _fmt_won(_util_sum))
    mc[2].metric("수도",           f"{_fmt_won(merged['water_total'].sum())} ({merged['water_total'].sum()/_util_sum*100:.0f}%)")
    mc[3].metric("온수",           f"{_fmt_won(merged['hw_total'].sum())} ({merged['hw_total'].sum()/_util_sum*100:.0f}%)")
    mc[4].metric("전기",           f"{_fmt_won(merged['elec_total'].sum())} ({merged['elec_total'].sum()/_util_sum*100:.0f}%)")

    tab_rank, tab_mix, tab_area, tab_bld, tab_cat, tab_mgmt = st.tabs(
        ["총 유틸리티 순위", "유틸리티 구성", "면적당 총비용", "건물별 비교", "🔍 항목별 분석", "📋 경영 보고"]
    )

    def _boxplot_with_labels(s: pd.Series, label_s: pd.Series,
                             x_title: str, key: str) -> None:
        """Horizontal box plot — dots for outliers, labels only on extremes."""
        _q1, _q3 = s.quantile(0.25), s.quantile(0.75)
        _iqr = _q3 - _q1
        _lo_w, _hi_w = float(_q1 - 1.5 * _iqr), float(_q3 + 1.5 * _iqr)

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
        st.plotly_chart(fig, use_container_width=True, key=key)

    # ═══════════════════════════ 총 유틸리티 순위 ═════════════════════════════
    with tab_rank:
        with st.expander("이 탭 설명"):
            st.markdown("""
**어떤 브랜드가 유틸리티 비용을 가장 많이 쓰고 있나요?**

수도·온수·전기 비용을 합산하거나 항목별로 나눠 브랜드 순위를 매기는 탭입니다.

- **분석 기준**: 전체(합산) 또는 수도·온수·전기 개별 항목 선택 가능
- **순위 차트**: 상위 N개 브랜드 비용 비교. 전체 선택 시 항목별 누적 막대, 개별 선택 시 단색 막대.
- **박스플롯**: 선택 기준의 통계 분포 및 이상치 브랜드 테이블.
- **히스토그램**: 비용 분포의 전체 형태. 주황색 구간은 테일(이상치 영역).
- **현재 데이터 해석**: 전체 + 항목별 종합 요약이 항상 표시됩니다.
""")

        # ── Category selector ────────────────────────────────────────────────
        _CAT_META = [
            ("전체",  "util_total",   None,      "합계 (만원)"),
            ("💧 수도", "water_total", "#4C72B0", "수도 (만원)"),
            ("🌡️ 온수", "hw_total",   "#C44E52", "온수 (만원)"),
            ("⚡ 전기", "elec_total",  "#DD8A00", "전기 (만원)"),
        ]
        _avail_cats = [label for label, col, _, _ in _CAT_META if merged[col].sum() > 0]
        _cat_sel = st.radio("분석 기준", _avail_cats, horizontal=True, key="sum_rank_cat")
        _sel_col  = next(col for label, col, _, _ in _CAT_META if label == _cat_sel)
        _sel_clr  = next(clr for label, col, clr, _ in _CAT_META if label == _cat_sel)
        _sel_label = next(lbl for label, col, clr, lbl in _CAT_META if label == _cat_sel)
        _sel_series = merged[_sel_col]

        # Derived stats for selected series
        _sel_pos     = _sel_series[_sel_series > 0]
        _r_up_sel    = _iqr_upper(_sel_pos) if len(_sel_pos) >= 4 else float("inf")
        _rq1s, _rq3s = _sel_series.quantile(0.25), _sel_series.quantile(0.75)
        _r_iqr_s     = _rq3s - _rq1s
        _r_lo_w      = float(_rq1s - 1.5 * _r_iqr_s)
        _r_hi_w      = float(_rq3s + 1.5 * _r_iqr_s)
        _n_above_sel  = int((_sel_series > _r_up_sel).sum())
        _avg_sel      = _sel_series.mean()
        _med_sel      = _sel_series.median()
        _total_sel    = _sel_series.sum()
        _top1_sel     = merged.loc[_sel_series.idxmax()]
        _top1_mult    = _top1_sel[_sel_col] / _med_sel if _med_sel > 0 else 0
        _low_thresh   = _med_sel * 0.1
        _n_suspicion  = int((_sel_series < _low_thresh).sum())
        _skew_msg     = ("평균이 중앙값보다 높아 고비용 브랜드 몇 곳이 평균을 견인하고 있습니다."
                         if _avg_sel > _med_sel * 1.1 else "분포가 비교적 균등합니다.")

        merged = merged.copy()
        merged["이상치"] = _sel_series.apply(
            lambda v: "▲ 상위" if v >= _r_hi_w else ("▼ 하위" if v <= _r_lo_w else "")
        )

        # ── Metrics row ──────────────────────────────────────────────────────
        sc = st.columns(4)
        sc[0].metric("합계",   _fmt_won(_total_sel))
        sc[1].metric("평균",   f"{_avg_sel/1e4:,.1f} 만원")
        sc[2].metric("중앙값", f"{_med_sel/1e4:,.1f} 만원")
        sc[3].metric("1위",    _top1_sel["brand"])

        # ── 현재 데이터 해석 — always shows all categories ──────────────────
        _util_total_all = merged["util_total"].sum()
        _dom_util = max(
            [("수도", merged["water_total"].sum()), ("온수", merged["hw_total"].sum()), ("전기", merged["elec_total"].sum())],
            key=lambda x: x[1],
        )
        _dom_pct = _dom_util[1] / _util_total_all * 100 if _util_total_all > 0 else 0
        _n_brands = len(merged)
        _top3_share = merged.nlargest(3, "util_total")["util_total"].sum() / _util_total_all * 100
        _top10p_n   = max(1, int(_n_brands * 0.1))
        _top10p_share = merged.nlargest(_top10p_n, "util_total")["util_total"].sum() / _util_total_all * 100
        _r_up_all   = _iqr_upper(merged["util_total"])
        _outlier_share = (merged[merged["util_total"] > _r_up_all]["util_total"].sum()
                          / _util_total_all * 100) if _r_up_all < float("inf") else 0.0
        _top1_all   = merged.loc[merged["util_total"].idxmax()]
        _top1_dom   = max([("수도", _top1_all["water_total"]), ("온수", _top1_all["hw_total"]), ("전기", _top1_all["elec_total"])], key=lambda x: x[1])

        def _cat_summary(col: str, label: str, icon: str) -> str:
            s = merged[col]
            if s.sum() == 0:
                return ""
            _pos = s[s > 0]
            _up  = _iqr_upper(_pos) if len(_pos) >= 4 else float("inf")
            _n_out = int((s > _up).sum())
            _top  = merged.loc[s.idxmax()]
            _med  = s.median()
            _avg  = s.mean()
            _skew = "⚠️ 평균>중앙값 (고비용 브랜드 견인)" if _avg > _med * 1.1 else "✅ 분포 균등"
            lines = [
                f"#### {icon} {label}",
                f"- 합계 **{_fmt_won(s.sum())}** | 평균 {s.mean()/1e4:,.0f}만 원 | 중앙값 {_med/1e4:,.0f}만 원 — {_skew}",
                f"- 1위: **{_top['brand']}** {_fmt_won(_top[col])} (중앙값의 {_top[col]/_med:.1f}배)" if _med > 0 else f"- 1위: **{_top['brand']}** {_fmt_won(_top[col])}",
                f"- 이상치(IQR 상한 초과): **{_n_out}개**" + (" — 계량기 점검 및 누수 여부 확인 권장" if _n_out > 0 else " — 없음"),
            ]
            return "\n".join(lines)

        with st.expander("현재 데이터 해석"):
            _med_all = merged["util_total"].median()
            _avg_all = merged["util_total"].mean()
            _skew_all = "평균이 중앙값보다 높아 고비용 브랜드가 전체 평균을 견인하고 있습니다." if _avg_all > _med_all * 1.1 else "비용 분포가 비교적 균등합니다."
            st.markdown(f"""
#### 전체 통합 요약
- **{_n_brands}개** 브랜드, 총 **{_fmt_won(_util_total_all)}** | 평균 {_avg_all/1e4:,.0f}만 원 | 중앙값 {_med_all/1e4:,.0f}만 원
- {_skew_all}
- 주요 항목: **{_dom_util[0]} ({_dom_pct:.0f}%)** — 이 유틸리티 관리에 우선 집중 권장
- 상위 3개 브랜드 비중 **{_top3_share:.1f}%** | 상위 10%({_top10p_n}개) 비중 **{_top10p_share:.1f}%**
  > {"⚠️ 비용이 소수에 집중되어 계량기 점검 및 임대 조건 재검토가 필요합니다." if _top10p_share > 50 else "✅ 비용 분포가 비교적 분산되어 있습니다."}
- IQR 이상치 브랜드 **{int((merged['util_total'] > _r_up_all).sum())}개** → 전체 비용의 **{_outlier_share:.1f}%** 부담
- 1위: **{_top1_all['brand']}** {_fmt_won(_top1_all['util_total'])} (주요 항목: {_top1_dom[0]} {_fmt_won(_top1_dom[1])})
""")
            for _col, _lbl, _ico in [("water_total","수도","💧"), ("hw_total","온수","🌡️"), ("elec_total","전기","⚡")]:
                _s = _cat_summary(_col, _lbl, _ico)
                if _s:
                    st.markdown(_s)

        # ── Shared table helpers ─────────────────────────────────────────────
        _RANK_DISP_COLS = ["brand", "이상치", "util_total", "elec_total", "water_total", "hw_total", "building", "floor"]
        _RANK_COL_CFG = {
            "brand":       st.column_config.TextColumn("브랜드"),
            "이상치":      st.column_config.TextColumn("이상치", width="small"),
            "building":    st.column_config.TextColumn("건물", width="small"),
            "floor":       st.column_config.TextColumn("층",   width="small"),
            "water_total": st.column_config.NumberColumn("수도 (만원)",  format="%.1f"),
            "hw_total":    st.column_config.NumberColumn("온수 (만원)",  format="%.1f"),
            "elec_total":  st.column_config.NumberColumn("전기 (만원)",  format="%.1f"),
            "util_total":  st.column_config.NumberColumn("합계 (만원)",  format="%.1f"),
        }
        def _to_manwon(df):
            d = df.copy()
            for _c in ["util_total", "water_total", "hw_total", "elec_total"]:
                if _c in d.columns:
                    d[_c] = d[_c] / 1e4
            return d

        def _rank_tables(df_sorted: pd.DataFrame, top_mask, bot_mask, mid_mask, prefix: str):
            _top_df = df_sorted[top_mask].sort_values(_sel_col, ascending=False)
            _bot_df = df_sorted[bot_mask].sort_values(_sel_col, ascending=False)
            _mid_df = df_sorted[~top_mask & ~bot_mask].sort_values(_sel_col, ascending=False)
            st.markdown(f"**▲ 상위 이상치** — 수염 상한 초과 ({len(_top_df)}개)")
            if not _top_df.empty:
                st.dataframe(_to_manwon(_top_df)[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**▼ 하위 이상치** — 수염 하한 미만 ({len(_bot_df)}개)")
            if not _bot_df.empty:
                st.dataframe(_to_manwon(_bot_df)[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**정상 범위** ({len(_mid_df)}개)")
            if not _mid_df.empty:
                st.dataframe(_to_manwon(_mid_df)[_RANK_DISP_COLS].reset_index(drop=True),
                             column_config=_RANK_COL_CFG, use_container_width=True, hide_index=True)

        st.divider()
        for _k, _v in [("sum_rank_hist_bins", 50), ("sum_rank_hist_bins_i", 50),
                       ("sum_rank_hist_tail", 20), ("sum_rank_hist_tail_i", 20)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v
        _rank_view = st.radio(
            "그래프 보기", ["순위 차트", "박스플롯", "히스토그램"],
            horizontal=True, key="sum_rank_view",
        )

        if _rank_view == "순위 차트":
            _n = st.slider("상위 N개", 1, len(merged), min(20, len(merged)), key="sum_rank_n")
            _top = merged.nlargest(_n, _sel_col).sort_values(_sel_col, ascending=True)
            fig_r = go.Figure()
            if _cat_sel == "전체":
                for label, col, clr in [("수도","water_total","#4C72B0"),
                                          ("온수","hw_total","#C44E52"),
                                          ("전기","elec_total","#DD8A00")]:
                    fig_r.add_trace(go.Bar(
                        x=_top[col].values, y=[str(b)[:26] for b in _top["brand"]],
                        name=label, orientation="h", marker_color=clr,
                        hovertemplate="<b>%{y}</b><br>" + label + ": %{x:,.0f} 원<extra></extra>",
                        text=[f"{v/1e3:.0f}k" if v >= 1e4 else ("" if v == 0 else f"{v:,.0f}") for v in _top[col].values],
                        textposition="inside", textfont=dict(size=9, color="white"),
                    ))
                _barmode = "stack"
            else:
                fig_r.add_trace(go.Bar(
                    x=_top[_sel_col].values, y=[str(b)[:26] for b in _top["brand"]],
                    name=_cat_sel, orientation="h", marker_color=_sel_clr,
                    hovertemplate="<b>%{y}</b><br>" + _cat_sel + ": %{x:,.0f} 원<extra></extra>",
                    text=[f"{v/1e3:.0f}k" if v >= 1e4 else ("" if v == 0 else f"{v:,.0f}") for v in _top[_sel_col].values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                ))
                _barmode = "relative"
            _r_up = _r_up_sel
            if _r_up < float("inf"):
                fig_r.add_vline(x=_r_up, line_dash="dash", line_color="#8B2BE2", line_width=2)
                fig_r.add_annotation(
                    x=_r_up, y=1, yref="paper",
                    text=f"⚠ IQR 상한 {_fmt_won(_r_up)}",
                    showarrow=False, xanchor="left", xshift=6,
                    font=dict(size=11, color="#8B2BE2"),
                    bgcolor="rgba(255,255,255,0.6)", bordercolor="#8B2BE2", borderwidth=1,
                )
            fig_r.update_layout(
                barmode=_barmode, height=max(480, _n * 22 + 80),
                xaxis_title="원", plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle"),
                margin=dict(l=10, r=100, t=30, b=40),
            )
            st.plotly_chart(fig_r, use_container_width=True, key="sum_rank_chart")
            _rank_tbl = (
                merged.nlargest(_n, _sel_col)
                .sort_values(_sel_col, ascending=False)
                .reset_index(drop=True)
            )
            _rank_tbl.index = _rank_tbl.index + 1
            st.dataframe(_to_manwon(_rank_tbl)[_RANK_DISP_COLS], column_config=_RANK_COL_CFG,
                         use_container_width=True)

        elif _rank_view == "박스플롯":
            _sel_man = _sel_series / 1e4
            _boxplot_with_labels(_sel_man, merged["brand"], f"{_cat_sel} 비용 (만 원)", "sum_rank_box")
            _top_mask = _sel_series >= _r_hi_w
            _bot_mask = _sel_series <= _r_lo_w
            _rank_tables(merged, _top_mask, _bot_mask, ~_top_mask & ~_bot_mask, "box")

        else:  # 히스토그램
            def _srh_sync_bs(): st.session_state["sum_rank_hist_bins_i"] = st.session_state["sum_rank_hist_bins"]
            def _srh_sync_bi(): st.session_state["sum_rank_hist_bins"]   = st.session_state["sum_rank_hist_bins_i"]
            def _srh_sync_ts(): st.session_state["sum_rank_hist_tail_i"] = st.session_state["sum_rank_hist_tail"]
            def _srh_sync_ti(): st.session_state["sum_rank_hist_tail"]   = st.session_state["sum_rank_hist_tail_i"]
            _b1, _b2 = st.columns([3, 1])
            with _b1:
                st.slider("Bins", 5, 200, value=st.session_state["sum_rank_hist_bins"], step=5,
                          key="sum_rank_hist_bins", on_change=_srh_sync_bs)
            with _b2:
                st.number_input("Bins", 5, 200, value=st.session_state["sum_rank_hist_bins_i"], step=5,
                                key="sum_rank_hist_bins_i", label_visibility="hidden", on_change=_srh_sync_bi)
            _t1, _t2 = st.columns([3, 1])
            with _t1:
                st.slider("Tail %", 1, 50, value=st.session_state["sum_rank_hist_tail"], step=1,
                          key="sum_rank_hist_tail", on_change=_srh_sync_ts)
            with _t2:
                st.number_input("Tail %", 1, 50, value=st.session_state["sum_rank_hist_tail_i"], step=1,
                                key="sum_rank_hist_tail_i", label_visibility="hidden", on_change=_srh_sync_ti)
            _h_bins = int(st.session_state.get("sum_rank_hist_bins", 50))
            _h_tail = int(st.session_state.get("sum_rank_hist_tail", 20))
            _lo_u, _hi_u = _sel_series.quantile([_h_tail / 100, 1 - _h_tail / 100])
            _plot_hist(_sel_series, _h_bins, float(_lo_u), float(_hi_u),
                       f"{_cat_sel} 비용 분포 (원)", tail_pct=_h_tail, key="sum_rank_hist")
            _top_tail_m = _sel_series >= _hi_u
            _bot_tail_m = _sel_series <= _lo_u
            _rank_tables(merged, _top_tail_m, _bot_tail_m, ~_top_tail_m & ~_bot_tail_m, "hist")

    # ═══════════════════════════ 유틸리티 구성 ════════════════════════════════
    with tab_mix:
        _mix_total = merged["util_total"].sum()
        _mix_w_pct = merged["water_total"].sum() / _mix_total * 100
        _mix_hw_pct = merged["hw_total"].sum() / _mix_total * 100
        _mix_el_pct = merged["elec_total"].sum() / _mix_total * 100
        _mix_dom = max([("수도", _mix_w_pct), ("온수", _mix_hw_pct), ("전기", _mix_el_pct)], key=lambda x: x[1])
        _elec_heavy = merged[merged["util_total"] > 0].copy()
        _elec_heavy["_ep"] = _elec_heavy["elec_total"] / _elec_heavy["util_total"]
        _top_elec = _elec_heavy.loc[_elec_heavy["_ep"].idxmax()]
        with st.expander("이 탭 설명"):
            st.markdown("""
**유틸리티 비용이 수도·온수·전기 중 어디에 집중되어 있나요?**

전체 및 브랜드별 유틸리티 항목 구성 비율을 분석하는 탭입니다.

- **도넛 차트**: 건물 전체 기준으로 수도·온수·전기가 총비용에서 차지하는 비중을 보여줍니다. 특정 항목이 지나치게 높다면 해당 설비 점검이 필요할 수 있습니다.
- **버블 스캐터**: 브랜드별 수도 비중(X)과 전기 비중(Y)을 동시에 비교합니다. 버블 크기는 총비용에 비례합니다. 우상단에 위치할수록 수도·전기 모두 높은 고비용 브랜드입니다.
- **요약 테이블**: 비용 상위 20개 브랜드의 항목별 청구 금액을 한눈에 확인할 수 있습니다.
""")
        with st.expander("현재 데이터 해석"):
            _mix_water_dom = _mix_w_pct > 40
            _mix_elec_dom  = _mix_el_pct > 50
            _mix_hw_dom    = _mix_hw_pct > 40
            st.markdown(f"""
#### 항목별 비중
- **{_mix_dom[0]}** 이 전체 유틸리티 비용의 가장 큰 비중을 차지합니다 ({_mix_dom[1]:.1f}%)
- 수도 **{_mix_w_pct:.1f}%** · 온수 **{_mix_hw_pct:.1f}%** · 전기 **{_mix_el_pct:.1f}%**

#### 설비 관리 포인트
{"- ⚠️ **전기 비중 과다** — 냉난방 설비 효율 점검 또는 전력 집약 업종 현황 확인 권장" if _mix_elec_dom else "- ✅ 전기 비중 정상 범위"}
{"- ⚠️ **수도 비중 과다** — 누수·고사용 업종(식음료 등) 집중 여부 확인 권장" if _mix_water_dom else "- ✅ 수도 비중 정상 범위"}
{"- ⚠️ **온수 비중 과다** — 난방·급탕 사용 패턴 점검 권장" if _mix_hw_dom else ""}

#### 브랜드별 의존도
- **전기 의존도 최고**: {_top_elec['brand']} — 유틸리티 비용의 **{_top_elec['_ep']*100:.1f}%** 가 전기요금
  {"→ 단독 전기 다소비 가능성, 계량기 점검 권장" if _top_elec['_ep'] > 0.7 else "→ 업종 특성상 일반적인 수준일 수 있음"}
""")
        _mv1, _mv2 = st.columns(2)
        with _mv1:
            # Overall donut
            _dvals = {"수도": merged["water_total"].sum(),
                      "온수": merged["hw_total"].sum(),
                      "전기": merged["elec_total"].sum()}
            fig_d = go.Figure(go.Pie(
                labels=list(_dvals.keys()), values=list(_dvals.values()), hole=0.45,
                marker=dict(colors=["#4C72B0","#C44E52","#DD8A00"]),
                textinfo="label+percent+value", textfont=dict(size=13),
            ))
            fig_d.update_layout(title="전체 유틸리티 비중", height=380,
                                margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_d, use_container_width=True, key="sum_mix_donut")

        with _mv2:
            # Per-brand mix: who is dominated by electricity vs water?
            merged_mix = merged[merged["util_total"]>0].copy()
            merged_mix["elec_pct"] = merged_mix["elec_total"] / merged_mix["util_total"] * 100
            merged_mix["water_pct"] = merged_mix["water_total"] / merged_mix["util_total"] * 100
            merged_mix["hw_pct"]   = merged_mix["hw_total"] / merged_mix["util_total"] * 100

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
                    customdata=sub[["brand","util_total","hw_pct"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "수도 %{x:.1f}%  전기 %{y:.1f}%  온수 %{customdata[2]:.1f}%<br>"
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
            st.plotly_chart(fig_mix, use_container_width=True, key="sum_mix_scatter")

        # Summary table: top 20 by util_total
        _tbl_cols = ["brand","building","floor","water_total","hw_total","elec_total","util_total"]
        _tbl = merged[_tbl_cols].head(20).copy()
        _tbl["water_total"] = _tbl["water_total"].apply(lambda v: f"{v:,.0f}")
        _tbl["hw_total"]    = _tbl["hw_total"].apply(lambda v: f"{v:,.0f}")
        _tbl["elec_total"]  = _tbl["elec_total"].apply(lambda v: f"{v:,.0f}")
        _tbl["util_total"]  = _tbl["util_total"].apply(lambda v: f"{v:,.0f}")
        _tbl = _tbl.rename(columns={"brand":"브랜드","building":"건물","floor":"층",
                                     "water_total":"수도 (원)","hw_total":"온수 (원)",
                                     "elec_total":"전기 (원)","util_total":"합계 (원)"})
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

    # ═══════════════════════════ 면적당 총비용 ════════════════════════════════
    with tab_area:
        _df_a = merged[merged["size_m2"] > 0].copy()
        _df_a["total_pm2"] = (_df_a["util_total"]  / _df_a["size_m2"]).round(0)
        _df_a["water_pm2"] = (_df_a["water_total"] / _df_a["size_m2"]).round(0)
        _df_a["hw_pm2"]    = (_df_a["hw_total"]    / _df_a["size_m2"]).round(0)
        _df_a["elec_pm2"]  = (_df_a["elec_total"]  / _df_a["size_m2"]).round(0)

        with st.expander("이 탭 설명"):
            st.markdown("""
**면적 대비 유틸리티 비용이 적정한가요?**

브랜드별 임대 면적(㎡)당 유틸리티 비용을 비교하는 탭입니다. 단순 총액이 아닌 면적 보정 지표이므로 규모가 다른 브랜드 간 공정한 비교가 가능합니다.

- **분석 기준**: 전체(합산) 또는 수도·온수·전기 개별 항목 선택 가능
- **순위 차트**: 선택 기준의 원/㎡ 순위. 전체 선택 시 항목별 누적 막대.
- **IQR 상한선(보라)**: 통계적 정상 범위 상단. 초과 브랜드는 에너지 감사 또는 누수 점검 대상.
- **중앙값(초록)**: 전체 중간값 기준선.
- **현재 데이터 해석**: 선택 기준 + 전체 항목별 요약이 함께 표시됩니다.
""")

        # ── Category selector ────────────────────────────────────────────────
        _AREA_CAT_META = [
            ("전체",    "total_pm2",  None,      "합계 (원/㎡)"),
            ("💧 수도", "water_pm2",  "#4C72B0", "수도 (원/㎡)"),
            ("🌡️ 온수", "hw_pm2",    "#C44E52", "온수 (원/㎡)"),
            ("⚡ 전기", "elec_pm2",   "#DD8A00", "전기 (원/㎡)"),
        ]
        _area_avail = [lbl for lbl, col, _, _ in _AREA_CAT_META if _df_a[col].sum() > 0]
        _area_cat = st.radio("분석 기준", _area_avail, horizontal=True, key="sum_area_cat")
        _asel_col = next(col for lbl, col, _, _ in _AREA_CAT_META if lbl == _area_cat)
        _asel_clr = next(clr for lbl, col, clr, _ in _AREA_CAT_META if lbl == _area_cat)

        _sf = _df_a[_asel_col]
        _f_up_sel  = _iqr_upper(_sf[_sf > 0]) if (_sf > 0).sum() >= 4 else float("inf")
        _aq1s, _aq3s = _sf.quantile(0.25), _sf.quantile(0.75)
        _a_iqr_s   = _aq3s - _aq1s
        _f_lo_w    = float(_aq1s - 1.5 * _a_iqr_s)
        _f_hi_w    = float(_aq3s + 1.5 * _a_iqr_s)
        _f_lo      = max(0.0, _f_lo_w)

        # ── Metrics row ──────────────────────────────────────────────────────
        ac = st.columns(4)
        ac[0].metric("중앙값",    f"{_sf.median():,.0f} 원/㎡")
        ac[1].metric("평균",      f"{_sf.mean():,.0f} 원/㎡")
        ac[2].metric("IQR 상한",  f"{_f_up_sel:,.0f} 원/㎡" if _f_up_sel < float("inf") else "—")
        ac[3].metric("상한 초과", f"{int((_sf > _f_up_sel).sum())}개")

        # ── 현재 데이터 해석 — all categories always shown ──────────────────
        def _area_cat_summary(col: str, label: str, icon: str) -> str:
            s = _df_a[col]
            if s.sum() == 0:
                return ""
            _pos = s[s > 0]
            _up  = _iqr_upper(_pos) if len(_pos) >= 4 else float("inf")
            _n_out = int((s > _up).sum())
            _med = s.median()
            _avg = s.mean()
            _top = _df_a.loc[s.idxmax()]
            _bot = _df_a.loc[s.idxmin()]
            _skew = "⚠️ 평균>중앙값 (고비용 브랜드 견인)" if _avg > _med * 1.1 else "✅ 분포 균등"
            lines = [
                f"#### {icon} {label}",
                f"- 중앙값 **{_med:,.0f} 원/㎡** · 평균 **{_avg:,.0f} 원/㎡** — {_skew}",
                f"- 최고: **{_top['brand']}** {_top[col]:,.0f} 원/㎡"
                + (f" (중앙값의 {_top[col]/_med:.1f}배)  ⚠️ IQR 초과" if _up < float("inf") and _top[col] > _up else ""),
                f"- 최저: **{_bot['brand']}** {_bot[col]:,.0f} 원/㎡"
                + (" → ⚠️ 미계량 의심" if _f_lo > 0 and _bot[col] < _f_lo else ""),
                f"- IQR 상한 초과: **{_n_out}개**" + (" — 누수·과소비·계량 오류 점검 권장" if _n_out > 0 else " — 없음"),
            ]
            return "\n".join(lines)

        with st.expander("현재 데이터 해석"):
            _lines_area = []
            for _col, _lbl, _ico in [
                ("total_pm2","전체 합산","📊"),
                ("water_pm2","수도","💧"),
                ("hw_pm2","온수","🌡️"),
                ("elec_pm2","전기","⚡"),
            ]:
                _s = _area_cat_summary(_col, _lbl, _ico)
                if _s:
                    _lines_area.append(_s)
            st.markdown("\n\n".join(_lines_area))

        st.divider()
        for _k, _v in [("sum_area_hist_bins", 50), ("sum_area_hist_bins_i", 50),
                       ("sum_area_hist_tail", 20), ("sum_area_hist_tail_i", 20)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v
        _area_view = st.radio(
            "그래프 보기", ["순위 차트", "박스플롯", "히스토그램"],
            horizontal=True, key="sum_area_view",
        )

        _AREA_DISP_COLS = ["brand", "total_pm2", "water_pm2", "hw_pm2", "elec_pm2", "building", "floor", "size_m2"]
        _AREA_COL_CFG = {
            "brand":      st.column_config.TextColumn("브랜드"),
            "total_pm2":  st.column_config.NumberColumn("합계 원/㎡",  format="%,.0f"),
            "water_pm2":  st.column_config.NumberColumn("수도 원/㎡",  format="%,.0f"),
            "hw_pm2":     st.column_config.NumberColumn("온수 원/㎡",  format="%,.0f"),
            "elec_pm2":   st.column_config.NumberColumn("전기 원/㎡",  format="%,.0f"),
            "building":   st.column_config.TextColumn("건물", width="small"),
            "floor":      st.column_config.TextColumn("층",   width="small"),
            "size_m2":    st.column_config.NumberColumn("면적(㎡)", format="%,.1f"),
        }

        def _area_tables(mask_top, mask_bot):
            _t = _df_a[mask_top].sort_values(_asel_col, ascending=False)
            _b = _df_a[mask_bot].sort_values(_asel_col, ascending=False)
            _m = _df_a[~mask_top & ~mask_bot].sort_values(_asel_col, ascending=False)
            st.markdown(f"**▲ 상위 이상치** — 수염 상한 초과 ({len(_t)}개)")
            if not _t.empty:
                st.dataframe(_t[_AREA_DISP_COLS].reset_index(drop=True),
                             column_config=_AREA_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**▼ 하위 이상치** — 수염 하한 미만 ({len(_b)}개)")
            if not _b.empty:
                st.dataframe(_b[_AREA_DISP_COLS].reset_index(drop=True),
                             column_config=_AREA_COL_CFG, use_container_width=True, hide_index=True)
            else:
                st.caption("해당 없음")
            st.markdown(f"**정상 범위** ({len(_m)}개)")
            if not _m.empty:
                st.dataframe(_m[_AREA_DISP_COLS].reset_index(drop=True),
                             column_config=_AREA_COL_CFG, use_container_width=True, hide_index=True)

        if _area_view == "순위 차트":
            _n_a = st.slider("상위 N개", 1, len(_df_a), min(20, len(_df_a)), key="sum_area_n")
            _top_a = _df_a.nlargest(_n_a, _asel_col).sort_values(_asel_col, ascending=True)
            fig_ab = go.Figure()
            if _area_cat == "전체":
                for _lbl, _col, _clr in [("수도","water_pm2","#4C72B0"),
                                           ("온수","hw_pm2","#C44E52"),
                                           ("전기","elec_pm2","#DD8A00")]:
                    fig_ab.add_trace(go.Bar(
                        x=_top_a[_col].values, y=[str(b)[:26] for b in _top_a["brand"]],
                        name=_lbl, orientation="h", marker_color=_clr,
                        text=[f"{v:,.0f}" if v > 0 else "" for v in _top_a[_col].values],
                        textposition="inside", textfont=dict(size=9, color="white"),
                        hovertemplate=f"<b>%{{y}}</b><br>{_lbl}: %{{x:,.0f}} 원/㎡<extra></extra>",
                    ))
                _barmode = "stack"
            else:
                fig_ab.add_trace(go.Bar(
                    x=_top_a[_asel_col].values, y=[str(b)[:26] for b in _top_a["brand"]],
                    name=_area_cat, orientation="h", marker_color=_asel_clr,
                    text=[f"{v:,.0f}" if v > 0 else "" for v in _top_a[_asel_col].values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                    hovertemplate=f"<b>%{{y}}</b><br>{_area_cat}: %{{x:,.0f}} 원/㎡<extra></extra>",
                ))
                _barmode = "relative"
            if _f_up_sel < float("inf"):
                fig_ab.add_vline(x=_f_up_sel, line_dash="dash", line_color="#8B2BE2", line_width=2)
                fig_ab.add_annotation(
                    x=_f_up_sel, y=1, yref="paper",
                    text=f"⚠ IQR 상한 {_f_up_sel:,.0f} 원/㎡",
                    showarrow=False, xanchor="left", xshift=6,
                    font=dict(size=11, color="#8B2BE2"),
                    bgcolor="rgba(255,255,255,0.6)", bordercolor="#8B2BE2", borderwidth=1,
                )
            fig_ab.add_vline(x=float(_sf.median()), line_dash="dash", line_color="#2CA02C", line_width=2)
            fig_ab.add_annotation(
                x=float(_sf.median()), y=1, yref="paper",
                text=f"중앙값 {_sf.median():,.0f} 원/㎡",
                showarrow=False, xanchor="right", xshift=-6,
                font=dict(size=11, color="#2CA02C"),
                bgcolor="rgba(255,255,255,0.6)", bordercolor="#2CA02C", borderwidth=1,
            )
            fig_ab.update_layout(
                barmode=_barmode,
                title=f"면적당 비용 — {_area_cat} (상위 {_n_a}개, 원/㎡)",
                height=max(420, _n_a * 22 + 80), xaxis_title="원/㎡",
                xaxis=dict(griddash="dot"),
                legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle"),
                margin=dict(l=10, r=100, t=30, b=40),
            )
            st.plotly_chart(fig_ab, use_container_width=True, key="sum_area_stacked")
            _rank_a = _df_a.nlargest(_n_a, _asel_col).sort_values(_asel_col, ascending=False).reset_index(drop=True)
            _rank_a.index = _rank_a.index + 1
            st.dataframe(_rank_a[_AREA_DISP_COLS], column_config=_AREA_COL_CFG, use_container_width=True)

        elif _area_view == "박스플롯":
            _boxplot_with_labels(_sf, _df_a["brand"], f"{_area_cat} 면적당 비용 (원/㎡)", "sum_area_box")
            _area_tables(_sf >= _f_hi_w, _sf <= _f_lo_w)

        else:  # 히스토그램
            def _sah_sync_bs(): st.session_state["sum_area_hist_bins_i"] = st.session_state["sum_area_hist_bins"]
            def _sah_sync_bi(): st.session_state["sum_area_hist_bins"]   = st.session_state["sum_area_hist_bins_i"]
            def _sah_sync_ts(): st.session_state["sum_area_hist_tail_i"] = st.session_state["sum_area_hist_tail"]
            def _sah_sync_ti(): st.session_state["sum_area_hist_tail"]   = st.session_state["sum_area_hist_tail_i"]
            _b1, _b2 = st.columns([3, 1])
            with _b1:
                st.slider("Bins", 5, 200, value=st.session_state["sum_area_hist_bins"], step=5,
                          key="sum_area_hist_bins", on_change=_sah_sync_bs)
            with _b2:
                st.number_input("Bins", 5, 200, value=st.session_state["sum_area_hist_bins_i"], step=5,
                                key="sum_area_hist_bins_i", label_visibility="hidden", on_change=_sah_sync_bi)
            _t1, _t2 = st.columns([3, 1])
            with _t1:
                st.slider("Tail %", 1, 50, value=st.session_state["sum_area_hist_tail"], step=1,
                          key="sum_area_hist_tail", on_change=_sah_sync_ts)
            with _t2:
                st.number_input("Tail %", 1, 50, value=st.session_state["sum_area_hist_tail_i"], step=1,
                                key="sum_area_hist_tail_i", label_visibility="hidden", on_change=_sah_sync_ti)
            _ah_bins = int(st.session_state.get("sum_area_hist_bins", 50))
            _ah_tail = int(st.session_state.get("sum_area_hist_tail", 20))
            _lo_a, _hi_a = _sf.quantile([_ah_tail / 100, 1 - _ah_tail / 100])
            _plot_hist(_sf, _ah_bins, float(_lo_a), float(_hi_a),
                       f"{_area_cat} 면적당 비용 분포 (원/㎡)", tail_pct=_ah_tail, key="sum_area_hist")
            _area_tables(_sf >= _hi_a, _sf <= _lo_a)

    # ═══════════════════════════ 건물별 비교 ══════════════════════════════════
    with tab_bld:
        _bld_pre = merged.groupby("building").agg(
            util=("util_total","sum"), area=("size_m2","sum"), cnt=("brand","count")
        ).reindex(["A","B","C","D"]).dropna(how="all")
        _bld_pre["pm2"] = _bld_pre["util"] / _bld_pre["area"].replace(0, float("nan"))
        _bld_max_total = _bld_pre["util"].idxmax()
        _bld_max_pm2   = _bld_pre["pm2"].idxmax()
        with st.expander("이 탭 설명"):
            st.markdown("""
**건물별로 유틸리티 비용 부담이 어떻게 다른가요?**

A·B·C·D 건물별 유틸리티 총비용과 면적 효율을 비교하는 탭입니다.

- **누적 막대 차트**: 건물별 수도·온수·전기 총비용을 항목별로 쌓아서 보여줍니다. 특정 건물의 비용이 두드러지게 높다면 해당 건물의 설비 상태나 입점 구성을 점검할 필요가 있습니다.
- **원/㎡ 차트**: 건물별 면적당 유틸리티 비용입니다. 총액이 아닌 효율 기준이므로 규모 차이를 제거하고 비교할 수 있습니다.
- **요약 테이블**: 건물별 브랜드 수, 항목별 금액, 총면적, 면적당 비용을 종합한 표입니다.
""")
        with st.expander("현재 데이터 해석"):
            _bld_pm2_med = _bld_pre['pm2'].median()
            _bld_util_total = _bld_pre['util'].sum()
            st.markdown(f"""
#### 건물별 요약
{chr(10).join(f"- **{idx}동**: 총 {row['util']/1e4:,.0f}만 원 ({row['util']/_bld_util_total*100:.0f}%) · {row['pm2']:,.0f} 원/㎡ · {int(row['cnt'])}개 브랜드" for idx, row in _bld_pre.iterrows() if pd.notna(row['util']))}

#### 주요 포인트
- **총비용 최고**: **{_bld_max_total}동** — {_bld_pre.loc[_bld_max_total, 'util']/1e4:,.0f}만 원 · 전체의 {_bld_pre.loc[_bld_max_total, 'util']/_bld_util_total*100:.0f}%
- **면적 효율 최고**: **{_bld_max_pm2}동** — {_bld_pre.loc[_bld_max_pm2, 'pm2']:,.0f} 원/㎡ {"⚠️ 중앙값({:.0f} 원/㎡) 대비 30% 이상 높습니다. 해당 건물 설비·입점 구성 점검 권장".format(_bld_pm2_med) if _bld_pre.loc[_bld_max_pm2, 'pm2'] > _bld_pm2_med * 1.3 else "→ 건물 간 효율 차이 허용 범위 내"}

#### 의사결정 가이드
{"- ⚠️ 특정 건물에 비용이 집중되어 있습니다. 설비 점검 및 입점 구성 재검토가 필요합니다." if _bld_pre['util'].max() / _bld_pre['util'].min() > 3 else "- ✅ 건물 간 비용 분포가 비교적 균형 잡혀 있습니다."}
- 면적당 비용이 가장 낮은 건물은 **{_bld_pre['pm2'].idxmin()}동** ({_bld_pre['pm2'].min():,.0f} 원/㎡) — 운영 효율 벤치마크로 참고 가능
""")
        _bld_agg = merged.groupby("building").agg(
            브랜드수=("brand","count"),
            수도=("water_total","sum"),
            온수=("hw_total","sum"),
            전기=("elec_total","sum"),
            합계=("util_total","sum"),
            총면적=("size_m2","sum"),
        ).reindex(["A","B","C","D"]).dropna(how="all").reset_index()
        _bld_agg["원/m²"] = (_bld_agg["합계"] / _bld_agg["총면적"]).round(0)

        # Stacked bar by building
        fig_bld = go.Figure()
        for label, col, clr in [("수도","수도","#4C72B0"),("온수","온수","#C44E52"),("전기","전기","#DD8A00")]:
            fig_bld.add_trace(go.Bar(
                x=[r["building"]+"동" for _,r in _bld_agg.iterrows()],
                y=[r[col] for _,r in _bld_agg.iterrows()],
                name=label, marker_color=clr,
                text=[f"{r[col]/1e6:.2f}M" for _,r in _bld_agg.iterrows()],
                textposition="inside", textfont=dict(size=11,color="white"),
            ))
        fig_bld.update_layout(
            barmode="stack", title="건물별 유틸리티 총비용 구성", height=380,
            plot_bgcolor="white", yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            legend=dict(orientation="h", x=0, y=1.08, xanchor="left", yanchor="bottom"),
            margin=dict(l=10, r=10, t=100, b=30),
        )
        st.plotly_chart(fig_bld, use_container_width=True, key="sum_bld_stacked")

        # 원/m² bar
        fig_bpm2 = go.Figure()
        for _,row in _bld_agg.iterrows():
            fig_bpm2.add_trace(go.Bar(
                x=[row["building"]+"동"], y=[row["원/m²"]],
                marker_color=_BLD_COLOR.get(row["building"],"#888"),
                text=[f"{row['원/m²']:,.0f}"], textposition="outside",
                textfont=dict(size=11), showlegend=False,
            ))
        fig_bpm2.update_layout(
            title="건물별 면적당 유틸리티 비용 (원/m²)", height=300,
            plot_bgcolor="white", yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            margin=dict(l=10,r=10,t=50,b=30),
        )
        st.plotly_chart(fig_bpm2, use_container_width=True, key="sum_bld_pm2")

        # Summary table
        _bld_disp = _bld_agg.copy()
        for col in ["수도","온수","전기","합계"]:
            _bld_disp[col] = _bld_disp[col].apply(lambda v: f"{v/1e6:.2f}M")
        _bld_disp["총면적"]  = _bld_disp["총면적"].apply(lambda v: f"{v:,.0f} m²")
        _bld_disp["원/m²"]  = _bld_disp["원/m²"].apply(lambda v: f"{v:,.0f}")
        _bld_disp["building"] = _bld_disp["building"] + "동"
        _bld_disp = _bld_disp.rename(columns={"building":"건물","브랜드수":"브랜드",
                                               "수도":"수도 (원)","온수":"온수 (원)",
                                               "전기":"전기 (원)","합계":"합계 (원)"})
        st.dataframe(_bld_disp, use_container_width=True, hide_index=True)

    # ═══════════════════════════ 항목별 분석 ════════════════════════════════════
    with tab_cat:
        with st.expander("이 탭 설명"):
            st.markdown("""
**각 유틸리티 내부의 비용이 어떻게 구성되어 있나요?**

수도·전기·온수 각각의 항목별 세부 내역을 브랜드 기준으로 분석합니다.

- **수도**: 상수도 / 하수도 / 부과금 + 전용/공용 분리
- **전기**: 비용 — 전용 / EHP / 공용 분리 | 사용량(kWh) — 일반전기 / HVAC(FCU·AHU·EHP) / 기타(펌프·환풍기)
- **온수**: 전용 / 공용 청구 분리

각 항목의 비중을 파악하면 설비 유형별 비용 드라이버를 특정하고 절감 레버를 찾는 데 도움이 됩니다.
""")

        with st.expander("현재 데이터 해석"):
            _interp_lines = []
            # 수도 insights
            if water_df is not None and not water_df.empty:
                _wi = water_df.groupby("brand").agg(
                    **{c: (c, "sum") for c in ["water_excl","water_comm","sewage_excl","sewage_comm",
                                                "levy_excl","levy_comm","total_excl","total_comm","usage_m3"]
                       if c in water_df.columns}
                ).reset_index()
                _w_total = (
                    (_wi["total_excl"].sum() if "total_excl" in _wi.columns else 0) +
                    (_wi["total_comm"].sum() if "total_comm" in _wi.columns else 0)
                )
                _w_comm = _wi["total_comm"].sum() if "total_comm" in _wi.columns else 0
                _w_comm_pct = _w_comm / _w_total * 100 if _w_total > 0 else 0
                _w_sewage = ((_wi["sewage_excl"].sum() if "sewage_excl" in _wi.columns else 0) +
                             (_wi["sewage_comm"].sum() if "sewage_comm" in _wi.columns else 0))
                _w_sewage_pct = _w_sewage / _w_total * 100 if _w_total > 0 else 0
                _w_levy = ((_wi["levy_excl"].sum() if "levy_excl" in _wi.columns else 0) +
                           (_wi["levy_comm"].sum() if "levy_comm" in _wi.columns else 0))
                _w_levy_pct = _w_levy / _w_total * 100 if _w_total > 0 else 0
                _interp_lines.append(f"""#### 💧 수도
- 공용 분담 비중 **{_w_comm_pct:.1f}%** ({_fmt_won(_w_comm)}) {"— ⚠️ 공용 비중이 30% 초과. 공용 구역 누수 또는 배분 기준 검토 권장" if _w_comm_pct > 30 else "— ✅ 공용 비중 정상"}
- 하수도 비중 **{_w_sewage_pct:.1f}%** · 부과금 비중 **{_w_levy_pct:.1f}%**
  {"→ ⚠️ 부과금 비중 과다 — 지자체 부과 내역 확인 필요" if _w_levy_pct > 15 else ""}""")

            # 전기 insights
            if elec_df is not None and not elec_df.empty:
                _ei = elec_df.groupby("brand").agg(
                    **{c: (c, "sum") for c in ["excl_total","ehp_total","comm_total",
                                               "kwh_elec01","kwh_elec02","kwh_fcu","kwh_ahu",
                                               "kwh_ehp","kwh_pump","kwh_kitchen_fan"]
                       if c in elec_df.columns}
                ).reset_index()
                _e_cost_total = sum(_ei[c].sum() for c in ["excl_total","ehp_total","comm_total"] if c in _ei.columns)
                _e_ehp_cost = _ei["ehp_total"].sum() if "ehp_total" in _ei.columns else 0
                _e_comm_cost = _ei["comm_total"].sum() if "comm_total" in _ei.columns else 0
                _e_ehp_pct = _e_ehp_cost / _e_cost_total * 100 if _e_cost_total > 0 else 0
                _e_comm_pct = _e_comm_cost / _e_cost_total * 100 if _e_cost_total > 0 else 0
                _kwh_hvac = sum(_ei[c].sum() for c in ["kwh_fcu","kwh_ahu","kwh_ehp"] if c in _ei.columns)
                _kwh_gen  = sum(_ei[c].sum() for c in ["kwh_elec01","kwh_elec02"] if c in _ei.columns)
                _kwh_etc  = sum(_ei[c].sum() for c in ["kwh_pump","kwh_kitchen_fan"] if c in _ei.columns)
                _kwh_total = _kwh_hvac + _kwh_gen + _kwh_etc
                _hvac_kwh_pct = _kwh_hvac / _kwh_total * 100 if _kwh_total > 0 else 0
                _top_ehp = _ei.loc[_ei["ehp_total"].idxmax()]["brand"] if "ehp_total" in _ei.columns else "—"
                _top_hvac_kwh = _ei.loc[(_ei[["kwh_fcu","kwh_ahu","kwh_ehp"]].sum(axis=1)).idxmax()]["brand"] if all(c in _ei.columns for c in ["kwh_fcu","kwh_ahu","kwh_ehp"]) else "—"
                _interp_lines.append(f"""#### ⚡ 전기
- EHP 비용 비중 **{_e_ehp_pct:.1f}%** ({_fmt_won(_e_ehp_cost)}) | 공용 전기 비중 **{_e_comm_pct:.1f}%**
  {"→ ⚠️ EHP 비중 과다 — 냉난방 설비 효율 점검 또는 고HVAC 업종 집중 확인" if _e_ehp_pct > 30 else "→ ✅ EHP 비중 정상"}
- HVAC 사용량 비중 **{_hvac_kwh_pct:.1f}%** (FCU·AHU·EHP 합산) | 일반전기 **{_kwh_gen/1e3:,.0f}k kWh** · HVAC **{_kwh_hvac/1e3:,.0f}k kWh**
- EHP 최다 비용 브랜드: **{_top_ehp}** | HVAC kWh 최다: **{_top_hvac_kwh}**""")

            # 온수 insights
            if hotwater_df is not None and not hotwater_df.empty:
                _hwi = hotwater_df.groupby("brand").agg(
                    **{c: (c, "sum") for c in ["fee_excl","fee_comm","total","usage_m3"] if c in hotwater_df.columns}
                ).reset_index()
                _hw_total = _hwi["total"].sum() if "total" in _hwi.columns else 0
                _hw_comm  = _hwi["fee_comm"].sum() if "fee_comm" in _hwi.columns else 0
                _hw_comm_pct = _hw_comm / _hw_total * 100 if _hw_total > 0 else 0
                _interp_lines.append(f"""#### 🌡️ 온수
- 공용 비중 **{_hw_comm_pct:.1f}%** ({_fmt_won(_hw_comm)})
  {"→ ⚠️ 공용 온수 비중이 예상보다 높습니다. 배관 손실 또는 공용 설비 사용 현황 확인 권장" if _hw_comm_pct > 20 else "→ ✅ 공용 온수 비중 정상"}""")

            if _interp_lines:
                st.markdown("\n\n".join(_interp_lines))
            else:
                st.info("분석 가능한 항목 데이터가 없습니다.")

        def _stacked_bar(df_agg: pd.DataFrame, brand_col: str, layers: list[tuple],
                         title: str, y_label: str, chart_key: str, top_n: int = 25) -> None:
            """Generic horizontal stacked bar: layers = [(label, col, color)]."""
            _layer_cols = [col for _, col, _ in layers]
            _df = df_agg.copy()
            _df["__total__"] = _df[_layer_cols].sum(axis=1)
            _plot_df = _df.nlargest(top_n, "__total__").sort_values("__total__", ascending=True)
            fig = go.Figure()
            for lbl, col, clr in layers:
                fig.add_trace(go.Bar(
                    y=_plot_df[brand_col], x=_plot_df[col],
                    name=lbl, marker_color=clr, orientation="h",
                    hovertemplate=f"<b>%{{y}}</b><br>{lbl}: %{{x:,.0f}}<extra></extra>",
                ))
            fig.update_layout(
                barmode="stack", title=title, height=max(300, min(top_n * 22, 600)),
                xaxis_title=y_label,
                legend=dict(orientation="h", x=0, y=1.04, xanchor="left", yanchor="bottom"),
                margin=dict(l=10, r=10, t=80, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # ── 수도 ─────────────────────────────────────────────────────────────
        if water_df is not None and not water_df.empty:
            st.subheader("💧 수도 — 항목별 분류")
            _agg_cols = {"building": ("building", "first")}
            for _c in ["usage_m3", "water_excl", "water_comm", "sewage_excl", "sewage_comm",
                       "levy_excl", "levy_comm", "total_excl", "total_comm"]:
                if _c in water_df.columns:
                    _agg_cols[_c] = (_c, "sum")
            _cat_w = water_df.groupby("brand").agg(**_agg_cols).reset_index()

            _has_water_comm = "water_comm" in _cat_w.columns
            if "water_excl" in _cat_w.columns:
                _cat_w["상수도"] = _cat_w["water_excl"] + (_cat_w["water_comm"] if _has_water_comm else 0)
            if "sewage_excl" in _cat_w.columns:
                _cat_w["하수도"] = _cat_w["sewage_excl"] + (_cat_w["sewage_comm"] if "sewage_comm" in _cat_w.columns else 0)
            if "levy_excl" in _cat_w.columns:
                _cat_w["부과금"] = _cat_w["levy_excl"] + (_cat_w["levy_comm"] if "levy_comm" in _cat_w.columns else 0)
            _cat_w["합계"] = sum(
                _cat_w[c] for c in ["상수도", "하수도", "부과금"] if c in _cat_w.columns
            )
            _cat_w = _cat_w[_cat_w["합계"] > 0]

            _wlayers_cat = [(lbl, lbl, clr) for lbl, clr in
                            [("상수도", "#4C72B0"), ("하수도", "#8172B3"), ("부과금", "#76B7B2")]
                            if lbl in _cat_w.columns]
            if _wlayers_cat:
                _stacked_bar(_cat_w, "brand", _wlayers_cat,
                             "수도 항목별 비용 (상위 25)", "원", "cat_water_items")

            # 전용 vs 공용
            if "total_excl" in _cat_w.columns and "total_comm" in _cat_w.columns:
                _wlayers_ec = [("전용", "total_excl", "#4C72B0"), ("공용", "total_comm", "#76B7B2")]
                _stacked_bar(_cat_w, "brand", _wlayers_ec,
                             "수도 전용 / 공용 분리 (상위 25)", "원", "cat_water_excl_comm")

            # Key insight
            if "total_comm" in _cat_w.columns and "합계" in _cat_w.columns:
                _wc_comm_rate = _cat_w["total_comm"].sum() / _cat_w["합계"].sum() * 100
                _wc_top_comm = _cat_w.loc[_cat_w["total_comm"].idxmax()]
                st.caption(
                    f"공용 수도 비중 **{_wc_comm_rate:.1f}%** — "
                    f"공용 최다 부담 브랜드: **{_wc_top_comm['brand']}** "
                    f"({_wc_top_comm['total_comm']/1e4:,.0f}만 원)"
                )

        # ── 전기 ─────────────────────────────────────────────────────────────
        if elec_df is not None and not elec_df.empty:
            st.subheader("⚡ 전기 — 항목별 분류")

            _ecols = {}
            for _c in ["excl_total", "ehp_total", "comm_total",
                       "kwh_elec01", "kwh_elec02", "kwh_fcu", "kwh_ahu",
                       "kwh_ehp", "kwh_pump", "kwh_kitchen_fan", "kwh_total"]:
                if _c in elec_df.columns:
                    _ecols[_c] = (_c, "sum")
            _cat_e = elec_df.groupby("brand").agg(
                building=("building", "first"), **_ecols
            ).reset_index()

            # Cost breakdown: 전용 / EHP / 공용
            _e_cost_layers = [(lbl, col, clr) for lbl, col, clr in [
                ("전용 전기", "excl_total", "#DD8A00"),
                ("EHP",      "ehp_total",  "#C44E52"),
                ("공용 전기", "comm_total", "#8172B3"),
            ] if col in _cat_e.columns]
            if _e_cost_layers:
                _cat_e["_ecost_total"] = sum(_cat_e[col] for _, col, _ in _e_cost_layers)
                _cat_e_cost = _cat_e[_cat_e["_ecost_total"] > 0].copy()
                _stacked_bar(_cat_e_cost, "brand", _e_cost_layers,
                             "전기 비용 구성 — 전용 / EHP / 공용 (상위 25)", "원", "cat_elec_cost")

            # KWH breakdown: 일반 / HVAC / 기타
            _kwh_layers: list[tuple] = []
            def _sum_cols(df, cols):
                present = [c for c in cols if c in df.columns]
                return df[present].sum(axis=1) if present else pd.Series(0, index=df.index)

            _cat_e["kwh_일반"] = _sum_cols(_cat_e, ["kwh_elec01", "kwh_elec02"])
            _cat_e["kwh_HVAC"] = _sum_cols(_cat_e, ["kwh_fcu", "kwh_ahu", "kwh_ehp"])
            _cat_e["kwh_기타"] = _sum_cols(_cat_e, ["kwh_pump", "kwh_kitchen_fan"])
            _cat_e["_kwh_sum"] = _cat_e["kwh_일반"] + _cat_e["kwh_HVAC"] + _cat_e["kwh_기타"]
            for _lbl, _clr in [("kwh_일반", "#DD8A00"), ("kwh_HVAC", "#C44E52"), ("kwh_기타", "#76B7B2")]:
                if _cat_e[_lbl].sum() > 0:
                    _kwh_layers.append((_lbl.replace("kwh_", ""), _lbl, _clr))
            if _kwh_layers:
                _cat_e_kwh = _cat_e[_cat_e["_kwh_sum"] > 0].copy()
                _stacked_bar(_cat_e_kwh, "brand", _kwh_layers,
                             "전기 사용량 구성 — 일반 / HVAC / 기타 (상위 25, kWh)", "kWh",
                             "cat_elec_kwh")

            # Key insight
            if "kwh_HVAC" in _cat_e.columns and "_kwh_sum" in _cat_e.columns:
                _e_hvac_sum = _cat_e["kwh_HVAC"].sum()
                _e_total_sum = _cat_e["_kwh_sum"].sum()
                if _e_total_sum > 0:
                    _hvac_rate = _e_hvac_sum / _e_total_sum * 100
                    _top_hvac = _cat_e.loc[_cat_e["kwh_HVAC"].idxmax()]
                    st.caption(
                        f"전체 전기 사용량 중 HVAC 비중 **{_hvac_rate:.1f}%** — "
                        f"HVAC 최다 브랜드: **{_top_hvac['brand']}** "
                        f"({_top_hvac['kwh_HVAC']:,.0f} kWh)"
                    )

        # ── 온수 ─────────────────────────────────────────────────────────────
        if hotwater_df is not None and not hotwater_df.empty:
            st.subheader("🌡️ 온수 — 전용 / 공용 분리")
            _hwcols = {}
            for _c in ["usage_m3", "fee_excl", "fee_comm", "total"]:
                if _c in hotwater_df.columns:
                    _hwcols[_c] = (_c, "sum")
            _cat_hw = hotwater_df.groupby("brand").agg(
                building=("building", "first"), **_hwcols
            ).reset_index()

            _hw_layers = [(lbl, col, clr) for lbl, col, clr in [
                ("전용", "fee_excl", "#C44E52"),
                ("공용", "fee_comm", "#E4A0A0"),
            ] if col in _cat_hw.columns]
            if _hw_layers:
                _cat_hw["_hw_total"] = sum(_cat_hw[col] for _, col, _ in _hw_layers)
                _cat_hw_plot = _cat_hw[_cat_hw["_hw_total"] > 0].copy()
                _stacked_bar(_cat_hw_plot, "brand", _hw_layers,
                             "온수 전용 / 공용 비용 (상위 25)", "원", "cat_hw_excl_comm")

            if "fee_comm" in _cat_hw.columns and "total" in _cat_hw.columns:
                _hw_comm_total = _cat_hw["fee_comm"].sum()
                _hw_total = _cat_hw["total"].sum()
                if _hw_total > 0:
                    st.caption(
                        f"온수 공용 비중 **{_hw_comm_total/_hw_total*100:.1f}%** "
                        f"({_hw_comm_total/1e4:,.0f}만 원)"
                    )

    # ═══════════════════════════ 경영 보고 ════════════════════════════════════
    with tab_mgmt:
        st.subheader("수익성 분석 — 경영 보고")
        st.caption("미청구 손실 추정, 이상 징후 브랜드, 비용 집중도를 종합한 경영용 보고서입니다.")
        with st.expander("이 탭 설명"):
            st.markdown("""
**유틸리티 관리에서 즉시 조치가 필요한 사안은 무엇인가요?**

경영진이 빠르게 의사결정을 내릴 수 있도록 핵심 리스크를 정리한 탭입니다.

- **미계량 브랜드**: 수도·온수·전기 중 하나 이상의 사용량이 0으로 기록된 브랜드입니다. 업종 특성상 정상일 수 있으나, 전 항목이 0인 경우 미터 설치 또는 계약 상태를 반드시 확인하세요.
- **이상 징후 브랜드**: 총 유틸리티 비용이 통계적 정상 범위(IQR 상한)를 초과한 브랜드입니다. 누수·설비 과부하·계량 오류 가능성이 있습니다.
- **우선 조치 테이블**: 이상치·전체 미계량·저납부 여부를 종합한 우선순위 점수로 정렬됩니다. 🔴 즉시 → 🟠 검토 → 🟡 관찰 → 🟢 정상 순서로 관리하세요.
- **비용 집중도(Pareto)**: 상위 몇 개 브랜드가 전체 유틸리티 비용의 대부분을 차지하는지 보여줍니다. 80% 기준선을 참고해 집중 관리 대상을 선정하세요.
""")

        # ── Leakage computation per sheet ──────────────────────────────────────
        _w_per_brand,  _w_total_leak  = _leakage_for(water_df,    "usage_m3",  "total")    if water_df    is not None else ({}, 0.0)
        _hw_per_brand, _hw_total_leak = _leakage_for(hotwater_df, "usage_m3",  "total")    if hotwater_df is not None else ({}, 0.0)
        _el_per_brand, _el_total_leak = _leakage_for(elec_df,     "kwh_total", "grand_total") if elec_df     is not None else ({}, 0.0)
        _total_leakage = _w_total_leak + _hw_total_leak + _el_total_leak

        # ── IQR anomaly count ──────────────────────────────────────────────────
        _util_up = _iqr_upper(merged["util_total"])
        _n_anomaly = int((merged["util_total"] > _util_up).sum())

        # ── Top-5 cost concentration ───────────────────────────────────────────
        _total_spend = merged["util_total"].sum()
        _top5_pct = merged.nlargest(5, "util_total")["util_total"].sum() / _total_spend * 100 if _total_spend else 0

        # ── Unmetered summary ──────────────────────────────────────────────────
        _all_unmet_brands = set(_w_per_brand) | set(_hw_per_brand) | set(_el_per_brand)
        # Brands unmetered across ALL three sheets are the most suspicious
        _all3_unmet = set(_w_per_brand) & set(_hw_per_brand) & set(_el_per_brand)

        # ── Data insight expander ──────────────────────────────────────────────
        _mgmt_top1_brand = merged.loc[merged["util_total"].idxmax(), "brand"]
        _mgmt_top1_val   = merged["util_total"].max()
        _mgmt_top5       = merged.nlargest(5, "util_total")
        _mgmt_top5_share = _mgmt_top5["util_total"].sum() / _total_spend * 100 if _total_spend else 0
        with st.expander("현재 데이터 해석"):
            st.markdown(f"""
#### 즉각 조치
- **고비용 이상치**: {"없음 — 현재 정상 범위 내 ✅" if _n_anomaly == 0 else f"⚠️ **{_n_anomaly}개** 브랜드가 IQR 상한({_util_up/1e4:,.0f}만 원) 초과 → 누수·설비 과부하·계량 오류 여부 점검"}
- **전 항목 미계량**: {"없음 ✅" if not _all3_unmet else f"⚠️ **{len(_all3_unmet)}개** 브랜드 수도·온수·전기 모두 미계량 ({', '.join(sorted(_all3_unmet)[:5])}{'…' if len(_all3_unmet)>5 else ''}) → 계약 및 미터 설치 즉시 확인"}

#### 비용 집중도 & 리스크
- 상위 5개 브랜드가 전체의 **{_mgmt_top5_share:.1f}%** 차지 ({', '.join(_mgmt_top5['brand'].astype(str).str[:10].tolist())})
{"- ⚠️ 집중도 과다 — 해당 브랜드 이탈 시 수익 구조에 큰 영향. 계약 조건 재검토 권장" if _mgmt_top5_share > 50 else "- ✅ 비용 집중도 분산 — 특정 브랜드 의존도 낮음"}
- **최고 비용 브랜드**: {_mgmt_top1_brand} — {_mgmt_top1_val/1e4:,.0f}만 원 (전체의 **{_mgmt_top1_val/_total_spend*100:.1f}%**)

#### 운영 효율
- 전체 브랜드 수: {len(merged)}개 · 총 유틸리티 지출: {_fmt_won(_total_spend)}
- 브랜드당 평균: {_fmt_won(_total_spend/len(merged))} · 미계량 포함 브랜드: {len(_all_unmet_brands)}개
""")

        # ── Cross-tab synthesis ─────────────────────────────────────────────────
        _es_util_total  = merged["util_total"].sum()
        _es_n           = len(merged)
        _es_avg         = _es_util_total / _es_n
        _es_med         = merged["util_total"].median()
        _es_top3_share  = merged.nlargest(3, "util_total")["util_total"].sum() / _es_util_total * 100
        _es_top10p_n    = max(1, int(_es_n * 0.1))
        _es_top10p_share = merged.nlargest(_es_top10p_n, "util_total")["util_total"].sum() / _es_util_total * 100
        _es_outlier_burden = merged[merged["util_total"] > _util_up]["util_total"].sum() / _es_util_total * 100 if _n_anomaly > 0 else 0.0
        _es_skew        = _es_avg > _es_med * 1.1

        # Area efficiency (recompute safe)
        _es_adf = merged[merged["size_m2"] > 0].copy()
        _es_adf["pm2"] = _es_adf["util_total"] / _es_adf["size_m2"]
        _es_a_iqr_up   = _iqr_upper(_es_adf["pm2"])
        _es_n_area_over = int((_es_adf["pm2"] > _es_a_iqr_up).sum())
        _es_area_top1  = _es_adf.loc[_es_adf["pm2"].idxmax()] if not _es_adf.empty else None

        # Building
        _es_bld = merged.groupby("building").agg(
            util=("util_total","sum"), cnt=("brand","count"), area=("size_m2","sum")
        ).reindex(["A","B","C","D"]).dropna(how="all")
        _es_bld["pm2"] = _es_bld["util"] / _es_bld["area"].replace(0, float("nan"))
        _es_bld_max    = _es_bld["util"].idxmax() if not _es_bld.empty else "-"
        _es_bld_eff    = _es_bld["pm2"].idxmin() if _es_bld["pm2"].notna().any() else "-"
        _es_bld_ratio  = _es_bld["util"].max() / _es_bld["util"].min() if _es_bld["util"].min() > 0 else 1

        # Utility mix
        _es_w_pct  = merged["water_total"].sum() / _es_util_total * 100
        _es_hw_pct = merged["hw_total"].sum()    / _es_util_total * 100
        _es_el_pct = merged["elec_total"].sum()  / _es_util_total * 100
        _es_dom    = max([("수도", _es_w_pct), ("온수", _es_hw_pct), ("전기", _es_el_pct)], key=lambda x: x[1])

        # Build prioritized findings
        _findings, _actions, _strategy = [], [], []

        if _n_anomaly > 0:
            _findings.append(f"고비용 이상치 **{_n_anomaly}개** 브랜드가 전체 유틸리티 비용의 **{_es_outlier_burden:.1f}%** 를 점유합니다.")
            _actions.append(f"🔴 **즉시** — 이상치 {_n_anomaly}개 브랜드 계량기·누수·설비 과부하 현장 점검")
        if _all3_unmet:
            _findings.append(f"수도·온수·전기 **전부** 미계량 브랜드가 **{len(_all3_unmet)}개** 있습니다. 미터 미설치 또는 계약 누락 가능성이 있습니다.")
            _actions.append(f"🔴 **즉시** — 전 항목 미계량 {len(_all3_unmet)}개 브랜드 계약서·미터 설치 현황 확인")
        if _es_n_area_over > 0:
            _top_pm2_brand = _es_area_top1["brand"] if _es_area_top1 is not None else "-"
            _findings.append(f"면적당 비용 이상 고점 **{_es_n_area_over}개** 브랜드 — 대표: {_top_pm2_brand} ({_es_area_top1['pm2']:,.0f} 원/㎡). 동일 면적 대비 과소비 수준입니다.")
            _actions.append(f"🟠 **검토** — 면적당 비용 상위 이상치 {_es_n_area_over}개 브랜드 사용 패턴 및 설비 상태 점검")
        if len(_all_unmet_brands) > 0:
            _findings.append(f"부분 미계량(수도·온수·전기 중 일부 누락) 브랜드가 **{len(_all_unmet_brands)}개** 있습니다. 업종 특성일 수 있으나 확인이 필요합니다.")
            _actions.append(f"🟡 **관찰** — 부분 미계량 {len(_all_unmet_brands)}개 브랜드 업종 특성 검토 후 계량 불필요 여부 판단")
        if _es_top10p_share > 50:
            _findings.append(f"상위 10% ({_es_top10p_n}개 브랜드)가 전체 비용의 **{_es_top10p_share:.1f}%** 를 차지합니다. 소수 브랜드 의존도가 높습니다.")
            _strategy.append(f"비용 집중 브랜드의 임대 계약 갱신 조건 재검토 — 이탈 시 수익 구조 충격 최소화 방안 마련")
        if _es_el_pct > 50:
            _findings.append(f"전기 비중이 **{_es_el_pct:.1f}%** 로 절반 이상을 차지합니다. 냉난방 설비 효율이 핵심 관리 포인트입니다.")
            _strategy.append("전력 다소비 구간 피크 저감 프로그램 도입 또는 LED·인버터 설비 교체 검토")
        if _es_skew:
            _findings.append(f"평균({_fmt_won(_es_avg)})이 중앙값({_fmt_won(_es_med)})보다 높아 소수 고비용 브랜드가 전체 평균을 왜곡하고 있습니다.")
        if _es_bld_ratio > 2.5:
            _findings.append(f"건물별 비용 격차가 크며({_es_bld_max}동 최고), **{_es_bld_eff}동** 이 면적당 비용 최저로 효율 벤치마크가 됩니다.")
            _strategy.append(f"{_es_bld_eff}동 운영 방식을 타 건물에 벤치마킹하여 면적당 비용 절감 목표 수립 권장")

        if not _findings:
            _findings.append("현재 데이터 기준으로 즉각 조치가 필요한 이상 징후는 발견되지 않았습니다.")

        st.markdown("### 📋 종합 경영 요약")
        st.markdown(f"분석 대상 **{_es_n}개 브랜드** · 총 유틸리티 지출 **{_fmt_won(_es_util_total)}** · 지배 항목 **{_es_dom[0]} ({_es_dom[1]:.0f}%)**")
        st.divider()

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            st.markdown("#### 🔍 주요 발견사항")
            for i, f in enumerate(_findings, 1):
                st.markdown(f"{i}. {f}")

        with fc2:
            st.markdown("#### ✅ 조치 항목 (우선순위순)")
            if _actions:
                for a in _actions:
                    st.markdown(f"- {a}")
            else:
                st.markdown("- 즉각 조치 항목 없음")
            if _strategy:
                st.markdown("#### 💡 전략적 제언")
                for s in _strategy:
                    st.markdown(f"- {s}")

        st.divider()

        # ── KPI row ────────────────────────────────────────────────────────────
        kc = st.columns(4)
        kc[0].metric("미계량 브랜드 (검토 필요)",
                     f"{len(_all_unmet_brands)}개",
                     help="수도/온수/전기 중 하나 이상 미계량. 업종 특성상 정상일 수 있습니다.")
        kc[1].metric("전 유틸리티 미계량",
                     f"{len(_all3_unmet)}개",
                     help="수도·온수·전기 모두 미계량 — 계약·미터 점검 권장")
        kc[2].metric("이상 징후 브랜드",
                     f"{_n_anomaly}개",
                     help="총 유틸리티 비용 IQR 상한 초과")
        kc[3].metric("상위 5개 비중",
                     f"{_top5_pct:.1f}%",
                     help="전체 유틸리티 지출에서 상위 5개 브랜드 비중")

        if _all3_unmet:
            st.markdown(f"#### 전체 미계량 브랜드 ({len(_all3_unmet)}개) — 계약·미터 즉시 확인")
            _unmet_rows = []
            for b in sorted(_all3_unmet):
                _row = merged[merged["brand"] == b]
                _bld  = _row["building"].iloc[0] if not _row.empty else "-"
                _flr  = str(_row["floor"].iloc[0]) if not _row.empty else "-"
                _util = int(_row["util_total"].iloc[0]) if not _row.empty else 0
                _unmet_rows.append({
                    "브랜드":        b,
                    "건물":          _bld,
                    "층":            _flr,
                    "총비용 (만원)": round(_util / 1e4, 1),
                })
            st.dataframe(
                pd.DataFrame(_unmet_rows),
                column_config={
                    "브랜드":        st.column_config.TextColumn("브랜드"),
                    "건물":          st.column_config.TextColumn("건물", width="small"),
                    "층":            st.column_config.TextColumn("층",   width="small"),
                    "총비용 (만원)": st.column_config.NumberColumn("총비용 (만원)", format="%.1f"),
                },
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── 공실 이상 소비 분석 ────────────────────────────────────────────────
        _vacancy_df = merged[merged["brand"].astype(str).str.contains("공실", na=False)].copy()
        if not _vacancy_df.empty:
            st.markdown(f"#### 🏚 공실 유틸리티 이상 소비 분석 ({len(_vacancy_df)}개 공실)")
            st.markdown("""
| 등급 | 기준 | 의심 원인 | 권고 조치 |
|------|------|-----------|-----------|
| 🔴 고의심 | 합계 ≥ 전체 중앙값 × 30% | 무단점거 · 대형 누수 | 즉시 현장 확인 |
| 🟠 주의   | 합계 > 0, 중앙값 × 30% 미만 | 소규모 누수 · 계량기 오류 · 배관 역류 | 점검 권장 |
| 🟢 정상   | 합계 = 0 | — | 이상 없음 |
""")
            _global_med = merged["util_total"].median()
            _vac_rows = []
            for _, vrow in _vacancy_df.iterrows():
                _w  = float(vrow.get("water_total", 0) or 0)
                _hw = float(vrow.get("hw_total",    0) or 0)
                _el = float(vrow.get("elec_total",  0) or 0)
                _tot = float(vrow["util_total"])
                _consuming = _tot > 0
                # Classify suspicion level
                if _tot > _global_med * 0.3:
                    _suspicion = "🔴 고의심 — 무단점거·대형 누수 가능성"
                elif _tot > 0:
                    _suspicion = "🟠 주의 — 소량 소비, 누수·계량기 오류 확인 필요"
                else:
                    _suspicion = "🟢 정상 — 소비 없음"
                _vac_rows.append({
                    "공실":          str(vrow["brand"]),
                    "건물":          str(vrow.get("building", "-")),
                    "층":            str(vrow.get("floor", "-")),
                    "수도 (만원)":   round(_w  / 1e4, 1),
                    "온수 (만원)":   round(_hw / 1e4, 1),
                    "전기 (만원)":   round(_el / 1e4, 1),
                    "합계 (만원)":   round(_tot / 1e4, 1),
                    "의심 수준":     _suspicion,
                })
            _vac_df = pd.DataFrame(_vac_rows).sort_values("합계 (만원)", ascending=False)
            _vac_consuming = (_vac_df["합계 (만원)"] > 0).sum()
            if _vac_consuming > 0:
                st.warning(f"⚠️ {_vac_consuming}개 공실에서 유틸리티 소비가 감지되었습니다. 무단점거·누수·계량기 오류 여부를 즉시 확인하세요.")
            else:
                st.success("✅ 모든 공실의 유틸리티 소비가 0입니다.")
            st.dataframe(
                _vac_df,
                column_config={
                    "공실":        st.column_config.TextColumn("공실"),
                    "건물":        st.column_config.TextColumn("건물", width="small"),
                    "층":          st.column_config.TextColumn("층",   width="small"),
                    "수도 (만원)": st.column_config.NumberColumn("수도 (만원)",  format="%.1f"),
                    "온수 (만원)": st.column_config.NumberColumn("온수 (만원)",  format="%.1f"),
                    "전기 (만원)": st.column_config.NumberColumn("전기 (만원)",  format="%.1f"),
                    "합계 (만원)": st.column_config.NumberColumn("합계 (만원)",  format="%.1f"),
                    "의심 수준":   st.column_config.TextColumn("의심 수준"),
                },
                use_container_width=True,
                hide_index=True,
            )
            st.divider()

        # ── Priority action table ──────────────────────────────────────────────
        st.markdown("#### 우선 조치 브랜드")
        st.markdown("""
| 등급 | 점수 | 구성 조건 | 의미 | 권고 조치 |
|------|------|-----------|------|-----------|
| 🔴 즉시 | 4점 이상 | 이상치 + 전체미계량 등 복합 | 복수의 심각한 이상 징후 동시 발생 | 즉시 현장 점검 및 계약 검토 |
| 🟠 검토 | 2–3점 | 이상치(3점) 또는 전체미계량(2점) | 단일 고위험 이상 징후 | 1개월 내 점검 예약 |
| 🟡 관찰 | 1점 | 저납부(1점) — 동일 건물 중앙값의 40% 미만 | 미계량·업종 특성 가능성, 저사용 의심 | 업종 확인 후 필요 시 점검 |
| 🟢 정상 | 0점 | 해당 없음 | 통계적 정상 범위 | 정기 모니터링 유지 |

> **점수 산식**: 이상치(IQR 상한 초과) ×3 · 전체미계량(수도·온수·전기 모두 0) ×2 · 저납부(동 건물 중앙값의 40% 미만) ×1
> 미계량은 업종 특성상 정상일 수 있으므로 단독으로 해석하지 마세요.
""")

        _action_rows = []
        for _, row in merged.iterrows():
            b = str(row["brand"])
            _is_anom = bool(row["util_total"] > _util_up)
            _w_u  = b in _w_per_brand
            _hw_u = b in _hw_per_brand
            _el_u = b in _el_per_brand
            _unmet_cnt = int(_w_u) + int(_hw_u) + int(_el_u)
            _all3 = _w_u and _hw_u and _el_u  # missing from ALL sheets — more suspicious
            _pm2 = row["util_total"] / row["size_m2"] if row.get("size_m2", 0) > 0 else np.nan
            _score = (_is_anom * 3) + (_all3 * 2)

            # Per-building median util/m² for underpay flag
            _bld = str(row.get("building", ""))
            _bld_sub = merged[merged["building"] == _bld]
            _bld_med_pm2 = (
                (_bld_sub["util_total"] / _bld_sub["size_m2"].replace(0, np.nan)).median()
                if not _bld_sub.empty else np.nan
            )
            _underpay = (pd.notna(_pm2) and pd.notna(_bld_med_pm2) and _bld_med_pm2 > 0
                         and _pm2 < _bld_med_pm2 * 0.4)
            if _underpay:
                _score += 1

            # Build plain-language reason
            _reasons = []
            if _is_anom:
                _mult = row["util_total"] / _util_up if _util_up > 0 else 0
                _reasons.append(f"총비용이 IQR 상한의 {_mult:.1f}배 — 누수·설비 과부하·계량 오류 의심")
            if _all3:
                _reasons.append("수도·온수·전기 전부 미계량 — 계약 누락 또는 미터 미설치 가능성")
            elif _w_u or _hw_u or _el_u:
                _items = ("수도" if _w_u else "") + ("·온수" if _hw_u else "") + ("·전기" if _el_u else "")
                _reasons.append(f"{_items.strip('·')} 미계량 — 업종 특성 또는 미터 오류 확인 필요")
            if _underpay and pd.notna(_bld_med_pm2):
                _ratio = _pm2 / _bld_med_pm2 * 100 if _bld_med_pm2 > 0 else 0
                _reasons.append(f"동 건물 중앙값의 {_ratio:.0f}% 수준 — 미계량·장기 휴업 가능성")

            _action_rows.append({
                "브랜드":        b,
                "건물":          _bld,
                "층":            str(row.get("floor", "")),
                "총비용 (만원)": round(row["util_total"] / 1e4, 1),
                "원/m²":         round(_pm2, 0) if pd.notna(_pm2) else None,
                "건물중앙 원/m²": round(_bld_med_pm2, 0) if pd.notna(_bld_med_pm2) else None,
                "주의 사유":     " / ".join(_reasons) if _reasons else "-",
                "우선순위 점수": _score,
            })

        _action_df = (
            pd.DataFrame(_action_rows)
            .sort_values(["우선순위 점수", "총비용 (만원)"], ascending=[False, False])
            .reset_index(drop=True)
        )
        _action_df.index = _action_df.index + 1  # 1-based rank
        _action_df.insert(0, "등급", _action_df["우선순위 점수"].map(
            lambda s: "🔴 즉시" if s >= 4 else ("🟠 검토" if s >= 2 else ("🟡 관찰" if s == 1 else "🟢 정상"))
        ))

        st.dataframe(
            _action_df,
            column_config={
                "등급":           st.column_config.TextColumn("등급", width="small"),
                "브랜드":         st.column_config.TextColumn("브랜드"),
                "건물":           st.column_config.TextColumn("건물", width="small"),
                "층":             st.column_config.TextColumn("층", width="small"),
                "총비용 (만원)":  st.column_config.NumberColumn("총비용 (만원)", format="%.1f"),
                "원/m²":          st.column_config.NumberColumn("원/m²", format="%,.0f"),
                "건물중앙 원/m²": st.column_config.NumberColumn("건물중앙 원/m²", format="%,.0f"),
                "주의 사유":      st.column_config.TextColumn("주의 사유", width="large"),
                "우선순위 점수":  st.column_config.ProgressColumn("우선순위", format="%d", min_value=0, max_value=6),
            },
            use_container_width=True,
        )

        # ── Download management report ─────────────────────────────────────────
        import io
        _buf = io.BytesIO()
        with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
            _action_df.to_excel(_xw, sheet_name="우선조치목록", index=True)
            # Unmetered brand detail per sheet (for manual review — not assumed leakage)
            for _sheet_label, _per_brand in [("수도_미계량", _w_per_brand),
                                              ("온수_미계량", _hw_per_brand),
                                              ("전기_미계량", _el_per_brand)]:
                if _per_brand:
                    _detail = pd.DataFrame(
                        [{"브랜드": k} for k in _per_brand]
                    )
                    _detail.to_excel(_xw, sheet_name=_sheet_label, index=False)
        st.download_button(
            "📥 경영 보고서 다운로드 (Excel)",
            data=_buf.getvalue(),
            file_name="utility_management_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.divider()

        # ── Cost concentration (Pareto) ────────────────────────────────────────
        st.markdown("#### 비용 집중도 (Pareto)")
        _sorted = merged.sort_values("util_total", ascending=False).reset_index(drop=True)
        _sorted["누적 비중 (%)"] = (_sorted["util_total"].cumsum() / _total_spend * 100).round(1)
        _sorted["순위"] = _sorted.index + 1

        _pareto_n = st.slider("표시 브랜드 수", 5, min(50, len(_sorted)), min(20, len(_sorted)), key="sum_pareto_n")
        _pareto = _sorted.head(_pareto_n)

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=[str(b)[:22] for b in _pareto["brand"]],
            y=_pareto["util_total"].values,
            marker_color=(
                [_BLD_COLOR.get(str(b), "#888") for b in _pareto["building"]]
                if split_by_building else "#4C72B0"
            ),
            text=[f"{v/1e6:.2f}M" for v in _pareto["util_total"].values],
            textposition="outside", textfont=dict(size=9),
            name="유틸리티 합계",
        ))
        fig_p.add_trace(go.Scatter(
            x=[str(b)[:22] for b in _pareto["brand"]],
            y=_pareto["누적 비중 (%)"].values,
            mode="lines+markers", name="누적 비중 (%)",
            yaxis="y2", line=dict(color="#C44E52", width=2),
            marker=dict(size=5),
        ))
        fig_p.update_layout(
            height=400, plot_bgcolor="white",
            yaxis=dict(title="원", gridcolor="#DDDDDD", griddash="dot"),
            yaxis2=dict(title="누적 비중 (%)", overlaying="y", side="right",
                        range=[0, 105], showgrid=False),
            xaxis=dict(tickangle=-40),
            legend=dict(orientation="h", x=1, y=1, xanchor="right", yanchor="top"),
            margin=dict(l=10, r=60, t=50, b=100),
        )
        st.plotly_chart(fig_p, use_container_width=True, key="sum_pareto_chart")

        _p80 = int((_sorted["누적 비중 (%)"] <= 80).sum()) + 1
        st.caption(
            f"상위 **{_p80}개** 브랜드가 전체 비용의 80%를 차지합니다. "
            f"(전체 {len(merged)}개 브랜드 중 {_p80/len(merged)*100:.0f}%)"
        )
