"""water.py — 수도 사용 내역 analysis view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_BLD_COLOR = {"A": "#4C72B0", "B": "#55A868", "C": "#C44E52", "D": "#DD8A00"}

_COMP_COLS = [
    ("상수도 전용",     "water_excl",  "#4C72B0"),
    ("하수도 전용",     "sewage_excl", "#C44E52"),
    ("분담금 전용",     "levy_excl",   "#8172B2"),
    ("공용합계",        "total_comm",  "#9EBADF"),
]


def _iqr_upper(s: pd.Series) -> float:
    s = s.dropna()
    s = s[s > 0]
    if len(s) < 4:
        return float("inf")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def render_water_view(df: pd.DataFrame) -> None:
    st.header("💧 수도 사용 내역 분석")

    # ── Building filter ────────────────────────────────────────────────────────
    buildings = sorted(df["building"].unique())
    sel_bld = st.multiselect(
        "건물 선택", ["All"] + buildings, default=["All"], key="water_bld"
    )
    if "All" not in sel_bld and sel_bld:
        df = df[df["building"].isin(sel_bld)].copy()

    if df.empty:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
        return

    n_total    = len(df)
    n_metered  = int((df["usage_m3"] > 0).sum())
    n_unmeterd = n_total - n_metered

    # ── Summary metrics ────────────────────────────────────────────────────────
    mc = st.columns(5)
    mc[0].metric("총 브랜드",    f"{n_total}개")
    mc[1].metric("총 부과금액",  f"{df['total'].sum()/1e6:.2f}M 원")
    _excl_pct = df["total_excl"].sum() / df["total"].sum() * 100 if df["total"].sum() else 0
    mc[2].metric("전용 부과",    f"{df['total_excl'].sum()/1e6:.2f}M ({_excl_pct:.0f}%)")
    mc[3].metric("총 사용량",    f"{int(df['usage_m3'].sum()):,} m³")
    mc[4].metric("계량 브랜드",  f"{n_metered} / {n_total}")

    # ── Anomaly flags (shared across all tabs) ────────────────────────────────
    _flags = pd.DataFrame(index=df["brand"].values)
    _flag_cols: list[str] = []

    _total_up = _iqr_upper(df["total"])
    _flags["총부과 이상치"]    = (df["total"].values > _total_up)
    _flag_cols.append("총부과 이상치")

    _cpm2     = df["total"] / df["size_m2"].replace(0, np.nan)
    _cpm2_up  = _iqr_upper(_cpm2)
    _flags["면적당비용 이상치"] = (_cpm2.values > _cpm2_up)
    _flag_cols.append("면적당비용 이상치")

    _metered_mask = df["usage_m3"] > 0
    _upm2    = (df["usage_m3"] / df["size_m2"].replace(0, np.nan)).where(_metered_mask)
    _upm2_up = _iqr_upper(_upm2.dropna())
    _flags["사용량/m² 이상치"]  = (_upm2.values > _upm2_up)
    _flag_cols.append("사용량/m² 이상치")

    _flags.index = df["brand"].values
    _flags["플래그 수"] = _flags[_flag_cols].sum(axis=1).astype(int)
    _flags["등급"] = _flags["플래그 수"].map(
        lambda n: "🔴 위험" if n >= 2 else ("🟠 주의" if n == 1 else "🟢 정상")
    )

    def _pfx(brand: str) -> str:
        if brand not in _flags.index:
            return ""
        val = _flags.loc[brand, "플래그 수"]
        n = int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)
        return "⛔ " if n >= 2 else ("⚠ " if n == 1 else "")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_rank, tab_comp, tab_fair, tab_usage, tab_anom = st.tabs(
        ["순위", "비중", "면적당 비용", "사용량", "이상 탐지"]
    )

    # ═══════════════════════════ 순위 ═════════════════════════════════════════
    with tab_rank:
        _metric = st.radio(
            "기준", ["총부과", "전용부과", "공용부과", "사용량 (m³)"],
            horizontal=True, key="water_rank_metric"
        )
        _col  = {"총부과": "total", "전용부과": "total_excl",
                 "공용부과": "total_comm", "사용량 (m³)": "usage_m3"}[_metric]
        _unit = "원" if _metric != "사용량 (m³)" else "m³"

        _df_r = df[["brand", "building", _col]].sort_values(_col, ascending=True)
        _ylbl = [_pfx(b) + str(b)[:26] for b in _df_r["brand"]]
        _clrs = [_BLD_COLOR.get(b, "#888") for b in _df_r["building"]]

        fig_r = go.Figure()
        for bld, clr in _BLD_COLOR.items():
            fig_r.add_trace(go.Bar(x=[None], y=[None], name=f"{bld}동",
                                   marker_color=clr, orientation="h"))
        fig_r.add_trace(go.Bar(
            x=_df_r[_col].values, y=_ylbl, orientation="h",
            marker_color=_clrs,
            text=[f"{v:,.0f}" for v in _df_r[_col].values],
            textposition="outside", textfont=dict(size=10),
            showlegend=False,
        ))

        # IQR upper line
        _r_up = _iqr_upper(df[_col])
        if _r_up < float("inf"):
            fig_r.add_vline(x=_r_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_r_up:,.0f}",
                            annotation_position="top left",
                            annotation_font_size=10)

        fig_r.update_layout(
            height=max(420, len(_df_r) * 22 + 80),
            margin=dict(l=10, r=130, t=40, b=40),
            xaxis_title=_unit,
            barmode="overlay",
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="water_rank_chart")

        _s = df[_col]
        sc = st.columns(5)
        sc[0].metric("합계",     f"{_s.sum():,.0f} {_unit}")
        sc[1].metric("평균",     f"{_s.mean():,.0f} {_unit}")
        sc[2].metric("중앙값",   f"{_s.median():,.0f} {_unit}")
        sc[3].metric("최대",     f"{_s.max():,.0f} {_unit}")
        sc[4].metric("1위",      df.loc[df[_col].idxmax(), "brand"])

    # ═══════════════════════════ 비중 ═════════════════════════════════════════
    with tab_comp:
        _cview = st.radio("보기", ["브랜드별 stacked", "전체 항목 donut"],
                          horizontal=True, key="water_comp_view")

        if _cview.startswith("브"):
            _n_show = st.slider("상위 N개", 10, min(60, n_total), min(30, n_total),
                                key="water_comp_n")
            _top = df.nlargest(_n_show, "total").sort_values("total", ascending=True)
            _cy  = [_pfx(b) + str(b)[:26] for b in _top["brand"]]

            fig_c = go.Figure()
            for label, col, clr in _COMP_COLS:
                fig_c.add_trace(go.Bar(
                    x=_top[col].values, y=_cy, name=label, orientation="h",
                    marker_color=clr,
                    text=[f"{v/1000:.0f}k" if v >= 1000 else "" for v in _top[col].values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                ))
            fig_c.update_layout(
                barmode="stack",
                height=max(420, _n_show * 22 + 80),
                margin=dict(l=10, r=20, t=40, b=40),
                xaxis_title="원",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_c, use_container_width=True, key="water_comp_stacked")

        else:
            _dvals = {
                "상수도 전용":   df["water_excl"].sum(),
                "하수도 전용":   df["sewage_excl"].sum(),
                "분담금 전용":   df["levy_excl"].sum(),
                "상수도 공용":   df["water_comm"].sum(),
                "하수도 공용":   df["sewage_comm"].sum(),
                "분담금 공용":   df["levy_comm"].sum(),
            }
            _dclrs = ["#4C72B0", "#C44E52", "#8172B2", "#9EBADF", "#F28E8E", "#C8B8E8"]
            fig_d = go.Figure(go.Pie(
                labels=list(_dvals.keys()),
                values=list(_dvals.values()),
                hole=0.45,
                marker=dict(colors=_dclrs),
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            fig_d.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_d, use_container_width=True, key="water_comp_donut")

            _tot_all = df["total"].sum()
            _tbl = pd.DataFrame({
                "항목":     list(_dvals.keys()),
                "금액 (원)": [f"{v:,.0f}" for v in _dvals.values()],
                "비중":      [f"{v/_tot_all*100:.1f}%" for v in _dvals.values()],
            })
            st.dataframe(_tbl, use_container_width=True, hide_index=True)

    # ═══════════════════════════ 면적당 비용 ══════════════════════════════════
    with tab_fair:
        _u_toggle = st.radio("면적 단위", ["㎡", "평"], horizontal=True, key="water_fair_unit")
        _a_col    = "size_m2" if _u_toggle == "㎡" else "size_py"

        _df_f = df[["brand", "building", "total", _a_col]].copy()
        _df_f = _df_f[_df_f[_a_col] > 0].copy()
        _df_f["cpa"] = _df_f["total"] / _df_f[_a_col]
        _df_f = _df_f.sort_values("cpa", ascending=True)

        _sf       = _df_f["cpa"]
        _f_up     = _iqr_upper(_sf)
        _q1f, _q3f = _sf.quantile(0.25), _sf.quantile(0.75)
        _f_lo     = max(0.0, float(_q1f - 1.5 * (_q3f - _q1f)))

        _bclr = [_BLD_COLOR.get(b, "#888") for b in _df_f["building"]]
        _bord_clr = [
            "#8B1A1A" if v > _f_up else ("#1A5C2A" if _f_lo > 0 and v < _f_lo else "white")
            for v in _sf.values
        ]
        _bord_w = [2.5 if v > _f_up or (_f_lo > 0 and v < _f_lo) else 0 for v in _sf.values]
        _fy = [_pfx(b) + str(b)[:26] for b in _df_f["brand"]]

        fig_f = go.Figure(go.Bar(
            x=_sf.values, y=_fy, orientation="h",
            marker_color=_bclr,
            marker_line=dict(color=_bord_clr, width=_bord_w),
            text=[f"{v:,.0f}" for v in _sf.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_f.add_vline(x=float(_sf.median()), line_dash="dash", line_color="#C44E52",
                        annotation_text=f"중앙값 {_sf.median():,.0f}",
                        annotation_position="top right", annotation_font_size=10)
        if _f_up < _sf.max() * 5:
            fig_f.add_vline(x=_f_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_f_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_f.update_layout(
            height=max(420, len(_df_f) * 22 + 80),
            xaxis_title=f"원/{_u_toggle}",
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            margin=dict(l=10, r=150, t=40, b=40),
        )
        st.plotly_chart(fig_f, use_container_width=True, key="water_fair_chart")

        fc = st.columns(4)
        fc[0].metric("중앙값",    f"{_sf.median():,.0f} 원/{_u_toggle}")
        fc[1].metric("평균",      f"{_sf.mean():,.0f} 원/{_u_toggle}")
        fc[2].metric("IQR 상한",  f"{_f_up:,.0f} 원/{_u_toggle}")
        fc[3].metric("상한 초과", f"{int((_sf > _f_up).sum())}개")

    # ═══════════════════════════ 사용량 ═══════════════════════════════════════
    with tab_usage:
        st.caption(
            f"계량 브랜드: **{n_metered}개** (직접 측정)  |  "
            f"미계량: **{n_unmeterd}개** (면적 비례 공용 배분만 부과)"
        )
        df_m = df[df["usage_m3"] > 0].copy()
        df_m["cost_per_m3"] = (df_m["total_excl"] / df_m["usage_m3"]).round(0)

        _uview = st.radio(
            "차트", ["사용량 순위", "사용량 vs 면적 산점도", "m³당 단가"],
            horizontal=True, key="water_usage_view"
        )

        if _uview == "사용량 순위":
            _df_u = df_m[["brand", "building", "usage_m3"]].sort_values("usage_m3", ascending=True)
            _uy   = [_pfx(b) + str(b)[:26] for b in _df_u["brand"]]
            _u_up = _iqr_upper(df_m["usage_m3"])

            fig_u = go.Figure(go.Bar(
                x=_df_u["usage_m3"].values, y=_uy, orientation="h",
                marker_color=[_BLD_COLOR.get(b, "#888") for b in _df_u["building"]],
                text=[f"{v:,}" for v in _df_u["usage_m3"].values],
                textposition="outside", textfont=dict(size=10),
            ))
            fig_u.add_vline(x=float(df_m["usage_m3"].median()), line_dash="dash",
                            line_color="#C44E52",
                            annotation_text=f"중앙값 {df_m['usage_m3'].median():.0f}",
                            annotation_font_size=10)
            if _u_up < float("inf"):
                fig_u.add_vline(x=_u_up, line_dash="dot", line_color="#DD8A00",
                                annotation_text=f"IQR 상한 {_u_up:.0f}",
                                annotation_position="top left", annotation_font_size=10)
            fig_u.update_layout(
                height=max(420, len(_df_u) * 22 + 80),
                xaxis_title="m³",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                margin=dict(l=10, r=120, t=40, b=40),
            )
            st.plotly_chart(fig_u, use_container_width=True, key="water_usage_rank")

            uc = st.columns(4)
            uc[0].metric("총 사용량",   f"{int(df_m['usage_m3'].sum()):,} m³")
            uc[1].metric("평균 사용량", f"{df_m['usage_m3'].mean():.0f} m³")
            uc[2].metric("중앙값",      f"{df_m['usage_m3'].median():.0f} m³")
            uc[3].metric("최대 사용",   f"{df_m.loc[df_m['usage_m3'].idxmax(), 'brand']} ({int(df_m['usage_m3'].max()):,} m³)")

        elif _uview == "사용량 vs 면적 산점도":
            fig_s = go.Figure()
            for bld in sorted(df_m["building"].unique()):
                sub = df_m[df_m["building"] == bld]
                fig_s.add_trace(go.Scatter(
                    x=sub["size_m2"], y=sub["usage_m3"],
                    mode="markers+text",
                    name=f"{bld}동",
                    marker=dict(color=_BLD_COLOR.get(bld, "#888"), size=9, opacity=0.8),
                    text=sub["brand"], textposition="top center",
                    textfont=dict(size=8),
                ))
            # OLS trendline
            _xa = df_m["size_m2"].values
            _ya = df_m["usage_m3"].values
            _c  = np.polyfit(_xa, _ya, 1)
            _xf = np.linspace(_xa.min(), _xa.max(), 120)
            fig_s.add_trace(go.Scatter(
                x=_xf, y=np.polyval(_c, _xf), mode="lines",
                name="추세선", line=dict(color="#888", dash="dash", width=1.5),
            ))
            fig_s.update_layout(
                height=520,
                xaxis_title="전용면적 (m²)",
                yaxis_title="사용량 (m³)",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                yaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                margin=dict(l=20, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_s, use_container_width=True, key="water_scatter")
            r2 = np.corrcoef(_xa, _ya)[0, 1] ** 2
            st.caption(f"R² = {r2:.3f}  |  면적으로 사용량의 {r2*100:.1f}%를 설명")

        else:  # m³당 단가
            _df_cpu = df_m[["brand", "building", "cost_per_m3"]].sort_values("cost_per_m3", ascending=True)
            _cpuy   = [str(b)[:26] for b in _df_cpu["brand"]]

            fig_cpu = go.Figure(go.Bar(
                x=_df_cpu["cost_per_m3"].values, y=_cpuy, orientation="h",
                marker_color=[_BLD_COLOR.get(b, "#888") for b in _df_cpu["building"]],
                text=[f"{v:,.0f}" for v in _df_cpu["cost_per_m3"].values],
                textposition="outside", textfont=dict(size=10),
            ))
            fig_cpu.add_vline(x=float(df_m["cost_per_m3"].median()), line_dash="dash",
                              line_color="#C44E52",
                              annotation_text=f"중앙값 {df_m['cost_per_m3'].median():,.0f}",
                              annotation_font_size=10)
            fig_cpu.update_layout(
                height=max(420, len(_df_cpu) * 22 + 80),
                xaxis_title="원/m³",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                margin=dict(l=10, r=120, t=40, b=40),
            )
            st.plotly_chart(fig_cpu, use_container_width=True, key="water_cpu_chart")
            st.caption("※ 전용부과 기준. 대부분 약 3,707원/m³의 단일 공정요금이 적용됩니다.")

    # ═══════════════════════════ 이상 탐지 ════════════════════════════════════
    with tab_anom:
        n_crit  = int((_flags["플래그 수"] >= 2).sum())
        n_watch = int((_flags["플래그 수"] == 1).sum())
        n_ok    = int((_flags["플래그 수"] == 0).sum())

        ac = st.columns(3)
        ac[0].metric("🔴 위험 (≥2 플래그)", f"{n_crit}개")
        ac[1].metric("🟠 주의 (1 플래그)",   f"{n_watch}개")
        ac[2].metric("🟢 정상",              f"{n_ok}개")

        st.divider()

        # Flag heatmap (top 40 by flag count)
        _hm = _flags.sort_values("플래그 수", ascending=False).head(40)
        _hm_brands = list(_hm.index)
        _hm_z    = [[int(_hm.at[b, c]) for c in _flag_cols] for b in _hm_brands]
        _hm_text = [["✓" if _hm.at[b, c] else "" for c in _flag_cols] for b in _hm_brands]

        fig_hm = go.Figure(go.Heatmap(
            z=_hm_z, x=_flag_cols, y=_hm_brands,
            colorscale=[[0, "#F0F0F0"], [1, "#C44E52"]],
            showscale=False,
            text=_hm_text, texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            xgap=3, ygap=3,
        ))
        fig_hm.update_layout(
            height=max(300, len(_hm_brands) * 22 + 80),
            margin=dict(l=10, r=20, t=40, b=40),
            plot_bgcolor="white",
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="water_anom_heatmap")

        # Flagged brand detail table
        _flagged_idx = _flags[_flags["플래그 수"] > 0].index
        if len(_flagged_idx) > 0:
            st.subheader("플래그 상세")
            _detail = df[df["brand"].isin(_flagged_idx)][
                ["brand", "building", "floor", "size_m2", "usage_m3",
                 "total_excl", "total_comm", "total"]
            ].copy()
            _flag_meta = _flags[["플래그 수", "등급"]].reset_index().rename(columns={"index": "brand"})
            _detail = _detail.merge(_flag_meta, on="brand", how="left")
            _detail = _detail.sort_values("플래그 수", ascending=False).reset_index(drop=True)
            st.dataframe(_detail, use_container_width=True, hide_index=True)
