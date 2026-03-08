"""summary.py — Cross-sheet utility summary: water + hotwater + electricity."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_BLD_COLOR = {"A": "#4C72B0", "B": "#55A868", "C": "#C44E52", "D": "#DD8A00"}
_UTIL_COLOR = {"수도": "#4C72B0", "온수": "#C44E52", "전기": "#DD8A00"}


def _iqr_upper(s: pd.Series) -> float:
    s = s.dropna(); s = s[s > 0]
    if len(s) < 4: return float("inf")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def render_summary_view(
    water_df: pd.DataFrame,
    hotwater_df: pd.DataFrame,
    elec_df: pd.DataFrame,
) -> None:
    st.header("📊 통합 유틸리티 분석")
    st.caption("수도 + 온수 + 전기 데이터를 브랜드 기준으로 통합한 종합 분석입니다.")

    # ── Building filter ────────────────────────────────────────────────────────
    all_blds = sorted(set(water_df["building"].unique()) |
                      set(hotwater_df["building"].unique()) |
                      set(elec_df["building"].unique()))
    sel_bld = st.multiselect("건물 선택", ["All"] + all_blds, default=["All"], key="sum_bld")
    if "All" not in sel_bld and sel_bld:
        water_df    = water_df[water_df["brand"].isin(water_df[water_df["building"].isin(sel_bld)]["brand"])].copy()
        hotwater_df = hotwater_df[hotwater_df["building"].isin(sel_bld)].copy()
        elec_df     = elec_df[elec_df["building"].isin(sel_bld)].copy()

    # ── Merge on brand ─────────────────────────────────────────────────────────
    _w  = water_df.groupby("brand").agg(
        building=("building","first"), floor=("floor","first"),
        size_m2=("size_m2","sum"),
        water_total=("total","sum"),
    ).reset_index()
    _hw = hotwater_df.groupby("brand").agg(hw_total=("total","sum")).reset_index()
    _el = elec_df.groupby("brand").agg(elec_total=("grand_total","sum"),
                                        kwh_total=("kwh_total","sum")).reset_index()

    merged = _w.merge(_hw, on="brand", how="outer").merge(_el, on="brand", how="outer")
    # Fill missing utilities with 0 (brand not in that sheet)
    for col in ["water_total","hw_total","elec_total","kwh_total","size_m2"]:
        merged[col] = merged[col].fillna(0)
    # Fill building/floor from any available source
    for df_src in [hotwater_df, elec_df]:
        _bld_map = df_src.groupby("brand")[["building","floor","size_m2"]].first()
        for idx, row in merged.iterrows():
            b = row["brand"]
            if pd.isna(row.get("building")) or row.get("building") == "" and b in _bld_map.index:
                merged.at[idx, "building"] = _bld_map.at[b,"building"]
                merged.at[idx, "floor"]    = _bld_map.at[b,"floor"]
                if merged.at[idx,"size_m2"] == 0:
                    merged.at[idx,"size_m2"] = _bld_map.at[b,"size_m2"]

    merged["util_total"] = merged["water_total"] + merged["hw_total"] + merged["elec_total"]
    merged = merged[merged["util_total"] > 0].sort_values("util_total", ascending=False).reset_index(drop=True)

    # ── Top metrics ────────────────────────────────────────────────────────────
    mc = st.columns(5)
    mc[0].metric("통합 브랜드",    f"{len(merged)}개")
    mc[1].metric("총 유틸리티 비용", f"{merged['util_total'].sum()/1e6:.2f}M 원")
    mc[2].metric("수도",           f"{merged['water_total'].sum()/1e6:.2f}M ({merged['water_total'].sum()/merged['util_total'].sum()*100:.0f}%)")
    mc[3].metric("온수",           f"{merged['hw_total'].sum()/1e6:.2f}M ({merged['hw_total'].sum()/merged['util_total'].sum()*100:.0f}%)")
    mc[4].metric("전기",           f"{merged['elec_total'].sum()/1e6:.2f}M ({merged['elec_total'].sum()/merged['util_total'].sum()*100:.0f}%)")

    tab_rank, tab_mix, tab_area, tab_bld = st.tabs(
        ["총 유틸리티 순위", "유틸리티 구성", "면적당 총비용", "건물별 비교"]
    )

    # ═══════════════════════════ 총 유틸리티 순위 ═════════════════════════════
    with tab_rank:
        _n = st.slider("상위 N개", 10, min(60, len(merged)), min(40, len(merged)), key="sum_rank_n")
        _top = merged.nlargest(_n, "util_total").sort_values("util_total", ascending=True)

        fig_r = go.Figure()
        for label, col, clr in [("수도","water_total","#4C72B0"),
                                  ("온수","hw_total","#C44E52"),
                                  ("전기","elec_total","#DD8A00")]:
            fig_r.add_trace(go.Bar(
                x=_top[col].values, y=[str(b)[:26] for b in _top["brand"]],
                name=label, orientation="h", marker_color=clr,
                text=[f"{v/1e3:.0f}k" if v >= 1e4 else ("" if v==0 else f"{v:,.0f}") for v in _top[col].values],
                textposition="inside", textfont=dict(size=9, color="white"),
            ))
        _r_up = _iqr_upper(merged["util_total"])
        if _r_up < float("inf"):
            fig_r.add_vline(x=_r_up, line_dash="dot", line_color="#888",
                            annotation_text=f"IQR 상한 {_r_up/1e6:.2f}M",
                            annotation_position="top left", annotation_font_size=10)
        fig_r.update_layout(
            barmode="stack",
            height=max(480, _n*22+80),
            xaxis_title="원", plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            legend=dict(orientation="h",y=1.02),
            margin=dict(l=10,r=20,t=50,b=40),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="sum_rank_chart")

        sc = st.columns(4)
        sc[0].metric("합계",   f"{merged['util_total'].sum()/1e6:.2f}M 원")
        sc[1].metric("평균",   f"{merged['util_total'].mean():,.0f} 원")
        sc[2].metric("중앙값", f"{merged['util_total'].median():,.0f} 원")
        sc[3].metric("1위",    merged.loc[merged["util_total"].idxmax(),"brand"])

    # ═══════════════════════════ 유틸리티 구성 ════════════════════════════════
    with tab_mix:
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
            fig_mix = go.Figure()
            for bld in sorted(merged_mix["building"].dropna().unique()):
                sub = merged_mix[merged_mix["building"]==bld]
                fig_mix.add_trace(go.Scatter(
                    x=sub["water_pct"], y=sub["elec_pct"],
                    mode="markers+text", name=f"{bld}동",
                    marker=dict(color=_BLD_COLOR.get(str(bld),"#888"),
                                size=(sub["util_total"]/merged_mix["util_total"].max()*40+6).round(0),
                                opacity=0.8),
                    text=sub["brand"], textposition="top center", textfont=dict(size=7),
                ))
            fig_mix.update_layout(
                title="수도 비중 vs 전기 비중 (버블=총비용)",
                height=380, xaxis_title="수도 비중 (%)", yaxis_title="전기 비중 (%)",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD",griddash="dot",range=[-5,105]),
                yaxis=dict(gridcolor="#DDDDDD",griddash="dot",range=[-5,105]),
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
        st.dataframe(_tbl, use_container_width=True, hide_index=True)

    # ═══════════════════════════ 면적당 총비용 ════════════════════════════════
    with tab_area:
        _df_a = merged[merged["size_m2"] > 0].copy()
        _df_a["total_per_m2"] = (_df_a["util_total"] / _df_a["size_m2"]).round(0)
        _df_a = _df_a.sort_values("total_per_m2", ascending=True)

        _sf = _df_a["total_per_m2"]
        _f_up = _iqr_upper(_sf)
        _q1f, _q3f = _sf.quantile(0.25), _sf.quantile(0.75)
        _f_lo = max(0.0, float(_q1f - 1.5*(_q3f-_q1f)))

        _bord_clr = ["#8B1A1A" if v>_f_up else ("#1A5C2A" if _f_lo>0 and v<_f_lo else "white") for v in _sf.values]
        _bord_w   = [2.5 if v>_f_up or (_f_lo>0 and v<_f_lo) else 0 for v in _sf.values]

        fig_a = go.Figure(go.Bar(
            x=_sf.values, y=[str(b)[:26] for b in _df_a["brand"]],
            orientation="h",
            marker_color=[_BLD_COLOR.get(str(b),"#888") for b in _df_a["building"]],
            marker_line=dict(color=_bord_clr, width=_bord_w),
            text=[f"{v:,.0f}" for v in _sf.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_a.add_vline(x=float(_sf.median()), line_dash="dash", line_color="#C44E52",
                        annotation_text=f"중앙값 {_sf.median():,.0f}",
                        annotation_position="top right", annotation_font_size=10)
        if _f_up < _sf.max()*5:
            fig_a.add_vline(x=_f_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_f_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_a.update_layout(
            height=max(480, len(_df_a)*22+80),
            xaxis_title="원/m² (수도+온수+전기 합산)", plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            margin=dict(l=10,r=150,t=40,b=40),
        )
        st.plotly_chart(fig_a, use_container_width=True, key="sum_area_chart")

        ac = st.columns(4)
        ac[0].metric("중앙값",    f"{_sf.median():,.0f} 원/m²")
        ac[1].metric("평균",      f"{_sf.mean():,.0f} 원/m²")
        ac[2].metric("IQR 상한",  f"{_f_up:,.0f} 원/m²")
        ac[3].metric("상한 초과", f"{int((_sf>_f_up).sum())}개")

        # Breakdown bar: per-m2 by utility type for top 30
        _n_a = st.slider("상위 N개", 10, min(60,len(_df_a)), min(30,len(_df_a)), key="sum_area_n")
        _top_a = _df_a.nlargest(_n_a,"total_per_m2").sort_values("total_per_m2",ascending=True)
        _top_a["water_pm2"] = (_top_a["water_total"]/_top_a["size_m2"]).round(0)
        _top_a["hw_pm2"]    = (_top_a["hw_total"]/_top_a["size_m2"]).round(0)
        _top_a["elec_pm2"]  = (_top_a["elec_total"]/_top_a["size_m2"]).round(0)

        fig_ab = go.Figure()
        for label, col, clr in [("수도","water_pm2","#4C72B0"),
                                  ("온수","hw_pm2","#C44E52"),
                                  ("전기","elec_pm2","#DD8A00")]:
            fig_ab.add_trace(go.Bar(
                x=_top_a[col].values, y=[str(b)[:26] for b in _top_a["brand"]],
                name=label, orientation="h", marker_color=clr,
                text=[f"{v:,.0f}" if v>0 else "" for v in _top_a[col].values],
                textposition="inside", textfont=dict(size=9,color="white"),
            ))
        fig_ab.update_layout(
            barmode="stack", title=f"면적당 비용 구성 (상위 {_n_a}개, 원/m²)",
            height=max(420,_n_a*22+80), xaxis_title="원/m²",
            plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            legend=dict(orientation="h",y=1.04),
            margin=dict(l=10,r=20,t=60,b=40),
        )
        st.plotly_chart(fig_ab, use_container_width=True, key="sum_area_stacked")

    # ═══════════════════════════ 건물별 비교 ══════════════════════════════════
    with tab_bld:
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
            legend=dict(orientation="h",y=1.04),
            margin=dict(l=10,r=10,t=70,b=30),
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
