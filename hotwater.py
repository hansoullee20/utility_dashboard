"""hotwater.py — 온수 사용 내역 analysis view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import BLD_COLOR as _BLD_COLOR, iqr_upper as _iqr_upper, flag_prefix as _flag_prefix


def render_hotwater_view(df: pd.DataFrame, season: str | None = None) -> None:
    st.header("🌡️ 온수 사용 내역 분석")
    if season:
        st.caption(f"적용 시즌: **{season}**")

    buildings = sorted(df["building"].unique())
    sel_bld = st.multiselect("건물 선택", ["All"] + buildings, default=["All"], key="hw_bld")
    if "All" not in sel_bld and sel_bld:
        df = df[df["building"].isin(sel_bld)].copy()
    if df.empty:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다."); return

    n_total   = len(df)
    n_metered = int((df["usage_m3"] > 0).sum())
    n_unmet   = n_total - n_metered

    mc = st.columns(5)
    mc[0].metric("총 브랜드",   f"{n_total}개")
    mc[1].metric("총 부과금액", f"{df['total'].sum()/1e6:.2f}M 원")
    mc[2].metric("계량 브랜드", f"{n_metered} / {n_total}")
    mc[3].metric("총 사용량",   f"{int(df['usage_m3'].sum()):,} m³")
    _avg = df["fee_excl"].sum() / df["usage_m3"].replace(0, np.nan).sum()
    mc[4].metric("평균 단가",   f"{_avg:,.0f} 원/m³" if pd.notna(_avg) else "-")

    # ── Anomaly flags ─────────────────────────────────────────────────────────
    _flags = pd.DataFrame(index=df["brand"].values)
    _flags["총부과 이상치"]    = df["total"].values > _iqr_upper(df["total"])
    _cpm2 = df["total"] / df["size_m2"].replace(0, np.nan)
    _flags["면적당비용 이상치"] = _cpm2.values > _iqr_upper(_cpm2)
    _mm   = df["usage_m3"] > 0
    _upm2 = (df["usage_m3"] / df["size_m2"].replace(0, np.nan)).where(_mm)
    _flags["사용량/m² 이상치"]  = _upm2.values > _iqr_upper(_upm2.dropna())
    _flag_cols = ["총부과 이상치", "면적당비용 이상치", "사용량/m² 이상치"]
    _flags.index = df["brand"].values
    _flags["플래그 수"] = _flags[_flag_cols].sum(axis=1).astype(int)
    _flags["등급"] = _flags["플래그 수"].map(
        lambda n: "🔴 위험" if n >= 2 else ("🟠 주의" if n == 1 else "🟢 정상"))

    tab_rank, tab_comp, tab_fair, tab_usage, tab_anom = st.tabs(
        ["순위", "비중", "면적당 비용", "사용량", "이상 탐지"]
    )

    # ═══════════════════════════ 순위 ═════════════════════════════════════════
    with tab_rank:
        _metric = st.radio("기준", ["총부과", "사용량 (m³)"], horizontal=True, key="hw_rank_metric")
        _col  = "total" if _metric == "총부과" else "usage_m3"
        _unit = "원" if _metric == "총부과" else "m³"
        _df_r = df[["brand","building",_col]].sort_values(_col, ascending=True)
        _r_up = _iqr_upper(df[_col])

        fig_r = go.Figure()
        for bld in ["A","B","C","D"]:
            sub = _df_r[_df_r["building"]==bld]
            if sub.empty: continue
            fig_r.add_trace(go.Bar(
                x=sub[_col].values, y=[_flag_prefix(_flags, b)+str(b)[:26] for b in sub["brand"]],
                name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                text=[f"{v:,.0f}" for v in sub[_col].values],
                textposition="outside", textfont=dict(size=10),
            ))
        if _r_up < float("inf"):
            fig_r.add_vline(x=_r_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_r_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_r.update_layout(
            height=max(420, len(_df_r)*22+80), margin=dict(l=10,r=130,t=40,b=40),
            xaxis_title=_unit, barmode="overlay", showlegend=True,
            legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),
            plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            yaxis=dict(tickfont=dict(size=10), categoryorder="total ascending"),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="hw_rank_chart")
        _s = df[_col]
        sc = st.columns(4)
        sc[0].metric("합계",   f"{_s.sum():,.0f} {_unit}")
        sc[1].metric("평균",   f"{_s.mean():,.0f} {_unit}")
        sc[2].metric("중앙값", f"{_s.median():,.0f} {_unit}")
        sc[3].metric("1위",    df.loc[df[_col].idxmax(), "brand"])

    # ═══════════════════════════ 비중 ═════════════════════════════════════════
    with tab_comp:
        _cview = st.radio("보기", ["브랜드별 순위", "계량/미계량 현황", "건물별 donut"],
                          horizontal=True, key="hw_comp_view")

        if _cview == "브랜드별 순위":
            _n_show = st.slider("상위 N개", 10, min(60, n_total), min(30, n_total), key="hw_comp_n")
            _top = df.nlargest(_n_show, "total").sort_values("total", ascending=True)
            fig_c = go.Figure()
            for bld in ["A","B","C","D"]:
                sub = _top[_top["building"]==bld]
                if sub.empty: continue
                fig_c.add_trace(go.Bar(
                    x=sub["fee_excl"].values, y=[_flag_prefix(_flags, b)+str(b)[:26] for b in sub["brand"]],
                    name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                    text=[f"{v/1000:.0f}k" if v >= 1000 else "" for v in sub["fee_excl"].values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                ))
            fig_c.update_layout(
                barmode="overlay",
                height=max(420, _n_show*22+80), margin=dict(l=10,r=20,t=40,b=40),
                xaxis_title="원 (전용요금)", plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig_c, use_container_width=True, key="hw_comp_brand")

        elif _cview == "계량/미계량 현황":
            # Donut: metered vs unmetered count by building
            _met_by_bld = df.groupby("building").apply(
                lambda g: pd.Series({"계량": (g["usage_m3"]>0).sum(), "미계량": (g["usage_m3"]==0).sum()})
            ).reset_index()
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = go.Figure(go.Pie(
                    labels=["계량", "미계량"],
                    values=[n_metered, n_unmet], hole=0.45,
                    marker=dict(colors=["#4C72B0","#DDDDDD"]),
                    textinfo="label+value+percent", textfont=dict(size=13),
                ))
                fig_pie.update_layout(title="계량 vs 미계량 (전체)", height=340,
                                      margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_pie, use_container_width=True, key="hw_comp_pie")
            with c2:
                fig_met = go.Figure()
                for label, clr in [("계량","#4C72B0"),("미계량","#DDDDDD")]:
                    fig_met.add_trace(go.Bar(
                        x=[r["building"]+"동" for _,r in _met_by_bld.iterrows()],
                        y=[r[label] for _,r in _met_by_bld.iterrows()],
                        name=label, marker_color=clr,
                        text=[str(int(r[label])) for _,r in _met_by_bld.iterrows()],
                        textposition="inside", textfont=dict(size=11, color="white" if label=="계량" else "black"),
                    ))
                fig_met.update_layout(barmode="stack", title="건물별 계량 현황", height=340,
                                      plot_bgcolor="white", yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                      margin=dict(l=10,r=10,t=50,b=30))
                st.plotly_chart(fig_met, use_container_width=True, key="hw_comp_bld_met")

            st.caption(f"미계량 브랜드 {n_unmet}개는 온수 사용요금이 없으며 (부과금액=0), 해당 시설이 없거나 해당 기간 미사용입니다.")

        else:  # 건물별 donut
            _bld_totals = df.groupby("building")["total"].sum().reindex(["A","B","C","D"]).dropna()
            fig_bd = go.Figure(go.Pie(
                labels=[b+"동" for b in _bld_totals.index],
                values=_bld_totals.values, hole=0.45,
                marker=dict(colors=[_BLD_COLOR[b] for b in _bld_totals.index]),
                textinfo="label+percent+value", textfont=dict(size=12),
            ))
            fig_bd.update_layout(title="건물별 온수요금 비중", height=420,
                                 margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_bd, use_container_width=True, key="hw_comp_bld_donut")

    # ═══════════════════════════ 면적당 비용 ══════════════════════════════════
    with tab_fair:
        _u = st.radio("면적 단위", ["㎡","평"], horizontal=True, key="hw_fair_unit")
        _a_col = "size_m2" if _u == "㎡" else "size_py"
        _df_f = df[["brand","building","total",_a_col]].copy()
        _df_f = _df_f[_df_f[_a_col] > 0].copy()
        _df_f["cpa"] = _df_f["total"] / _df_f[_a_col]
        _df_f = _df_f.sort_values("cpa", ascending=True)
        _sf = _df_f["cpa"]
        _f_up = _iqr_upper(_sf)
        _q1f, _q3f = _sf.quantile(0.25), _sf.quantile(0.75)
        _f_lo = max(0.0, float(_q1f - 1.5*(_q3f-_q1f)))
        _bord_clr = ["#8B1A1A" if v>_f_up else ("#1A5C2A" if _f_lo>0 and v<_f_lo else "white") for v in _sf.values]
        _bord_w   = [2.5 if v>_f_up or (_f_lo>0 and v<_f_lo) else 0 for v in _sf.values]
        fig_f = go.Figure(go.Bar(
            x=_sf.values, y=[_flag_prefix(_flags, b)+str(b)[:26] for b in _df_f["brand"]],
            orientation="h",
            marker_color=[_BLD_COLOR.get(b,"#888") for b in _df_f["building"]],
            marker_line=dict(color=_bord_clr, width=_bord_w),
            text=[f"{v:,.0f}" for v in _sf.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_f.add_vline(x=float(_sf.median()), line_dash="dash", line_color="#C44E52",
                        annotation_text=f"중앙값 {_sf.median():,.0f}",
                        annotation_position="top right", annotation_font_size=10)
        if _f_up < _sf.max()*5:
            fig_f.add_vline(x=_f_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_f_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_f.update_layout(
            height=max(420, len(_df_f)*22+80), xaxis_title=f"원/{_u}",
            plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
            margin=dict(l=10,r=150,t=40,b=40),
        )
        st.plotly_chart(fig_f, use_container_width=True, key="hw_fair_chart")
        fc = st.columns(4)
        fc[0].metric("중앙값",    f"{_sf.median():,.0f} 원/{_u}")
        fc[1].metric("평균",      f"{_sf.mean():,.0f} 원/{_u}")
        fc[2].metric("IQR 상한",  f"{_f_up:,.0f} 원/{_u}")
        fc[3].metric("상한 초과", f"{int((_sf>_f_up).sum())}개")

    # ═══════════════════════════ 사용량 ═══════════════════════════════════════
    with tab_usage:
        st.caption(f"계량 브랜드: **{n_metered}개**  |  미계량 (부과 없음): **{n_unmet}개**")

        # Building summary
        st.subheader("건물별 집계")
        _bgrp = df.groupby("building").agg(
            브랜드수=("brand","count"),
            계량브랜드=("usage_m3", lambda x: (x>0).sum()),
            총사용량=("usage_m3","sum"),
            총부과=("total","sum"),
            총면적=("size_m2","sum"),
        ).reindex(["A","B","C","D"]).dropna(how="all").reset_index()

        _bc1, _bc2 = st.columns(2)
        with _bc1:
            fig_bu = go.Figure()
            for _,row in _bgrp.iterrows():
                fig_bu.add_trace(go.Bar(
                    x=[row["building"]+"동"], y=[row["총사용량"]],
                    marker_color=_BLD_COLOR.get(row["building"],"#888"),
                    text=[f"{int(row['총사용량']):,} m³"], textposition="outside",
                    textfont=dict(size=11), showlegend=False,
                ))
            fig_bu.update_layout(title="건물별 총 사용량 (m³)", height=300, plot_bgcolor="white",
                                 yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                 margin=dict(l=10,r=10,t=50,b=30))
            st.plotly_chart(fig_bu, use_container_width=True, key="hw_bld_usage")
        with _bc2:
            fig_bt = go.Figure()
            for _,row in _bgrp.iterrows():
                fig_bt.add_trace(go.Bar(
                    x=[row["building"]+"동"], y=[row["총부과"]],
                    marker_color=_BLD_COLOR.get(row["building"],"#888"),
                    text=[f"{row['총부과']/1e6:.2f}M"], textposition="outside",
                    textfont=dict(size=11), showlegend=False,
                ))
            fig_bt.update_layout(title="건물별 총 부과금액 (원)", height=300, plot_bgcolor="white",
                                 yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                 margin=dict(l=10,r=10,t=50,b=30))
            st.plotly_chart(fig_bt, use_container_width=True, key="hw_bld_total")

        _bgrp_disp = _bgrp.copy()
        _bgrp_disp["총사용량"] = _bgrp_disp["총사용량"].apply(lambda v: f"{int(v):,} m³")
        _bgrp_disp["총부과"]   = _bgrp_disp["총부과"].apply(lambda v: f"{v:,.0f} 원")
        _bgrp_disp["원/m²"]   = (_bgrp["총부과"] / _bgrp["총면적"]).apply(lambda v: f"{v:,.0f}")
        _bgrp_disp["building"] = _bgrp_disp["building"] + "동"
        _bgrp_disp = _bgrp_disp.rename(columns={"building":"건물","브랜드수":"브랜드","계량브랜드":"계량",
                                                  "총사용량":"사용량 합계","총부과":"부과 합계"})
        st.dataframe(_bgrp_disp[["건물","브랜드","계량","사용량 합계","부과 합계","원/m²"]],
                     use_container_width=True, hide_index=True)

        st.divider()

        # 무사용 브랜드 expander
        df_zero = df[df["usage_m3"] == 0].copy()
        with st.expander(f"무사용 브랜드 ({n_unmet}개) — 온수 미부과", expanded=False):
            _zero_by_bld = df_zero.groupby("building").agg(
                브랜드수=("brand","count"), 총면적=("size_m2","sum")).reset_index()
            st.caption("온수 미터기가 없거나 해당 기간 사용량이 없어 부과금액이 0입니다.")
            zc1, zc2 = st.columns(2)
            with zc1:
                fig_z = go.Figure()
                for _,row in _zero_by_bld.iterrows():
                    fig_z.add_trace(go.Bar(
                        x=[row["building"]+"동"], y=[row["브랜드수"]],
                        marker_color=_BLD_COLOR.get(row["building"],"#888"),
                        text=[str(int(row["브랜드수"]))], textposition="outside",
                        showlegend=False,
                    ))
                fig_z.update_layout(title="건물별 무사용 브랜드 수", height=250,
                                    plot_bgcolor="white", yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                    margin=dict(l=10,r=10,t=50,b=30))
                st.plotly_chart(fig_z, use_container_width=True, key="hw_zero_bld")
            with zc2:
                st.dataframe(
                    df_zero[["brand","building","floor","size_m2"]].sort_values(["building","size_m2"], ascending=[True,False]),
                    use_container_width=True, hide_index=True, height=220,
                )

        st.divider()
        df_m = df[df["usage_m3"] > 0].copy()
        _uview = st.radio("차트", ["사용량 순위", "사용량 vs 면적 산점도"],
                          horizontal=True, key="hw_usage_view")

        if _uview == "사용량 순위":
            _df_u = df_m[["brand","building","usage_m3"]].sort_values("usage_m3", ascending=True)
            _u_up = _iqr_upper(df_m["usage_m3"])
            fig_u = go.Figure()
            for bld in ["A","B","C","D"]:
                sub = _df_u[_df_u["building"]==bld]
                if sub.empty: continue
                fig_u.add_trace(go.Bar(
                    x=sub["usage_m3"].values, y=[_flag_prefix(_flags, b)+str(b)[:26] for b in sub["brand"]],
                    name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                    text=[f"{v:,}" for v in sub["usage_m3"].values],
                    textposition="outside", textfont=dict(size=10),
                ))
            fig_u.add_vline(x=float(df_m["usage_m3"].median()), line_dash="dash", line_color="#C44E52",
                            annotation_text=f"중앙값 {df_m['usage_m3'].median():.0f}", annotation_font_size=10)
            if _u_up < float("inf"):
                fig_u.add_vline(x=_u_up, line_dash="dot", line_color="#DD8A00",
                                annotation_text=f"IQR 상한 {_u_up:.0f}",
                                annotation_position="top left", annotation_font_size=10)
            fig_u.update_layout(
                height=max(420,len(_df_u)*22+80), xaxis_title="m³", barmode="overlay",
                plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                yaxis=dict(categoryorder="total ascending"),
                legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),
                margin=dict(l=10,r=120,t=40,b=40),
            )
            st.plotly_chart(fig_u, use_container_width=True, key="hw_usage_rank")
        else:
            fig_s = go.Figure()
            for bld in sorted(df_m["building"].unique()):
                sub = df_m[df_m["building"]==bld]
                fig_s.add_trace(go.Scatter(
                    x=sub["size_m2"], y=sub["usage_m3"], mode="markers+text", name=f"{bld}동",
                    marker=dict(color=_BLD_COLOR.get(bld,"#888"), size=9, opacity=0.8),
                    text=sub["brand"], textposition="top center", textfont=dict(size=8),
                ))
            xa, ya = df_m["size_m2"].values, df_m["usage_m3"].values
            c = np.polyfit(xa, ya, 1)
            xf = np.linspace(xa.min(), xa.max(), 100)
            r2 = np.corrcoef(xa, ya)[0, 1] ** 2
            fig_s.add_trace(go.Scatter(
                x=xf, y=np.polyval(c,xf), mode="lines",
                name=f"추세선 (R²={r2:.3f})", line=dict(color="#888",dash="dash",width=1.5),
            ))
            fig_s.update_layout(
                height=520, xaxis_title="전용면적 (m²)", yaxis_title="사용량 (m³)",
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                margin=dict(l=20,r=20,t=40,b=40),
            )
            st.plotly_chart(fig_s, use_container_width=True, key="hw_scatter")
            st.caption(f"R² = {r2:.3f}  |  면적으로 사용량의 {r2*100:.1f}%를 설명")

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

        _hm = _flags.sort_values("플래그 수", ascending=False).head(40)
        _hm_brands = list(_hm.index)
        fig_hm = go.Figure(go.Heatmap(
            z=[[int(_hm.at[b,c]) for c in _flag_cols] for b in _hm_brands],
            x=_flag_cols, y=_hm_brands,
            colorscale=[[0,"#F0F0F0"],[1,"#C44E52"]], showscale=False,
            text=[["✓" if _hm.at[b,c] else "" for c in _flag_cols] for b in _hm_brands],
            texttemplate="%{text}", textfont=dict(size=13,color="white"),
            xgap=3, ygap=3,
        ))
        fig_hm.update_layout(
            height=max(300,len(_hm_brands)*22+80), margin=dict(l=10,r=20,t=40,b=40),
            plot_bgcolor="white", yaxis=dict(autorange="reversed",tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="hw_anom_heatmap")

        _fi = _flags[_flags["플래그 수"]>0].index
        if len(_fi)>0:
            st.subheader("플래그 상세")
            _detail = df[df["brand"].isin(_fi)][
                ["brand","building","floor","size_m2","usage_m3","fee_excl","total"]].copy()
            _meta = _flags[["플래그 수","등급"]].reset_index().rename(columns={"index":"brand"})
            _detail = _detail.merge(_meta, on="brand", how="left")
            _detail = _detail.sort_values("플래그 수", ascending=False).reset_index(drop=True)
            st.dataframe(_detail, use_container_width=True, hide_index=True)
