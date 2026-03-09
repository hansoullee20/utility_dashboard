"""water.py — 수도 사용 내역 analysis view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import BLD_COLOR as _BLD_COLOR, iqr_upper as _iqr_upper, flag_prefix as _flag_prefix
from filters import render_sheet_filters

_COMP_COLS = [
    ("상수도 전용",  "water_excl",  "#4C72B0"),
    ("하수도 전용",  "sewage_excl", "#C44E52"),
    ("분담금 전용",  "levy_excl",   "#8172B2"),
    ("공용합계",     "total_comm",  "#9EBADF"),
]


def _scatter_with_trendline(df_sub, x_col, y_col, color_col, title, xlab, ylab, key):
    fig = go.Figure()
    for bld in sorted(df_sub[color_col].unique()):
        sub = df_sub[df_sub[color_col] == bld]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers+text",
            name=f"{bld}동",
            marker=dict(color=_BLD_COLOR.get(bld, "#888"), size=8, opacity=0.8),
            text=sub["brand"], textposition="top center", textfont=dict(size=8),
        ))
    xa, ya = df_sub[x_col].values, df_sub[y_col].values
    if len(xa) >= 2:
        c = np.polyfit(xa, ya, 1)
        xf = np.linspace(xa.min(), xa.max(), 100)
        r2 = np.corrcoef(xa, ya)[0, 1] ** 2
        fig.add_trace(go.Scatter(
            x=xf, y=np.polyval(c, xf), mode="lines",
            name=f"추세선 (R²={r2:.3f})",
            line=dict(color="#888", dash="dash", width=1.5),
        ))
    fig.update_layout(
        title=title, height=480,
        xaxis_title=xlab, yaxis_title=ylab,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
        yaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    _ev_sc = st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")
    _sel_sc = _ev_sc.selection.points if _ev_sc and hasattr(_ev_sc, "selection") else []
    if _sel_sc:
        _pt_sc = _sel_sc[0]
        _cd_sc = _pt_sc.get("customdata", [])
        _brand_sc = _cd_sc[0] if isinstance(_cd_sc, list) and _cd_sc else str(_cd_sc) if _cd_sc else _pt_sc.get("text") or ""
        _fdf_sc = df_sub[df_sub["brand"] == _brand_sc] if _brand_sc else pd.DataFrame()
        if not _fdf_sc.empty:
            st.caption(f"선택됨: **{_brand_sc}**")
            st.dataframe(_fdf_sc.reset_index(drop=True), hide_index=True, use_container_width=True)
    if len(xa) >= 2:
        r2 = np.corrcoef(xa, ya)[0, 1] ** 2
        st.caption(f"R² = {r2:.3f}  |  면적으로 {y_col}의 {r2*100:.1f}%를 설명")


def render_water_view(df: pd.DataFrame) -> None:
    st.header("💧 수도 사용 내역 분석")

    df = render_sheet_filters(df, key_prefix="water")
    if df.empty:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다."); return

    n_total   = len(df)
    n_metered = int((df["usage_m3"] > 0).sum())
    n_unmet   = n_total - n_metered

    mc = st.columns(5)
    mc[0].metric("총 브랜드",   f"{n_total}개")
    mc[1].metric("총 부과금액", f"{df['total'].sum()/1e6:.2f}M 원")
    _ep = df["total_excl"].sum() / df["total"].sum() * 100 if df["total"].sum() else 0
    mc[2].metric("전용 부과",   f"{df['total_excl'].sum()/1e6:.2f}M ({_ep:.0f}%)")
    mc[3].metric("총 사용량",   f"{int(df['usage_m3'].sum()):,} m³")
    mc[4].metric("계량 브랜드", f"{n_metered} / {n_total}")

    # ── Anomaly flags ─────────────────────────────────────────────────────────
    _flags = pd.DataFrame(index=df["brand"].values)
    _total_up = _iqr_upper(df["total"])
    _flags["총부과 이상치"]    = df["total"].values > _total_up
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

    tab_rank, tab_comp, tab_fair, tab_usage, tab_excl, tab_anom = st.tabs(
        ["순위", "비중", "면적당 비용", "사용량", "전용/공용 분석", "이상 탐지"]
    )

    # ═══════════════════════════ 순위 ═════════════════════════════════════════
    with tab_rank:
        _metric = st.radio("기준", ["총부과", "전용부과", "공용부과", "사용량 (m³)"],
                           horizontal=True, key="water_rank_metric")
        _col  = {"총부과": "total", "전용부과": "total_excl",
                 "공용부과": "total_comm", "사용량 (m³)": "usage_m3"}[_metric]
        _unit = "원" if _metric != "사용량 (m³)" else "m³"

        _col_label = {"총부과": "총부과", "전용부과": "전용부과", "공용부과": "공용부과", "사용량 (m³)": "사용량 (m³)"}[_metric]
        _df_r = df[["brand", "building", "floor", "size_m2", _col]].sort_values(_col, ascending=True)
        _r_up = _iqr_upper(df[_col])

        fig_r = go.Figure()
        for bld in ["A", "B", "C", "D"]:
            sub = _df_r[_df_r["building"] == bld]
            if sub.empty: continue
            fig_r.add_trace(go.Bar(
                x=sub[_col].values, y=[_flag_prefix(_flags, b) + str(b)[:26] for b in sub["brand"]],
                name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                customdata=sub[["floor", "size_m2"]].fillna("").values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{_col_label}: " + "%{x:,.0f}<br>"
                    + "층: %{customdata[0]}<br>"
                    + "면적: %{customdata[1]:.0f} m²"
                    + "<extra>%{fullData.name}</extra>"
                ),
                text=[f"{v:,.0f}" for v in sub[_col].values],
                textposition="outside" if len(_df_r) <= 25 else "none", textfont=dict(size=10),
            ))
        if _r_up < float("inf"):
            fig_r.add_vline(x=_r_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_r_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_r.update_layout(
            height=max(420, len(_df_r) * 22 + 80), margin=dict(l=10, r=130, t=40, b=40),
            xaxis_title=_unit, barmode="overlay", showlegend=True,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            yaxis=dict(tickfont=dict(size=10), categoryorder="total ascending"),
        )
        _ev_rank = st.plotly_chart(fig_r, use_container_width=True, key="water_rank_chart", on_select="rerun")
        _sel_rank = _ev_rank.selection.points if _ev_rank and hasattr(_ev_rank, "selection") else []
        if _sel_rank:
            _pt = _sel_rank[0]
            _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
            if isinstance(_brand, (list, tuple)):
                _brand = _brand[0]
            if isinstance(_brand, str):
                _brand = _brand.lstrip("🔴 ").lstrip("🟠 ").lstrip("🟢 ")
            _fdf = df[df["brand"].str.contains(_brand[:20], regex=False)] if _brand else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_brand}**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

        _s = df[_col]
        sc = st.columns(5)
        sc[0].metric("합계",   f"{_s.sum():,.0f} {_unit}")
        sc[1].metric("평균",   f"{_s.mean():,.0f} {_unit}")
        sc[2].metric("중앙값", f"{_s.median():,.0f} {_unit}")
        sc[3].metric("최대",   f"{_s.max():,.0f} {_unit}")
        sc[4].metric("1위",    df.loc[df[_col].idxmax(), "brand"])

    # ═══════════════════════════ 비중 ═════════════════════════════════════════
    with tab_comp:
        _cview = st.radio("보기", ["브랜드별 stacked", "전체 항목 donut"],
                          horizontal=True, key="water_comp_view")
        if _cview.startswith("브"):
            _n_show = st.slider("상위 N개", 10, min(60, n_total), min(30, n_total), key="water_comp_n")
            _top = df.nlargest(_n_show, "total").sort_values("total", ascending=True)
            fig_c = go.Figure()
            for label, col, clr in _COMP_COLS:
                fig_c.add_trace(go.Bar(
                    x=_top[col].values, y=[_flag_prefix(_flags, b) + str(b)[:26] for b in _top["brand"]],
                    name=label, orientation="h", marker_color=clr,
                    text=[f"{v/1000:.0f}k" if v >= 1000 else "" for v in _top[col].values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                ))
            fig_c.update_layout(
                barmode="stack", height=max(420, _n_show * 22 + 80),
                margin=dict(l=10, r=20, t=40, b=40), xaxis_title="원",
                plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                legend=dict(orientation="h", y=1.02),
            )
            _ev_comp = st.plotly_chart(fig_c, use_container_width=True, key="water_comp_stacked", on_select="rerun")
            _sel_comp = _ev_comp.selection.points if _ev_comp and hasattr(_ev_comp, "selection") else []
            if _sel_comp:
                _pt = _sel_comp[0]
                _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
                if isinstance(_brand, (list, tuple)):
                    _brand = _brand[0]
                if isinstance(_brand, str):
                    _brand = _brand.lstrip("🔴 ").lstrip("🟠 ").lstrip("🟢 ")
                _fdf = df[df["brand"].str.contains(_brand[:20], regex=False)] if _brand else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_brand}**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
        else:
            _dvals = {
                "상수도 전용": df["water_excl"].sum(), "하수도 전용": df["sewage_excl"].sum(),
                "분담금 전용": df["levy_excl"].sum(),  "상수도 공용": df["water_comm"].sum(),
                "하수도 공용": df["sewage_comm"].sum(), "분담금 공용": df["levy_comm"].sum(),
            }
            fig_d = go.Figure(go.Pie(
                labels=list(_dvals.keys()), values=list(_dvals.values()), hole=0.45,
                marker=dict(colors=["#4C72B0","#C44E52","#8172B2","#9EBADF","#F28E8E","#C8B8E8"]),
                textinfo="label+percent", textfont=dict(size=12),
            ))
            fig_d.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_d, use_container_width=True, key="water_comp_donut")
            _tot_all = df["total"].sum()
            st.dataframe(pd.DataFrame({
                "항목": list(_dvals.keys()),
                "금액 (원)": [f"{v:,.0f}" for v in _dvals.values()],
                "비중": [f"{v/_tot_all*100:.1f}%" for v in _dvals.values()],
            }), use_container_width=True, hide_index=True)

    # ═══════════════════════════ 면적당 비용 ══════════════════════════════════
    with tab_fair:
        _u_toggle = st.radio("면적 단위", ["㎡", "평"], horizontal=True, key="water_fair_unit")
        _a_col    = "size_m2" if _u_toggle == "㎡" else "size_py"
        _df_f = df[["brand", "building", "total", _a_col]].copy()
        _df_f = _df_f[_df_f[_a_col] > 0].copy()
        _df_f["cpa"] = _df_f["total"] / _df_f[_a_col]
        _df_f = _df_f.sort_values("cpa", ascending=True)
        _sf = _df_f["cpa"]
        _f_up = _iqr_upper(_sf)
        _q1f, _q3f = _sf.quantile(0.25), _sf.quantile(0.75)
        _f_lo = max(0.0, float(_q1f - 1.5 * (_q3f - _q1f)))
        _bord_clr = ["#8B1A1A" if v > _f_up else ("#1A5C2A" if _f_lo > 0 and v < _f_lo else "white") for v in _sf.values]
        _bord_w   = [2.5 if v > _f_up or (_f_lo > 0 and v < _f_lo) else 0 for v in _sf.values]
        fig_f = go.Figure(go.Bar(
            x=_sf.values, y=[_flag_prefix(_flags, b) + str(b)[:26] for b in _df_f["brand"]],
            orientation="h",
            marker_color=[_BLD_COLOR.get(b, "#888") for b in _df_f["building"]],
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
            height=max(420, len(_df_f) * 22 + 80), xaxis_title=f"원/{_u_toggle}",
            plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            margin=dict(l=10, r=150, t=40, b=40),
        )
        _ev_fair = st.plotly_chart(fig_f, use_container_width=True, key="water_fair_chart", on_select="rerun")
        _sel_fair = _ev_fair.selection.points if _ev_fair and hasattr(_ev_fair, "selection") else []
        if _sel_fair:
            _pt = _sel_fair[0]
            _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
            if isinstance(_brand, (list, tuple)):
                _brand = _brand[0]
            if isinstance(_brand, str):
                _brand = _brand.lstrip("🔴 ").lstrip("🟠 ").lstrip("🟢 ")
            _fdf = _df_f[_df_f["brand"].str.contains(_brand[:20], regex=False)] if _brand else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_brand}**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
        fc = st.columns(4)
        fc[0].metric("중앙값",    f"{_sf.median():,.0f} 원/{_u_toggle}")
        fc[1].metric("평균",      f"{_sf.mean():,.0f} 원/{_u_toggle}")
        fc[2].metric("IQR 상한",  f"{_f_up:,.0f} 원/{_u_toggle}")
        fc[3].metric("상한 초과", f"{int((_sf > _f_up).sum())}개")

    # ═══════════════════════════ 사용량 ═══════════════════════════════════════
    with tab_usage:
        st.caption(f"계량 브랜드: **{n_metered}개** (직접 측정)  |  미계량: **{n_unmet}개** (면적 비례 공용 배분만 부과)")
        df_m = df[df["usage_m3"] > 0].copy()
        df_m["cost_per_m3"] = (df_m["total_excl"] / df_m["usage_m3"]).round(0)

        # Building summary
        st.subheader("건물별 집계")
        _bld_grp = df.groupby("building").agg(
            브랜드수=("brand", "count"),
            계량브랜드=("usage_m3", lambda x: (x > 0).sum()),
            총사용량=("usage_m3", "sum"),
            총부과=("total", "sum"),
            총면적=("size_m2", "sum"),
        ).reindex(["A","B","C","D"]).dropna(how="all").reset_index()
        _bld_grp["면적당비용"]    = (_bld_grp["총부과"] / _bld_grp["총면적"]).round(0)
        _bld_grp["브랜드당사용량"] = (_bld_grp["총사용량"] / _bld_grp["계량브랜드"].replace(0, np.nan)).round(1)

        _bc1, _bc2 = st.columns(2)
        with _bc1:
            fig_bu = go.Figure()
            for _, row in _bld_grp.iterrows():
                fig_bu.add_trace(go.Bar(
                    x=[row["building"]+"동"], y=[row["총사용량"]],
                    marker_color=_BLD_COLOR.get(row["building"],"#888"),
                    text=[f"{int(row['총사용량']):,} m³"], textposition="outside",
                    textfont=dict(size=11), showlegend=False,
                ))
            fig_bu.update_layout(title="건물별 총 사용량 (m³)", height=300, plot_bgcolor="white",
                                 yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                 margin=dict(l=10,r=10,t=50,b=30))
            _ev_bld_usage = st.plotly_chart(fig_bu, use_container_width=True, key="water_bld_usage", on_select="rerun")
            _sel_bld_usage = _ev_bld_usage.selection.points if _ev_bld_usage and hasattr(_ev_bld_usage, "selection") else []
            if _sel_bld_usage:
                _pt = _sel_bld_usage[0]
                _bld = str(_pt.get("x") or "").replace("동", "")
                _fdf = _bld_grp[_bld_grp["building"] == _bld] if _bld else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_bld}동**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
        with _bc2:
            fig_bt = go.Figure()
            for _, row in _bld_grp.iterrows():
                fig_bt.add_trace(go.Bar(
                    x=[row["building"]+"동"], y=[row["총부과"]],
                    marker_color=_BLD_COLOR.get(row["building"],"#888"),
                    text=[f"{row['총부과']/1e6:.2f}M"], textposition="outside",
                    textfont=dict(size=11), showlegend=False,
                ))
            fig_bt.update_layout(title="건물별 총 부과금액 (원)", height=300, plot_bgcolor="white",
                                 yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                 margin=dict(l=10,r=10,t=50,b=30))
            _ev_bld_total = st.plotly_chart(fig_bt, use_container_width=True, key="water_bld_total", on_select="rerun")
            _sel_bld_total = _ev_bld_total.selection.points if _ev_bld_total and hasattr(_ev_bld_total, "selection") else []
            if _sel_bld_total:
                _pt = _sel_bld_total[0]
                _bld = str(_pt.get("x") or "").replace("동", "")
                _fdf = _bld_grp[_bld_grp["building"] == _bld] if _bld else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_bld}동**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

        fig_bcomp = go.Figure()
        for label, col, clr in _COMP_COLS:
            fig_bcomp.add_trace(go.Bar(
                x=[r["building"]+"동" for _,r in _bld_grp.iterrows()],
                y=[df[df["building"]==r["building"]][col].sum() for _,r in _bld_grp.iterrows()],
                name=label, marker_color=clr,
                text=[f"{df[df['building']==r['building']][col].sum()/1e6:.2f}M" for _,r in _bld_grp.iterrows()],
                textposition="inside", textfont=dict(size=10, color="white"),
            ))
        fig_bcomp.update_layout(barmode="stack", title="건물별 요금 구성", height=320,
                                plot_bgcolor="white", yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                legend=dict(orientation="h",y=1.12), margin=dict(l=10,r=10,t=70,b=30))
        _ev_bld_comp = st.plotly_chart(fig_bcomp, use_container_width=True, key="water_bld_comp", on_select="rerun")
        _sel_bld_comp = _ev_bld_comp.selection.points if _ev_bld_comp and hasattr(_ev_bld_comp, "selection") else []
        if _sel_bld_comp:
            _pt = _sel_bld_comp[0]
            _bld = str(_pt.get("x") or "").replace("동", "")
            _fdf = _bld_grp[_bld_grp["building"] == _bld] if _bld else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_bld}동**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

        _bld_disp = _bld_grp.copy()
        _bld_disp["총사용량"]      = _bld_disp["총사용량"].apply(lambda v: f"{int(v):,} m³")
        _bld_disp["총부과"]        = _bld_disp["총부과"].apply(lambda v: f"{v:,.0f} 원")
        _bld_disp["면적당비용"]    = _bld_disp["면적당비용"].apply(lambda v: f"{v:,.0f} 원/m²")
        _bld_disp["브랜드당사용량"] = _bld_disp["브랜드당사용량"].apply(lambda v: f"{v:.1f} m³" if pd.notna(v) else "-")
        _bld_disp["building"]     = _bld_disp["building"] + "동"
        _bld_disp = _bld_disp.rename(columns={"building":"건물","브랜드수":"브랜드","계량브랜드":"계량",
                                               "총사용량":"사용량 합계","총부과":"부과 합계",
                                               "면적당비용":"원/m²","브랜드당사용량":"계량브랜드당 평균"})
        st.dataframe(_bld_disp, use_container_width=True, hide_index=True)
        st.divider()

        # 미계량 브랜드 expander
        df_unmet = df[df["usage_m3"] == 0].copy()
        with st.expander(f"미계량 브랜드 상세 ({n_unmet}개) — 공용요금만 부과", expanded=False):
            _um_disp = df_unmet[["brand","building","floor","size_m2","total_comm","total"]].copy()
            _um_disp = _um_disp.sort_values("total", ascending=False).reset_index(drop=True)
            st.caption("이 브랜드들은 수도 미터기가 없거나 해당 월 사용량이 0으로, 면적 비례 공용요금만 부과됩니다.")
            st.dataframe(_um_disp, use_container_width=True, hide_index=True)

        st.divider()
        _uview = st.radio("차트", ["사용량 순위", "사용량 vs 면적 산점도", "m³당 단가 (효율 지도)"],
                          horizontal=True, key="water_usage_view")

        if _uview == "사용량 순위":
            _df_u = df_m[["brand","building","floor","size_m2","usage_m3"]].sort_values("usage_m3", ascending=True)
            _u_up = _iqr_upper(df_m["usage_m3"])
            fig_u = go.Figure()
            for bld in ["A","B","C","D"]:
                sub = _df_u[_df_u["building"]==bld]
                if sub.empty: continue
                fig_u.add_trace(go.Bar(
                    x=sub["usage_m3"].values, y=[_flag_prefix(_flags, b)+str(b)[:26] for b in sub["brand"]],
                    name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                    customdata=sub[["floor", "size_m2"]].fillna("").values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        + "사용량 (m³): %{x:,.0f}<br>"
                        + "층: %{customdata[0]}<br>"
                        + "면적: %{customdata[1]:.0f} m²"
                        + "<extra>%{fullData.name}</extra>"
                    ),
                    text=[f"{v:,}" for v in sub["usage_m3"].values],
                    textposition="outside" if len(_df_u) <= 25 else "none", textfont=dict(size=10),
                ))
            fig_u.add_vline(x=float(df_m["usage_m3"].median()), line_dash="dash", line_color="#C44E52",
                            annotation_text=f"중앙값 {df_m['usage_m3'].median():.0f}", annotation_font_size=10)
            if _u_up < float("inf"):
                fig_u.add_vline(x=_u_up, line_dash="dot", line_color="#DD8A00",
                                annotation_text=f"IQR 상한 {_u_up:.0f}",
                                annotation_position="top left", annotation_font_size=10)
            fig_u.update_layout(
                height=max(420, len(_df_u)*22+80), xaxis_title="m³", barmode="overlay",
                plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                yaxis=dict(categoryorder="total ascending"),
                legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),
                margin=dict(l=10,r=120,t=40,b=40),
            )
            _ev_usage = st.plotly_chart(fig_u, use_container_width=True, key="water_usage_rank", on_select="rerun")
            _sel_usage = _ev_usage.selection.points if _ev_usage and hasattr(_ev_usage, "selection") else []
            if _sel_usage:
                _pt = _sel_usage[0]
                _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
                if isinstance(_brand, (list, tuple)):
                    _brand = _brand[0]
                if isinstance(_brand, str):
                    _brand = _brand.lstrip("🔴 ").lstrip("🟠 ").lstrip("🟢 ")
                _fdf = _df_u[_df_u["brand"].str.contains(_brand[:20], regex=False)] if _brand else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_brand}**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
            uc = st.columns(4)
            uc[0].metric("총 사용량",   f"{int(df_m['usage_m3'].sum()):,} m³")
            uc[1].metric("평균 사용량", f"{df_m['usage_m3'].mean():.0f} m³")
            uc[2].metric("중앙값",      f"{df_m['usage_m3'].median():.0f} m³")
            uc[3].metric("최대 사용",   f"{df_m.loc[df_m['usage_m3'].idxmax(),'brand']} ({int(df_m['usage_m3'].max()):,} m³)")

        elif _uview == "사용량 vs 면적 산점도":
            _scatter_with_trendline(df_m, "size_m2", "usage_m3", "building",
                                    "전용면적 vs 수도 사용량 (계량 브랜드)",
                                    "전용면적 (m²)", "사용량 (m³)", "water_scatter")

        else:  # m³당 단가 효율 지도
            st.caption("※ 공정 수도요금은 모든 브랜드에 동일하게 적용 (~3,707원/m³). 이탈값은 부과 오류 또는 데이터 이슈일 수 있습니다.")
            _df_cpu = df_m[["brand","building","cost_per_m3"]].sort_values("cost_per_m3", ascending=True)
            _cpu_up = _iqr_upper(df_m["cost_per_m3"])
            _cpu_med = float(df_m["cost_per_m3"].median())

            _bord_cpu_clr = ["#8B1A1A" if v > _cpu_up else "white" for v in _df_cpu["cost_per_m3"].values]
            _bord_cpu_w   = [2.5 if v > _cpu_up else 0 for v in _df_cpu["cost_per_m3"].values]

            fig_cpu = go.Figure()
            for bld in ["A","B","C","D"]:
                sub = _df_cpu[_df_cpu["building"]==bld]
                if sub.empty: continue
                fig_cpu.add_trace(go.Bar(
                    x=sub["cost_per_m3"].values, y=[str(b)[:26] for b in sub["brand"]],
                    name=f"{bld}동", orientation="h", marker_color=_BLD_COLOR[bld],
                    marker_line=dict(color=[_bord_cpu_clr[i] for i,r in enumerate(_df_cpu["building"]) if r==bld],
                                     width=[_bord_cpu_w[i] for i,r in enumerate(_df_cpu["building"]) if r==bld]),
                    text=[f"{v:,.0f}" for v in sub["cost_per_m3"].values],
                    textposition="outside", textfont=dict(size=10),
                ))
            fig_cpu.add_vline(x=_cpu_med, line_dash="dash", line_color="#C44E52",
                              annotation_text=f"중앙값 {_cpu_med:,.0f}", annotation_font_size=10)
            if _cpu_up < float("inf") and _cpu_up < df_m["cost_per_m3"].max()*5:
                fig_cpu.add_vline(x=_cpu_up, line_dash="dot", line_color="#DD8A00",
                                  annotation_text=f"IQR 상한 {_cpu_up:,.0f}",
                                  annotation_position="top left", annotation_font_size=10)
            fig_cpu.update_layout(
                height=max(420, len(_df_cpu)*22+80), xaxis_title="원/m³", barmode="overlay",
                plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                yaxis=dict(categoryorder="total ascending"),
                legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),
                margin=dict(l=10,r=130,t=40,b=40),
            )
            _ev_cpu = st.plotly_chart(fig_cpu, use_container_width=True, key="water_cpu_chart", on_select="rerun")
            _sel_cpu = _ev_cpu.selection.points if _ev_cpu and hasattr(_ev_cpu, "selection") else []
            if _sel_cpu:
                _pt = _sel_cpu[0]
                _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
                if isinstance(_brand, (list, tuple)):
                    _brand = _brand[0]
                _fdf = _df_cpu[_df_cpu["brand"].str.contains(str(_brand)[:20], regex=False)] if _brand else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_brand}**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
            _outliers = df_m[df_m["cost_per_m3"] > _cpu_up] if _cpu_up < float("inf") else pd.DataFrame()
            if not _outliers.empty:
                st.warning(f"단가 이탈 브랜드 {len(_outliers)}개 (IQR 상한 {_cpu_up:,.0f}원/m³ 초과)")
                st.dataframe(_outliers[["brand","building","usage_m3","total_excl","cost_per_m3"]],
                             use_container_width=True, hide_index=True)

    # ═══════════════════════════ 전용/공용 분석 ═══════════════════════════════
    with tab_excl:
        st.markdown("""
        **공용요금**은 전용면적에 비례해 자동 배분됩니다 (R²≈1.0).
        **전용요금**은 실제 사용량 기반이므로 면적과의 상관이 훨씬 낮습니다.
        추세선 위쪽 브랜드 = 면적 대비 사용량/비용이 높은 테넌트입니다.
        """)

        _df_ec = df[df["size_m2"] > 0].copy()
        ec1, ec2 = st.columns(2)
        with ec1:
            _scatter_with_trendline(_df_ec, "size_m2", "total_comm", "building",
                                    "전용면적 vs 공용요금 (면적 비례 배분)",
                                    "전용면적 (m²)", "공용요금 (원)", "water_excl_comm_scatter")
        with ec2:
            _scatter_with_trendline(_df_ec, "size_m2", "total_excl", "building",
                                    "전용면적 vs 전용요금 (사용량 기반)",
                                    "전용면적 (m²)", "전용요금 (원)", "water_excl_excl_scatter")

        st.divider()
        st.subheader("전용요금 이탈 브랜드 (면적 대비 전용요금 과다)")
        # Fit line and find residuals for total_excl
        _df_m2 = _df_ec[_df_ec["total_excl"] > 0].copy()
        if len(_df_m2) >= 4:
            _c2 = np.polyfit(_df_m2["size_m2"].values, _df_m2["total_excl"].values, 1)
            _df_m2["예상_전용요금"] = np.polyval(_c2, _df_m2["size_m2"].values)
            _df_m2["초과율(%)"]    = ((_df_m2["total_excl"] - _df_m2["예상_전용요금"]) / _df_m2["예상_전용요금"] * 100).round(1)
            _res_up = _iqr_upper(_df_m2["초과율(%)"])
            _over = _df_m2[_df_m2["초과율(%)"] > max(0, _res_up)].sort_values("초과율(%)", ascending=False)
            if not _over.empty:
                st.dataframe(_over[["brand","building","floor","size_m2","total_excl","예상_전용요금","초과율(%)"]],
                             use_container_width=True, hide_index=True)
            else:
                st.success("전용요금 이탈 브랜드 없음")

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
            texttemplate="%{text}", textfont=dict(size=13, color="white"),
            xgap=3, ygap=3,
        ))
        fig_hm.update_layout(
            height=max(300, len(_hm_brands)*22+80), margin=dict(l=10,r=20,t=40,b=40),
            plot_bgcolor="white", yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        )
        _ev_heatmap = st.plotly_chart(fig_hm, use_container_width=True, key="water_anom_heatmap", on_select="rerun")
        _sel_heatmap = _ev_heatmap.selection.points if _ev_heatmap and hasattr(_ev_heatmap, "selection") else []
        if _sel_heatmap:
            _pt = _sel_heatmap[0]
            _brand = _pt.get("y") or ""
            if isinstance(_brand, (list, tuple)):
                _brand = _brand[0]
            _fdf = df[df["brand"] == _brand] if _brand else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_brand}**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

        _fi = _flags[_flags["플래그 수"] > 0].index
        if len(_fi) > 0:
            st.subheader("플래그 상세")
            _detail = df[df["brand"].isin(_fi)][
                ["brand","building","floor","size_m2","usage_m3","total_excl","total_comm","total"]].copy()
            _meta = _flags[["플래그 수","등급"]].reset_index().rename(columns={"index":"brand"})
            _detail = _detail.merge(_meta, on="brand", how="left")
            _detail = _detail.sort_values("플래그 수", ascending=False).reset_index(drop=True)
            st.dataframe(
                _detail,
                column_config={
                    "brand":       st.column_config.TextColumn("브랜드"),
                    "building":    st.column_config.TextColumn("건물"),
                    "floor":       st.column_config.TextColumn("층"),
                    "size_m2":     st.column_config.NumberColumn("면적 (m²)", format="%.0f"),
                    "usage_m3":    st.column_config.ProgressColumn(
                        "사용량 (m³)", format="%,.0f", min_value=0,
                        max_value=int(_detail["usage_m3"].max()) if not _detail["usage_m3"].empty and _detail["usage_m3"].max() > 0 else 1,
                    ),
                    "total_excl":  st.column_config.NumberColumn("전용부과 (원)", format="%,.0f"),
                    "total_comm":  st.column_config.NumberColumn("공용부과 (원)", format="%,.0f"),
                    "total":       st.column_config.NumberColumn("총부과 (원)", format="%,.0f"),
                    "플래그 수":   st.column_config.NumberColumn("플래그", format="%d"),
                    "등급":        st.column_config.TextColumn("등급"),
                },
                use_container_width=True,
                hide_index=True,
            )
