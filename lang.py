"""lang.py — Korean / English internationalisation.

Usage:
    from lang import t
    st.header(t("nav_sheet_view"))

Language is stored in st.session_state["lang"] ("ko" | "en").
Default is "ko".
"""
import streamlit as st

_S: dict[str, dict[str, str]] = {
    # ── Sidebar ──────────────────────────────────────────────────────────────
    "lang_label":       {"ko": "언어 / Language", "en": "언어 / Language"},
    "upload":           {"ko": "업로드",           "en": "Upload"},
    "upload_label":     {"ko": "CSV / XLSX / Parquet 업로드", "en": "Upload CSV / XLSX / Parquet"},
    "settings":         {"ko": "⚙️ 설정",          "en": "⚙️ Settings"},
    "quick_presets":    {"ko": "⚡ 빠른 설정",      "en": "⚡ Quick presets"},
    "preset_custom":    {"ko": "사용자 정의",       "en": "Custom"},
    "preset_default":   {"ko": "기본 (20%)",       "en": "Default (20%)"},
    "preset_gentle":    {"ko": "완만 (10%)",        "en": "Gentle (10%)"},
    "preset_dense":     {"ko": "조밀 (30%)",        "en": "Dense (30%)"},
    "bins":             {"ko": "구간 수",           "en": "Bins"},
    "tail_pct":         {"ko": "꼬리 %",           "en": "Tail %"},
    "tail_help":        {"ko": "변화량과 % 값의 하위/상위 N% 표시", "en": "Show bottom N% and top N% of change and pct values"},
    "debug":            {"ko": "디버그",            "en": "Debug"},
    # ── App / Navigation ─────────────────────────────────────────────────────
    "upload_prompt":    {"ko": "최소 하나의 파일을 업로드하세요.", "en": "Upload at least one file."},
    "select_file":      {"ko": "파일 선택",         "en": "Select file"},
    "no_sheets_warn":   {"ko": "지원되는 시트를 찾을 수 없습니다.", "en": "No supported sheets found in this file."},
    "nav_anomaly":      {"ko": "🚨 이상감지",        "en": "🚨 Anomaly"},
    "nav_insight":      {"ko": "📊 인사이트",       "en": "📊 Insight"},
    "nav_profile":      {"ko": "🏢 브랜드",         "en": "🏢 Brand"},
    "nav_detail":       {"ko": "📋 상세",           "en": "📋 Detail"},
    "nav_sheet_view":   {"ko": "📋 시트 보기",      "en": "📋 Sheet View"},
    "nav_analysis":     {"ko": "📊 분석",           "en": "📊 Analysis"},
    "analysis_select":  {"ko": "분석 선택",         "en": "Select Analysis"},
    "select_sheet":     {"ko": "시트 선택",         "en": "Select sheet"},
    "summary_analysis": {"ko": "📋 유틸리티 요약",   "en": "📋 Utility Summary"},
    "biz_analysis":     {"ko": "📊 비즈니스 분석",   "en": "📊 Business Analysis"},
    "biz_header":       {"ko": "📊 비즈니스 분석 — 사용량 · 비용 · 이상감지", "en": "📊 Business Analysis — Usage · Cost · Anomaly"},
    "biz_tab_cost":     {"ko": "💰 비용 분석",       "en": "💰 Cost Analysis"},
    "biz_tab_eff":      {"ko": "📈 사용량 · 효율",   "en": "📈 Usage & Efficiency"},
    "biz_tab_anom":     {"ko": "🚨 이상감지",        "en": "🚨 Anomaly Detection"},
    "eff_header":       {"ko": "📈 사용량 · 효율 — 면적당 에너지 사용량", "en": "📈 Usage & Efficiency — Energy Usage per Area"},
    "anomaly_header":   {"ko": "🚨 이상감지 — 교차 시트 통합 이상 점수", "en": "🚨 Anomaly Detection — Cross-Sheet Composite Signals"},
    "summary_header":   {"ko": "📋 유틸리티 요약 — 시트별 사용량 · 비용", "en": "📋 Utility Summary — Sheet-Level Usage & Cost"},
    "cost_header":      {"ko": "💰 비용 분석 — 교차 시트 분석",    "en": "💰 Cost Analysis — Cross-Sheet Breakdown"},
    "no_util_sheets":   {"ko": "로드 가능한 유틸리티 시트가 없습니다.", "en": "No loadable utility sheets found."},
    "meter_load_fail":  {"ko": "검침 내역 로드 실패",  "en": "Failed to load 검침 내역"},
    # ── Meter view — filters ─────────────────────────────────────────────────
    "building":         {"ko": "건물",             "en": "Building"},
    "floor":            {"ko": "층",               "en": "Floor"},
    "category":         {"ko": "분류",             "en": "Category"},
    "vacancy":          {"ko": "공실 필터",         "en": "Vacancy (공실)"},
    "vacancy_all":      {"ko": "전체",             "en": "All"},
    "vacancy_exclude":  {"ko": "공실 제외",         "en": "Exclude Vacancy"},
    "vacancy_only":     {"ko": "공실만",            "en": "Vacancy Only"},
    "no_data_building": {"ko": "선택한 건물에 데이터가 없습니다.", "en": "No data for the selected building."},
    "no_data_floor":    {"ko": "선택한 층 조합에 데이터가 없습니다.", "en": "No data for the selected floor combination."},
    "no_numeric":       {"ko": "수치 데이터가 없습니다.", "en": "No numeric data."},
    # ── Meter view — KPIs ────────────────────────────────────────────────────
    "tenants":          {"ko": "입점 업체",         "en": "Tenants"},
    "critical":         {"ko": "🔴 위험",          "en": "🔴 Critical"},
    "watch":            {"ko": "🟠 주의",          "en": "🟠 Watch"},
    "alert":            {"ko": "🟡 경보",          "en": "🟡 Alert"},
    "kpi_across":       {"ko": "전체 유틸리티",     "en": "across any utility"},
    "kpi_elevated":     {"ko": "사용량 상승",       "en": "elevated usage"},
    "kpi_sharp_rise":   {"ko": "급격한 % 상승",    "en": "sharp % rise"},
    # ── Meter view — backward readings ───────────────────────────────────────
    "backward_expander":{"ko": "⚠️ 데이터 품질 — 역방향 검침 감지됨", "en": "⚠️ Data Quality — backward meter reading(s) detected"},
    "backward_warning": {"ko": "다음 업체의 현재 검침값이 이전 검침값보다 낮습니다. 계량기 교체 없이 물리적으로 불가능하며 데이터 입력 오류일 가능성이 높습니다. 해당 행은 분석에 포함되나 변화량은 음수로 표시됩니다.",
                         "en": "The following tenants have a current meter reading lower than the previous reading. This is physically impossible without a meter reset and likely indicates a data entry error. These rows are still included but their change values will appear negative."},
    # ── Meter view — report download ─────────────────────────────────────────
    "download_report":  {"ko": "요약 보고서 다운로드", "en": "Download Summary Report"},
    "report_caption":   {"ko": "모든 유틸리티 유형에 대한 차트, 설명, 주요 업체가 포함된 보고서를 생성합니다.",
                         "en": "Generates a PDF report covering all utility types — with charts, explanations, and flagged tenants."},
    "report_lang":      {"ko": "보고서 언어",       "en": "Report language / 보고서 언어"},
    "gen_report_btn":   {"ko": "보고서 생성",       "en": "Generate Report"},
    "report_spinning":  {"ko": "보고서 생성 중…",  "en": "Building PDF report…"},
    "dl_pdf_btn":       {"ko": "PDF 다운로드",     "en": "Download PDF Report"},
    # ── Meter view — histograms ───────────────────────────────────────────────
    "hist_view":        {"ko": "히스토그램 보기",   "en": "Histogram view"},
    "hist_side":        {"ko": "나란히",            "en": "Side by Side"},
    "hist_change_only": {"ko": "변화량만",          "en": "Change only"},
    "hist_pct_only":    {"ko": "% 변화만",          "en": "% Change only"},
    # ── Meter view — tabs ────────────────────────────────────────────────────
    "tab_efficiency":   {"ko": "효율성",           "en": "Efficiency"},
    "tab_change":       {"ko": "수치 변화",         "en": "Quantitative Change"},
    "tab_pct":          {"ko": "비율 변화",         "en": "Percentage Change"},
    "tab_quadrant":     {"ko": "사분면 분석",       "en": "Quadrant Analysis"},
    "tab_ranking":      {"ko": "브랜드 순위",       "en": "Brand Ranking"},
    "tab_corr":         {"ko": "상관관계",          "en": "Correlation"},
    # ── Meter view — change/pct tab content ──────────────────────────────────
    "show":             {"ko": "표시",             "en": "Show"},
    "show_all":         {"ko": "전체",             "en": "All"},
    "show_top":         {"ko": "상위",             "en": "Top"},
    "show_bottom":      {"ko": "하위",             "en": "Bottom"},
    "all_entries":      {"ko": "전체 항목",         "en": "All entries"},
    "sorted_hl":        {"ko": "높은순 정렬",       "en": "sorted high→low"},
    "no_data_nan":      {"ko": "데이터 없음 (NaN)", "en": "No Data (NaN)"},
    "missing_change":   {"ko": "수치 변화 누락",    "en": "missing quantitative change"},
    "missing_pct":      {"ko": "비율 변화 누락",    "en": "missing percentage change"},
    # ── Meter view — quadrant tab ─────────────────────────────────────────────
    "quadrant_title":   {"ko": "이상치 사분면 분석 — 변화량 × 비율 교차 필터",
                         "en": "Outlier Quadrant Analysis — Change × Pct Cross-Filter"},
    "q_HH":             {"ko": "**위험 급등** — 변화 HIGH · 비율 HIGH",   "en": "**Critical Surge** — Change HIGH · Pct HIGH"},
    "q_HL":             {"ko": "**큰 기저, 완만한 급등** — 변화 HIGH · 비율 LOW", "en": "**Large Base, Moderate Surge** — Change HIGH · Pct LOW"},
    "q_LH":             {"ko": "**작은 기저, 급락** — 변화 LOW · 비율 HIGH",      "en": "**Small Base, Sharp Drop** — Change LOW · Pct HIGH"},
    "q_LL":             {"ko": "**안정 / 유의한 변화 없음** — 변화 LOW · 비율 LOW",
                         "en": "**Stable / No Significant Change** — Change LOW · Pct LOW"},
    "q_normal":         {"ko": "**정상 (비이상치)** — 어떤 사분면에도 속하지 않음",
                         "en": "**Normal (non-outliers)** — not in any quadrant"},
    # ── Meter view — brand ranking tab ───────────────────────────────────────
    "ranking_title":    {"ko": "브랜드 중요도 순위",  "en": "Brand Significance Ranking"},
    "no_brand_data":    {"ko": "브랜드 데이터가 없습니다.", "en": "No brand data available."},
    "score_explain":    {"ko": "중요도 점수 계산 방법?", "en": "How is the significance score calculated?"},
    # ── Utility category labels ───────────────────────────────────────────────
    "cat_water":        {"ko": "💧 수도",          "en": "💧 Water"},
    "cat_hwater":       {"ko": "🌡️ 온수",         "en": "🌡️ Hot Water"},
    "cat_elect":        {"ko": "⚡ 전기",          "en": "⚡ Electricity"},
    "cat_heat":         {"ko": "🔥 난방",          "en": "🔥 Heat"},
    # ── Efficiency tab ────────────────────────────────────────────────────────
    "eff_single_title": {"ko": "면적당 사용량 — 단일 유틸리티", "en": "Per-Area Usage — Single Utility"},
    "eff_utility_sel":  {"ko": "유틸리티",          "en": "Utility"},
    "eff_ehp_title":    {"ko": "EHP 전기 사용량 (m²당)", "en": "EHP Electricity Usage per m²"},
    "eff_no_ehp":       {"ko": "EHP 사용 데이터가 없습니다.", "en": "No EHP usage data found."},
    "eff_no_ehp_match": {"ko": "현재 선택에서 EHP 데이터와 브랜드를 매칭할 수 없습니다.", "en": "Could not match EHP data to brands in current view."},
    "eff_no_size":      {"ko": "면적 데이터가 없습니다. 시트에 size_m2 값이 있는지 확인하세요.", "en": "No per-area data available. Ensure size_m2 values are present in the sheet."},
    "eff_combined_title":{"ko": "종합 효율성 점수",  "en": "Combined Efficiency Score"},
    "eff_combined_cap": {"ko": "각 유틸리티의 m²당 사용량을 [0, 1]로 정규화합니다 (0 = 최소, 1 = 최대). 종합 점수는 평균. **높을수록 비효율적.**",
                         "en": "Each utility's per-m² usage is normalized to [0, 1] (0 = lowest, 1 = highest). The combined score is the mean. **Higher = more consumption per m² = less efficient.**"},
    "eff_need_two":     {"ko": "종합 점수는 2개 이상의 유틸리티 데이터가 필요합니다.", "en": "Combined score requires at least 2 utilities with size data."},
    "load_ehp_btn":     {"ko": "EHP 효율 데이터 불러오기", "en": "Load EHP Efficiency Data"},
    "ehp_spinner":      {"ko": "EHP 데이터 분석 중…",   "en": "Analyzing EHP data…"},
    "ehp_load_fail":    {"ko": "EHP 데이터를 불러올 수 없음", "en": "Could not load EHP data"},
    # ── Cost & Breakdown tab ──────────────────────────────────────────────────
    "cross_load_btn":   {"ko": "교차 시트 데이터 불러오기", "en": "Load Cross-Sheet Features"},
    "cross_loading":    {"ko": "교차 시트 데이터 로딩 중…", "en": "Loading cross-sheet data…"},
    "cross_no_sheets":  {"ko": "추가 시트가 없습니다. 청구 또는 전기 시트가 필요합니다.", "en": "No additional sheets found. Billing and/or electricity detail sheet required."},
    "cross_avail":      {"ko": "사용 가능한 시트",    "en": "Available sheets"},
    "cross_unit_title": {"ko": "단가 분석",           "en": "Unit Cost Analysis"},
    "cross_unit_cap":   {"ko": "소비 단위당 비용. **Z-점수**는 그룹 평균으로부터의 편차입니다.",
                         "en": "Cost per unit of consumption. **Z-score** shows how far each brand's unit cost is from the group mean."},
    "cross_anomaly":    {"ko": "이상치 브랜드",        "en": "anomaly brand(s)"},
    "cross_full_table": {"ko": "전체 테이블",          "en": "Full table"},
    "cross_elec_title": {"ko": "전기 사용 카테고리 분류", "en": "Electricity Breakdown by Category"},
    "cross_elec_cap":   {"ko": "브랜드별 전기 소비 카테고리 비율. **HVAC** = EHP + FCU + AHU. **기본** = 일반 + 펌프 + 주방 환풍기.",
                         "en": "Share of total electricity by category. **HVAC** = EHP + FCU + AHU. **Base** = general + pump + kitchen fan."},
    "cross_elec_full":  {"ko": "전체 전기 분류 테이블", "en": "Full electricity breakdown table"},
    "cross_unit_fail":  {"ko": "단가 계산 실패",       "en": "Unit cost computation failed"},
    "cross_elec_fail":  {"ko": "전기 분류 실패",       "en": "Electricity breakdown failed"},
    "cross_no_data":    {"ko": "어떤 시트에서도 데이터를 불러올 수 없습니다.", "en": "No data could be loaded from either sheet."},
    # ── Reconciliation ────────────────────────────────────────────────────────
    "recon_expander":   {"ko": "청구 ↔ 검침 대사",    "en": "Billing ↔ Meter Reconciliation"},
    "recon_caption":    {"ko": "청구 시트(수도광열비 부과 내역)와 검침 데이터를 비교하여 누락 업체를 표시합니다.",
                         "en": "Compares the billing sheet against meter readings. Flags tenants present in one source but missing in the other."},
    "recon_billed_no_meter": {"ko": "청구됐지만 검침 없음",  "en": "Billed but no meter reading"},
    "recon_metered_no_bill": {"ko": "검침됐지만 청구 없음",  "en": "Metered but not billed"},
    "recon_all_billed": {"ko": "청구된 모든 업체에 검침 데이터가 있습니다.", "en": "All billed tenants have meter readings."},
    "recon_all_metered":{"ko": "검침된 모든 업체가 청구 시트에 있습니다.", "en": "All metered tenants appear on the billing sheet."},
    "recon_billed_zero":{"ko": "청구됐지만 검침 사용량이 0인 업체",       "en": "Billed non-zero but zero meter usage"},
    "recon_fail":       {"ko": "대사 실패",            "en": "Reconciliation failed"},
}


def t(key: str) -> str:
    """Return the UI string for the current language (defaults to Korean)."""
    lang = st.session_state.get("lang", "ko")
    entry = _S.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("en", key))


def lang() -> str:
    """Return the current language code: 'ko' or 'en'."""
    return st.session_state.get("lang", "ko")
