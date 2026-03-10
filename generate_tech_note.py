"""Generate technical design note PDF for utility_analysis dashboard."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted,
)
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = "utility_analysis_tech_note.pdf"

# ── Korean fonts ───────────────────────────────────────────────────────────────
_NANUM_REG  = "/home/hansoullee20/.fonts/NanumGothic-Regular.ttf"
_NANUM_BOLD = "/home/hansoullee20/.fonts/NanumGothic-Bold.ttf"
pdfmetrics.registerFont(TTFont("NanumGothic",     _NANUM_REG))
pdfmetrics.registerFont(TTFont("NanumGothic-Bold", _NANUM_BOLD))

_KO   = "NanumGothic"
_KO_B = "NanumGothic-Bold"
_MONO = "Courier"

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

H1 = ParagraphStyle("H1", fontName=_KO_B, fontSize=18, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a2e"), leading=26)
H2 = ParagraphStyle("H2", fontName=_KO_B, fontSize=13, spaceAfter=4,
                    spaceBefore=14, textColor=colors.HexColor("#16213e"), leading=20)
H3 = ParagraphStyle("H3", fontName=_KO_B, fontSize=11, spaceAfter=3,
                    spaceBefore=8,  textColor=colors.HexColor("#0f3460"), leading=17)
BODY = ParagraphStyle("Body", fontName=_KO, fontSize=9.5, leading=16,
                      spaceAfter=4, textColor=colors.HexColor("#222222"))
CAPTION = ParagraphStyle("Caption", fontName=_KO, fontSize=8.5,
                          textColor=colors.HexColor("#555555"), leading=13, spaceAfter=3)
MONO = ParagraphStyle("Mono", fontName=_MONO, fontSize=8, leading=11,
                      spaceAfter=2, textColor=colors.HexColor("#1a1a1a"),
                      backColor=colors.HexColor("#f5f5f5"),
                      leftIndent=6, rightIndent=6)

DIVIDER_COLOR = colors.HexColor("#cccccc")


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6, spaceBefore=2)

def h1(text):  return Paragraph(text, H1)
def h2(text):  return Paragraph(text, H2)
def h3(text):  return Paragraph(text, H3)
def p(text):   return Paragraph(text, BODY)
def cap(text): return Paragraph(text, CAPTION)
def sp(h=4):   return Spacer(1, h * mm)
def code(text): return Preformatted(text, MONO)


def table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME",      (0, 0), (-1, -1), _KO),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), _KO_B),
        ]
    for i in range(1, len(data)):
        bg = colors.HexColor("#f9f9f9") if i % 2 == 0 else colors.white
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


# ── Content ────────────────────────────────────────────────────────────────────

def build():
    M = 18 * mm
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
    )

    story = []

    # ── Title ──────────────────────────────────────────────────────────────────
    story += [
        sp(2),
        h1("Utility Analysis Dashboard"),
        Paragraph("기술 설계 노트 (Technical Design Note)",
                  ParagraphStyle("Sub", fontName=_KO, fontSize=12,
                                 textColor=colors.HexColor("#555555"), spaceAfter=2, leading=18)),
        cap("작성일: 2026-03-10  ·  내부 참고용"),
        hr(), sp(2),
    ]

    # ── 1. 스택 ────────────────────────────────────────────────────────────────
    story += [h2("1. 스택 (Stack)"), hr()]
    story += [
        p("<b>런타임:</b> Python 3.11+, Streamlit, Plotly (go + px), pandas / numpy"),
        p("<b>가상환경:</b> 공유 venv — <font name='Courier'>../finance_vis/venv_finance</font> "
          "(새로 생성 금지)"),
        p("<b>실행:</b>"),
        code("source ../finance_vis/venv_finance/bin/activate && streamlit run app.py"),
        p("<b>입력 파일:</b> 한국어 <font name='Courier'>.xlsm/.xlsx</font> — "
          "<font name='Courier'>./data/</font> 폴더 (사이드바에서 경로 변경 가능)"),
        sp(),
    ]

    # ── 2. 모듈 구조 ────────────────────────────────────────────────────────────
    story += [h2("2. 모듈 구조 (Module Map)"), hr()]
    story += [code("""\
app.py              <- 페이지 설정, 최상위 라우팅, 파일 로딩
sidebar.py          <- 파일 로더 (폴더 스캔 -> _FileEntry 목록), bins/tail 기본값
data.py             <- Excel I/O (@st.cache_data), 시트명 상수, 검침 기간 파싱
features.py         <- 컬럼 엔지니어링: create_change_columns, build_from_two_files,
                       aggregate_by_brand, split_brand_by_floor, 층 파싱 헬퍼
filters.py          <- render_meter_filters, show_filter_widgets, apply_sheet_filter,
                       brand_search_bar
viz.py              <- plot_hist_with_tails (Plotly go.Bar 히스토그램 + IQR 음영)
lang.py             <- t() 번역 헬퍼 (한국어 전용, key->string)

meter_view.py       <- 검침내역 전체 파이프라인: 로드->필터->히스토그램->5개 탭
summary.py          <- 총 유틸리티 순위 + 면적당 총비용
billing.py          <- 수도광열비 부과 내역 뷰
ehp.py              <- EHP(OAC)검침자료 뷰
water.py / hotwater.py / electricity.py  <- 시트별 뷰
brand_profile.py    <- 브랜드 프로필 탭 (피어 비교 차트)

tab_anomaly.py      <- 이상감지 탭 (복합 점수, 히트맵, 급등/비용/HVAC 서브탭)
tab_cross.py        <- 비용분석 탭 (단위 비용, 전기 분류)
tab_efficiency.py   <- 효율분析 탭 (m2당 사용량 벤치마킹)
tab_corr.py / tab_reconciliation.py  <- 상관관계 & 대사 탭

anomaly_features.py <- build_anomaly_df: 검침+청구+전기+수도 -> 이상 점수
cross_features.py   <- build_unit_costs, build_elec_breakdown
report.py           <- generate_report_pdf (검침내역 메인 리포트)
biz_report.py       <- generate_anomaly_pdf, generate_cross_pdf, generate_efficiency_pdf
billing_report.py / ehp_report.py / hvac_report.py  <- 시트별 PDF 생성기"""), sp()]

    # ── 3. 내비게이션 ──────────────────────────────────────────────────────────
    story += [h2("3. 내비게이션 구조"), hr()]
    story += [code("""\
sac.tabs (사이드바, 세로 배치) -> nav_mode
  +-- 시트 보기     -> 시트 selectbox -> billing/ehp/water/hotwater/electricity/meter_view
  +-- 분析          -> 분析 selectbox
  |     +-- 요약 분析 -> render_summary_view(water_df, hw_df, el_df)
  |     +-- 업체 분析 -> render_meter_filters -> [이상감지|비용분析|효율분析] 탭
  +-- 브랜드 프로필  -> render_brand_profile_tab"""),
        sp(2),
        p("<b>핵심 주의사항:</b> <font name='Courier'>sac.tabs</font>는 반드시 "
          "<font name='Courier'>with st.sidebar:</font> 내부에서 직접 렌더링해야 합니다. "
          "<font name='Courier'>st.empty()</font> 안에 넣으면 커스텀 컴포넌트가 "
          "매 리런마다 재초기화되어 선택 상태가 초기화됩니다."),
        sp(),
    ]

    # ── 4. 데이터 파이프라인 ────────────────────────────────────────────────────
    story += [h2("4. 데이터 파이프라인 (검침내역)"), hr()]
    story += [code("""\
read_sheet()                       # raw Excel, header=[2, 3, 4]
  └─ apply_header_rows()           # MultiIndex 헤더 -> 명명된 컬럼으로 평탄화
       └─ build_from_two_files()   # 이전 파일 -> *_previous = 전월 사용량
            └─ create_change_columns()  # *_change = curr-prev, *_pct = change/prev*100
                 └─ aggregate_by_brand()   # (brand, building) 그룹별 사용량 합산
                      └─ split_brand_by_floor()  # 층 필터 시 브랜드 합계를 층수로 균등 분할"""),
        sp(2),
        p("<b>누적 vs 사용량:</b> Excel의 검침값은 누적값입니다. "
          "<font name='Courier'>build_from_two_files</font>가 이를 "
          "<font name='Courier'>*_meter_curr/prev</font>로 이름 변경하고, "
          "미리 계산된 사용량 컬럼(<font name='Courier'>water_usage_m3</font> 등)을 "
          "당월 실제 사용량으로 사용합니다."),
        sp(2),
        h3("컬럼 명명 규칙"),
        table([
            ["컬럼명", "의미"],
            ["water_previous / water_current", "이전 파일/현재 파일의 당월 사용량"],
            ["water_change", "current - previous (사용량 변화)"],
            ["water_pct", "change / previous × 100 (변화율 %)"],
            ["water_meter_prev / water_meter_curr", "원본 누적 검침값 (역방향 감지용으로 보존)"],
        ], col_widths=[80*mm, 100*mm]),
        sp(),
    ]

    # ── 5. 층 로직 ─────────────────────────────────────────────────────────────
    story += [h2("5. 층 로직 (Floor Logic)"), hr()]
    story += [
        p("층 값은 복합 문자열입니다: "
          "<font name='Courier'>\"1F/2F\"</font>, "
          "<font name='Courier'>\"2F~5F\"</font>, "
          "<font name='Courier'>\"B2F/B1F\"</font>."),
        sp(1),
        table([
            ["함수", "역할"],
            ["parse_floor_value(s)", "복합 층 값에서 개별 층 문자열 리스트 반환"],
            ["get_simple_floors(df)", "multiselect 위젯용 정렬된 고유 층 목록"],
            ["split_brand_by_floor(df, sel_floors)", "브랜드 합계를 매칭 층수로 균등 분할"],
        ], col_widths=[75*mm, 105*mm]),
        sp(2),
        p("<font name='Courier'>sel_floors == [\"All\"]</font> 이고 "
          "<font name='Courier'>sel_bldg == [\"All\"]</font> 이면 → 분할 없이 합계 사용."),
        sp(),
    ]

    # ── 6. 히스토그램 시스템 ────────────────────────────────────────────────────
    story += [h2("6. 히스토그램 시스템 (viz.py: plot_hist_with_tails)"), hr()]
    story += [
        h3("주요 파라미터"),
        code("""\
plot_hist_with_tails(s, bins, lo, hi, title,
    source_df=None, val_col=None, key="hist",
    display_cols=None, tail_pct=None, val_scale=1.0)"""),
        sp(2),
        p("• <b>lo / hi</b>: 플롯된 시리즈 s와 <b>동일한 단위</b>여야 함"),
        p("• <b>val_scale</b>: s가 스케일된 경우(예: /1e4 → 만원) source_df의 원본 컬럼과 "
          "단위를 맞추는 브리지 파라미터"),
        code("mask = (source_df[val_col] / val_scale >= x0) & (source_df[val_col] / val_scale <= x1)"),
        p("• <b>스타일 고정</b>: 파란 정상 막대 (#4C72B0), 주황 꼬리 막대 (#DD8A00), "
          "빨간 중앙값 선 (#C44E52), 흰 배경, 높이 380px"),
        p("• <b>이상치 테이블</b>: source_df 제공 시 lo/hi 범위 밖 항목을 "
          "자동으로 '이상치 목록' 익스팬더에 표시"),
        sp(2),
        h3("IQR 이상치 감지 패턴"),
        code("""\
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr    = q3 - q1
lo     = q1 - k * iqr   # k 슬라이더: 0.5~3.0, 기본값 1.5, 스텝 0.25
hi     = q3 + k * iqr
# LaTeX 수식 표시:
st.markdown(
    f"$$Q_1={q1:,.0f},\\;Q_3={q3:,.0f},\\;IQR={iqr:,.0f}$$\\n\\n"
    f"$$\\text{{Lower}}={lo:,.0f},\\;\\text{{Upper}}={hi:,.0f}\\;(k={k})$$"
)"""),
        sp(),
    ]

    # ── 7. 원화 단위 포매팅 ─────────────────────────────────────────────────────
    story += [h2("7. 원화 단위 포매팅"), hr()]
    story += [code("""\
def _fmt_won(v):
    if abs(v) >= 1e8: return f"{v/1e8:,.0f} 억원"
    if abs(v) >= 1e4: return f"{v/1e4:,.0f} 만원"
    return f"{v:,.0f} 원"

# 막대 차트: max(series)로 _div 계산 후 val_scale에 동일하게 전달
_div, _unit = (1e8, "억원") if _max >= 1e8 else (1e4, "만원")
_xv = series / _div
# plot_hist_with_tails(..., val_scale=_div) 로 bin 클릭 필터도 정상 동작"""),
        p("규칙: <b>원 단위 값에 소수점 없음</b> — 항상 <font name='Courier'>:.0f</font>."),
        sp(),
    ]

    # ── 8. 이상감지 점수 ────────────────────────────────────────────────────────
    story += [h2("8. 이상감지 점수 (anomaly_features.py)"), hr()]
    story += [
        p("복합 점수 = 5개 구성 요소의 가중 합산, 각각 [0, 1]로 정규화:"),
        sp(1),
        table([
            ["구성 요소", "가중치", "신호", "원본 시트"],
            ["급등 (Spike)",       "30%", "전월 대비 증가율 절댓값 기준 (100/50/20%)", "검침내역"],
            ["소비 (Consumption)", "25%", "사분면 점수: HH=4, HL=3, LH=2, Normal=1, LL=0", "검침내역"],
            ["비용 (Cost)",        "25%", "Z-점수: 원/m3, 원/kWh, 만원/m2", "수도광열비 부과 내역"],
            ["HVAC",               "10%", "kWh/m2 IQR 정규화", "전체 전기 사용내역"],
            ["일관성 (Consistency)", "10%", "사용량=0 유틸리티 항목 수", "검침내역 + 수도/온수 시트"],
        ], col_widths=[35*mm, 18*mm, 82*mm, 45*mm]),
        sp(2),
        p("<b>위험 등급:</b> "
          "위험(빨강) >= 0.65  ·  주의(주황) >= 0.40  ·  관찰(노랑) >= 0.20  ·  정상(초록) < 0.20"),
        sp(),
    ]

    # ── 9. PDF 생성 패턴 ────────────────────────────────────────────────────────
    story += [h2("9. PDF 생성 패턴 (업체 분析 탭 공통)"), hr()]
    story += [code("""\
_pdf_key = f"{tab}_pdf_{file_name}"
_col_gen, _col_dl = st.columns([1, 2])
with _col_gen:
    if st.button("PDF 리포트 생성", key=f"gen_{tab}_pdf_{file_name}"):
        with st.spinner("PDF 생성 중..."):
            st.session_state[_pdf_key] = generate_X_pdf(df)
if _pdf_key in st.session_state:
    with _col_dl:
        st.download_button("다운로드", st.session_state[_pdf_key], ...)"""),
        p("생성 후 캐시 패턴: PDF 바이트를 session_state에 저장하여 "
          "Streamlit 리런 시 재생성 방지. 생성 버튼과 다운로드 버튼을 인접 컬럼에 분리 배치."),
        sp(),
    ]

    # ── 10. 흔한 실수 ───────────────────────────────────────────────────────────
    story += [h2("10. 흔한 실수 (Common Pitfalls)"), hr()]
    pitfalls = [
        ["#", "문제", "해결책"],
        ["1", "루프 클로저 버그",
         "기본 인수 바인딩 사용: def fn(key, _p=p, _cc=cc): — 루프 변수 직접 참조 금지"],
        ["2", "sac.tabs 상태 초기화",
         "st.empty() 안에 넣지 말 것. with st.sidebar: 내부에 직접 렌더링"],
        ["3", "bin 클릭 시 단위 불일치",
         "시리즈를 스케일한 경우 val_scale=스케일 인수를 plot_hist_with_tails에 전달"],
        ["4", "라디오 옵션 순서",
         "히스토그램 옵션 항상 첫 번째: [\"히스토그램\", ...] 또는 [\"% Change only\", \"Change only\", \"Side by Side\"]"],
        ["5", "원 단위 소수점",
         "항상 :.0f, 절대로 :.2f 사용 금지"],
        ["6", "EHP 컬럼 파싱",
         "누적 컬럼: M~DG (0-based 12~110); 다음 ▣ 섹션 헤더에서 파싱 중단; 두 테이블 병합 금지"],
        ["7", "sac.tabs ValueError",
         "저장된 인덱스가 탭 수 이상일 때 발생. nav_{file_name} 세션 스테이트 키 삭제로 초기화"],
    ]
    story += [table(pitfalls, col_widths=[8*mm, 55*mm, 117*mm]), sp()]

    # ── 11. 세션 상태 키 ────────────────────────────────────────────────────────
    story += [h2("11. 주요 세션 상태 키"), hr()]
    story += [
        table([
            ["키", "역할"],
            ["nav_{file_name}",              "사이드바 내비게이션 탭 인덱스 (sac.tabs)"],
            ["anomaly_loaded_{file_name}",   "이상감지 분析 시작 여부 게이트"],
            ["cross_loaded_{file_name}",     "크로스 시트 데이터 로드 여부 게이트"],
            ["anomaly_pdf_{file_name}",      "캐시된 이상감지 PDF 바이트"],
            ["cross_pdf_{file_name}",        "캐시된 비용분析 PDF 바이트"],
            ["efficiency_pdf_{file_name}",   "캐시된 효율분析 PDF 바이트"],
            ["{prefix}_iqr_k",              "히스토그램별 IQR k 배수"],
            ["{prefix}_bins / {prefix}_bins_i", "Bins 슬라이더/숫자입력 동기화 쌍"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # ── 12. 미결 사항 ────────────────────────────────────────────────────────────
    story += [h2("12. 미결 사항 / 알려진 이슈"), hr()]
    story += [
        p("• <b>면적당 총비용 단위 스케일링</b>: 만원/m2 표시 필요 (현재 원/m2 또는 만원). "
          "사용자 요청으로 롤백됨 — 재구현 준비 완료."),
        p("• 2026-03-10 기준 다른 미결 사항 없음."),
        sp(4),
    ]

    doc.build(story)
    print(f"PDF 생성 완료 -> {OUT}")


if __name__ == "__main__":
    build()
