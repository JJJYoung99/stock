# Magnitude 측정 실패 원인과 알고리즘 개선

작성: 2026-05-02 (브랜치 `claude/stock-trading-algorithm-CrPJz`)

---

## 1. 왜 magnitude 가 측정 안 됐나 — 실제 사례 5건

### Case A. KLAC EARNINGS_BEAT (2026-04-30)
- 보도: "KLA reported $9.40 EPS, beat consensus $9.16 by $0.24" — [Daily Political](https://www.dailypolitical.com/2026/04/30/kla-nasdaqklac-announces-quarterly-earnings-results-beats-expectations-by-0-24-eps.html)
- 계산 가능했던 것: 단순 surprise% = (9.40−9.16)/9.16 = **+2.62%**
- **누락된 것**: 컨센서스의 표준편차 (`consensus_stdev`). 12명 애널리스트의 EPS 추정치가 9.10~9.20 사이로 좁았다면 σ≈0.05 → SUE = 0.24/0.05 = **4.0** 으로 강한 신호. 반대로 8.5~9.5 로 넓었다면 σ≈0.4 → SUE = 0.6 으로 약한 신호. **저널리즘은 둘을 구분하지 않는다.**

### Case B. KLAC ANALYST_TARGET_UP — Wells Fargo $2,100, Needham $2,000
- 보도: 두 신규 목표가만 명시
- **누락**: 직전 목표가, 그리고 active 애널리스트들의 목표가 dispersion. 둘 다 없으면 (new−old)/old 도, dispersion-scaled revision-z 도 계산 불가.
- 측정의 정석 (Stickel 1991): `revision_z = (new − old) / σ(active_targets)`. σ 는 "5명 애널리스트 목표가의 표준편차" — 보도에 절대 안 나오는 숫자.

### Case C. MSFT, AMZN, RDDT EARNINGS_BEAT (2026-04-29~30)
- 보도: "earnings and revenue beat", "above estimates", "+4% post-print"
- **누락**: 실제 actual EPS / consensus EPS / surprise %. 정성적 표현으로 압축됨.
- → magnitude 가 정의 자체로 산출 불가능. plahceholder 0.02~0.025 로 두었으나 framework 의 `confidence=0.5` 로 표시.

### Case D. 한국 증권사 목표주가 보고서 (삼성전자 27만원, SK하이닉스 180만원, 등)
- 보도: 새 목표가만 게시
- **누락**: 동일 증권사·동일 애널리스트의 직전 목표가, 발표일, dispersion across brokers
- **추가 누락**: 한국 보도는 종종 "영업이익 컨센서스 +85% 상향" 같이 *fundamentals* 변동을 보고하는데, 이건 우리 EventType 에 직접 매핑이 안 됨 → `GUIDANCE_RAISE` 로 가려면 회사 가이던스 발표 자체를 구해야 함.

### Case E. TWLO GUIDANCE_RAISE (2026-04-30)
- 보도: "raised full-year organic revenue forecast"
- **누락**: 직전 가이던스의 구체적 수치 (range mid-point). magnitude = (new_guide − old_guide)/|old_guide| 산출 불가.

---

## 2. 공통 패턴 — 뉴스 기사는 "구조화 데이터"가 아니다

저널리즘이 압축하는 3종류 데이터:

| 측정 입력 | 뉴스 기사에서 보존? | 어디서 구해야 하나 |
|---|---|---|
| actual EPS, consensus EPS | 자주 보존 | 그대로 사용 |
| **consensus_stdev** (애널리스트 분산) | 거의 안 보존 | **finnhub `earnings`**, **IBES** (paid), **Refinitiv** |
| **prior target / target dispersion** | 거의 안 보존 | finnhub `analyst_recommendations`, 네이버 증권 컨센서스, 한경 컨센서스 |
| **계약금액 / 시가총액** | KR 단일판매·공급계약 공시는 *원본 공시* 에 명시 (뉴스에선 종종 누락) | **OpenDartReader** 가 공시 본문에서 직접 파싱 |
| **이전 가이던스 mid-point** | 회사 IR 페이지 / 직전 분기 컨퍼런스콜 transcript | finnhub, AlphaSense (paid), 직접 IR 페이지 스크랩 |

→ **결론**: 뉴스 헤드라인을 점수화하는 detector 는 fallback 일 뿐, 진짜 신호는 **구조화 데이터 어그리게이터**(finnhub, IBES) 또는 **원본 공시**(DART, SEC EDGAR)에서 와야 한다.

---

## 3. 알고리즘 수정 — 이번 커밋에 들어간 것

깃허브 레퍼런스와 학술 컨센서스를 참고해 framework 코드를 6곳 손봤어.

### 3.1 Earnings: SUE 우선 산정 (Bernard-Thomas 1989; Foster-Olsen-Shevlin 1984)

`src/catalyst/events/detectors.py` 의 `EarningsSurpriseDetector` 가 **세 단계 폴백**으로 magnitude 를 산출:

```text
1. consensus_stdev 제공 시:    SUE = (actual − consensus) / consensus_stdev
2. season_baseline 제공 시:    max(|surprise%| − baseline%, 0)
3. 둘 다 없으면 (legacy):       |surprise%|
```

`extras["magnitude_source"]` 에 어느 경로를 탔는지 기록(`"sue" | "baseline_adjusted" | "fallback_pct"`).

레퍼런스: [quantopian/research_public — PEAD lectures](https://github.com/quantopian/research_public), [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) ch.7.

### 3.2 Analyst: dispersion-scaled revision-z (Stickel 1991; Womack 1996; Jegadeesh-Kim 2010)

`AnalystTargetDetector` 가 `consensus_dispersion` 을 받으면 `revision_z = (new − old)/σ(active_targets)` 로 magnitude 산출. 없으면 `(new − old)/old` 로 폴백. `extras["magnitude_source"]` 에 `"revision_z" | "fallback_pct"` 기록.

레퍼런스: [alpha-vantage/alpha_vantage](https://github.com/alpha-vantage/alpha_vantage) (`ANALYST_TARGET_PRICE` + dispersion 노출), [robertmartin8/MachineLearningStocks](https://github.com/robertmartin8/MachineLearningStocks).

### 3.3 Magnitude_source 별 별도 (scale, cap)

`configs/event_rules.yaml` 에 `magnitude_source_overrides` 추가. SUE 와 fallback%는 **단위가 다르기 때문에** 하나의 (scale, cap) 으로 누르면 둘 중 하나가 망가짐.

| event_type | source | scale | cap |
|---|---|---|---|
| EARNINGS_BEAT | sue | 1.0 | 4.0 |
| EARNINGS_BEAT | baseline_adjusted | 0.10 | 2.5 |
| EARNINGS_BEAT | fallback_pct (default) | 0.20 | 2.5 |
| ANALYST_TARGET_UP | revision_z | 0.5 | 3.0 |
| ANALYST_TARGET_UP | fallback_pct (default) | 0.15 | 1.5 |

### 3.4 Pre-event runup dampener (Lee-Swaminathan 2000; Hong-Stein 1999)

`scoring.rule_score()` 에 곱:

```text
dampener(runup) = max(floor, 1 − max(0, runup) × coef)
```

기본값 coef=1.5, floor=0.2. 즉 카탈리스트 직전 20거래일에 +30% 오른 종목이면 rule_score 가 0.55배로 줄어듦; +50% 이상이면 floor 0.20 으로 고정. 이미 가격에 반영된 카탈리스트는 신호로 보지 않는 것.

레퍼런스: [hudson-and-thames/research](https://github.com/hudson-and-thames/research) — residual momentum / Lee-Swaminathan replication.

### 3.5 Abnormal returns (MacKinlay 1997)

`windowing.abnormal_window_returns()` 추가, `Scorer` 가 기본적으로 사용. analog event 의 실현 수익률을 raw cum_return 대신 (asset − benchmark) 로 계산. KOSPI/SPY 가 같은 기간 +5% 면 종목의 +10% 수익은 abnormal +5% 로 계상.

레퍼런스: [LemaireJean-Baptiste/eventstudy](https://github.com/LemaireJean-Baptiste/eventstudy), [QuantLet/EventStudy](https://github.com/QuantLet/EventStudy). 단순화 버전 — 정식 market-model (alpha+beta, [-250,-30] estimation window) 은 follow-up 단계.

### 3.6 `Event` 에 `extras` 필드 + `CsvEventStore` 호환

위 모든 신호는 `Event.extras: dict[str, Any]` 로 전달. CSV 는 `extras_json` 컬럼이 새로 추가됐고, 컬럼이 없는 *기존 CSV 도 그대로 로드* (호환 보존, `tests/test_sue_and_revision.py::test_csv_store_loads_legacy_csv_without_extras_column` 확인).

---

## 4. KLAC 사례 — 알고리즘이 어떻게 다르게 반응하는가

같은 KLAC 어닝(EPS $9.40 vs $9.16) 을 세 가지 입력으로 점수화 비교:

| 입력 | magnitude_source | magnitude | rule_score | 해석 |
|---|---|---|---|---|
| 뉴스 기사만 (현재) | `fallback_pct` | 0.026 | **+0.00** (5% min 미달) | 차단 — 시즌 평균 +20.7% 대비 평범한 비트 |
| + season_baseline=0.207 | `baseline_adjusted` | 0.000 | **+0.00** | 명시적으로 "코호트 미달" |
| + consensus_stdev=0.06 (finnhub 데이터 가정) | `sue` | **4.000** (capped) | **+4.80** | 분산이 작은 컨센서스에서 4σ surprise → 강한 매수 신호 |

→ **같은 사건, 다른 데이터 → 정반대 액션**. 이게 핵심. Framework 가 자동으로 "어떤 데이터가 있느냐에 따라" 신호의 의미를 다르게 해석.

(`python -c "..."` 으로 실측. 자세한 한 줄짜리 재현은 README 업데이트 예정.)

---

## 5. KLAC 애널리스트 사례 — 같은 효과

Wells Fargo 의 새 목표가 $2,100 만 알고 직전 목표가 미상이면 magnitude = 0 (산출 불가). 하지만:

| 입력 | magnitude_source | magnitude | rule_score |
|---|---|---|---|
| (직전 $1,900, dispersion $80) — finnhub `analyst_price_targets` 풀링 시 | `revision_z` | 2.5 | **+2.10** |
| (직전 $1,900, dispersion 미상) | `fallback_pct` | 0.105 | +0.49 |
| (직전 미상) | (산출 불가) | 0 | +0.00 |

→ 중요한 것은 SUE/revision-z 가 *항상 더 강한 신호를 주지 않는다*는 점. 분산이 큰 종목(예: dispersion $200) 에선 같은 +$200 revision 도 z=1.0 으로 약해진다. 이게 정확히 우리가 원하는 동작.

---

## 6. 데이터 소스 — 우선순위로 정리한 작업

### 미국 (US) — finnhub 무료 티어 (60 req/min) 으로 대부분 해결

| 데이터 | 엔드포인트 / 라이브러리 | 비용 |
|---|---|---|
| actual_eps, consensus_eps, surprise | `finnhub.earnings_surprise(symbol)` | 무료 |
| consensus_stdev, n_estimates (SUE 의 핵심) | `finnhub.eps_estimates(symbol)` 의 `epsHigh`, `epsLow`, `numberAnalysts` 로 σ 추정 | 무료 |
| prior_target / target_dispersion | `finnhub.price_target(symbol)` (`targetMean`, `targetHigh`, `targetLow`, `numberOfAnalysts`) | 무료 |
| 8-K item 1.01 (계약 수주) | `sec-edgar-downloader` 로 본문 수집 후 자체 파서 | 무료 |
| 8-K item 2.02 (어닝 발표 본문) | 동일 | 무료 |

### 한국 (KR) — OpenDartReader + 네이버 컨센서스 조합

| 데이터 | 라이브러리 / 페이지 | 비용 |
|---|---|---|
| 단일판매·공급계약 (계약금액, 매출액 대비 비율) | `OpenDartReader.list(corp, kind='B')` 후 본문 파싱 — `josw123/dart-fss` 가 XBRL 헬퍼 | 무료 (DART API key 필요) |
| 유상증자, 자기주식취득, CB 발행 | 동일 (DART 본문 정형 필드) | 무료 |
| consensus_stdev, prior_target | 네이버 증권 컨센서스 (`finance.naver.com/research/`) — 비공식 스크랩 | 무료 |
| 영업이익 컨센서스 | 한경 컨센서스 (`consensus.hankyung.com`) — 비공식 스크랩 | 무료 |
| 정밀 IBES 데이터 | 에프앤가이드 / FnDataGuide | 유료 |

→ 다음 단계는 `src/catalyst/ingest/` 패키지를 만들어 위 소스에서 정형 dict 를 뽑은 뒤 기존 detector 에 그대로 넣는 것. 본 커밋에는 detector 만 들어갔고, ingestion 은 사용자의 API key 와 함께 별도 단계로 분리.

---

## 7. 다음으로 더 강화할 수 있는 것 (우선순위 순)

1. **정식 market-model abnormal return**: 현재는 단순 (asset − benchmark). MacKinlay 1997 의 [-250, -30] estimation window 에서 OLS 로 alpha/beta 추정 → AR_t = r_i,t − (α̂ + β̂·r_m,t). β 가 1 이 아닌 종목에서 유의미한 차이.
2. **Quintile-based posterior**: 현재는 magnitude 밴드 ±50% 로 analog 추출. SUE / revision_z 의 quintile 로 conditioning 하면 더 정밀 (Bernard-Thomas 1989 의 standard).
3. **Volume confirmation gate**: 이벤트일 거래량 / 20d ADV < 1.5 면 confidence 를 절반으로. Gervais & Odean (2001).
4. **News sentiment 가중치**: 헤드라인 분류기 결과 (KR-FinBert-SC / finBERT) 를 `confidence` 의 추가 곱으로. NLP 모델이 부정적으로 라벨링한 "비트" 는 신호 약화.
5. **Cross-event correlation**: 같은 섹터에서 동시 다발 어닝 비트 시 individual 신호 약화 (cross-sectional baseline).

---

## 8. 출처 (학술 + 깃허브)

**학술**

- Bernard, V., & Thomas, J. (1989). "Post-Earnings-Announcement Drift". *Journal of Accounting Research*.
- Foster, G., Olsen, C., & Shevlin, T. (1984). "Earnings releases, anomalies, and the behavior of security returns". *The Accounting Review*.
- Stickel, S. (1991). "Common stock returns surrounding earnings forecast revisions". *The Accounting Review*.
- Womack, K. (1996). "Do brokerage analysts' recommendations have investment value?". *Journal of Finance*.
- Jegadeesh, N., & Kim, W. (2010). "Do analysts herd? An analysis of recommendations and market reactions". *Review of Financial Studies*.
- Lee, C., & Swaminathan, B. (2000). "Price momentum and trading volume". *Journal of Finance*.
- Hong, H., & Stein, J. (1999). "A unified theory of underreaction, momentum trading, and overreaction". *Journal of Finance*.
- MacKinlay, A. C. (1997). "Event Studies in Economics and Finance". *Journal of Economic Literature*.
- Brown, S. J., & Warner, J. B. (1985). "Using daily stock returns: The case of event studies". *Journal of Financial Economics*.

**깃허브 (이번 커밋에 직접 영향)**

- [quantopian/research_public](https://github.com/quantopian/research_public) — PEAD / 애널리스트 revision factor 노트북 패턴
- [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) — ch.7 alpha factor pipeline, ch.10 베이지안
- [LemaireJean-Baptiste/eventstudy](https://github.com/LemaireJean-Baptiste/eventstudy) — market-model + CAR + t-stats 참조 구현
- [hudson-and-thames/research](https://github.com/hudson-and-thames/research) — residual momentum / Lee-Swaminathan
- [josw123/dart-fss](https://github.com/josw123/dart-fss), [FinanceData/OpenDartReader](https://github.com/FinanceData/OpenDartReader) — DART 본문 파서
- [Finnhub-Stock-API/finnhub-python](https://github.com/Finnhub-Stock-API/finnhub-python) — `epsHigh/epsLow`, `targetHigh/targetLow` 가 SUE/revision-z 의 핵심 입력
- [alpha-vantage/alpha_vantage](https://github.com/alpha-vantage/alpha_vantage) — `ANALYST_TARGET_PRICE` 노출
- [sharebook-kr/pykrx](https://github.com/sharebook-kr/pykrx), [FinanceData/FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) — KR 가격/시총
