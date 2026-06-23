# Grading Report — Group 7
**Course:** Computational Finance in Python, Universität Tübingen (Summer 2026)  
**Instructor:** Dr. Thomas Schön  
**Files reviewed:** `assessment_notebook_group7.ipynb`, `research_notebook_group7_backup_Changes_Flo.ipynb`, `module.py`  
**Reference:** `Lecture_02_Assessment.pdf`  

---

## 1. Rubric Checklist

| Criterion | Requirement | Met? | Notes |
|---|---|---|---|
| 3 signals implemented | assessment_notebook | ✅ | MA Crossover, RSI, Donchian |
| Statistics | Reasonable metrics | ✅ | Sharpe, Sortino, CAGR, MaxDD, Calmar, Win Rate, DSR |
| Graphs | Reasonable plots | ✅ | Equity curve, drawdown, signal plots, heatmaps |
| Standalone (assessment) | Runs on another computer | ⚠️ | yfinance fallback present but OOS2 depends on separate CSV files |
| Standalone (research) | Runs on another computer | ❌ | `_spx_slice` called but never defined — notebook crashes at cells 21 & 23 |
| LaTeX formulas | Both notebooks | ✅ | Properly rendered in Markdown cells |
| NumPy only | No pandas/built-in numerical | ✅ | All computations use NumPy arrays; no `rolling().mean()` etc. |
| module.py reuse | Imported in both notebooks | ✅ | All shared logic in module.py; both notebooks import and reload it |
| Readable naming | Self-explanatory variables | ✅ | `df_finance_is`, `best_rsi_params`, `pv_ma_oos1` etc. are clear |
| Comments | Context-providing comments | ✅ | module.py has concise, non-obvious comments throughout |
| Empirical evidence | research_notebook | ✅ | 70-pair exhaustive screening + IS/OOS split + stress test |
| IS/OOS split | Parameters frozen before OOS | ✅ | IS 2010-2019, OOS1 2020-2025, OOS2 2000-2009 (pre-sample stress) |

---

## 2. Per-File Assessment

### `module.py`
- **NumPy compliance:** All numerical functions (`moving_average`, `rolling_std`, `compute_rsi`, `compute_sortino`, etc.) use raw NumPy — no pandas built-ins. `_vectorised_signal` uses the cumsum/accumulate trick to avoid a Python loop. ✅
- **Reusability:** Functions used in both notebooks without copy-paste. The module exports 30+ well-named functions covering signal generation, performance metrics, portfolio simulation, and visualisation. ✅
- **Signal correctness:**
  - `ma_signal`: Correct Golden/Death Cross logic using a cumsum-based MA. ✅
  - `rsi_signal`: Correct Wilder EMA seeded with plain mean. 1-day execution lag applied correctly via `_vectorised_signal`. ✅
  - `donchian_signal`: Uses `sliding_window_view` on `prices[:-1]` for correct look-ahead-free channel computation. ✅
- **Comments:** Brief but informative: e.g. `# Wilder's EMA (alpha = 1/period); loop is irreducible without numba`, `# cumsum group-key trick`. ✅
- **Minor issue:** `_collect_trade_returns` contains a Python loop that could in principle be vectorised, but the comment acknowledges it is structurally irreducible; acceptable.
- **DSR implementation:** `compute_deflated_sharpe` uses the rational-polynomial normal CDF approximation without scipy — satisfies NumPy-only constraint and is correctly attributed to Bailey & López de Prado (2014). ✅

---

### `assessment_notebook_group7.ipynb`

**Structure:** Setup → Signals → Portfolio & Performance → Robustness → Walk-Forward Validation

**Signal 0 – MA Crossover (XLF):**
- Formula correctly states $s_t = 1 \iff \text{MA}_{w_s}(t) > \text{MA}_{w_l}(t)$. ✅
- Economic rationale: interest-rate cycles → sustained trends in Financials. Credible. ✅
- IS-optimal params ($w_s=20$, $w_l=100$) inherited from research notebook grid. ✅
- Brock, Lakonishok & LeBaron (1992) cited. ✅

**Signal 1 – RSI (XLB):**
- Formula correctly presents Wilder's EMA formulation. ✅
- Parameters: oversold=45, overbought=65 — notably wider than textbook 30/70. Rationale is IS-optimised but not explicitly discussed as a departure from convention. Minor gap. ⚠️
- Economic rationale: commodity PMI cycles → mean reversion. Credible. ✅

**Signal 2 – Donchian Channel (XLK):**
- Formula correctly distinguishes asymmetric entry (N=125-day high) and exit (M=100-day low). ✅
- Economic rationale: S-curve technology adoption → sustained breakouts. Credible. ✅
- Donchian (1960) cited. ✅

**Performance table (full 2010-2025):**

| Metric | Strategy | S&P 500 |
|---|---|---|
| CAGR | 11.3% | 12.0% |
| Ann. Volatility | 12.4% | 17.3% |
| Sharpe | 0.92 | 0.74 |
| Sortino | 1.30 | 1.04 |
| Calmar | 0.48 | 0.35 |
| Max Drawdown | -23.4% | -33.9% |
| Win Rate | 74.0% | — |

- Lower CAGR than buy-and-hold but significantly better risk-adjusted metrics. Cash drag from unallocated capital is correctly acknowledged in a benchmark note. ✅

**Deflated Sharpe Ratio:**
- All three signals show DSR = 100.000% ("Genuine skill"). With SR ≈ 0.75–0.85 and only 24-35 trials, the threshold SR* ≈ 0.061 is trivially low, making DSR saturate to machine precision. The result is mathematically correct but should be accompanied by a caveat that DSR = 100% is due to the small trial count relative to the observed SR, not evidence of literal certainty. The "Genuine skill" label is misleading in its strength. ⚠️

**Walk-Forward Validation:**
- IS (2010-2019): Sharpe 0.95 vs S&P 0.79 ✅
- OOS1 (2020-2025): Sharpe 1.09 vs S&P 0.70 — OOS improvement, strong result ✅
- OOS2 (2000-2009, pre-sample stress): CAGR 5.3% vs S&P -2.6%, MaxDD -21.5% vs S&P -56.8% ✅

**Heatmaps:** Parameter sensitivity heatmaps shown for all three signals with ★ marking chosen parameters. Confirms plateau-like stability around the optimal. ✅

**Data loading concern:** OOS2 requires `xlf_ext.csv` and `spx_ext.csv`. If these are missing and `yfinance_ok = False`, the OOS2 walk-forward block raises a `FileNotFoundError`. The primary data path should fall back more robustly. ⚠️

---

### `research_notebook_group7_backup_Changes_Flo.ipynb`

**Structure:** Intro → Setup → Signal Catalogue → ETF Assignment → IS Parameter Optimisation → OOS Validation → Return Curves → Drawdown Curves → Conclusion

**Signal catalogue:** Concise mathematical treatment of all 7 signals with LaTeX formulas. The 4 non-primary signals (MACD, Bollinger, Stochastic, Z-Score) are catalogued but not deeply evaluated — acceptable given scope. ✅

**ETF Screening (Section 3):** Exhaustive 70-pair evaluation (7 signals × 10 SPDR sector ETFs) with IS optimisation and OOS evaluation across two windows. Table ranked by Min OOS Sortino. This is the strongest analytical contribution in the submission. ✅

- 9 / 70 pairs beat the S&P 500 in **both** OOS periods
- Final picks:
  - MA Cross on XLF: #7 overall, beats SPX both OOS ✅
  - RSI on XLB: #10 overall, **does not beat SPX in OOS1** (0.787 vs 0.992) ⚠️
  - Donchian on XLK: #6 overall, beats SPX both OOS ✅

The assessment notebook description says "XLB ranked first in our §3 screening by Min OOS Sortino for RSI" — technically accurate (it is the top RSI-only result) but the absolute rank (#10) and OOS1 underperformance are not highlighted. The economic rationale for XLB is still sound but this framing is slightly misleading.

**IS Parameter Optimisation (Section 4):**
- Grid search uses `basket_sortino` (Sortino ratio as objective) on IS 2010-2019 only. ✅
- Heatmaps generated with `module.draw_heatmap`. ✅
- Final IS-optimal params recovered consistently with assessment notebook. ✅

**OOS Validation (Section 5):**
- `basket_portfolio_value` produces cumulative PV with 1-day lag. ✅
- Master summary table is well-structured with vs-SPX Sortino delta. ✅
- RSI OOS1 Sortino = 0.787, negative delta vs SPX (-0.206) — noted in table but not discussed in text. ⚠️

**Critical Bug — `_spx_slice` undefined (Sections 6 & 7):**
- Cells for net cumulative return curves and drawdown curves call `_spx_slice(df_, spx_ref)`, which is **never defined** anywhere in the notebook.
- On a fresh run, both `plt.show()` calls at cells 21 and 23 will raise `NameError: name '_spx_slice' is not defined`.
- This directly violates the "needs to be able to run stand alone on another computer" requirement. ❌

**Conclusion (Section 8):** Well-written summary linking each section to the analytical chain. References are extensive and correctly formatted. ✅

**No transaction costs:** Explicitly stated ("Gross Returns = Net Returns"), which is acceptable for a course submission but limits real-world interpretability. The strategy executes 83 trades over 16 years, so costs are unlikely to flip the conclusion, but it should be noted. ⚠️

---

## 3. Strengths

- **Exceptional analytical depth:** The 70-pair exhaustive screening (Section 3 of research notebook) is above course expectations and provides rigorous empirical grounding for the signal-ETF choices.
- **Three structurally distinct signals:** Trend-following (MA), mean-reversion (RSI), and breakout (Donchian) cover diverse market regimes — sound portfolio design.
- **Two OOS windows:** OOS2 (2000-2009, including the dot-com bust and GFC) is a genuine stress test; the strategy's performance (CAGR 5.3%, MaxDD -21.5% vs S&P -56.8%) is convincing.
- **DSR implementation:** Computing the Deflated Sharpe Ratio from scratch in NumPy (without scipy) is technically impressive and methodologically rigorous.
- **Module hygiene:** `module.py` is genuinely reusable; both notebooks import it cleanly with `importlib.reload`. Zero copy-paste between notebooks.
- **LaTeX throughout:** All signal formulas rendered in LaTeX markdown cells in both notebooks.
- **OOS1 result is strong:** Combined portfolio Sharpe 1.09 vs S&P 0.70 in OOS1 (2020-2025) including COVID crash is a meaningful out-of-sample result.
- **Code readability:** Variable and function naming is consistently self-explanatory across all three files.

---

## 4. Weaknesses & Issues

**Priority: high (grade-impacting)**

1. **`_spx_slice` undefined in research notebook** (`research_notebook_group7_backup_Changes_Flo.ipynb`, cells 21 and 23): The function is called but not defined anywhere in the notebook. Running the notebook on another machine raises `NameError` and aborts Sections 6 and 7 (return curves and drawdown curves). This is the most significant single deficiency — it directly fails the standalone requirement. The plots themselves exist as saved outputs in the notebook, but a fresh run will crash.

2. **OOS2 data-loading fragility in assessment notebook** (`assessment_notebook_group7.ipynb`, walk-forward section): OOS2 requires `data/xlf_ext.csv` and `data/spx_ext.csv` to exist. There is no yfinance fallback for these files in this section, and the primary `load_csv` / fallback logic from Section 1 is not reused. A fresh machine without these CSVs will fail during the OOS2 `period_stats` call.

3. **RSI on XLB OOS1 underperformance not flagged:** RSI on XLB produces a Sortino ratio of 0.787 in OOS1 vs the S&P 500's 0.992 — a -0.206 delta. The assessment notebook describes this signal as performing well but does not acknowledge this shortfall. The research notebook shows it in the master table but provides no discussion.

**Priority: medium**

4. **DSR = 100.000% needs contextual caveat** (`assessment_notebook_group7.ipynb`, DSR cell): The result is mathematically correct (small trial counts vs high observed SR) but the label "Genuine skill" at 100% without explanation of saturation reads as overclaiming. A sentence noting that SR* ≈ 0.061 is trivially low with n ≈ 24-35 trials would improve intellectual honesty.

5. **RSI parameter justification** (`assessment_notebook_group7.ipynb`, Signal 1 cell): oversold=45, overbought=65 are far from conventional (30/70). These are IS-optimised, but no comment acknowledges that these parameters move RSI toward a momentum rather than mean-reversion regime. The economic framing of XLB as "mean reverting" is partially inconsistent with the chosen thresholds.

6. **No transaction cost sensitivity:** Both notebooks treat gross returns as net returns. Even a brief sensitivity paragraph (e.g., "at 0.1% per trade, 83 trades over 16 years reduces total return by ~8%") would strengthen the investor-facing narrative.

**Priority: low**

7. **Research notebook intro table says "Gross Returns = Net Returns":** This is clearly labelled, so not a factual error, but a single quantitative note on cost sensitivity would strengthen the argument for a real investor.

8. **Assessment notebook `_spx_slice` is not referenced**: The assessment notebook does not call `_spx_slice`, so only the research notebook is affected. However, one of these notebooks appears to be an earlier draft (the "backup_Changes_Flo" suffix) — it is unclear if this is the final submission file.

---

## 5. Overall Assessment

Group 7 submits a technically strong and analytically ambitious project that substantially exceeds the baseline requirements in several areas: the exhaustive 70-pair screening, the Deflated Sharpe Ratio from scratch, and the pre-sample OOS2 stress test are all well above what the rubric demands. Module structure is clean, naming is professional, and LaTeX formulas are present and correct throughout.

The primary weakness is a **reproducibility-breaking bug in the research notebook**: `_spx_slice` is called in the return-curve and drawdown cells but never defined, which will crash any clean-environment run and directly fails the standalone requirement. If the submitted file is indeed the "backup_Changes_Flo" version rather than a polished final, this may be a version-control artefact — but as graded, it is a significant technical failure. The OOS1 underperformance of RSI on XLB is also insufficiently discussed. Fixing the undefined function and adding a brief discussion of the RSI OOS1 result are the highest-leverage improvements.

**Indicative German grade: 2.0**  
*(Would be 1.3–1.7 with the `_spx_slice` bug resolved and the RSI OOS1 discussion added; the underlying analysis quality is at 1.3 level.)*

---

*Assessment conducted against `Lecture_02_Assessment.pdf` rubric (Regular Assessment track). AI tools used per course policy (Lecture_02_Assessment.pdf, p. 6).*
