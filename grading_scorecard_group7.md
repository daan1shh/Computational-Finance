# Grade Scorecard — Group 7
**Rubric source:** `grading_report_group7.md`  
**Files graded:** `assessment_notebook_group7.ipynb` · `research_notebook_group7_backup_Changes_Flo.ipynb` · `module.py`  
**Verified against:** actual notebook cell content (not static report)

---

## Criterion Scores

| # | Criterion | Weight | Score | Justification |
|---|---|---|---|---|
| 1 | 3 signals implemented correctly | High | **Full** | MA Crossover, RSI, Donchian — all mathematically correct, LaTeX-documented, economically justified |
| 2 | Reasonable statistics | High | **Full** | Sharpe, Sortino, CAGR, MaxDD, Calmar, Win Rate, DSR — well above minimum |
| 3 | Reasonable graphs | High | **Full** | Equity curve, drawdown, per-signal plots, 3×3 IS/OOS curves, drawdown grid, heatmaps |
| 4 | assessment_notebook runs standalone | High | **Partial** | yfinance fallback present for primary data; OOS2 block uses `pd.read_csv` directly on 3 CSV files with no download fallback — **however** all three files (`xlf_ext.csv`, `sector_etfs_ext.csv`, `spx_ext.csv`) are present in `data/` so the notebook runs as submitted |
| 5 | research_notebook runs standalone | High | **Full** | `_spx_slice` is defined in Cell 21 (`return module.spx_normalise(df_basket, df_spx_full)`). Notebook runs end-to-end on a fresh machine. *(The grading report claimed this was missing — it was already fixed before submission.)* |
| 6 | LaTeX formulas | Medium | **Full** | All signal formulas in LaTeX in both notebooks; portfolio construction equation in research notebook |
| 7 | NumPy only (no pandas built-ins) | High | **Full** | All numerical computation via raw NumPy arrays; no `rolling().mean()` or equivalent anywhere |
| 8 | module.py reuse in both notebooks | High | **Full** | 30+ functions; both notebooks `import module` + `importlib.reload`; zero copy-paste |
| 9 | Readable variable/function naming | Medium | **Full** | `df_finance_is`, `pv_ma_oos1`, `best_rsi_params`, `basket_sortino` — all self-explanatory |
| 10 | Comments in code | Medium | **Full** | module.py has concise non-obvious comments throughout (e.g. `# cumsum group-key trick`, `# Wilder's EMA … loop is irreducible without numba`) |
| 11 | Empirical evidence (research) | High | **Full** | Exhaustive 70-pair screening (7 signals × 10 ETFs) ranked by Min OOS Sortino; IS/OOS/OOS2 |
| 12 | IS/OOS split discipline | High | **Full** | Parameters frozen on IS 2010-2019; applied unchanged to OOS1 2020-2025 and OOS2 2000-2009 |
| 13 | Economic reasoning for signal choice | Medium | **Partial** | MA (XLF) and Donchian (XLK) well-justified; RSI (XLB) rationale is credible but oversold=45/overbought=65 is inconsistent with "mean reversion" framing — not acknowledged |
| 14 | OOS results honestly discussed | Medium | **Partial** | Research notebook conclusion says "RSI–XLB beat S&P 500 in OOS1 or OOS2 (or both)" — technically true but evasive. RSI OOS1 Sortino 0.787 vs SPX 0.992 is visible in the master table but never explicitly flagged in text |
| 15 | DSR interpretation | Low | **Partial** | Correct computation; SR* printed; but DSR=100.000% labelled "Genuine skill" without caveat that SR*≈0.061 is trivially low at n≈24-35 trials — result is mathematically correct but overclaims certainty |

---

## Penalty Summary

| Issue | Severity | Grade Impact |
|---|---|---|
| ~~`_spx_slice` undefined~~ | ~~Critical~~ | ~~−0.7~~ **NOT A BUG — function defined in Cell 21** |
| OOS2 CSV fallback absent (files present; theoretical fragility) | Minor | −0.1 |
| RSI OOS1 underperformance evasively framed in research notebook conclusion | Minor | −0.2 |
| RSI parameter/framing inconsistency (oversold=45 ≠ mean reversion) | Minor | −0.1 |
| DSR 100% overclaiming without saturation caveat | Minor | −0.1 |

---

## Final Grade

| Component | Baseline quality | Penalties applied | Component grade |
|---|---|---|---|
| module.py | 1.0 (near-perfect) | None | **1.0** |
| assessment_notebook | 1.3 (strong) | −0.2 (DSR caveat, RSI framing, OOS2 fallback) | **1.5** |
| research_notebook | 1.0 (exceptional analysis) | −0.3 (RSI OOS1 evasive framing) | **1.3** |

**Weighted combined grade: 1.3–1.7**

> The `_spx_slice` function IS defined in Cell 21 of the research notebook — the grading report's "critical bug" was already resolved before submission. With that penalty removed, the submission reflects its actual quality: exhaustive 70-pair screening, correct IS/OOS discipline, DSR from scratch, pre-sample stress test, and clean module architecture. The remaining minor issues (RSI OOS1 framing, DSR saturation caveat, RSI parameter justification) are all medium-priority improvements that would push the grade toward 1.3.

---

## Remaining improvements (priority order)

1. **Explicitly acknowledge RSI OOS1 underperformance** in research notebook conclusion: the current phrasing "beat S&P 500 in OOS1 or OOS2" obscures that RSI specifically trails in OOS1 (Sortino 0.787 vs 0.992). A single sentence noting this and explaining why XLB was retained (OOS2 resilience, diversification) would resolve criterion 14.

2. **Add a DSR saturation caveat** in assessment notebook Cell 22/23: note that SR* ≈ 0.061 at n ≈ 24-35 trials is low enough that DSR→100% is expected, not evidence of certainty.

3. **Clarify RSI parameter framing**: acknowledge that oversold=45/overbought=65 shifts RSI toward a momentum regime rather than classical mean-reversion, or relabel as "RSI momentum". Currently inconsistent with the XLB "mean reversion" economic rationale.

4. **Add yfinance fallback for OOS2 CSVs** in assessment notebook walk-forward cell (theoretical robustness; currently not needed since files are present).
