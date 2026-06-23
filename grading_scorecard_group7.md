# Grade Scorecard — Group 7
**Rubric source:** `grading_report_group7.md`  
**Files graded:** `assessment_notebook_group7.ipynb` · `research_notebook_group7_backup_Changes_Flo.ipynb` · `module.py`

---

## Criterion Scores

| # | Criterion | Weight | Score | Justification |
|---|---|---|---|---|
| 1 | 3 signals implemented correctly | High | **Full** | MA Crossover, RSI, Donchian — all mathematically correct, LaTeX-documented, economically justified |
| 2 | Reasonable statistics | High | **Full** | Sharpe, Sortino, CAGR, MaxDD, Calmar, Win Rate, DSR — well above minimum |
| 3 | Reasonable graphs | High | **Full** | Equity curve, drawdown, per-signal plots, 3×3 IS/OOS curves, drawdown grid, heatmaps |
| 4 | assessment_notebook runs standalone | High | **Partial** | yfinance fallback present for primary data; OOS2 block depends on `xlf_ext.csv` / `spx_ext.csv` with no download fallback — fails on clean machine without CSVs |
| 5 | research_notebook runs standalone | **Critical** | **Fail** | `_spx_slice` called in Sections 6 & 7 but never defined anywhere → `NameError` on fresh run; directly violates standalone requirement |
| 6 | LaTeX formulas | Medium | **Full** | All signal formulas in LaTeX in both notebooks; portfolio construction equation in research notebook |
| 7 | NumPy only (no pandas built-ins) | High | **Full** | All numerical computation via raw NumPy arrays; no `rolling().mean()` or equivalent anywhere |
| 8 | module.py reuse in both notebooks | High | **Full** | 30+ functions; both notebooks `import module` + `importlib.reload`; zero copy-paste |
| 9 | Readable variable/function naming | Medium | **Full** | `df_finance_is`, `pv_ma_oos1`, `best_rsi_params`, `basket_sortino` — all self-explanatory |
| 10 | Comments in code | Medium | **Full** | module.py has concise non-obvious comments throughout (e.g. `# cumsum group-key trick`, `# Wilder's EMA … loop is irreducible without numba`) |
| 11 | Empirical evidence (research) | High | **Full** | Exhaustive 70-pair screening (7 signals × 10 ETFs) ranked by Min OOS Sortino; IS/OOS/OOS2 |
| 12 | IS/OOS split discipline | High | **Full** | Parameters frozen on IS 2010-2019; applied unchanged to OOS1 2020-2025 and OOS2 2000-2009 |
| 13 | Economic reasoning for signal choice | Medium | **Partial** | MA (XLF) and Donchian (XLK) well-justified; RSI (XLB) rationale is credible but oversold=45/overbought=65 is inconsistent with "mean reversion" framing — not acknowledged |
| 14 | OOS results honestly discussed | Medium | **Partial** | RSI OOS1 Sortino 0.787 vs SPX 0.992 visible in master table but never discussed in text of either notebook |
| 15 | DSR interpretation | Low | **Partial** | Correct computation; DSR=100.000% labelled "Genuine skill" without caveat that SR*≈0.061 is trivially low at n≈24-35 trials |

---

## Penalty Summary

| Issue | Severity | Grade Impact |
|---|---|---|
| `_spx_slice` undefined → research notebook crashes on fresh run | Critical | −0.7 |
| OOS2 data fragility in assessment notebook | Moderate | −0.3 |
| RSI OOS1 underperformance undiscussed | Minor | −0.3 |
| RSI parameter/framing inconsistency | Minor | −0.1 |
| DSR 100% overclaiming | Minor | −0.1 |

---

## Final Grade

| Component | Baseline quality | Penalties applied | Component grade |
|---|---|---|---|
| module.py | 1.0 (near-perfect) | None | **1.0** |
| assessment_notebook | 1.3 (strong) | −0.3 (OOS2 fragility, DSR caveat, RSI framing) | **1.7** |
| research_notebook | 1.0 (exceptional analysis) | −0.7 (`_spx_slice` crash) −0.3 (RSI OOS1 undiscussed) | **2.0** |

**Weighted combined grade: 2.0**

> The underlying research and analytical depth is comfortably 1.3 material. The single `_spx_slice` bug in the research notebook is the controlling deficiency — it fails the explicit "runs standalone" requirement. Resolving it (one function definition, ~5 lines) and adding a paragraph on RSI OOS1 underperformance would move the grade to **1.3–1.7**.

---

## What to fix before submission (priority order)

1. **Define `_spx_slice`** in `research_notebook_group7_backup_Changes_Flo.ipynb` before the Section 6 plotting cell.  
   Simplest fix — inline from `module.spx_normalise`:
   ```python
   def _spx_slice(df_basket, df_spx_full):
       return module.spx_normalise(df_basket, df_spx_full)
   ```

2. **Add yfinance fallback for `xlf_ext.csv` and `spx_ext.csv`** in `assessment_notebook_group7.ipynb` walk-forward section so OOS2 data downloads if the CSV is missing.

3. **Add one sentence in the research notebook** acknowledging RSI OOS1 Sortino (0.787) falls below the S&P 500 benchmark (0.992) and why this was still retained (e.g. diversification benefit, OOS2 resilience).

4. **Add a DSR caveat** in the assessment notebook: note that SR* ≈ 0.061 at n ≈ 24-35 trials is low enough that DSR saturating to 100% is expected, not evidence of certainty.

5. **Clarify RSI parameter framing**: acknowledge that oversold=45/overbought=65 shifts RSI toward a momentum regime rather than classical mean-reversion (or re-label the signal as "RSI momentum").
