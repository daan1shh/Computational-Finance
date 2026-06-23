# Changes: `research_notebook_group7_backup_Changes_Flo.ipynb`

Compared against `research_notebook_group7_backup.ipynb`.

---

## 1. Gross Returns → Net Returns terminology (Cell 0)

**Approach section rewritten.** The original said "We report gross returns, no transaction costs are included." The updated version clarifies that since no transaction costs are modelled, Gross Returns = Net Returns, and commits to calling them **Net Returns** consistently throughout the notebook.

> *Before:* "We report gross returns, no transaction costs are included."
>
> *After:* "We report Gross Returns, no transaction costs are included. Therefore Gross Returns = Net Returns throughout this notebook. For consistency we will call the returns from now on Net Returns."

---

## 2. Section numbering renumbered (Cells 10, 11, 15, 18, 22, 23, 24 → new cell 25)

All section headers were renumbered to start from **2** (the original started at **3** after a setup section numbered 1):

| Original | Updated |
|----------|---------|
| §3 Signal Catalogue | 2. Signal Catalogue |
| §4 ETF Assignment | 3. ETF Assignment |
| §5 In-Sample Parameter Optimisation | 4. In-Sample Parameter Optimisation |
| §6 OOS Validation | 5. Out-Of-Sample Validation |
| (subsection) Net Cumulative Return Curves | 6. Net Cumulative Return Curves |
| (subsection) Drawdown Curves | 7. Drawdown Curves |
| §9 Conclusion | 8. Conclusion & Final Parameter Justification |

---

## 3. Section 5 (OOS Validation) renamed and description trimmed (Cell 18)

- Header changed from "OOS Validation" to **"Out-Of-Sample Validation"** for clarity.
- Description of portfolio construction simplified: removed the sentence explaining one-day lag look-ahead bias avoidance and the explicit statement that "all returns are gross."
- "gross portfolio value" changed to **"net portfolio value"** in the formula description, consistent with the new terminology.

---

## 4. Section 6 — Return curves promoted from subsection to top-level section (Cell 20)

The heading "### Net Cumulative Return Curves - IS / OOS1 / OOS2" was promoted to a top-level `##` section: **"## 6. Net Cumulative Return Curves"**. The description was also trimmed slightly (removed "IS-optimal parameters are frozen throughout").

---

## 5. Section 7 — Drawdown curves promoted from subsection to top-level section (Cell 22)

The heading "### Drawdown Curves - IS / OOS1 / OOS2" was promoted to a top-level `##` section: **"## 7. Drawdown Curves"**. The definition of drawdown (`pv / running_max - 1 × 100`) was removed from the description.

---

## 6. Plot title simplified (Cell 23)

The drawdown chart `suptitle` was shortened:

> *Before:* `'Drawdown (%) vs. S&P 500  |  IS · OOS1 · OOS2'`
>
> *After:* `'Drawdown (%) vs. S&P 500'`

---

## 7. Return curves plot title simplified (Cell 21)

The cumulative return chart `suptitle` was shortened:

> *Before:* `'Net Cumulative Return (%) vs. S&P 500  |  Gross Returns  |  Start-of-Period Normalised'`
>
> *After:* `'Net Cumulative Return (%) vs. S&P 500  |  Start-of-Period Normalised'`

(`period_titles` also switched from Unicode escape `–` to literal `–` dashes — functionally identical.)

---

## 8. Portfolio construction code: comments cleaned up (Cell 19)

Two comment blocks were simplified:

- `# --- Portfolio Construction Helper Functions ---` → `# Portfolio Construction Helper Functions`
- `# --- Compute Portfolio Values: All Nine Window-Signal Combinations ---` with its multi-line explanation → `# Portfolio Values Computation`
- `# --- Consolidated Master Summary Table ---` → `# Consolidated Master Summary Table`

The **MinOOS / Sortino decay robustness block** was removed from `master_summary()`. In the original, the function printed a second table showing `MinOOS`, `Δ Sort OOS1`, `Δ Sort OOS2`, and an "All OOS > 0?" flag. This block is gone in the updated version; the function now only prints the main three-window performance table.

---

## 9. Conclusion rewritten (Cell 24 cleared → new Cell 25 added)

The original Cell 24 was a detailed conclusion referencing internal section numbers (§3, §4, §5, §6) and including extensive inline justification. It was **cleared** (now blank) and replaced by two new cells:

**New Cell 25** contains a rewritten conclusion that:
- References sections by their updated numbers (2 through 7).
- Uses shorter, more direct bullet points.
- Removes the detailed justification of the Donchian asymmetric channel and the sensitivity heatmap discussion.
- Keeps the same 12-reference bibliography unchanged.

**New Cell 26** is an empty markdown cell (structural placeholder).

---

## 10. Signal-ETF results table: leading `---` separator removed (Cell 14)

The "Signal-ETF Assignment Results" cell lost its leading `---` horizontal rule. The table content is identical.

---

## Summary

| Category | Nature |
|----------|--------|
| Terminology | "Gross Returns" → "Net Returns" throughout |
| Structure | All section numbers renumbered; subsections promoted to top-level |
| Plot titles | Shortened to remove redundant labels |
| Code comments | Simplified/de-noised |
| Summary table | MinOOS robustness block removed from `master_summary()` |
| Conclusion | Rewritten to reference new section numbers; kept same references |
