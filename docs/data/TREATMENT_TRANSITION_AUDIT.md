# Treatment-to-Follow-up Transition Feasibility Audit

**Run date:** 2026-10-27  
**Input:** `clinical.project-tcga-ov.2026-10-09.json` (608 cases)  
**Script:** [`audit_treatment_transitions.py`](../../scripts/audit_treatment_transitions.py)  
**Machine-readable output:** [`treatment_transition_audit.json`](../../data/processed/treatment_transition_audit.json)  
**Purpose:** Determine how many candidate observational treatment-to-response links can be extracted before building a transition model. This is not a causal analysis and does not approve RL use.

## Time origin and inclusion

The TCGA [Ovary Pharmaceutical Supplemental Form](https://gdc.cancer.gov/system/files/public/file/Pharmaceutical%20Form.pdf) defines treatment-start and treatment-end intervals from initial pathological diagnosis. The TCGA [Ovary Followup Form](https://gdc.cancer.gov/system/files/public/file/Ovary%20Followup%20Form.pdf) uses that same reference for follow-up intervals. The export has `index_date = Diagnosis` for 587 cases; 21 cases were excluded from candidate linkage because that reference is not explicit. Four negative treatment-start days were counted as data-quality exceptions and excluded from the candidate timeline.

For each eligible patient, timed treatment records on the same numeric start day were bundled as a **provisional same-day treatment set**. The audit then found the first unambiguous, known response at a strictly later follow-up day. It separately counted outcomes after treatment start and after all end days in that same-day set. A more conservative recorded-time subset also excluded any later *dated* treatment start before the response and required a single parent diagnosis record.

The form definitions support a common diagnosis-based clock. They do not establish the clinical meaning of every treatment bundle, prove that a follow-up response was caused by a particular treatment, or show that no untimed treatment occurred in the interval.

## Main results

| Measure | Result |
|---|---:|
| Eligible cases with diagnosis as index date | 587 / 608 |
| Treatment records | 4,125 |
| Treatment records with numeric start day | 2,501 (60.6%) |
| Treatment records with no numeric start day | 1,624 (39.4%) |
| Eligible patients with at least one treatment missing its start day | 575 / 587 |
| Eligible patients with start days on every recorded treatment | 4 / 587 |
| Treatment records with an end day | 2,188 (53.0%) |
| Records with no dose value | 2,419 (58.6%) |
| Records without a known therapeutic agent | 1,631 (39.5%) |
| Same-day treatment sets among patients with timed starts | 1,563 across 517 patients |
| Sets with an unambiguous later response after treatment start | 1,479 across 480 patients |
| Sets with all member end days and a later response after completion | 1,212 across 439 patients |
| Post-completion links with no *dated* later treatment start before response and one parent diagnosis | 366 across 359 patients |
| Those links in patients whose **entire** recorded treatment history has start days | 2 across 2 patients |

The last row is the strongest completeness check. Because 575 eligible patients have at least one undated treatment, the 366 links above are only **upper-bound candidates** for treatment-isolated outcomes. Missing start dates mean an intervening treatment cannot be ruled out. The full-timeline subset of two cases is too small to support RL or a reliable action comparison.

## Timing and action support

For the 366 recorded-time-isolated post-completion links, the first response was observed **1–3,072 days** after the latest treatment end (median **207.5 days**, 90th percentile **1,469 days**). Labels were `WT-With Tumor` for 263 links and `TF-Tumor Free` for 103. This is a variable, often long observation gap rather than a fixed clinical response horizon.

Coarse action types in these links are highly concentrated: 327 chemotherapy sets; 13 external-beam radiation; 10 targeted molecular therapy; 8 chemotherapy plus targeted therapy; 6 hormone therapy; and two other sets. Dose is missing for most treatment records, and detailed agents/regimens are not consistently recorded. These coarse counts do not establish adequate treatment overlap or clinically interchangeable alternatives.

There were 511 duplicate known-response records beyond the first at the same patient-day, but no same-day groups with conflicting known response labels. Identical same-day labels were collapsed. Ten candidate treatment sets had a known response on the same day as treatment start; their within-day order is ambiguous, so that response was not used as a strictly later outcome.

## Manual spot check

One anonymized example has two chemotherapy records (carboplatin and docetaxel) starting on day 53 and ending on day 221, followed by `WT-With Tumor` at day 1,036. No other **dated** treatment start appears between the action and that response. However, this patient has five treatment records total, three without start days and three without end days. The example illustrates why a clean-looking dated sequence still cannot prove treatment isolation.

## Conclusion and next decision

The export can produce many retrospective associations, but **it does not provide a complete enough treatment timeline for treatment-effect learning or RL**. The dataset has 366 promising records under a dated-event-only filter, but only 2 survive the requirement that all of that patient's recorded treatment starts be known. Outcomes are sparse and separated from treatment end by irregular, sometimes multi-year gaps.

Recommended project scope now: use TCGA-OV for retrospective, patient-level response forecasting with an explicitly defined horizon and strong caveats; keep RL approval withheld. If action-conditioned treatment optimization remains essential, first obtain a dataset with much more complete longitudinal treatment, response, and toxicity records, or secure source-level clarification for the missing treatment dates. Do not train an RL policy from the 366 candidate links as though they were causal transitions.

## Relationship to the paper-style preprocessing counts

The newer `scripts/paper_method_pipeline.py` answers a related but different question: it first maps records into the paper's 11 therapy classes, then forms same-day action sets only when at least one action class is available. It reports 1,453 such sets (1,449 with every member mapped), 51 local action combinations, and 364 `dated_event_only_candidate` rows. The 364 count is not a correction to this audit's 366: the pipelines apply different action-class and event-group eligibility rules. Both yield only 2 links among patients with a complete recorded treatment-start timeline. See [`PAPER_METHOD_REPLICATION.md`](../architecture/PAPER_METHOD_REPLICATION.md) and [`pipeline_summary.json`](../../data/processed/paper_method/pipeline_summary.json).

## Reproducibility

```powershell
python scripts/audit_treatment_transitions.py --input clinical.project-tcga-ov.2026-10-09.json
```

The JSON output includes counts by action category, lag summaries, anonymized sample links, and the explicit linkage rules and limitations.
