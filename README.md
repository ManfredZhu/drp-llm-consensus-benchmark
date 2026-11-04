# Data and code for: Benchmarking nine large language models against
# human consensus for drug-related problem coding

Version 1.1.1 · released 2026-11-04

This deposit contains the model outputs, the human rating tallies and the
analysis code behind every number reported in the manuscript. All reported
statistics can be regenerated from the files below; no network access and no
model credentials are required.

## What the study measured

Nine large language models from eight vendors coded 15 clinical vignettes for
drug-related problems (DRPs) under two prompt conditions: arm A, with the
codebook, the workflow and worked examples; arm B, with the codebook only.
Each model answered each case five times, giving 1,980 sessions in total,
including four single-run ablation arms and a ten-case transfer arm that used
the external PCNE V9.1 codebook.

The reference standard is not an expert panel but a tally: 139 pharmacists
rated 20 cases and 46 pharmacists rated 15 cases, back to back, without seeing
one another's answers. A cell counts as *consensus* when its reference code set
reaches the k-th highest vote (k = number of reference codes); otherwise it is
*contested* and no reference set is imposed. The AI test set therefore holds 98
consensus cells and 7 contested cells.

## Contents

```
.
├── README.md                     this file
├── CHECKSUMS.sha256              checksums of every file below
├── requirements.txt              Python dependencies
├── CITATION.cff                  citation metadata
├── .zenodo.json                  deposit metadata
├── LICENSE-DATA-CC-BY-4.0.txt
├── LICENSE-CODE-MIT.txt
├── drp_eval/
│   ├── runs/                     63 record files, 1,980 sessions
│   ├── data/                     human ratings, reference sets, case list
│   ├── *.json                    analysis outputs shipped as released
│   ├── *.py                      analysis and figure code
│   ├── icons_png/                icons pre-composed for the figure code
│   └── _vendor/                  panel-alignment auditor used by the figure code
└── icon/                         source icons (PNG and SVG)
```

| Component | Files | Contents |
|---|---|---|
| `drp_eval/runs/` | 63 | One JSONL per model and arm; 1,980 sessions, raw model text included |
| `drp_eval/data/` | 7 | Per-cell vote tallies, reference sets, the rater-by-cell matrix, the case list |
| `drp_eval/*.json` | 4 | Primary endpoint, derived figure data, cross-model agreement, PCNE arm endpoints |
| `drp_eval/*.py` | 13 | Scoring, resampling, agreement, error typing, figure generation |
| `icon/` | 26 | Identity icons (PNG and SVG) referenced by Figs. 2, 4 and 9 |

## Data dictionary

### `drp_eval/runs/<arm>__<model>.jsonl`

One JSON object per session. Arms are `A`, `B`, `ablation__<model>__<variant>`
and `pcne__<model>`; variants are `minus_examples`, `minus_check`,
`minus_followup` and `minus_output`.

| Field | Meaning |
|---|---|
| `run_id` | Unique session identifier: arm, model, case and repetition |
| `case_id` | Case number, 1–15 (the same vignettes throughout) |
| `repetition` | 1–5 in arms A and B; 1 in the ablation and PCNE arms |
| `raw_output` | Verbatim model output, unedited (3.3 MB in total) |
| `parsed_codes` | Codes extracted from the output, by dimension: `P, E1, E2, C, I, A, O` |
| `parse_status` | `ok`, `partial` or `fail` |
| `latency_ms` | End-to-end wall-clock time of the call |
| `channel` | Acquisition path recorded at run time (blank for early WorkBuddy runs) |
| `model_version_reported` | Version string returned by the model, where available |
| `shape` | Output shape flags: character count, Chinese-character share, presence of JSON |

### `drp_eval/data/`

| File | Contents |
|---|---|
| `cases.json` | The 35 vignettes, each with the Chinese original (`narrative_zh`) and an English translation (`narrative_en`), plus batch, gold label, setting, number of DRPs and the dual-DRP flag |
| `matrix_batch1.json`, `matrix_batch2.json` | Rater-by-cell answers; one record per rater, case and dimension. Rater labels are pseudonyms |
| `cells_batch1.json`, `cells_batch2.json` | Per-cell tallies: valid raters, required k, threshold, reference codes, vote counts, distinct answers |
| `gold_batch1.json`, `gold_batch2.json` | Reference code set per case and dimension |

A cell is *consensus* when the reference set reaches the k-th highest vote; the
threshold and k are given explicitly in the cell files so the rule can be
checked without reading the code.

### `drp_eval/*.json`

| File | Contents |
|---|---|
| `results_stage4.json` | Primary endpoint per model with cluster-bootstrap intervals, dimension-level accuracy, contested-cell tiers, ablation arms, latency, shape |
| `derived_figs.json` | Per-cell correctness, error typology, Jaccard similarity of error sets, the shared-failure cells, the human-agreement join |
| `consistency9.json` | Pairwise agreement and Fleiss κ across the ten model modes |
| `pcne_endpoints.json` | PCNE transfer arm: five descriptive endpoints, potential/manifest split |

## Reproducing the reported numbers

Python 3.12 with the packages in `requirements.txt`. Run from the package root:

```bash
python drp_eval/stage4.py score          # primary endpoint and all secondary endpoints
python drp_eval/score.py audit --batch 2 # rebuild the 98/7 split from the rating matrix
python drp_eval/derive_fig_data.py       # rewrite derived_figs.json
python drp_eval/consistency9.py          # rewrite consistency9.json
python drp_eval/pcne_endpoints.py        # rewrite pcne_endpoints.json
python drp_eval/shape_audit.py --json    # shape audit, printed as JSON
python drp_eval/make_pub_figures.py all  # redraw Figs. 2-9 into figures_nature/
```

`stage4.py score` prints the endpoint table without writing anything.
`stage4.py export` and the three derive scripts overwrite the shipped JSON
files; run them in a copy if you want to keep the released versions.

Expected values, for orientation: accuracy on the 98 consensus cells ranges
from 69.4% to 88.8% across the nine models; the structured-output ablation
parses 0 of 135 responses; pairwise agreement among model modes is 92.8% in
arm A and 89.8% in arm B.

## Licensing

Data (everything under `drp_eval/data/`, `drp_eval/runs/`, the JSON outputs and
this README) is released under CC BY 4.0. Code (everything under `drp_eval/*.py`
and `drp_eval/_vendor/`) is released under the MIT licence. The two licences are
provided in full alongside this file.

## Ethics and privacy

The cases are constructed vignettes: each reproduces a situation that occurs in
routine pharmacy practice, but the narratives were written for this study and do
not derive from, or correspond to, any individual patient. No patient was
involved, no clinical record was accessed, and no patient-identifiable
information is present in this deposit. The authors confirm that no ethics
approval was required under institutional and national regulations. Pharmacists
took part voluntarily and are identified only by pseudonymous labels.

Both language versions of every vignette are included, so that the benchmark can
be extended to new models. The English text is a translation of the Chinese
original; where the source contains an apparent clinical slip the translation
preserves it rather than correcting it, and the Chinese text remains
authoritative.

## Contact

Chao Fang (corresponding author), Department of Pharmacy, Tongren Hospital,
Shanghai Jiao Tong University School of Medicine, Shanghai, China.
