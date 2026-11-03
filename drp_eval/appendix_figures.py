# -*- coding: utf-8 -*-
"""附录 Extended Data 19 张图（E01–E19）。

只读冻结数据：score.py 的判定、derived_figs.json、data/v2_evidence.json 与原始 runs。
每张图单独一个文件夹产物：figures_appendix/E01_*.{pdf,svg,png} 等。
E09–E17 每个模型一张，A／B 两个条件作为内部子图并排。

用法：
    python drp_eval/appendix_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "drp_eval"))

from score import Dataset, classifications, score_cell, DIMS  # noqa: E402
import make_pub_figures as mp  # noqa: E402
from shape_audit import audit as shape_audit_file  # noqa: E402

OUT = ROOT / "figures_appendix"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "drp_eval" / "runs"
MM = mp.MM
HALF = mp.HALF
COLW = mp.COLW
MANIFEST: list[dict] = []

PLAN_MODELS = [
    ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
    ("glm-5.3-flash", "GLM-5.3-Flash"),
    ("hy4-preview", "Hy4 preview"),
    ("minimax-m3", "MiniMax-M3"),
    ("mimo-v2.6-flash", "MiMo-V2.6-Flash"),
    ("qwen3.8-flash", "Qwen3.8-Flash"),
    ("space-bunny", "MiniMax-M3.1-Flash"),
    ("ling-3.1-flash", "Ling-3.1-Flash"),
    ("kimi-k2.8-preview", "Kimi K2.8 Preview"),
]
MODEL_KEYS = [k for k, _ in PLAN_MODELS]
PRETTY = dict(PLAN_MODELS)

CH = {**{k: "WorkBuddy" for k in ["deepseek-v4.1-flash", "glm-5.3-flash",
                                  "hy4-preview", "minimax-m3", "kimi-k2.8-preview"]},
      "space-bunny": "OpenCode", "mimo-v2.6-flash": "OpenCode",
      "ling-3.1-flash": "OpenCode", "qwen3.8-flash": "Qoder"}
CH_COLOR = {c: mp.pal.sw_color(c) for c in ("WorkBuddy", "OpenCode", "Qoder")}
CH_COLOR["OpenRouter"] = mp.NEU_M

# 2026-11-02 用户裁定：附录图统一低饱和马卡龙色
P_RED = "#E4A3A3"
P_ORANGE = "#F0B58E"
P_BLUE = "#9DB9D8"
P_TEAL = "#9FC9C9"
P_CREAM = "#F7F5EE"
P_GRAY = "#D9D9D4"
P_INK = "#333333"


def save(fig, code, name, title, data_source):
    base = OUT / f"{code}_{name}"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append({"id": code, "title": title, "data": data_source,
                     "files": [f"{base.name}{e}" for e in (".pdf", ".svg", ".png")]})


def _load_runs(pattern):
    p = RUNS / pattern
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _human_rows(cls, ds, batch):
    rows = []
    for (cid, dim), v in sorted(cls.items()):
        cnt, n = ds.distribution(batch, cid, dim)
        rows.append({"case": cid, "dim": dim, "consensus": bool(v["consensus"]),
                     "n_valid": n, "n_answers": len(cnt),
                     "gold": v["gold"], "votes": v["votes"],
                     "top": v["top"], "threshold": v["threshold"]})
    return rows


DS = Dataset()
CLS1 = classifications(DS, 1, "topk")
CLS2 = classifications(DS, 2, "topk")
ROWS1 = _human_rows(CLS1, DS, 1)
ROWS2 = _human_rows(CLS2, DS, 2)
CASES1 = sorted({r["case"] for r in ROWS1})
CASES2 = sorted({r["case"] for r in ROWS2})


# ---------------------------------------------------------------- human E1–E8
def e1_e2_map(rows, cases, code, name, title, source):
    ncon = sum(r["consensus"] for r in rows)
    mat = np.zeros((len(cases), len(DIMS)))
    for r in rows:
        mat[cases.index(r["case"]), DIMS.index(r["dim"])] = 1 if r["consensus"] else 0
    fig, ax = plt.subplots(figsize=(HALF, (52 + 2.2 * len(cases)) * MM))
    cmap = ListedColormap([P_RED, P_BLUE])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(DIMS, fontsize=5)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([str(c) for c in cases], fontsize=4.2)
    ax.set_xlabel("Dimension", fontsize=5)
    ax.set_ylabel("Case", fontsize=5)
    ax.set_title(f"{title}\nconsensus {ncon} / contested {len(rows)-ncon}", fontsize=6,
                 loc="left", pad=5)
    ax.legend(handles=[Patch(color=P_BLUE, label="consensus"),
                       Patch(color=P_RED, label="contested")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, code, name, title, source)


def e3_candidates():
    c1 = [r["n_answers"] for r in ROWS1]
    c2 = [r["n_answers"] for r in ROWS2]
    bins = np.arange(1, max(max(c1), max(c2)) + 2) - 0.5
    fig, ax = plt.subplots(figsize=(HALF, 55 * MM))
    ax.hist([c1, c2], bins=bins, label=["Batch 1 (139 raters)", "Batch 2 (46 raters)"],
            color=[P_BLUE, P_RED], alpha=0.85, rwidth=0.8)
    ax.set_xlabel("Distinct candidate codes per cell", fontsize=5)
    ax.set_ylabel("Cells", fontsize=5)
    ax.set_title("Candidate-code diversity per cell\n(batch 1 vs batch 2)", fontsize=6,
                 loc="left", pad=5)
    ax.legend(frameon=False, fontsize=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E03", "candidate_codes", "Candidate-code diversity per cell",
         "matrix_batch1/2.json via score.Dataset.distribution")


def e4_contested_cases():
    c1 = Counter(r["case"] for r in ROWS1 if not r["consensus"])
    c2 = Counter(r["case"] for r in ROWS2 if not r["consensus"])
    xs = np.arange(1, max(CASES1) + 1)
    fig, ax = plt.subplots(figsize=(HALF, 55 * MM))
    ax.bar(xs - 0.2, [c1.get(x, 0) for x in xs], width=0.4,
           color=P_BLUE, label="Batch 1 (28 contested)")
    ax.bar(xs + 0.2, [c2.get(x, 0) for x in xs], width=0.4,
           color=P_RED, label="Batch 2 (7 contested)")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs], fontsize=4.2)
    ax.set_xlabel("Case", fontsize=5); ax.set_ylabel("Contested cells", fontsize=5)
    ax.set_title("Contested cells by case\n(batch 1 vs batch 2)", fontsize=6,
                 loc="left", pad=5)
    ax.legend(frameon=False, fontsize=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E04", "contested_by_case", "Contested cells by case",
         "matrix_batch1/2.json + gold_batch1/2.json")


def e5_low_kappa():
    d = json.loads((ROOT / "drp_eval" / "data" / "v2_evidence.json").read_text(encoding="utf-8"))
    rows = sorted(d["low_kappa_rows"], key=lambda r: r[0])[:30]
    fig, ax = plt.subplots(figsize=(HALF, 105 * MM))
    y = np.arange(len(rows))[::-1]
    for yi, (k, batch, dim, code, ncase) in zip(y, rows):
        col = P_RED if dim == "C" else P_ORANGE
        ax.plot([0, k], [yi, yi], color=col, lw=0.7, zorder=2)
        ax.plot(k, yi, "o", ms=3.2, color=col, zorder=3)
        ax.text(k - 0.001, yi, f"{code} · b{batch} · n={ncase}", ha="right", va="center",
                fontsize=4.2, color=mp.INK)
    ax.axvline(0, color=mp.INK, lw=0.6)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(min(r[0] for r in rows) - 0.012, 0.002)
    ax.set_yticks([])
    ax.set_xlabel("Fleiss κ (sub-code)", fontsize=5)
    ax.set_title("Lowest-κ 30 of 111 C/I sub-codes\nC = red, I = orange",
                 fontsize=6, loc="left", pad=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E05", "low_kappa_subcodes", "Lowest-κ sub-codes (bottom 30)",
         "data/v2_evidence.json low_kappa_rows")


def e6_low_kappa_panorama():
    d = json.loads((ROOT / "drp_eval" / "data" / "v2_evidence.json").read_text(encoding="utf-8"))
    rows = d["low_kappa_rows"]
    fig, ax = plt.subplots(figsize=(HALF, 70 * MM))
    for dim, col in (("C", P_RED), ("I", P_ORANGE)):
        rr = [r for r in rows if r[2] == dim]
        ax.scatter([r[0] for r in rr], [r[4] for r in rr], s=10, color=col,
                   alpha=0.55, edgecolor="white", linewidth=0.2, label=f"{dim} (n={len(rr)})")
    ax.axvline(0, color=mp.INK, lw=0.5)
    ax.set_xlabel("Fleiss κ (sub-code)", fontsize=5)
    ax.set_ylabel("Cases selecting code", fontsize=5)
    ax.set_title("C/I sub-code κ landscape (111)\nC=71, I=40; 64 with κ≤0.10", fontsize=6,
                 loc="left", pad=5)
    ax.legend(frameon=False, fontsize=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E06", "low_kappa_panorama", "C/I sub-code κ landscape (111)",
         "data/v2_evidence.json low_kappa_rows")


def e7_valid_raters():
    fig, ax = plt.subplots(figsize=(HALF, 55 * MM))
    med1 = [float(np.median([r["n_valid"] for r in ROWS1 if r["dim"] == d])) for d in DIMS]
    med2 = [float(np.median([r["n_valid"] for r in ROWS2 if r["dim"] == d])) for d in DIMS]
    pos = np.arange(len(DIMS))
    ax.bar(pos - 0.19, med1, width=0.36, color=P_BLUE, label="Batch 1")
    ax.bar(pos + 0.19, med2, width=0.36, color=P_RED, label="Batch 2")
    for i, (a, b) in enumerate(zip(med1, med2)):
        ax.text(i - 0.19, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=4.6)
        ax.text(i + 0.19, b + 1.5, f"{b:.0f}", ha="center", va="bottom", fontsize=4.6)
    ax.set_xticks(pos); ax.set_xticklabels(DIMS, fontsize=5)
    ax.set_ylim(0, 155)
    ax.set_ylabel("Median valid raters per cell", fontsize=5)
    ax.set_title("Valid raters per cell\n(batch 1 blue · batch 2 red)", fontsize=6,
                 loc="left", pad=5)
    ax.legend(frameon=False, fontsize=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E07", "valid_raters", "Valid raters per cell",
         "matrix_batch1/2.json via score.Dataset.distribution")


def e8_v2_strength():
    vals = [("P1", 7), ("P2", 4), ("P3", 2)]
    fig, ax = plt.subplots(figsize=(HALF, 45 * MM))
    ax.bar([x[0] for x in vals], [x[1] for x in vals],
           color=[P_RED, P_ORANGE, P_GRAY], width=0.55)
    for i, (_k, v) in enumerate(vals):
        ax.text(i, v + 0.15, str(v), ha="center", va="bottom", fontsize=6,
                color=mp.INK, fontweight="bold")
    ax.set_ylabel("Recommendations", fontsize=5)
    ax.set_title("V2 revision recommendations by evidence strength\n(13 items total)",
                 fontsize=6, loc="left", pad=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E08", "v2_evidence_strength", "V2 recommendations by evidence strength",
         "产出物/V2修订建议清单.md §1")


# ------------------------------------------------------- per-model E09–E17
def _model_stats(arm, model):
    correct, reps = Counter(), Counter()
    err = Counter()
    for r in _load_runs(f"{arm}__{model}.jsonl"):
        if r.get("parse_status") != "ok" or not isinstance(r.get("parsed_codes"), dict):
            continue
        cid = int(r["case_id"])
        pc = r["parsed_codes"]
        for dim in DIMS:
            key = (cid, dim)
            if key not in CLS2 or not CLS2[key]["consensus"]:
                continue
            s = score_cell(DS, 2, cid, dim, pc.get(dim, []))
            reps[key] += 1
            if s["correct"]:
                correct[key] += 1
            else:
                g, a = set(s["gold"]), set(s["ai"])
                err["over_report" if (a - g) else
                    "under_report" if (g - a) else "wrong_code"] += 1
    return correct, reps, err


ARM_LABEL = {"A": "A: full arm", "B": "B: codebook only"}


def _per_model_ab(code, name, model):
    """同一模型的 A／B 合成一张图：左 A、右 B，共享案例行。"""
    cases = CASES2
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 100 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.09})
    cmap = LinearSegmentedColormap.from_list("corr", [P_RED, P_CREAM, P_BLUE])
    cmap.set_bad(P_GRAY)
    for ax, arm, letter in zip(axes, ("A", "B"), "ab"):
        correct, reps, err = _model_stats(arm, model)
        val = np.full((len(cases), len(DIMS)), np.nan)
        for (cid, dim), n in reps.items():
            if n:
                val[cases.index(cid), DIMS.index(dim)] = 5.0 * correct.get((cid, dim), 0) / n
        ax.imshow(np.ma.masked_invalid(val), cmap=cmap, vmin=0, vmax=5, aspect="auto")
        rates = []
        for dim in DIMS:
            cc = sum(correct.get((cid, dim), 0) for cid in cases)
            nn = sum(reps.get((cid, dim), 0) for cid in cases)
            rates.append(100 * cc / nn if nn else float("nan"))
        ax.set_xticks(range(len(DIMS)))
        ax.set_xticklabels([f"{d}\n{rr:.0f}%" for d, rr in zip(DIMS, rates)], fontsize=4.5)
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels([str(c) for c in cases], fontsize=4)
        if arm == "A":
            ax.set_ylabel("Case", fontsize=5)
        cc = sum(correct.values()); nn = sum(reps.values())
        overall = 100 * cc / nn if nn else float("nan")
        ax.set_title(f"{ARM_LABEL[arm]} · {overall:.1f}%  |  "
                     f"over {err.get('over_report', 0)} · "
                     f"under {err.get('under_report', 0)} · "
                     f"wrong {err.get('wrong_code', 0)}",
                     fontsize=5.2, loc="left", pad=2.3)
        mp.panel_label(ax, letter, dx_pt=-13, dy_pt=5)
        mp.clean_axes(ax, keep_left=True)
    fig.subplots_adjust(left=0.070, right=0.992, top=0.925, bottom=0.160)
    left = axes[0].get_position().x0
    fig.text(left, 0.045,
             f"{PRETTY[model]} · A vs B    98 consensus cells · "
             "0/5 red → 5/5 blue · gray = contested",
             ha="left", va="bottom", fontsize=5.2)
    save(fig, code, name, f"{PRETTY[model]} · A vs B per-cell correctness",
         f"runs/{{A,B}}__{model}.jsonl via score.score_cell")


# ---------------------------------------------------------- quality E18–E19
def e18_shape():
    arms = [("A", "A__{m}.jsonl"), ("B", "B__{m}.jsonl"),
            ("−examples", "ablation__{m}__minus_examples.jsonl"),
            ("−checklist", "ablation__{m}__minus_check.jsonl"),
            ("−follow-up", "ablation__{m}__minus_followup.jsonl"),
            ("−output", "ablation__{m}__minus_output.jsonl"),
            ("PCNE", "pcne__{m}.jsonl")]
    labels, ra, rb, ns = [], [], [], []
    for lab, pat in arms:
        na = nb = n = 0
        for m in MODEL_KEYS:
            p = RUNS / pat.format(m=m)
            if not p.exists():
                continue
            x, a, b, _ = shape_audit_file(str(p))
            n += x; na += a; nb += b
        if n:
            labels.append(lab); ns.append(n)
            ra.append(100 * na / n); rb.append(100 * nb / n)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(HALF, 60 * MM))
    ax.bar(x - 0.19, ra, width=0.36, color=P_BLUE, label="Ruler A: Chinese <5%")
    ax.bar(x + 0.19, rb, width=0.36, color=P_TEAL, label="Ruler B: <300 chars OR Chinese <5%")
    for i, (a, b, n) in enumerate(zip(ra, rb, ns)):
        ax.text(i - 0.19, a + 1.2, f"{a:.1f}", ha="center", va="bottom", fontsize=4.2)
        ax.text(i + 0.19, b + 1.2, f"{b:.1f}", ha="center", va="bottom", fontsize=4.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={nn})" for lab, nn in zip(labels, ns)], fontsize=4.4)
    ax.set_ylabel("Degenerate sessions (%)", fontsize=5)
    ax.set_ylim(0, 108)
    ax.set_title("Shape degeneration across 1980 sessions\n"
                 "−output is parse-fail by design", fontsize=6, loc="left", pad=5)
    ax.legend(frameon=False, fontsize=4.6, loc="upper right")
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E18", "shape_degeneration", "Shape degeneration across 1980 sessions",
         "runs/*.jsonl via shape_audit.audit")


def e19_ablation_parse():
    arms = [("−examples", "minus_examples"), ("−checklist", "minus_check"),
            ("−follow-up", "minus_followup"), ("−output", "minus_output")]
    mat = np.zeros((len(MODEL_KEYS), len(arms)))
    for i, m in enumerate(MODEL_KEYS):
        for j, (_lab, arm) in enumerate(arms):
            rows = _load_runs(f"ablation__{m}__{arm}.jsonl")
            oks = sum(1 for r in rows if r.get("parse_status") == "ok")
            mat[i, j] = 100 * oks / len(rows) if rows else np.nan
    fig, ax = plt.subplots(figsize=(HALF, 70 * MM))
    cmap = LinearSegmentedColormap.from_list("parse", [P_RED, P_CREAM, P_BLUE])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=4.4,
                    color=P_INK)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[0] for a in arms], fontsize=4.6)
    ax.set_yticks(range(len(MODEL_KEYS)))
    ax.set_yticklabels([PRETTY[m] for m in MODEL_KEYS], fontsize=4.4)
    ax.set_title("Parse success by ablation arm (%)\n−output = 0 by design",
                 fontsize=6, loc="left", pad=5)
    mp.clean_axes(ax, keep_left=True)
    save(fig, "E19", "ablation_parse", "Parse success by ablation arm",
         "runs/ablation__*.jsonl")


def main():
    e1_e2_map(ROWS1, CASES1, "E01", "consensus_map_batch1",
               "Batch 1 · consensus map", "matrix_batch1.json + gold_batch1.json")
    e1_e2_map(ROWS2, CASES2, "E02", "consensus_map_batch2",
               "Batch 2 · consensus map", "matrix_batch2.json + gold_batch2.json")
    e3_candidates()
    e4_contested_cases()
    e5_low_kappa()
    e6_low_kappa_panorama()
    e7_valid_raters()
    e8_v2_strength()
    for i, (m, _pretty) in enumerate(PLAN_MODELS):
        _per_model_ab(f"E{9+i:02d}", f"{m}_AB", m)
    e18_shape()
    e19_ablation_parse()
    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"✅ {len(MANIFEST)} 张附录图 → {OUT}")


if __name__ == "__main__":
    main()
