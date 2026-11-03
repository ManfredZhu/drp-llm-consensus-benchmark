# -*- coding: utf-8 -*-
"""新图（Fig.1／6／7）所需的派生读数——**一次算好落盘，出图脚本只读不算**

**为什么单独落盘**（AGENTS.md 原则 1「可读性」＋ 原则 2「模块化」）：
本模块把「错误类型学／逐格难度谱／跨模型共现」三类派生指标从原始 jsonl 算出，
落成`derived_figs.json`。出图脚本 `make_pub_figures.py` **只读该 json**，
不在绘图函数里做任何统计——与现有 `results_stage4.json` 的分工一致。

**判定逻辑全部 import 自 `score.py`**（冻结口径的唯一来源），无任何重写阈值。

用法：
    python drp_eval/derive_fig_data.py            # 落盘 derived_figs.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "drp_eval"))

from score import Dataset, classifications, score_cell, DIMS, MULTI  # noqa: E402

RUNS = ROOT / "drp_eval" / "runs"
OUT = ROOT / "drp_eval" / "derived_figs.json"

MODELS = ["kimi-k2.8-preview", "glm-5.3-flash", "hy4-preview",
          "deepseek-v4.1-flash", "minimax-m3", "ling-3.1-flash",
          "mimo-v2.6-flash", "space-bunny", "qwen3.8-flash"]

PRETTY = {
    "kimi-k2.8-preview": "Kimi K2.8 Preview", "glm-5.3-flash": "GLM-5.3-Flash",
    "hy4-preview": "Hy4 preview", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "minimax-m3": "MiniMax-M3", "ling-3.1-flash": "Ling-3.1-Flash",
    "mimo-v2.6-flash": "MiMo-V2.6-Flash", "space-bunny": "MiniMax-M3.1-Flash",
    "qwen3.8-flash": "Qwen3.8-Flash",
}
# 采集通道（Fig.1b 规模矩阵与 Fig.8c 用）
CHANNEL = {
    "deepseek-v4.1-flash": "WorkBuddy CLI", "glm-5.3-flash": "WorkBuddy CLI",
    "hy4-preview": "WorkBuddy CLI", "minimax-m3": "WorkBuddy CLI",
    "kimi-k2.8-preview": "WorkBuddy CLI", "space-bunny": "OpenCode CLI",
    "mimo-v2.6-flash": "OpenCode CLI", "ling-3.1-flash": "OpenRouter CLI",
    "qwen3.8-flash": "Qoder CN CLI",
}


def load(arm, model):
    f = RUNS / f"{arm}__{model}.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def build():
    ds = Dataset()
    cls = classifications(ds, 2, "topk")          # 第二批 · top-k 主口径
    cons = {k: v for k, v in cls.items() if v["consensus"]}
    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "sap_version": "v1.25",
        "n_cells": len(cons),
        "models": MODELS,
        "pretty": PRETTY,
        "channel": CHANNEL,
    }

    # ---------- 逐格层：每格每模型的答对遍数 + 错因 ----------
    # R40教训：每格每模型只有 5 遍 → 「全对」= 5/5，不是 9/9
    per_model_correct = {}          # model -> {(cid,dim): 答对遍数}
    per_model_reps = {}             # model -> {(cid,dim): 有效遍数}
    err_kind = {}                   # model -> Counter{错因}
    err_kind_by_dim = {}            # (model,dim) -> Counter{错因}
    for m in MODELS:
        c = Counter(); n = Counter(); ek = Counter(); ekd = defaultdict(Counter)
        for r in load("A", m):
            cid = int(r["case_id"])
            pc = r.get("parsed_codes")
            if r.get("parse_status") != "ok" or not isinstance(pc, dict):
                continue
            for dim in DIMS:
                key = (cid, dim)
                if key not in cons:
                    continue
                s = score_cell(ds, 2, cid, dim, pc.get(dim, []))
                n[key] += 1
                if s["correct"]:
                    c[key] += 1
                    continue
                g, a = set(s["gold"]), set(s["ai"])
                # 🔴 错因判据（R42教训：分类须互斥且穷尽，按优先级顺序判定）
                kind = ("over_report" if (a - g) else      # 多报：多给了码
                        "under_report" if (g - a) else     # 漏报：缺了码
                        "wrong_code")                     # 码错位（互有出入）
                ek[kind] += 1
                ekd[dim][kind] += 1
        per_model_correct[m] = dict(c)
        per_model_reps[m] = dict(n)
        err_kind[m] = dict(ek)
        err_kind_by_dim[m] = {d: dict(v) for d, v in ekd.items()}
    out["per_model_correct"] = {m: {f"{k[0]}|{k[1]}": v for k, v in per_model_correct[m].items()}
                                for m in MODELS}
    out["per_model_reps"] = {m: {f"{k[0]}|{k[1]}": v for k, v in per_model_reps[m].items()}
                             for m in MODELS}
    out["err_kind"] = err_kind
    out["err_kind_by_dim"] = {m: err_kind_by_dim[m] for m in MODELS}

    # ---------- 错误类型学汇总（全体 / 分维度 / 严格 vs 包含口径）----------
    tot = Counter()
    tot_dim = defaultdict(Counter)
    loose_pass = 0                  # 「金标准⊆AI 但多了码」被严格判错的次数
    strict_fail = 0
    scored = 0
    for m in MODELS:
        tot.update(err_kind[m])
        for d, v in err_kind_by_dim[m].items():
            tot_dim[d].update(v)
        strict_fail += sum(err_kind[m].values())
        scored += sum(per_model_reps[m].values())
    # 包含口径：仅对多选维（严格集合相等的口径敏感处）可放宽
    for m in MODELS:
        for r in load("A", m):
            cid = int(r["case_id"])
            pc = r.get("parsed_codes")
            if r.get("parse_status") != "ok" or not isinstance(pc, dict):
                continue
            for dim in DIMS:
                key = (cid, dim)
                if key not in cons:
                    continue
                s = score_cell(ds, 2, cid, dim, pc.get(dim, []))
                if s["correct"]:
                    continue
                g, a = set(s["gold"]), set(s["ai"])
                if dim in MULTI and g and g <= a:
                    loose_pass += 1
    out["err_summary"] = {
        "by_kind": dict(tot), "total_errors": sum(tot.values()),
        "by_dim": {d: dict(v) for d, v in tot_dim.items()},
        "n_scored_records": scored,
        "loose_gain_records": loose_pass,
    }

    # ---------- 逐格难度谱：每格被几个模型「5遍全对」 ----------
    #🔴 键型必须统一：内部一律用元组 (cid,dim)，落盘时才转 "cid|dim" 字符串。
    #   （本轮首版在循环里用元组查 `per_model_reps`、却用字符串回查，
    #   导致rates全空、全对分布恒为 0 —— 与 R42 同类的「静默归零」失败。）
    n_all = Counter()
    cell_detail = {}
    for key in cons:                      # key = (cid, dim)
        full = 0
        rates = {}
        for m in MODELS:
            tot_r = per_model_reps[m].get(key, 0)
            if tot_r == 0:
                continue
            rt = per_model_correct[m].get(key, 0) / tot_r
            rates[m] = round(rt, 4)
            if rt == 1.0:
                full += 1
        n_all[full] += 1
        cell_detail[f"{key[0]}|{key[1]}"] = {
            "n_models_perfect": full, "rates": rates, "dim": key[1], "case": key[0]}
    out["cell_perfect_distribution"] = {str(k): v for k, v in n_all.items()}
    out["cell_detail"] = cell_detail

    # ---------- 跨模型错误共现（Jaccard） ----------
    wrong_sets = {}
    for m in MODELS:
        wrong_sets[m] = {k for k, v in per_model_correct[m].items()
                         if v < per_model_reps[m].get(k, 0)}
    J = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            u = len(wrong_sets[a] | wrong_sets[b])
            J[f"{a}|{b}"] = round(len(wrong_sets[a] & wrong_sets[b]) / u, 4) if u else 0.0
    # 独立期望：两模型错误集**独立**时的 Jaccard 期望。
    # 公式：J = |A∩B| / |A∪B|；独立时 E|A∩B| = p²·N、E|A∪B| = 2pN − p²·N
    #      → E[J] = p² / (2p − p²) = p / (2 − p)，p = 单模型平均错误率。
    # ⚠️ 不可写成 p²（那是「交／并都按交算」的错式，会把期望压低一个量级）。
    p = (sum(len(wrong_sets[m]) for m in MODELS) / len(MODELS)) / len(cons)
    exp_indep = p / (2 - p) if p < 2 else 0.0
    out["jaccard"] = J
    out["jaccard_median"] = round(sorted(J.values())[len(J) // 2], 4)
    out["jaccard_expected_if_independent"] = round(exp_indep, 4)
    out["n_wrong_cells_any_model"] = len(set().union(*wrong_sets.values()))
    out["hard_cells"] = sorted(
        [k for k, v in cell_detail.items() if v["n_models_perfect"] == 0],
        key=lambda s: -cons[(int(s.split("|")[0]), s.split("|")[1])]["n_valid"])

    # ---------- 人类共识度 vs AI 错误数 ----------
    rows = []
    for key, v in cons.items():
        nv = v["n_valid"]
        if not nv:
            continue
        share = min(v["votes"].get(c, 0) for c in v["gold"]) / nv
        n_err = sum(1 for m in MODELS if per_model_correct[m].get(key, 0) < per_model_reps[m].get(key, 0))
        rows.append({"cell": f"{key[0]}|{key[1]}", "gold_share": round(share, 4),
                     "n_candidates": v["n_answers"], "n_models_wrong": n_err})
    def pearson(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return round(num / (dx * dy), 4) if dx and dy else 0.0
    out["human_vs_ai"] = {
        "rows": rows,
        "r_share_vs_errors": pearson([r["gold_share"] for r in rows],
                                     [r["n_models_wrong"] for r in rows]),
        "r_candidates_vs_errors": pearson([r["n_candidates"] for r in rows],
                                          [r["n_models_wrong"] for r in rows]),
        "median_share_hard": statistics_median(
            [r["gold_share"] for r in rows if r["n_models_wrong"] == len(MODELS)]),
        "median_share_all": statistics_median([r["gold_share"] for r in rows]),
    }

    # ---------- 硬格的完整信息（供 d 面板标注） ----------
    hard = []
    for k in out["hard_cells"]:
        cid, dim = int(k.split("|")[0]), k.split("|")[1]
        v = cls[(cid, dim)]
        nv = v["n_valid"]
        hard.append({"case": cid, "dim": dim, "gold": v["gold"],
                     "gold_share": round(min(v["votes"].get(c, 0) for c in v["gold"]) / nv, 4),
                     "n_candidates": v["n_answers"]})
    out["hard_cell_detail"] = sorted(hard, key=lambda r: -r["gold_share"])
    return out


def statistics_median(xs):
    xs = sorted(xs)
    n = len(xs)
    return round(xs[n // 2], 4) if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 4)


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    e = d["err_summary"]
    print(f"读数已写出 → {OUT}")
    print(f"  有共识格 {d['n_cells']}")
    print(f"  错因 {e['by_kind']}  共 {e['total_errors']}")
    print(f"  逐格全对分布 {d['cell_perfect_distribution']}")
    print(f"  Jaccard 中位 {d['jaccard_median']} vs 独立期望 {d['jaccard_expected_if_independent']}")
    print(f"  人类得票占比 vs AI错误数 r = {d['human_vs_ai']['r_share_vs_errors']}")
    print(f"  全错硬格 {len(d['hard_cell_detail'])} 个")