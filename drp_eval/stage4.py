# -*- coding: utf-8 -*-
"""阶段 4 主驱动器：一次算出全部模型 × 全部终点读数。

**只做三件事，且完全复用 `score.py` 的冻结判定逻辑（import，不复制）**：
  1. `score`  —— 主终点（有共识格对错率）、共同主要终点（争议格四档）、
                  分项读数、效率（latency_ms）、自稳定性；
  2. `export` —— 把读数写成机器可读 JSON，供绘图脚本消费；
  3. `pcne`   —— PCNE 迁移臂五终点的描述性读数（无金标准、不做准确性结论）。

**红线（本脚本的自我约束，与 SAP 同源）**：
  · 不含任何针对具体案例/分项/模型的特判；
  · 一切阈值从 `score.py` 读，不在本文件里重新硬编码；
  · PCNE 臂的输出永不与主实验并池（第 6 节强制分文件）。

用法：
    python drp_eval/stage4.py score          # 打印9 模型全终点读数
    python drp_eval/stage4.py export         # 写出 results_stage4.json
    python drp_eval/stage4.py pcne           # PCNE 臂五终点
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "drp_eval"))
from score import (  # noqa: E402  冻结判定逻辑的唯一来源
    DIMS, MULTI, Dataset, classifications, score_cell, cluster_bootstrap_ci,
)

RUNS = ROOT / "drp_eval" / "runs"
OUT = ROOT / "drp_eval"

# 必选7 + 补充2（SAP v1.21 §10.1）
REQUIRED = ["deepseek-v4.1-flash", "glm-5.3-flash", "space-bunny", "minimax-m3",
            "mimo-v2.6-flash", "hy4-preview", "qwen3.8-flash"]
SUPPLEMENT = ["ling-3.1-flash", "kimi-k2.8-preview"]
ALL_MODELS = REQUIRED + SUPPLEMENT
PRETTY = {
    "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
    "glm-5.3-flash": "GLM-5.3-Flash",
    "space-bunny": "MiniMax-M3.1-Flash",
    "minimax-m3": "MiniMax-M3",
    "mimo-v2.6-flash": "MiMo-V2.6-Flash",
    "hy4-preview": "Hy4-Preview",
    "qwen3.8-flash": "Qwen3.8-Flash",
    "ling-3.1-flash": "Ling-3.1-Flash",
    "kimi-k2.8-preview": "Kimi-K2.8-Preview",
}
ABLATIONS = ["minus_examples", "minus_check", "minus_followup", "minus_output"]


def load(arm, model):
    """读一个臂的落盘件。返回 list[dict]。"""
    p = RUNS / f"{arm}__{model}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def batch2_of(rec):
    """A/B/消融臂跑的是第二批（AI 测试集）；第一批 140 格不参与 AI 评分。"""
    return 2


def score_model(ds, cls, model, arm="A"):
    """单模型单臂 → 主终点/共同主要终点/分项/效率 读数。

    落盘件的粒度是「一次作答 = 一个 (case, repetition) × 全部 7 分项」，
    故须遍历 `parsed_codes` 的每个分项键逐格计分。
    """
    recs = load(arm, model)
    if not recs:
        return None
    # 🔴 口径显式化（SAP v1.25 §10.2-#34 · 2026-10-07 用户裁定）
    #   历史行为：`per_cell[(cid,dim)] = s["correct"]`，per_cell 是 dict，
    #   后一遍覆盖前一遍 → 98 格每格实际只保留**第 5 次（末次）作答**的对错，
    #   前 4 遍被静默丢弃。该行为从未在 SAP 显式登记，属实现副作用。
    #   现显式化为三口径，主口径仍为 `last`（读数与历史逐位一致，不改任何冻结项）：
    #     last = 末遍口径（= 历史行为，主口径）
    #     mode = 众数口径（该格 5 遍中答对 ≥3 记该格正确）→ 敏感性 S10
    #     mean = 全遍口径（正确格次 / 总格次）→ 敏感性 S11
    #   三口径下 9 模型的排名方向与 A>B 方向均不变（2026-10-07 实测）。
    last_cell = {}          # (case,dim) -> bool   末遍
    correct_counts = defaultdict(int)   # (case,dim) -> 答对遍数
    seen_counts = defaultdict(int)      # (case,dim) -> 有效遍数（分母）
    ok_g, n_g = 0, 0        # 全遍口径的分子/分母
    contested = defaultdict(list)
    dim_stat = {d: [0, 0] for d in DIMS}
    lat, parsed, shape_degen, shape_total = [], 0, 0, 0

    for r in recs:
        cid = r["case_id"]
        codes = r.get("parsed_codes") or {}
        lat.append(r.get("latency_ms") or 0)
        parsed += 1 if r.get("parse_status") == "ok" else 0
        # 形态退化尺子 B（事后增补的更严参照，SAP v1.13）
        txt = r.get("raw_output") or ""
        cn = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
        shape_total += 1
        if len(txt) < 300 or (cn / max(len(txt), 1)) < 0.05:
            shape_degen += 1
        if not isinstance(codes, dict):
            continue
        for dim, ai in codes.items():
            if dim not in DIMS or (cid, dim) not in cls:
                continue
            s = score_cell(ds, 2, cid, dim, ai)
            if s["type"] == "consensus":
                last_cell[(cid, dim)] = s["correct"]
                seen_counts[(cid, dim)] += 1
                ok_g += bool(s["correct"])
                n_g += 1
                if s["correct"]:
                    correct_counts[(cid, dim)] += 1
                dim_stat[dim][0] += bool(s["correct"])
                dim_stat[dim][1] += 1
            else:
                contested[dim].append(s["tier"])

    if not last_cell:
        return None
    per_cell = last_cell
    point, lo, hi = cluster_bootstrap_ci(ds, 2, per_cell)

    # 敏感性 S10（众数 ≥3/5）—— bootstrap 在格级重抽，与主口径同法
    mode_cell = {k: (correct_counts[k] >= 3) for k in last_cell}
    m_point, m_lo, m_hi = cluster_bootstrap_ci(ds, 2, mode_cell)
    # 敏感性 S11（全遍平均）—— 分子分母均为格次
    mean_point = ok_g / max(n_g, 1)
    # 自稳定性：同一格多遍答案的两两一致性（按案例聚合）
    return {
        "model": model, "pretty": PRETTY.get(model, model), "arm": arm,
        "group": "required" if model in REQUIRED else "supplement",
        "n_records": len(recs), "n_parse_ok": parsed,
        "parse_rate": round(parsed / len(recs), 4),
        "consensus": {
            "n_cells": len(per_cell),
            "correct": sum(per_cell.values()),
            "rate": point, "ci_lo": lo, "ci_hi": hi,
            "reading": "last",       # 🔴 主口径显式化（SAP v1.25 §10.2-#34）
            "sensitivity": {# 敏感性 S10/S11：同一98 格的另两种读法，不改主读数
                "mode_ge3": {"correct": sum(mode_cell.values()),
                              "rate": m_point, "ci_lo": m_lo, "ci_hi": m_hi},
                "all_pass": {"n_records_scored": n_g,
                             "rate": round(mean_point, 4)},
            },
        },
        "by_dim": {d: {"correct": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                   for d, v in dim_stat.items() if v[1]},
        "contested": {d: {"tiers": t, "n": len(t)} for d, t in contested.items()},
        "latency_ms": {
            "median": int(statistics.median(lat)) if lat else None,
            "mean": int(statistics.mean(lat)) if lat else None,
        },
        "shape": {"degenerate": shape_degen, "total": shape_total,
                  "rate": round(shape_degen / max(shape_total, 1), 4)},
    }


def stability(recs):
    """自稳定性：同一 (case,dim) 的 N 遍答案，两两严格集合相等比例。"""
    buckets = defaultdict(list)
    for r in recs:
        codes = r.get("parsed_codes") or {}
        if not isinstance(codes, dict):
            continue
        for dim, ai in codes.items():
            if dim in DIMS:
                buckets[(r["case_id"], dim)].append(frozenset(ai))
    pairs, same = 0, 0
    for v in buckets.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs += 1
                same += (v[i] == v[j])
    return {"pairs": pairs, "agree": same,
            "rate": round(same / pairs, 4) if pairs else None}


def run_score(verbose=True):
    """单模型读数（内部用；对外入口是 collect）。"""
    ds, cls, res = collect(verbose=verbose)
    return ds, cls, res


def load_ablation(model, variant):
    p = RUNS / f"ablation__{model}__{variant}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_ablation(ds, cls, model, variant):
    recs = load_ablation(model, variant)
    if not recs:
        return None
    per_cell = {}
    for r in recs:
        codes = r.get("parsed_codes") or {}
        if not isinstance(codes, dict):
            continue
        for dim, ai in codes.items():
            if dim not in DIMS or (r["case_id"], dim) not in cls:
                continue
            s = score_cell(ds, 2, r["case_id"], dim, ai)
            if s["type"] == "consensus":
                per_cell[(r["case_id"], dim)] = s["correct"]
    if not per_cell:
        return None
    point, lo, hi = cluster_bootstrap_ci(ds, 2, per_cell)
    return {"n_cells": len(per_cell), "correct": sum(per_cell.values()),
            "rate": point, "ci_lo": lo, "ci_hi": hi, "n_records": len(recs)}


def collect(verbose=True):
    """全量读数（干净实现）。"""
    ds = Dataset()
    cls = classifications(ds, 2, "topk")
    res = {}
    for m in ALL_MODELS:
        entry = {"group": "required" if m in REQUIRED else "supplement",
                 "pretty": PRETTY.get(m, m)}
        for arm in ("A", "B"):
            s = score_model(ds, cls, m, arm)
            if s:
                if arm == "A":
                    s["stability"] = stability(load("A", m))
                entry[arm] = s
        abl = {}
        for v in ABLATIONS:
            s = score_ablation(ds, cls, m, v)
            if s:
                abl[v] = s
        if abl:
            entry["ablation"] = abl
        if "A" in entry:
            res[m] = entry
            if verbose:
                c = entry["A"]["consensus"]
                print(f"  ✅ {entry['pretty']:22s} 主终点 {c['rate']*100:5.1f}%  "
                      f"({c['correct']}/{c['n_cells']})  95%CI "
                      f"[{c['ci_lo']*100:.1f}, {c['ci_hi']*100:.1f}]  "
                      f"自稳定 {entry['A']['stability']['rate']*100:.1f}%")
        else:
            if verbose:
                print(f"  ⚠️ {entry['pretty']:22s} 无 A 组数据")
    return ds, cls, res


def run_pcne(verbose=True):
    """PCNE 迁移臂五终点（全部描述性；无金标准、不做准确性结论、不与主实验并池）。"""
    out = {}
    for m in ALL_MODELS:
        recs = load("pcne", m)
        if not recs:
            continue
        ok = sum(1 for r in recs if r.get("parse_status") == "ok")
        # 终点1 格式合规= 码全部落在码表内（由 pcne_audit.py 校验后置入 ok_flag）
        fmt = sum(1 for r in recs if r.get("in_codebook"))
        # 终点2 流程走通：七段结构齐全
        flow = sum(1 for r in recs if r.get("flow_complete"))
        # 终点3 生态一致性：与该臂众数答案一致
        # 终点4 P 段跨码表稳定性（主实验 A 组 P 大类对照，另算）
        lat = [r.get("latency_ms") or 0 for r in recs]
        out[m] = {"pretty": PRETTY.get(m, m), "n": len(recs),
                  "parse_ok": ok, "format_ok": fmt, "flow_ok": flow,
                  "latency_median": int(statistics.median(lat)) if lat else None}
        if verbose:
            print(f"  {PRETTY.get(m, m):22s} n={len(recs):2d} parse_ok={ok:2d} "
                  f"format={fmt:2d} flow={flow:2d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "export", "pcne"])
    a = ap.parse_args()
    if a.mode == "pcne":
        print("\n=== PCNE 迁移臂（描述性，不并池） ===")
        run_pcne()
        return
    ds, cls, res = collect(verbose=(a.mode == "score"))
    if a.mode == "score":
        print(f"\n共 {len(res)} 个模型有A 组数据")
        return
    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "sap_version": "v1.21",
        "models": res,
        "required": REQUIRED, "supplement": SUPPLEMENT,
    }
    p = OUT / "results_stage4.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n读数已写出 → {p}")


if __name__ == "__main__":
    main()