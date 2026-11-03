# -*- coding: utf-8 -*-
"""跨模型一致性 / 离散度：按 9 个模型重算（SAP v1.22 口径统一用）。

**口径变更说明（2026-10-06 · 用户裁定）**
  旧口径：跨模型离散度与 Fleiss κ 只按「必选 7 个模型」计算，补充名单 2 个
        （Ling-3.1-Flash、Kimi-K2.8-Preview）不进合并主分析的分母与 Holm 族。
  新口径：**9 个模型统一参与**主终点、自稳定率与跨模型一致性的计算。
        全部 9 个模型 A/B 组各 75/75 满格，数据完整性已核验，具备重算条件。

**本脚本只做重算，不改任何冻结项**：主终点定义、98／7 分母、50%／10% 四档阈值、
严格集合相等、top-k 读法、多重比较方法全部逐字沿用（判定逻辑 import 自 score.py）。

**同质性前提**：跨模型一致性要求各模型在同一案例同一条件下有同质的5 遍作答。
  A／B 两条件9 个模型均为 75/75（15 例 × 5 遍）满格 → 满足。

用法：
    python drp_eval/consistency9.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "drp_eval"))
from score import Dataset, DIMS, MULTI, classifications  # noqa: E402

RUNS = ROOT / "drp_eval" / "runs"

# 现行 9 个模型（SAP v1.22：取消必选／补充分组，统一为 9 个模型）
MODELS = [
    "deepseek-v4.1-flash", "glm-5.3-flash", "space-bunny", "minimax-m3",
    "mimo-v2.6-flash", "hy4-preview", "qwen3.8-flash",
    "ling-3.1-flash", "kimi-k2.8-preview",
]


def load_arm(arm, model):
    """读一个臂的落盘件 → {(case,dim): [5 遍答案]}"""
    p = RUNS / f"{arm}__{model}.jsonl"
    if not p.exists():
        return None
    buckets = defaultdict(list)
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            codes = r.get("parsed_codes") or {}
            if not isinstance(codes, dict):
                continue
            for dim, ai in codes.items():
                if dim in DIMS:
                    buckets[(r["case_id"], dim)].append(tuple(sorted(set(ai))))
    return buckets


def modal(buckets):
    """每(案例,维) 取 5 遍的众数（并列时取字典序最小者，保证确定性）。"""
    out = {}
    for key, answers in buckets.items():
        cnt = Counter(answers)
        top = max(cnt.values())
        out[key] = sorted([a for a, c in cnt.items() if c == top])[0]
    return out


def fleiss_kappa(counts):
    """Fleiss κ，**沿用项目权威实现 kappa_analysis.py 的变长 n_i 口径**。

    ⚠️ 不可用固定 n 的简化公式：n_i 变化时分母错配会算出 κ>1 的非法值。
    counts: [{label: n}, ...]，每格一个 dict（此处 n恒为 9 个模型）。
    """
    rows = []
    for c in counts:
        n = sum(c.values())
        if n < 2:
            continue
        agree = sum(v * (v - 1) for v in c.values()) / (n * (n - 1))
        rows.append((c, n, agree))
    if not rows:
        return None, 0
    p_bar = sum(r[2] for r in rows) / len(rows)
    total = sum(r[1] for r in rows)
    labels = set()
    for c, _, _ in rows:
        labels |= set(c)
    p_e = sum((sum(c.get(l, 0) for c, _, _ in rows) / total) ** 2 for l in labels)
    if abs(1 - p_e) < 1e-12:
        return None, len(rows)
    return (p_bar - p_e) / (1 - p_e), len(rows)


def consensus_agreement(models_modal, cls, dim_filter=None):
    """在有共识格上：各模型众数与金标准的一致比例（跨模型一致性）。"""
    per_model = {}
    gold_all = cls
    for name, modal_map in models_modal.items():
        hit = tot = 0
        for (cid, dim), ans in modal_map.items():
            if dim_filter and dim not in dim_filter:
                continue
            cell = gold_all[(cid, dim)]
            if not cell["consensus"]:
                continue
            tot += 1
            if set(ans) == set(cell["gold"]):
                hit += 1
        per_model[name] = {"hit": hit, "n": tot,
                           "rate": round(hit / tot, 4) if tot else None}
    return per_model


def main():
    ds = Dataset()
    cls = classifications(ds, 2, "topk")
    out = {"sap_version": "v1.22", "n_models": len(MODELS), "models": MODELS}

    for arm, label in (("A", "full arm (A)"), ("B", "codebook only (B)")):
        modal_map, buckets_all = {}, {}
        missing = []
        for m in MODELS:
            b = load_arm(arm, m)
            if b is None:
                missing.append(m)
                continue
            buckets_all[m] = b
            modal_map[m] = modal(b)
        print(f"\n{'='*68}")
        print(f"条件 {arm} · {label} · 参与模型 {len(modal_map)}/{len(MODELS)}")
        if missing:
            print(f"  ⚠️ 缺数据：{missing}")
        print(f"{'='*68}")

        # ---- ① 各模型主终点（众数答案 vs 金标准，有共识格） ----
        per_model = consensus_agreement(modal_map, cls)
        print("\n① 各模型众数答案在有共识格上的对错率")
        for m in MODELS:
            if m not in per_model:
                continue
            d = per_model[m]
            print(f"   {m:<24} {d['rate']*100:5.1f}%  ({d['hit']}/{d['n']})")

        # ---- ② 跨模型一致性：众数答案两两完全一致的比例 ----
        keys = sorted(set.intersection(*[set(m) for m in modal_map.values()]))
        keys = [k for k in keys if cls[k]["consensus"]]
        # 同一格上，9 个模型的众数答案两两比较
        same = 0
        pairs = 0
        for k in keys:
            answers = [set(modal_map[m].get(k, ())) for m in modal_map]
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    pairs += 1
                    if answers[i] == answers[j]:
                        same += 1
        pair_rate = same / pairs if pairs else None
        print(f"\n② 跨模型一致性（同格众数答案两两完全相等）")
        print(f"   有共识格 {len(keys)} 个 · 模型对 {pairs} 对")
        print(f"   **两两一致率 = {pair_rate*100:.1f}%**")

        # ---- ③ Fleiss κ（每格 = 一个评定项，9 个评定者 = 9 个模型众数） ----
        # 类别数取该格全部出现过的码
        per_dim = {}
        for dim in DIMS:
            counts = []
            for k in keys:
                if k[1] != dim:
                    continue
                ans = [set(modal_map[m].get(k, ())) for m in modal_map]
                # 多选格用其码集合的排序串作为类别标签（保证可哈希可比）
                labels = ["|".join(sorted(a)) if a else "(none)" for a in ans]
                counts.append(dict(Counter(labels)))
            if not counts:
                continue
            kappa, n_eff = fleiss_kappa(counts)
            if kappa is not None:
                per_dim[dim] = round(kappa, 3)
        if per_dim:
            print("\n③ Fleiss κ（9 个模型众数，按分项）")
            for d, v in per_dim.items():
                print(f"   {d:<4} κ = {v:+.3f}")
            vals = list(per_dim.values())
            print(f"   ——均值 {sum(vals)/len(vals):+.3f}"
                  f"｜ 跨度 {max(vals)-min(vals):.3f}"
                  f"｜ 最高 {max(per_dim, key=per_dim.get)}"
                  f"／最低 {min(per_dim, key=per_dim.get)}")

        out[arm] = {"per_model": per_model,
                    "pair_agreement": round(pair_rate, 4) if pair_rate else None,
                    "n_consensus_cells": len(keys),
                    "fleiss_kappa_by_dim": per_dim}

    p = ROOT / "drp_eval" / "consistency9.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n读数已写出 → {p}")


if __name__ == "__main__":
    main()
