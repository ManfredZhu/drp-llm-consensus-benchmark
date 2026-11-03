# -*- coding: utf-8 -*-
"""评分脚本（阶段 2.3 的核心件）

**只读** `drp_eval/data/*.json`，不含任何针对具体案例/分项的特判。

三件事：
  1. `audit`：复现两批的「有共识格 / 争议格」分母（期望 112/140、98/7）；
  2. `score`：按 SAP §3.2 判定 AI 输出对错（有共识格）/ 按 §4.2 判四档（争议格）；
  3. `bootstrap`：按案例聚类的 bootstrap CI（SAP §5）。

四种「有共识」读法（全部都算，主口径取 topk）：
  topk        主口径  每个金标准码得票 >= 该格第 k 高票（k = 金标准码数）——见 SAP §3.1
  tied        S8     每个金标准码得票 == 最高票（"全部并列第一"）
  strict_top  S4     按冻结原文「**仅对并列最高票的格子**改判（须唯一最高票）」——
                     k=1 时要求得票 > 第二高票；k>=2 时维持 topk 不变。
  strict_all  （新增发现，冻结文本未覆盖）"任何名次都不许并列"的一致化应用：
                     k 个码得票均须 > 第 (k+1) 高票。比 S4 多改一格（案例 12·C，
                     第 2/3 高票并列 11:11）→ 第二批 96／9 而非 97／8。已回报待用户裁定。

用法：
  python drp_eval/score.py audit
  python drp_eval/score.py audit --batch 2
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "drp_eval" / "data"

DIMS = ["P", "E1", "E2", "C", "I", "A", "O"]
MULTI = {"C", "I"}

# ---- 冻结的期望值（SAP §10.1 #2；改动须走 §7 敏感性分析，不得静默漂移） ----
EXPECT = {
    1: {"consensus": 112, "contested": 28, "total": 140},
    2: {"consensus": 98, "contested": 7, "total": 105},
}
EXPECT_CONTESTED_2 = {(1, "E2"), (9, "E2"), (10, "E2"), (11, "E2"), (11, "C"), (2, "I"), (11, "O")}


# ---------------- 载入 ----------------
class Dataset:
    def __init__(self, batches=(1, 2)):
        self.gold = {}
        self.matrix = {}
        self.cases = {}
        for b in batches:
            g = json.loads((DATA / f"gold_batch{b}.json").read_text(encoding="utf-8"))
            self.gold[b] = {rec["case"]: rec for rec in g}
            raw = json.loads((DATA / f"matrix_batch{b}.json").read_text(encoding="utf-8"))
            m = defaultdict(lambda: defaultdict(list))
            for r in raw["records"]:
                m[r["case_id"]][r["dim"]].append(r)
            self.matrix[b] = m
        for c in json.loads((DATA / "cases.json").read_text(encoding="utf-8")):
            self.cases[(c["batch"], c["case_id"])] = c

    def distribution(self, batch, case_id, dim):
        """返回 (Counter{code: 人数}, 有效作答人数)。一名评议者同一码只计一次。"""
        cnt, n = Counter(), 0
        for r in self.matrix[batch][case_id][dim]:
            if not r["codes"]:
                continue
            n += 1
            for c in set(r["codes"]):
                cnt[c] += 1
        return cnt, n

    def gold_codes(self, batch, case_id, dim):
        return list(self.gold[batch][case_id][dim])


# ---------------- 有共识 / 争议 ----------------
def judge(cnt, gold, reading="topk"):
    """返回 (is_consensus, threshold, k)。"""
    if not gold or not cnt:
        return None, 0, 0
    votes = sorted(cnt.values(), reverse=True)
    k = min(len(gold), len(votes))
    if reading == "topk":
        thr = votes[k - 1]
        return all(cnt.get(c, 0) >= thr for c in gold), thr, len(gold)
    if reading == "tied":
        thr = votes[0]
        return all(cnt.get(c, 0) >= thr for c in gold), thr, len(gold)
    if reading == "strict_top":            # S4：仅要求 k=1 时为唯一最高票
        if len(gold) == 1:
            second = votes[1] if len(votes) > 1 else 0
            gv = cnt.get(gold[0], 0)
            return gv > second, second + 1, 1
        thr = votes[k - 1]
        return all(cnt.get(c, 0) >= thr for c in gold), thr, len(gold)
    if reading == "strict_all":            # 一致化：任何名次都不许并列
        thr = votes[k] if len(votes) > k else 0
        return all(cnt.get(c, 0) > thr for c in gold), thr + 1, len(gold)
    raise ValueError(reading)


def classifications(ds, batch, reading="topk"):
    """返回 {(case_id, dim): dict}。"""
    out = {}
    for cid in sorted(ds.gold[batch]):
        for dim in DIMS:
            cnt, n = ds.distribution(batch, cid, dim)
            gold = ds.gold_codes(batch, cid, dim)
            ok, thr, k = judge(cnt, gold, reading)
            if ok is None:
                raise AssertionError(f"!! 第{batch}批 案例{cid}·{dim} 无法判定（有效人数 {n}，金标准 {gold}）")
            out[(cid, dim)] = {"consensus": ok, "n_valid": n, "k": k, "threshold": thr,
                               "gold": gold, "votes": {c: cnt.get(c, 0) for c in gold},
                               "n_answers": len(cnt), "top": cnt.most_common(1)[0] if cnt else ("", 0)}
    return out


# ---------------- AI 输出评分 ----------------
def score_cell(ds, batch, case_id, dim, ai_codes, reading="topk"):
    """按 SAP §3.2 / §4.2 给单个格子打分。

    有共识格：返回 {"type":"consensus","correct":bool}
    争议格：返回 {"type":"contested","tier":str,"tier_min":str}（tier_min = 最低档主口径）
    """
    cnt, n = ds.distribution(batch, case_id, dim)
    gold = ds.gold_codes(batch, case_id, dim)
    ok, _, _ = judge(cnt, gold, reading)
    ai = sorted(set(ai_codes))

    if ok:
        if dim in MULTI:
            correct = set(ai) == set(gold)                    # 主口径：严格集合相等
        else:
            correct = len(ai) == 1 and ai[0] == gold[0]       # 单选：等于金标准码
        return {"type": "consensus", "correct": correct, "gold": gold, "ai": ai}

    tier_of = {}
    for c in ai:
        share = (cnt.get(c, 0) / n) if n else 0
        tier_of[c] = ("多数派档" if share >= 0.50 else
                      "可接受档" if share >= 0.10 else
                      "边缘档" if share > 0 else "无人档")
    if not ai:
        tiers = ["无人档"]
    else:
        tiers = [tier_of[c] for c in ai]
    order = {"多数派档": 0, "可接受档": 1, "边缘档": 2, "无人档": 3}
    tier_min = max(tiers, key=lambda t: order[t])              # 主口径：最低档（保守）
    tier_max = min(tiers, key=lambda t: order[t])              # S7 宽松口径
    return {"type": "contested", "tier": tier_min, "tier_max": tier_max,
            "tiers": tier_of, "gold": gold, "ai": ai}


# ---------------- 聚类 bootstrap（SAP §5） ----------------
def cluster_bootstrap_ci(ds, batch, per_cell, n_boot=10000, seed=20261002):
    """按案例聚类重抽；per_cell: {(case_id, dim): bool} -> (point, lo, hi)。"""
    by_case = defaultdict(list)
    for (cid, dim), v in per_cell.items():
        by_case[cid].append(1.0 if v else 0.0)
    cids = sorted(by_case)
    point = sum(sum(by_case[c]) for c in cids) / sum(len(by_case[c]) for c in cids)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        picks = [rng.choice(cids) for _ in cids]
        vals = [v for c in picks for v in by_case[c]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


# ---------------- audit ----------------
def run_audit(ds, batches, verbose=True):
    ok_all = True
    for b in batches:
        cls = classifications(ds, b, "topk")
        cls_tied = classifications(ds, b, "tied")
        cls_s4 = classifications(ds, b, "strict_top")
        cls_sall = classifications(ds, b, "strict_all")

        per_dim = defaultdict(lambda: [0, 0])
        diffs_tied, diffs_s4, diffs_sall, contested = [], [], [], []
        for key, v in cls.items():
            per_dim[key[1]][1] += 1
            per_dim[key[1]][0] += bool(v["consensus"])
            if not v["consensus"]:
                contested.append(key)
            if v["consensus"] != cls_tied[key]["consensus"]:
                diffs_tied.append(key)
            if v["consensus"] != cls_s4[key]["consensus"]:
                diffs_s4.append(key)
            if v["consensus"] != cls_sall[key]["consensus"]:
                diffs_sall.append(key)

        n_con = sum(v[0] for v in per_dim.values())
        n_tot = sum(v[1] for v in per_dim.values())
        exp = EXPECT[b]
        flag = "✅ 复现" if (n_con == exp["consensus"] and n_tot == exp["total"]) else "❌ 与冻结值不符"
        ok_all &= flag.startswith("✅")

        print(f"\n{'='*70}")
        print(f"第 {b} 批 · {len(ds.gold[b])} 例 × 7 分项 = {n_tot} 格")
        print(f"{'='*70}")
        print(f"{'分项':<6}{'有共识':>7}{'争议':>6}")
        for dim in DIMS:
            con, tot = per_dim[dim]
            print(f"{dim:<8}{con:>5}{tot - con:>6}")
        print(f"{'合计':<6}{n_con:>7}{n_tot - n_con:>6}")
        print(f"\n{flag}：有共识 {n_con} / 争议 {n_tot - n_con}（冻结值 {exp['consensus']} / {exp['contested']}）")
        print(f"S8 替代读法 tied       ：有共识 {sum(1 for v in cls_tied.values() if v['consensus'])}"
              f" / 争议 {sum(1 for v in cls_tied.values() if not v['consensus'])}"
              f"（{len(diffs_tied)} 格判定不同）")
        print(f"S4 严格口径 strict_top ：有共识 {sum(1 for v in cls_s4.values() if v['consensus'])}"
              f" / 争议 {sum(1 for v in cls_s4.values() if not v['consensus'])}"
              f"（{len(diffs_s4)} 格判定不同）")
        print(f"   [新增发现] strict_all：有共识 {sum(1 for v in cls_sall.values() if v['consensus'])}"
              f" / 争议 {sum(1 for v in cls_sall.values() if not v['consensus'])}"
              f"（{len(diffs_sall)} 格判定不同：{diffs_sall}）")

        if b == 2:
            got = set(contested)
            match = got == EXPECT_CONTESTED_2
            ok_all &= match
            print(f"\n争议格名单 {'✅ 与冻结名单一致' if match else '❌ 不一致'}：")
            for cid, dim in sorted(contested):
                v = cls[(cid, dim)]
                print(f"  · 案例{cid}·{dim}  金标准={v['gold']}（得票 "
                      f"{[v['votes'][c] for c in v['gold']]}）  最高票={v['top'][0]}"
                      f"({v['top'][1]}人)  答案种数={v['n_answers']}")

        (DATA / f"cells_batch{b}.json").write_text(json.dumps(
            {f"{k[0]}|{k[1]}": v for k, v in cls.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    print(f"\n逐格判定表已写出 → {DATA}/cells_batch*.json")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit"])
    ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args()
    ds = Dataset()
    batches = [a.batch] if a.batch else [1, 2]
    ok = run_audit(ds, batches)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
