# -*- coding: utf-8 -*-
"""人类多评议者分项 Kappa 重算。

输出：项目根目录 结果.md（更新第 1 节）。

口径：
- 每分项在每批数据内独立计算多评议者 Fleiss' kappa（允许各案例有效评议人数不同；有效人数<2 的案例跳过）。
- 单选分项 {P, E1, E2, A, O}：类别 = 所选码；单格出现多个码按缺失剔除（计数记录）。
- 多选分项 {C, I}：拆子码，逐子码做 0/1 Fleiss kappa（仅该子码被选过的案例），分项值 = 宏平均；
  无定义的子码剔除并记录。
- 格值统一用码模式提取，兼容 |、/、、 与无分隔连写；提不出码 = 缺失。
"""
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "结果.md"

SHEET_MAP = [
    ("问题", "P"), ("涉及人员", "E1"), ("严重程度", "E2"),
    ("原因", "C"), ("干预", "I"), ("介入", "I"),
    ("接受", "A"), ("结果", "O"),
]
SINGLE = {"P", "E1", "E2", "A", "O"}
MULTI = {"C", "I"}
CODE_RE = re.compile(r"[A-Z]\d*\.\d+")

FILES = [
    ("第一批", ROOT / "2-第一批案例.xlsx"),
    ("第二批", ROOT / "3-第二批案例.xlsx"),
]


def landis_koch(k):
    if k is None:
        return "—"
    for hi, name in [(0.20, "轻微"), (0.40, "一般"), (0.60, "中等"),
                     (0.80, "实质性"), (1.01, "几乎完全")]:
        if k <= hi:
            return name
    return "几乎完全"


def fmt(k):
    return f"{k:.3f}" if k is not None else "—"


def fleiss_kappa(counts):
    """counts: list per case of dict {label: n}，允许 n_i 不同。返回 (kappa, 有效案例数)。"""
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


def sheet_kind(name):
    for key, kind in SHEET_MAP:
        if key in name:
            return kind
    return None


def load(path):
    """返回 {kind: {'cases': [{rater: [codes]}], 'n_raters', 'stats'}}"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        kind = sheet_kind(ws.title)
        if kind is None:
            continue
        rows = list(ws.iter_rows(values_only=True))
        header = rows[0]
        label_col = header[0] is None
        raters = [str(h).strip() for h in (header[1:] if label_col else header) if h]
        stats = {"drop_multi_single": 0, "no_code": 0, "cells": 0, "missing": 0,
                 "excluded_rows": 0}
        cases = []
        for r in rows[1:]:
            vals = (r[1:] if label_col else r)[:len(raters)]
            cell_map = {}
            row_stats = {"cells": 0, "missing": 0, "no_code": 0, "drop_multi_single": 0}
            for name, v in zip(raters, vals):
                row_stats["cells"] += 1
                if v is None or (isinstance(v, str) and not v.strip()):
                    row_stats["missing"] += 1
                    continue
                codes = CODE_RE.findall(str(v))
                if not codes:
                    row_stats["no_code"] += 1
                    continue
                if kind in SINGLE:
                    if len(codes) > 1:
                        row_stats["drop_multi_single"] += 1
                        continue
                    cell_map[name] = [codes[0]]
                else:
                    cell_map[name] = sorted(set(codes))
            if cell_map:
                cases.append(cell_map)
                for k in stats:
                    if k in row_stats:
                        stats[k] += row_stats[k]
            else:
                stats["excluded_rows"] += 1
        out[kind] = {"cases": cases, "n_raters": len(raters), "stats": stats}
    wb.close()
    return out


def kappa_for_kind(kind, data):
    cases = data["cases"]
    if kind in SINGLE:
        counts = []
        for cm in cases:
            c = {}
            for codes in cm.values():
                c[codes[0]] = c.get(codes[0], 0) + 1
            counts.append(c)
        k, n_used = fleiss_kappa(counts)
        return {"kappa": k, "n_cases": len(cases), "n_used": n_used, "per_code": None}
    # 多选：逐子码 0/1
    subcodes = sorted({code for cm in cases for codes in cm.values() for code in codes})
    per_code, undef = [], []
    for sc in subcodes:
        involved = [cm for cm in cases if any(sc in codes for codes in cm.values())]
        if not involved:
            undef.append(f"{sc}（未出现）")
            continue
        counts_sub = []
        for cm in involved:
            sel = sum(1 for codes in cm.values() if sc in codes)
            n = len(cm)
            counts_sub.append({1: sel, 0: n - sel})
        k, n_used = fleiss_kappa(counts_sub)
        if k is None:
            undef.append(sc)
            continue
        per_code.append({"code": sc, "kappa": k, "n_sel": len(involved)})
    ks = [p["kappa"] for p in per_code]
    macro = sum(ks) / len(ks) if ks else None
    robust = [p["kappa"] for p in per_code if p["n_sel"] >= 3]
    macro_ge3 = sum(robust) / len(robust) if robust else None
    return {"kappa": macro, "kappa_ge3": macro_ge3, "n_codes_ge3": len(robust),
            "n_cases": len(cases), "n_used": None,
            "per_code": per_code, "undef": undef}


def main():
    results = {}
    for batch, path in FILES:
        loaded = load(path)
        results[batch] = {}
        for kind, d in loaded.items():
            r = kappa_for_kind(kind, d)
            r["stats"] = d["stats"]
            r["n_raters"] = d["n_raters"]
            results[batch][kind] = r

    L = []
    a = L.append
    a("# 结果")
    a("")
    a("> 项目所有分析结果的默认落点——新结果一律先写进本文件。")
    a("")
    a("## 1. 人类分项 Kappa 重算")
    a("")
    a("**与旧报告的区别**：旧 `drp_eval/DRP人机一致性评测报告.md` 中的 κ 全部是 AI 相关指标（AI 3 次运行间的 Fleiss κ、AI-金标准的 Cohen κ），**从未计算过人类评议者之间的一致性**。本节为人类基线。")
    a("")
    a("**口径**：")
    a("")
    a("- 每分项在每批内独立计算多评议者 **Fleiss' κ**，允许各案例有效评议人数不同（有效人数 < 2 的案例跳过）；")
    a("- 单选分项 P/E1/E2/A/O：类别 = 所选码；单格出现多个码按缺失剔除（计数见 §1.4）；")
    a("- 多选分项 C/I：拆子码后逐子码做 0/1 Fleiss κ（仅计该子码被至少 1 人选中的案例），分项值 = **宏平均**；无定义子码剔除（§1.3）；")
    a("- 格值统一按码模式提取，兼容 `|`、`/`、`、` 与无分隔连写；提不出码的格按缺失处理；")
    a("- 判读区间：Landis & Koch (1977)。")
    a("")
    for idx, batch in enumerate(results, 1):
        nr = next(iter(results[batch].values()))["n_raters"]
        a(f"### 1.{idx} {batch}（{nr} 名评议者）")
        a("")
        a("| 分项 | 案例数 | 有效案例数 | Fleiss κ | 判读 | 缺失格 | 多值剔除 | 无效文本格 |")
        a("|---|---|---|---|---|---|---|---|")
        for kind in ["P", "E1", "E2", "C", "I", "A", "O"]:
            r = results[batch].get(kind)
            if not r:
                continue
            st = r["stats"]
            n_used = str(r["n_used"]) if r["n_used"] is not None else "逐码"
            a(f"| {kind} | {r['n_cases']} | {n_used} | {fmt(r['kappa'])} "
              f"| {landis_koch(r['kappa'])} | {st['missing']} | {st['drop_multi_single']} "
              f"| {st['no_code']} |")
        a("")
    # 多选子码明细
    a("### 1.3 多选分项子码明细（宏平均的构成）")
    a("")
    for batch in results:
        for kind in ["C", "I"]:
            r = results[batch].get(kind)
            if not r or not r.get("per_code"):
                continue
            a(f"**{batch} · {kind}**（分项宏平均 κ = {fmt(r['kappa'])}，{r['n_cases']} 案例）")
            a("")
            a("| 子码 | 被选案例数 | κ |")
            a("|---|---|---|")
            for p in r["per_code"]:
                a(f"| {p['code']} | {p['n_sel']} | {fmt(p['kappa'])} |")
            if r.get("undef"):
                a("")
                a(f"无定义已剔除：{'、'.join(r['undef'])}")
            a("")
            a(f"- 全部子码宏平均 κ = **{fmt(r['kappa'])}**（主表值）；"
              f"被选 ≥ 3 案例的子码（{r['n_codes_ge3']} 个）宏平均 κ = **{fmt(r['kappa_ge3'])}**（稳健参考）")
            a("")
    # 附注
    a("### 1.4 数据处理附注")
    a("")
    a("- **有效案例数**：第一批各分项统一为 20 例、第二批 15 例。原始 sheet 行数不齐（P/C/I/A 尾部有空行与编辑残留，已整行剔除，见剔除行数），此前「案例数不齐」的印象来自这些空行；")
    for batch in results:
        exc = "、".join(f"{k}:{results[batch][k]['stats']['excluded_rows']}"
                        for k in ["P", "C", "I", "A"]
                        if k in results[batch] and results[batch][k]["stats"]["excluded_rows"])
        if exc:
            a(f"  - {batch}剔除的空/无效行：{exc}")
    a("- **第一批「接受程度A」sheet 结构特殊**：首行无空标签列（首格即评议者名）、仅 67 名评议者；含非编码文本行（如「总体协议a,b」），整行或整格按无效剔除；")
    a("- 各分项 sheet 命名差异已兼容（干预I/介入I、接受程度A/接受度A）；")
    a("- 单选分项中的零星多值（第二批 P/E1/E2/O 出现 `P2.1/P1.1` 形式）按缺失剔除，计数见主表「多值剔除」列；")
    a("- **低频子码警告**：仅在 1-2 个案例被选中的子码，κ 由极少变异决定、噪声很大（明细中大量 ≈0 或负值属正常统计现象）；判断多选分项一致性请以「被选 ≥ 3 案例子码的宏平均」为准（§1.3 各表末行）。")
    a("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)
    # 控制台摘要
    for batch in results:
        print(f"== {batch}")
        for kind in ["P", "E1", "E2", "C", "I", "A", "O"]:
            r = results[batch].get(kind)
            if r:
                print(f"  {kind}: kappa={fmt(r['kappa'])} cases={r['n_cases']}")


if __name__ == "__main__":
    main()
