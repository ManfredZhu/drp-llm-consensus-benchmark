# -*- coding: utf-8 -*-
"""形态退化率全批盘点：两把尺子并列（跨通道可复现的唯一依据）。

为什么需要两把尺子
------------------
SAP §10.2-#19 的**文字**把退化定义为「裸 JSON ＝ 中文占比 < 5%」（单判据），
而 Qwen（Qoder CN CLI）与 K2.8（workbuddy CLI）两条通道在落盘时把判据写进了
`shape` 字段：**长度 < 300 字符 或 中文占比 < 5%**（双判据，比文字更严）。

同一份 SAP 里两把尺子量同一个东西，会导致阶段 4 复核时对不上数。本脚本
把 9 个模型 × 全部臂 × 两把尺子一次算清，供 SAP 登记时并列披露。

用法
----
    python3 drp_eval/shape_audit.py            # 打印两尺对照表
    python3 drp_eval/shape_audit.py --json     # 输出 JSON，便于引用

口径说明
--------
· 统计单位＝每臂；**剔除 `−结构化输出` 臂**（该臂按设计不强制格式，
  解析失败是预期读数，SAP §10.2-#19 明确规定不计入退化率）。
· 尺子 A（#19 文字）＝ raw_output 的中文占比 < 5%。
· 尺子 B（落盘实现）＝ 长度 < 300 字符 **或** 中文占比 < 5%；
  与 `shape.degraded` 字段逐条一致时会在输出中标注，不一致则显式报警
  （防止脚本日后改动导致历史判据漂移）。
· LongCat 已排除，其数据在 `runs/_aux/excluded_longcat_20261004/`，
  本脚本默认不纳入；加 --longcat 可一并盘点（仅供留痕核对）。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "drp_eval", "runs")

# 模型 → 名单归属（写死在此，避免调用方自己猜；变更须同步改 SAP §10.1）
MODELS = {
    "deepseek-v4.1-flash": "必选",
    "glm-5.3-flash": "必选",
    "space-bunny": "必选",
    "minimax-m3": "必选",
    "mimo-v2.6-flash": "必选",
    "hy4-preview": "必选",
    "qwen3.8-flash": "必选",           # v1.9 由补充名单升入
    "ling-3.1-flash": "补充名单",
    "kimi-k2.8-preview": "补充名单",   # v1.10 登记
}
EXCLUDED = {"longcat-2.5-preview": "已排除（S-污染判定形态退化）"}
MIN_CHARS = 300      # 尺子 B 的长度门槛
MIN_CN_PCT = 5.0     # 尺子 A/B 共用的中文占比门槛（%）

ARMS = [("A 手册组", "A__{m}.jsonl"),
        ("B 裸考组", "B__{m}.jsonl"),
        ("−完整性检查", "ablation__{m}__minus_check.jsonl"),
        ("−引导追问", "ablation__{m}__minus_followup.jsonl"),
        ("−内置范例", "ablation__{m}__minus_examples.jsonl")]
# `−结构化输出` 按 #19 规定剔除，不计入退化率分母


def cn_pct(s):
    """raw_output 的中文占比（百分数，保留一位小数）。"""
    if not s:
        return 0.0
    n = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return round(100.0 * n / len(s), 1)


def audit(path):
    """返回该臂的 (n, 尺子A 命中数, 尺子B 命中数, 与 shape 字段不一致条数)。"""
    n = a = b = mismatch = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out = r.get("raw_output") or ""
            n += 1
            hit_a = cn_pct(out) < MIN_CN_PCT
            hit_b = hit_a or len(out) < MIN_CHARS
            a += hit_a
            b += hit_b
            sh = r.get("shape") or {}
            if isinstance(sh, dict) and "degraded" in sh:
                if bool(sh["degraded"]) != hit_b:
                    mismatch += 1
    return n, a, b, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--longcat", action="store_true", help="一并盘点已排除的 LongCat（仅供留痕核对）")
    args = ap.parse_args()

    models = dict(MODELS)
    if args.longcat:
        models.update(EXCLUDED)

    out = {}
    for m, belong in models.items():
        rows = []
        for arm, pat in ARMS:
            p = os.path.join(RUNS, pat.format(m=m))
            if not os.path.exists(p):
                continue
            n, a, b, mm = audit(p)
            if n:
                rows.append({"arm": arm, "n": n, "尺子A": f"{a}/{n}（{100*a/n:.1f}%）",
                             "尺子B": f"{b}/{n}（{100*b/n:.1f}%）", "shape字段不一致": mm})
        if rows:
            out[m] = {"名单": belong, "arms": rows}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("形态退化率全批盘点 · 两把尺子并列")
    print(f"尺子A（SAP #19 文字）＝ 中文占比 < {MIN_CN_PCT}%")
    print(f"尺子B（落盘实现）＝ 长度 < {MIN_CHARS} 字符 或 中文占比 < {MIN_CN_PCT}%")
    print(f"统计单位＝每臂；已剔除 `−结构化输出` 臂（SAP §10.2-#19）\n")
    worst_a = worst_b = 0.0
    for m, d in out.items():
        print(f"【{m}】{d['名单']}")
        for r in d["arms"]:
            flag = ""
            pa = float(r["尺子A"].split("（")[1].rstrip("%）"))
            pb = float(r["尺子B"].split("（")[1].rstrip("%）"))
            if pa >= 50 or pb >= 50:
                flag = "  ← 停机线"
            elif pa >= 20 or pb >= 20:
                flag = "  ← 披露线"
            if r["shape字段不一致"]:
                flag += f"  ⚠️ shape 字段与尺子B 不一致 {r['shape字段不一致']} 条"
            print(f"    {r['arm']:10s} n={r['n']:3d}   A {r['尺子A']:>16s}   B {r['尺子B']:>16s}{flag}")
            worst_a, worst_b = max(worst_a, pa), max(worst_b, pb)
        print()
    print(f"全批最高：尺子A {worst_a:.1f}% ／ 尺子B {worst_b:.1f}%")
    print("档位：停机线 ≥50% ｜ 披露线 20–50% ｜ 通过线 <20%")


if __name__ == "__main__":
    main()
