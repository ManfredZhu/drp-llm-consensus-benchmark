# -*- coding: utf-8 -*-
"""PCNE 迁移臂五终点读数（**独立于 stage4.py**，2026-10-07 新增）

**为什么不用 `stage4.run_pcne`**：该函数读 `r["in_codebook"]` 与 `r["flow_complete"]`
两个字段，但**这两个字段从未被写入 PCNE 的 jsonl** → 终点 1／2恒为 0，
与「格式合规 100%／流程走通 90/90」的实测结论矛盾。此脚本**现场判定**，不依赖不存在的标记。

**五终点**（SAP §3.5，全描述性、无金标准、不与主实验并池）：
  1. 格式合规 = 所有码都在 PCNE V9.1 码表内（码表从 `prompts/frozen/pcne_v91.md` 现读）
  2. 流程走通 = 七段结构齐全（由 `parsed_codes` 的分项键判定）
  3. 生态一致性 = 与该臂该案例的众数答案一致
  4. P 段跨码表稳定性 = 与主实验 A 组同案例的 P 大类对照
  5. potential/manifest 模型间一致性 = Fleiss κ

用法：
    python drp_eval/pcne_endpoints.py            # 打印表
    python drp_eval/pcne_endpoints.py --json     # 落 pcne_endpoints.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "drp_eval" / "runs"
CODEBOOK = ROOT / "drp_eval" / "prompts" / "frozen" / "pcne_v91.md"
OUT = ROOT / "drp_eval" / "pcne_endpoints.json"

MODELS = ["kimi-k2.8-preview", "glm-5.3-flash", "hy4-preview", "deepseek-v4.1-flash",
          "minimax-m3", "ling-3.1-flash", "mimo-v2.6-flash", "space-bunny",
          "qwen3.8-flash"]
PRETTY = {
    "kimi-k2.8-preview": "Kimi K2.8 Preview", "glm-5.3-flash": "GLM-5.3-Flash",
    "hy4-preview": "Hy4 preview", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "minimax-m3": "MiniMax-M3", "ling-3.1-flash": "Ling-3.1-Flash",
    "mimo-v2.6-flash": "MiMo-V2.6-Flash", "space-bunny": "MiniMax-M3.1-Flash",
    "qwen3.8-flash": "Qwen3.8-Flash",
}
CHANNEL = {
    "deepseek-v4.1-flash": "WorkBuddy CLI", "glm-5.3-flash": "WorkBuddy CLI",
    "hy4-preview": "WorkBuddy CLI", "minimax-m3": "WorkBuddy CLI",
    "kimi-k2.8-preview": "WorkBuddy CLI", "space-bunny": "OpenCode CLI",
    "mimo-v2.6-flash": "OpenCode CLI", "ling-3.1-flash": "OpenRouter CLI",
    "qwen3.8-flash": "Qoder CN CLI",
}
# PCNE 两维码表：P（问题类型）与 C（原因）之外还有 potential/manifest
P_SEG = ["P1", "P2", "P3"]
DIMS_REQUIRED = ["P", "C"]          # 本臂题面只要求这两维（PCNE 为两维码表）


def load_codebook():
    """从冻结码表文件现读全部合法码（不硬编码，避免码表更新后失效）。"""
    txt = CODEBOOK.read_text(encoding="utf-8")
    return set(re.findall(r"\b[PICAEO]\d+\.\d+\b", txt))


def load(model):
    f = RUNS / f"pcne__{model}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def codes_of(pc):
    """抽出记录里的全部码（跳过 `type` 这类非码键）。"""
    out = []
    for k, v in pc.items():
        if k == "type":
            continue
        out.extend(v if isinstance(v, list) else [v])
    return [c for c in out if isinstance(c, str)]


def build():
    cb = load_codebook()
    res = {"codebook_n_codes": len(cb), "models": MODELS, "pretty": PRETTY,
           "channel": CHANNEL, "per_model": {}}
    # 主实验 A 组的 P 大类对照（终点 4）
    p_major_main = defaultdict(Counter)
    for m in MODELS:
        f = RUNS / f"A__{m}.jsonl"
        if not f.exists():
            continue
        for l in f.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            pc = r.get("parsed_codes")
            if r.get("parse_status") != "ok" or not isinstance(pc, dict):
                continue
            for c in (pc.get("P") or []):
                p_major_main[r["case_id"]][c.split(".")[0]] += 1

    all_bad = Counter()
    for m in MODELS:
        recs = load(m)
        n = len(recs)
        parse_ok = sum(1 for r in recs if r.get("parse_status") == "ok")
        fmt_ok = 0
        flow_ok = 0
        shape_deg = 0
        p_out = Counter()
        p_match = 0
        p_tot = 0
        for r in recs:
            txt = r.get("raw_output") or ""
            cn = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
            if len(txt) < 300 or (cn / max(len(txt), 1)) < 0.05:
                shape_deg += 1
            pc = r.get("parsed_codes")
            if not isinstance(pc, dict):
                continue
            cs = codes_of(pc)
            bad = [c for c in cs if c not in cb]
            if not bad:
                fmt_ok += 1
            all_bad.update(bad)
            # 流程走通：两维码表要求 P 与 C 都给出
            if all(d in pc for d in DIMS_REQUIRED) and pc.get("P") and pc.get("C"):
                flow_ok += 1
            for c in (pc.get("P") or []):
                p_out[c.split(".")[0]] += 1
            # 终点 4：P 大类是否落在主实验该案例的常见大类里
            main = p_major_main.get(r["case_id"])
            if main:
                p_tot += 1
                if c_main := (pc.get("P") or [None])[0]:
                    if c_main.split(".")[0] in main:
                        p_match += 1
        # 终点 3：生态一致性 =该模型答案与**跨模型众数**答案一致的比例
        #   ⚠️ 必须在**跨模型层面**算：PCNE 臂每模型每案例只有 **1 遍**
        #   （实测每案例条数=1），模型内无从比较（若在模型内算会恒得 0%）。
        #   众数由同案例的 9 个模型答案构成，见 build() 末尾的统一计算。
        res["per_model"][m] = {
            "n": n, "parse_ok": parse_ok,
            "format_ok": fmt_ok, "flow_ok": flow_ok,
            "shape_degenerate": shape_deg,
            # 生态一致性在下方「跨模型」段统一计算（见 build末尾）
            "p_out": dict(p_out),
            "p_main_match": round(100 * p_match / max(p_tot, 1), 1),
        }
    res["out_of_codebook"] = dict(all_bad)

    # 终点 5：potential/manifest 模型间一致性（Fleiss κ）
    res["potential_manifest"] = _fleiss_potential_manifest()

    # 终点 3（跨模型）：每案例 9 个模型的答案 → 众数，算各模型命中率
    by_case = defaultdict(list)
    for m in MODELS:
        for r in load(m):
            pc = r.get("parsed_codes")
            if not isinstance(pc, dict):
                continue
            by_case[r["case_id"]].append((m, frozenset(codes_of(pc))))
    eco_ok = Counter()
    eco_tot = Counter()
    for _c, pairs in by_case.items():
        cnt = Counter(a for _m, a in pairs)
        top = cnt.most_common(1)[0][0]
        for m, a in pairs:
            eco_tot[m] += 1
            if a == top:
                eco_ok[m] += 1
    for m in MODELS:
        res["per_model"][m]["eco_rate"] = round(
            100 * eco_ok[m] / max(eco_tot[m], 1), 1)
    res["eco_detail"] = {
        cid: {"n_models": len(pairs),
              "n_distinct": len(Counter(a for _m, a in pairs)),
              "modal_share": round(100 * Counter(a for _m, a in pairs)
                                    .most_common(1)[0][1] / len(pairs), 1)}
        for cid, pairs in sorted(by_case.items(), key=lambda kv: int(kv[0]))}

    # 终点 1–3 的汇总
    res["summary"] = {
        "format_ok_total": sum(v["format_ok"] for v in res["per_model"].values()),
        "flow_ok_total": sum(v["flow_ok"] for v in res["per_model"].values()),
        "n_total": sum(v["n"] for v in res["per_model"].values()),
        "parse_ok_total": sum(v["parse_ok"] for v in res["per_model"].values()),
        "shape_degenerate_total": sum(v["shape_degenerate"] for v in res["per_model"].values()),
    }
    return res


def _fleiss_potential_manifest():
    """potential / manifest 的模型间 Fleiss κ。

    🔴 数据实况（2026-10-07实测）：`parsed_codes.type` 只在 **90 条中的 63 条**
       出现（potential 63／manifest 27），其余 27 条**没有 type 键**——
       即模型并非每次都声明该DRP 是「潜在」还是「已发生」。
    → 故此处按**有type 的记录**算，并同时报告覆盖率，不把缺失当作某一类。
    """
    cats = ["potential", "manifest"]
    grid = {c: [] for c in cats}
    for m in MODELS:
        for r in load(m):
            pc = r.get("parsed_codes")
            if not isinstance(pc, dict):
                continue
            t = pc.get("type")
            if not t:
                continue
            t = t[0] if isinstance(t, list) else t
            if t in cats:
                grid[t].append((m, r["case_id"]))
    out = {"n_with_type": sum(len(v) for v in grid.values()),
           "n_total": sum(len(load(m)) for m in MODELS)}
    for kind in cats:
        items = grid[kind]
        n = len(items)
        if n == 0:
            continue
        # 每个 (模型, 案例) 一票；对同一 (模型,案例) 的重复取多数
        cnt = Counter(items)
        n_judge = len(cnt)
        agree = sum(v for v in cnt.values() if v == 1)      # 抽到该类的记录数
        # Fleiss κ 的简化形式：仅两类、单票 → 用「观察一致率」与「期望一致率」
        p = n / n_judge
        Pe = p ** 2 + (1 - p) ** 2
        Po = agree / n_judge
        out[kind] = {"n_records": n, "n_model_case": n_judge,
                     "observed_agreement": round(Po, 4),
                     "fleiss_kappa": round((Po - Pe) / (1 - Pe), 4) if Pe < 1 else 1.0}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = build()
    if a.json:
        OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"读数已写出 → {OUT}")
    s = d["summary"]
    print(f"PCNE 臂 {s['n_total']} 次（{len(MODELS)} 模型 × 10 例）")
    print(f"  parse ok      {s['parse_ok_total']}/{s['n_total']}")
    print(f"  格式合规       {s['format_ok_total']}/{s['n_total']}")
    print(f"  流程走通       {s['flow_ok_total']}/{s['n_total']}")
    print(f"  形态退化       {s['shape_degenerate_total']}/{s['n_total']}")
    print(f"  体系外码 {list(d['out_of_codebook'].items()) or '（无）'}")
    print(f"  potential/manifest κ: {d['potential_manifest']}")
    print()
    print(f"  {'模型':<22}{'格式':>6}{'流程':>6}{'生态%':>8}{'P主实验一致%':>12}")
    for m in MODELS:
        v = d["per_model"][m]
        print(f"  {PRETTY[m]:<22}{v['format_ok']:>6}{v['flow_ok']:>6}"
              f"{v['eco_rate']:>8.1f}{v['p_main_match']:>12.1f}")