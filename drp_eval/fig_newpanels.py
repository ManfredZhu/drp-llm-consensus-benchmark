# -*- coding: utf-8 -*-
"""Fig.6 与 Fig.7 的绘图函数（追加进 make_pub_figures.py 的模块）

**为什么单独成文件再被导入**（AGENTS.md 原则 2「模块化」）：
Fig.6／7 依赖 `derived_figs.json`（由 `derive_fig_data.py` 产出），
与既有 5 图的数据源不同；且体量较大。独立成模块可避免主脚本膨胀，
两个模块都只读 json、不在绘图时做统计。

用法（由 make_pub_figures.py 导入后走统一的 save_pub）：
    fig6()  # 错误从哪来：错因 × 逐格 × 严格性
    fig7()  # 错误不是随机的：共现结构与硬格
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "drp_eval" / "derived_figs.json"
STAGE4 = ROOT / "drp_eval" / "results_stage4.json"

# ---- 色板（与主脚本同源：语义色恒定，身份色走chan_color）----
NEU_L = "#D3D1C7"
NEU_M = "#B4B2A9"
NEU_D = "#5F5E5A"
INK = "#2C2C2A"
RED = "#A32D2D"
BLUE = "#185FA5"
BLUE_L = "#85B7EB"
TEAL = "#0F6E56"
# 🔴 mm → 英寸 的换算（2026-10-07 修·曾致页面尺寸放大 65 倍）：
#   `figsize` 接受**英寸**，图宽以毫米给（双栏 180 mm）。
#   **毫米 → 英寸 = ÷ 25.4**（1 inch = 25.4 mm）。
#   ⚠️ 本轮踩的坑：先写成 `MM = 72/25.4`（pt/mm，与主脚本一致）再 `180 * MM`
#   → 得到 180 × 2.835 = **510 英寸**（= 12954 mm），页面放大 65 倍，
#   连带 TIFF 渲染报`FzErrorLimit: Overly large image`、PNG writer 必被 kill。
#   **教训：跨模块复用常量前必须确认它的量纲**——同名`MM` 在两处含义不同。
MM_IN = 1/ 25.4         # mm → 英寸
COLW_MM = 180.0         # 双栏宽（毫米）
COLW = COLW_MM * MM_IN          # = 7.087 英寸
COLH_MM = {"fig6": 132.0, "fig7": 138.0}

DIMS7 = ["P", "E1", "E2", "C", "I", "A", "O"]


def tiny(name, n=11):
    """极短标签（热图/矩阵图用）——9×9 网格的轴标签必须够短，
    否则旋转 90° 后纵向拉出 30+ pt，会与面板外的横向元素相撞。"""
    return name if len(name) <= n else name[:n - 1] + "…"
# 错因 = **语义**（两个不同模型可能同错因）→ 用语义色，不占身份色
KIND_COLOR = {"over_report": RED, "under_report": BLUE, "wrong_code": NEU_M}
KIND_LABEL = {"over_report": "over-report\n(extra codes)",
              "under_report": "under-report\n(missing codes)",
              "wrong_code": "wrong code"}


def load_derived():
    return json.loads(DERIVED.read_text(encoding="utf-8"))


def load_stage4():
    return json.loads(STAGE4.read_text(encoding="utf-8"))["models"]


def fig6(D, M, save_pub, panel_label, clean_axes, chan_color, short):
    """Fig.6 · 错误从哪来：错因 × 逐格 × 严格性

    **这一张是主终点的解释性分析**，不是新主终点。
    四面板各自承担不同推断角色：
      a 错误类型学（9 模型 × 维度）——81% 是多报
      b 单选维100% 多报 / 漏报全在多选格
      c 逐格难度谱（几模型 5 遍全对）
      d 严格集合相等 vs 包含口径的差（口径敏感性）
    """
    D = load_derived()
    order = sorted(M.items(), key=lambda kv: -kv[1]["A"]["consensus"]["rate"])
    # 2026-11-02 用户裁定：上/下两行间距（a–c，b–d 同行共用）再缩短 6 mm。
    #   累计：24.67 → 18.66 → 12.66 mm；轴高 121.3 → 129.8 pt；
    #   hspace = 35.90 / 129.85 ≈ 0.2765。
    fig = plt.figure(figsize=(COLW, COLH_MM["fig6"] * MM_IN))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.2765, wspace=0.30,
                          left=0.115, right=0.985, top=0.905, bottom=0.115)

    # ---------- a · 错误类型学（堆叠横条）----------
    axA = ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    ek = D["err_kind"]
    for kind in ("over_report", "under_report", "wrong_code"):
        vals = np.array([ek.get(k, {}).get(kind, 0) for k, _ in order], float)
        if vals.sum() == 0:
            continue
        ax.barh(ys, vals, left=left, height=0.62, color=KIND_COLOR[kind],
                lw=0, label=KIND_LABEL[kind], zorder=3)
        for yi, (v, l0) in zip(ys, zip(vals, left)):
            if v >= 40:                     # 只标够宽的段，避免小数字压叠
                ax.text(l0 + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=5, color="white", zorder=5)
        left += vals
    ax.set_yticks(ys)
    # 2026-11-02 用户裁定：模型名**完整显示**（不再 tiny 截断）；
    #   颜色仍走 chan_color —— 与 Fig.3a 完全同源的身份色。
    _ylabels = ax.set_yticklabels([M[k]["pretty"] for k, _ in order],
                                  fontsize=5)
    for _tick, (k, _) in zip(_ylabels, order):
        _tick.set_color(chan_color(k))
    ax.set_xlabel("Wrong answers on consensus cells (count)")
    ax.set_xlim(0, max(left) * 1.02)
    ax.set_ylim(-0.6, 9.2)
    tot_ov = D["err_summary"]["by_kind"].get("over_report", 0)
    tot_un = D["err_summary"]["by_kind"].get("under_report", 0)
    # 2026-11-02 用户裁定：标题合并为一行，移入面板内、距 Kimi 横柱 1 mm。
    _axh_pt = ax.get_position().height * fig.get_size_inches()[1] * 72
    _kimi_top = ys[0] + 0.31
    _note_y = ((_kimi_top - ax.get_ylim()[0]) /
               (ax.get_ylim()[1] - ax.get_ylim()[0]) + 2.835 / _axh_pt)
    note_a = ax.text(
           0.0, _note_y,
           f"81% of errors are over-reporting, not omission "
           f"(over {tot_ov} · under {tot_un} of {D['err_summary']['total_errors']})",
           transform=ax.transAxes, ha="left", va="bottom",
           fontsize=5.0, color=NEU_D)
    ax.tick_params(axis="y", length=0)
    # 2026-11-02 用户裁定：两个图例移到面板内右上部空白区，各两行、左对齐。
    ax.legend(loc="upper left", bbox_to_anchor=(103, 7.9),
              bbox_transform=ax.transData, ncol=1,
              handlelength=1.0, handletextpad=0.4, fontsize=5,
              labelspacing=0.5, borderpad=0, frameon=False)
    clean_axes(ax)

    # ---------- b · 错因 × 维度类型 ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    by_dim = D["err_summary"]["by_dim"]
    SINGLE = ["P", "E1", "E2", "A", "O"]
    MULTI_D = ["C", "I"]
    groups = [("single-choice\nP / E1 / E2 / A / O", SINGLE),
              ("multi-choice\nC / I", MULTI_D)]
    for gi, (glab, dims) in enumerate(groups):
        agg = {"over_report": 0, "under_report": 0, "wrong_code": 0}
        for dm in dims:
            for kind, v in by_dim.get(dm, {}).items():
                agg[kind] = agg.get(kind, 0) + v
        tot = sum(agg.values()) or 1
        left = 0.0
        for kind in ("over_report", "under_report", "wrong_code"):
            v = agg[kind]
            if v == 0:
                continue
            ax.barh(gi, 100 * v / tot, left=left, height=0.5,
                    color=KIND_COLOR[kind], lw=0, zorder=3)
            if 100 * v / tot >= 8:
                ax.text(left + 50 * v / tot, gi, f"{100*v/tot:.0f}%", ha="center",
                        va="center", fontsize=5, color="white", zorder=5)
            left += 100 * v / tot
    ax.set_yticks([0, 1])
    ax.set_yticklabels([g[0] for g in groups], fontsize=5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of wrong answers (%)")
    #🔴 关键数字：单选维零漏报（漏报数学上不可能——单选答错即多给或答非所问）
    ax.set_ylim(-0.55, 1.75)
    # 2026-11-02 用户裁定：b 顶部两行文字顶边对齐 a 的 81% 行。
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _nb = note_a.get_window_extent(_r)
    _bb = ax.get_window_extent(_r)
    _b_top = (_nb.y1 - _bb.y0) / _bb.height
    ax.text(0.5, _b_top, "under-reporting occurs\nonly in multi-choice cells",
            transform=ax.transAxes, ha="center", va="top", fontsize=5,
            color=BLUE, fontweight="bold", linespacing=1.3)
    clean_axes(ax)

    # ---------- c · 逐格难度谱 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    dist = {int(k): v for k, v in D["cell_perfect_distribution"].items()}
    n_cells = D["n_cells"]
    xs = np.arange(len(dist))
    # 语义色：越少模型全对越红（表达「难」，不是身份）
    cols = [BLUE if k >= 8 else (NEU_L if k >= 4 else RED) for k in sorted(dist)]
    ax.bar(xs, [dist[k] for k in sorted(dist)], width=0.72, color=cols, lw=0, zorder=3)
    for xi, k in enumerate(sorted(dist)):
        n = dist[k]
        if n >= 15:
            ax.text(xi, n + 1.2, str(n), ha="center", va="bottom", fontsize=5,
                    color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in sorted(dist)], fontsize=5)
    ax.set_xlabel("Models with all 5 repeats correct (out of 9)")
    ax.set_ylabel("Consensus cells (count)")
    ax.set_ylim(0, max(dist.values()) * 1.16)
    _c_txt = (f"most cells are easy for almost every model\n"
              f"but {dist.get(0,0)} cells defeat all nine")
    _c_title = ax.set_title(_c_txt, fontsize=5.0, color=NEU_D, pad=5,
                            loc="left", linespacing=1.35)
    # 2026-11-02 用户裁定：两行文字整体下移 8 mm（8 mm = 8/25.4 in）。
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _tb = _c_title.get_window_extent(_r)
    _bb = ax.get_window_extent(_r)
    _c_top = (_tb.y1 - 8 * fig.dpi / 25.4 - _bb.y0) / _bb.height
    ax.set_title("", loc="left")
    ax.text(0.0, _c_top, _c_txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.0, color=NEU_D, linespacing=1.35)
    # 2026-11-02 用户裁定：显示 c 面板纵轴线。
    clean_axes(ax, keep_left=True)

    # ---------- d · 严格性口径的代价 ----------
    axD = ax = fig.add_subplot(gs[1, 1])
    # 🔴 主终点的两个读数（严格集合相等 vs 包含口径）。
    #   数值取自 results_stage4 的 A 组全体逐格读数，**不在此处重算**——
    #   主读数只有一个来源（AGENTS.md 原则 2：不在绘图函数里做统计）。
    #   包含口径仅对多选维放宽，Δ 的分子 = err_summary.loose_gain_records。
    loose = D["err_summary"]["loose_gain_records"]
    bars = [("strict set equality\n(main reading)", 82.5, INK),
            ("gold code included\n(sensitivity)", 85.7, NEU_M)]
    # 🔴 数值标签画在**柱内右端**（va/ha 反过来），不放在柱外（2026-10-07 修 FAIL）：
    #   柱外侧 +0.4 处正好落在annotate 引线的箭杆上（PDF 里一条 18x23pt 的
    #   灰色 stroked path）→ 判 text-stroke。柱内白字既避开线又更易读。
    for i, (lab, v, c) in enumerate(bars):
        ax.barh(i, v, height=0.46, color=c, lw=0, zorder=3)
        ax.text(v - 0.4, i, f"{v:.1f}%", va="center", ha="right", fontsize=5,
                color="white", fontweight="bold", zorder=6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([b[0] for b in bars], fontsize=5)
    ax.set_xlim(78, 90)
    ax.set_xlabel("Accuracy on consensus cells (%)")
    ax.annotate(f"+3.2 pp = {loose} records answered correctly\nbut carried extra codes",
                xy=(85.7, 1.0), xytext=(87.4, 1.35), fontsize=5, color=NEU_D,
                ha="center", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=NEU_M, lw=0.7))
    ax.set_ylim(-0.6, 1.75)
    clean_axes(ax)

    # 2026-11-02 用户裁定：a 标识左缘对齐 "DeepSeek-V4.1-Flash"，
    #   b 标识左缘对齐 "P / E1 / E2 / A / O"；c/d 分别沿用 a/b 的绝对 x。
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    def _tick_x(_ax, _needle):
        return next(t.get_window_extent(_r).x0 for t in _ax.get_yticklabels()
                    if _needle in t.get_text())
    _x_ds = _tick_x(axA, "DeepSeek-V4.1-Flash")
    _x_pe = _tick_x(axB, "P / E1 / E2 / A / O")
    def _dx_to(_ax, _target):
        return (_target - _ax.get_window_extent(_r).x0) / fig.dpi * 72
    # 2026-11-02 用户裁定：a/b/c/d 四个标识各下移 1 mm（2.835 pt），
    #   面板位置不动；默认 dy_pt=4.0 → 1.165。
    _DY_F6 = 4.0 - 72 / 25.4
    panel_label(axA, "a", dx_pt=_dx_to(axA, _x_ds), dy_pt=_DY_F6)
    panel_label(axB, "b", dx_pt=_dx_to(axB, _x_pe), dy_pt=_DY_F6)
    panel_label(axC, "c", dx_pt=_dx_to(axC, _x_ds), dy_pt=_DY_F6)
    panel_label(axD, "d", dx_pt=_dx_to(axD, _x_pe), dy_pt=_DY_F6)

    # 🔴 panel_ids 显式声明（对齐闸门的必要入参）：不声明时闸门靠
    #   infer 自动分组，会把 cax（色标inset_axes）当独立面板而误判列宽不等。
    save_pub(fig, "Fig6_error_composition", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exemptions=[{"panels": ["a", "b", "c", "d"],
                          "checks": ["panel-label"],
                          "reason": "a/b 对齐本面板最长 y 轴文字，c/d 分别沿用 a/b 的标签列"}])
    return n_cells


def fig7(D, M, save_pub, panel_label, clean_axes, chan_color, short):
    """Fig.7 · 错误不是随机的：共现结构与硬格

    **独立主张**：失败是系统性的，不是随机噪声——
    模型之间共享同一批「系统性困难格」，换模型无效。
    四面板：
      a 98 格按「被几个模型答错」分组
      b 两两错误 Jaccard 矩阵 vs 独立期望
      c 人类共识度 vs AI 错误数（相关性）
      d 全错硬格清单（人类高共识、AI 全错）
    """
    D = load_derived()
    n = len(D["models"])
    order = sorted(M.items(), key=lambda kv: -kv[1]["A"]["consensus"]["rate"])
    # 2026-11-02 用户裁定：上/下两行（a–c、b–d 同行共用）缝缩短 10 mm。
    #   原 gap 30.39 mm → 20.39 mm；行高比 1:1.05 不变，
    #   平均行高 113.2 → 124.6 pt → hspace = 57.81 / 124.63 ≈ 0.464。
    fig = plt.figure(figsize=(COLW, COLH_MM["fig7"] * MM_IN))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.464, wspace=0.30,
                          left=0.105, right=0.985, top=0.90, bottom=0.115)

    # ---------- a · 每格被几个模型答错 ----------
    axA = ax = fig.add_subplot(gs[0, 0])
    # 语义色梯度：错得越多的模型数越红
    dist = {}
    for v in D["cell_detail"].values():
        k = n - v["n_models_perfect"]# n_models_wrong
        dist[k] = dist.get(k, 0) + 1
    xs = np.arange(n + 1)
    cols = [NEU_L if k <= 1 else (BLUE_L if k <= 3 else
                                  (NEU_M if k <= 6 else RED)) for k in xs]
    ys_ = [dist.get(k, 0) for k in xs]
    ax.bar(xs, ys_, width=0.74, color=cols, lw=0, zorder=3)
    for xi, v in zip(xs, ys_):
        if v >= 8:
            ax.text(xi, v + 0.7, str(v), ha="center", va="bottom", fontsize=5, color=INK)
    n_hard = dist.get(n, 0)
    ax.set_xticks(xs)
    ax.set_xlabel("Models answering this cell wrongly (out of 9)")
    ax.set_ylabel("Consensus cells (count)")
    ax.set_ylim(0, max(ys_) * 1.2)
    # 2026-11-02 用户裁定：面板说明移入图注，面板内只留 a/b/c/d 标识；显示 a 纵轴线。
    clean_axes(ax, keep_left=True)

    # ---------- b · 两两错误 Jaccard ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    keys = [k for k, _ in order]
    Z = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                Z[i, j] = 1.0
            else:
                v = D["jaccard"].get(f"{a}|{b}", D["jaccard"].get(f"{b}|{a}", 0.0))
                Z[i, j] = v
    # 🔴 `aspect="auto"`（2026-10-07 修对齐闸门 FAIL「同列面板宽度不等」）：
    #   `imshow` 默认 `aspect="equal"` → 9×9 的**正方形**矩阵会把坐标区
    #   在高度方向撑满、宽度方向压窄，实测 b 面板宽仅 **107.8 pt**，
    #   而同列的 a 面板为 195.2 pt（窄 45%）→ 面板左右边界与同列不一致。
    #   该图的矩阵是 9×9 的**对称相关矩阵**，不是需要等比例的图像，
    #   故 `aspect="auto"` 正确：格子按面板宽高比铺开，可读性不受影响。
    im = ax.imshow(Z, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    # 2026-11-02 用户裁定：x 标签改 45°（从左下到右上），名称完整；
    #   x/y 两轴模型名均用 chan_color（与 Fig.3 同一身份色）。
    # 45° 单行时长名（DeepSeek-V4.1-Flash 等）会互相压字；提到 65° 以压缩
    # 水平投影，保留完整模型名与「左下→右上」方向（2026-11-03 碰撞审计 FAIL→0）。
    _xl = ax.set_xticklabels([M[k]["pretty"] for k in keys], fontsize=4.2,
                             rotation=65, ha="right", rotation_mode="anchor")
    for _t, k in zip(_xl, keys):
        _t.set_color(chan_color(k))
    ax.set_yticks(range(n))
    _yl = ax.set_yticklabels([M[k]["pretty"] for k in keys], fontsize=4.4)
    for _t, k in zip(_yl, keys):
        _t.set_color(chan_color(k))
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center",
                        fontsize=4.2, color="white" if Z[i, j] > 0.55 else INK)
    # 🔴 色标改用**文字图例**，不画 inset 色条（2026-10-07 修 FAIL×9→0）：
    #   原inset_axes 色条（含其上下边框线）在 PDF 里留下两条横贯 108pt 的
    #   stroked path，恰好穿过 b 面板**旋转 90° 的 9 个 y 轴模型名**
    #   （旋转后文字 bbox 宽仅 4pt、高 36pt，极易被任何横线命中）。
    #   下移色条位置只能减少、不能根除——**文字图例不产生任何几何体**，
    #   彻底断掉这条碰撞路径。色标含义用一句话说清。
    #   ⚠️ y偏移须小于 −0.20（−0.20 时文字落在 y 标签的纵向范围内 → text-text 碰撞）；
    #   取 −0.34 让它落到 x 轴刻度标签（90°旋转、高约 30pt）之下。
    # 2026-11-02 用户裁定：色标说明从轴下搬到 b 标识下方顶部，保持居中、不左移。
    ax.text(0.5, 1.02,
            "shared red scale: 0 (disjoint errors) → 1 (identical error sets)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=4.4, color=NEU_D)
    ax.tick_params(length=0)

    # ---------- c · 人类共识度 vs AI 错误 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    hv = D["human_vs_ai"]
    xs_ = [r["gold_share"] * 100 for r in hv["rows"]]
    ys2 = [r["n_models_wrong"] for r in hv["rows"]]
    ax.plot(xs_, ys2, "o", ms=2.6, color=BLUE, ls="none",
            markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    # 趋势线（最小二乘）
    if len(xs_) > 2:
        mx, my = np.mean(xs_), np.mean(ys2)
        b = np.cov(xs_, ys2)[0, 1] / np.var(xs_) if np.var(xs_) else 0
        a0 = my - b * mx
        xr = np.linspace(min(xs_), max(xs_), 20)
        ax.plot(xr, a0 + b * xr, color=NEU_D, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel("Human agreement on the gold code (%)")
    ax.set_ylabel("Models answering wrongly (out of 9)")
    ax.set_ylim(-0.5, n + 0.5)
    # 2026-11-02 用户裁定：显示 c 面板纵轴线。
    clean_axes(ax, keep_left=True)

    # ---------- d · 全错硬格 ----------
    axD = ax = fig.add_subplot(gs[1, 1])
    hard = D["hard_cell_detail"]
    ys3 = np.arange(len(hard))[::-1]
    labs = [f"T{h['case']}·{h['dim']}" for h in hard]
    shares = [h["gold_share"] * 100 for h in hard]
    # 语义色：按维度分组（语义，非身份）
    DIMC = {"C": RED, "I": "#D85A30", "P": BLUE, "E1": TEAL, "E2": NEU_M}
    ax.barh(ys3, shares, height=0.6, color=[DIMC.get(h["dim"], NEU_M) for h in hard],
            lw=0, zorder=3)
    for yi, (h, s_) in zip(ys3, zip(hard, shares)):
        ax.text(s_ + 1.5, yi, f"{s_:.0f}%", va="center", ha="left", fontsize=4.6,
                color=NEU_D)
    ax.set_yticks(ys3)
    ax.set_yticklabels(labs, fontsize=4.6)
    ax.set_xlabel("Human agreement on the gold code (%)")
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.7, len(hard) - 0.3)
    clean_axes(ax)

    # 2026-11-02 用户裁定：a/c 标识左缘分别对齐本面板 y 轴标题；
    #   b/d 标识左缘对齐 b 面板 "DeepSeek-V4.1-Flash" 的左端。
    fig.canvas.draw()
    _r7 = fig.canvas.get_renderer()
    def _x_of(artist):
        return artist.get_window_extent(_r7).x0
    _x_a = _x_of(axA.yaxis.label)
    _x_c = _x_of(axC.yaxis.label)
    _x_ds = next(_x_of(t) for t in axB.get_yticklabels()
                 if "DeepSeek-V4.1-Flash" in t.get_text())
    def _dx7(_ax, _target):
        return (_target - _ax.get_window_extent(_r7).x0) / fig.dpi * 72
    panel_label(axA, "a", dx_pt=_dx7(axA, _x_a))
    panel_label(axB, "b", dx_pt=_dx7(axB, _x_ds))
    panel_label(axC, "c", dx_pt=_dx7(axC, _x_c))
    panel_label(axD, "d", dx_pt=_dx7(axD, _x_ds))

    # 🔴 panel_ids 必传：否则闸门 infer 会按 axes 顺序猜分组，
    #   本图 a/c 与 b/d 的轴宽不同，会被误判列宽不等。
    #   （b 面板原先的 inset 色条已改为文字图例，故不再需要 exclude_axes。）
    save_pub(fig, "Fig7_error_structure", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exemptions=[{"panels": ["a", "b", "c", "d"],
                          "checks": ["panel-label"],
                          "reason": "a/c 对齐各自 y 轴标题，b/d 对齐 DeepSeek 左端（用户 2026-11-02）"}])

# ============================================================
# Fig.8 · PCNE 迁移臂（外部码表 V9.1）
# ============================================================
def fig8(P, M, save_pub, panel_label, clean_axes, chan_color, short):
    """Fig.8 · 换用外部码表（PCNE V9.1）后的五终点

    **本臂无金标准、不与主实验并池**（SAP §3.5），全部描述性。
    d 面板按用户裁定（2026-10-07）改为「V1.0 与 PCNE 的 P 段码表对照」，
    **合规声明移入图注**（须写致函日期）。
    """
    order = sorted(M.items(), key=lambda kv: -kv[1]["A"]["consensus"]["rate"])
    pm = P["per_model"]
    keys = [k for k, _ in order]
    fig = plt.figure(figsize=(COLW, 134 * MM_IN))
    # 2026-11-02 用户裁定：上下行（a–c、b–d）缝增加 1 mm。
    #   13.95 → 14.95 mm；行高比 1:0.86 不变，
    #   平均行高 127.0 pt → hspace = 42.38 / 126.95 ≈ 0.334。
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.334, wspace=0.28,
                          left=0.10, right=0.985, top=0.90, bottom=0.12)

    # ---------- a · 五终点读数矩阵 ----------
    axA = ax = fig.add_subplot(gs[0, 0])
    # 列＝终点（语义：不同性质 → 语义色），行＝模型
    cols = [
        ("format\ncompliance", lambda v: v["format_ok"]),
        ("flow\ncompleted", lambda v: v["flow_ok"]),
        ("cross-model\nagreement", lambda v: round(v["eco_rate"] / 10)),
        ("P-segment\nstability", lambda v: round(v["p_main_match"] / 10)),
    ]
    Z = np.zeros((len(keys), len(cols)))
    for i, k in enumerate(keys):
        v = pm[k]
        for j, (_lab, fn) in enumerate(cols):
            Z[i, j] = fn(v)
    im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=4.6)
    ax.set_yticks(range(len(keys)))
    # 2026-11-02 用户裁定：模型名完整、身份色与 Fig.3 同源。
    _ylA = ax.set_yticklabels([M[k]["pretty"] for k in keys], fontsize=4.4)
    for _t, k in zip(_ylA, keys):
        _t.set_color(chan_color(k))
    for i in range(len(keys)):
        for j in range(len(cols)):
            full = 10 if cols[j][0].startswith(("format", "flow")) else None
            val = Z[i, j]
            # 格式/流程是 0–10 计数，其余已是百分制/10
            raw = (pm[keys[i]]["format_ok"] if j == 0 else
                   pm[keys[i]]["flow_ok"] if j == 1 else
                   pm[keys[i]]["eco_rate"] if j == 2 else
                   pm[keys[i]]["p_main_match"])
            ax.text(j, i, f"{raw:.0f}", ha="center", va="center", fontsize=4.4,
                    color="white" if val > 6 else INK)
    # 2026-11-02 用户裁定：面板说明取消，仅留 a/b/c/d 标识（图注再解释）。
    cax = ax.inset_axes([0.0, -0.16, 1.0, 0.028])
    cax.imshow(np.linspace(0, 10, 120).reshape(1, -1), cmap="Blues", aspect="auto")
    cax.set_yticks([]); cax.set_xticks([0, 119])
    cax.set_xticklabels(["0", "10"], fontsize=4.6)
    cax.xaxis.set_ticks_position("bottom")
    cax.tick_params(length=1.2, pad=1.0)
    for s_ in ("top", "right", "left"):
        cax.spines[s_].set_visible(False)
    ax.tick_params(length=0)

    # ---------- b · 生态一致性逐案例 ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    det = P["eco_detail"]
    cids = list(det.keys())
    shares = [det[c]["modal_share"] for c in cids]
    xs = np.arange(len(cids))
    cols_b = [BLUE if s >= 78 else (NEU_L if s >= 60 else RED) for s in shares]
    ax.bar(xs, shares, width=0.66, color=cols_b, lw=0, zorder=3)
    for xi, c in zip(xs, cids):
        ax.text(xi, det[c]["modal_share"] + 1.8,
                f"{det[c]['n_distinct']}", ha="center", va="bottom",
                fontsize=4.6, color=NEU_D)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"T{c}" for c in cids], fontsize=4.6)
    ax.set_xlabel("Case (10 cases in the PCNE arm)")
    ax.set_ylabel("Modal answer share across 9 models (%)")
    ax.set_ylim(0, 112)
    # 2026-11-02 用户裁定：显示 b 面板纵轴线。
    clean_axes(ax, keep_left=True)

    # ---------- c · P 段跨码表稳定性 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    vals = [pm[k]["p_main_match"] for k in keys]
    ys = np.arange(len(keys))[::-1]
    # 身份色：与 a/b/c 面板同序，便于纵向追踪
    ax.barh(ys, vals, height=0.6,
            color=[chan_color(k) for k in keys], lw=0, zorder=3)
    for yi, v, k in zip(ys, vals, keys):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", ha="left", fontsize=4.6,
                color=NEU_D)
    ax.set_yticks(ys)
    _ylC = ax.set_yticklabels([M[k]["pretty"] for k in keys], fontsize=4.4)
    for _t, k in zip(_ylC, keys):
        _t.set_color(chan_color(k))
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlabel("P-segment agreement with the main experiment (%)")
    clean_axes(ax)

    # ---------- d · V1.0 与 PCNE V9.1 的 P 段对照（🔴 逐码核实后重画）----------
    # 🔴🔴 本面板首版的四条描述**全错**，且错在合规敏感处（2026-10-07 逐字核实后修正）：
    #   首版写「P1 problem not treated」「P2 unnecessary or ineffective drug」
    #   「P3 administrative issue」——**三条都不是任何码表的官方定义**。
    #   其中给 P2 写的「unnecessary / ineffective drug」实为 **P3.2** 的义项，
    #   属R24「码号撞车比缺失更危险」的实例。
    #
    # ✅ 核实依据（本面板所有文字均从两处**逐字抄录**，不凭记忆）：
    #   ① V1.0 → `0-《中国药物相关问题分类系统(V1.0)》表单.docx` 表格行（提取 w:tr 文本）
    #   ② V9.1 → `drp_eval/prompts/frozen/pcne_v91.md` §2.1/ §2.2
    #
    # ✅ 核实结论（与首版的假设相反，务必如实呈现）：
    #   · **一级类目三对全部语义相同**（P1 治疗有效性 / P2 治疗安全性 / P3 其它）
    #   · **但三级码有实质差异**，且存在**一处真正的码号撞车**：
    #       V1.0 的 P3.1 =「有未治疗的适应症」→ 在 V9.1 里对应 **P1.3**
    #       V9.1 的 P3.1 =「治疗成本效益问题」→ V1.0 **无此义项**
    #     → 故 **P3.1 在两套体系里含义完全不同**，跨码表对照时必须显式声明。
    #   · 另：V1.0 的 P1.1 =「药物治疗无效/不佳」= V9.1 的 **P1.1 + P1.2 合并**
    #
    # 📊 撞车码的实际暴露情况（决定图上能说多强）：
    #   PCNE 臂 90 次输出中，**P3.1 出现 0 次** → 撞车**未实际发生**；
    #   而 V9.1 独有的细分码 P1.2(18) / P1.3(3) 被用了 21 次（V1.0 无对应细分）。
    #   → 故可说「码号结构兼容」且「撞车码零暴露」，**不可说「完全一一对应」**。
    axD = ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ROW, ROW_STEP = 0.900, 0.075      # 一级类目行
    SUB, SUB_STEP = 0.670, 0.072      # 三级码区
    HDR = 0.985                       # 表头（顶部不变）
    DIV = 0.270                       # 表下分隔线
    # ---- 表头两列 ----
    ax.text(0.25, HDR, "V1.0 (this study)", ha="center", va="top",
            fontsize=4.8, color=BLUE, fontweight="bold")
    ax.text(0.75, HDR, "PCNE V9.1", ha="center", va="top",
            fontsize=4.8, color=TEAL, fontweight="bold")
    # ---- 一级类目：三对全部语义相同 ----
    top = [("P1", "treatment efficacy", "P1", "treatment efficacy", True),
           ("P2", "drug safety / ADE", "P2", "drug safety / ADE", True),
           ("P3", "other", "P3", "other", True)]
    for k, (c1, e1, c2, e2, same) in enumerate(top):
        y = ROW - k * ROW_STEP
        ax.add_patch(Rectangle((0.01, y - 0.026), 0.48, 0.052,
                               facecolor=BLUE, alpha=0.16, edgecolor=BLUE,
                               lw=0.6, zorder=2))
        ax.add_patch(Rectangle((0.51, y - 0.026), 0.48, 0.052,
                               facecolor=TEAL, alpha=0.16, edgecolor=TEAL,
                               lw=0.6, zorder=2))
        ax.text(0.03, y, c1, ha="left", va="center", fontsize=4.6,
                color=BLUE, fontweight="bold", zorder=3)
        ax.text(0.10, y, e1, ha="left", va="center", fontsize=4.2,
                color=INK, zorder=3)
        ax.text(0.53, y, c2, ha="left", va="center", fontsize=4.6,
                color=TEAL, fontweight="bold", zorder=3)
        ax.text(0.60, y, e2, ha="left", va="center", fontsize=4.2,
                color=INK, zorder=3)
    # ---- 三级码：逐码对照，用颜色标出「同义/ 合并 / 撞车 / 仅一方有」 ----
    # （实测 90 次输出中 P3.1 零暴露，故撞车码标注为「本实验未触发」）
    sub = [
        ("P1.1", "no effect / suboptimal", "P1.1+2", "merged in V1.0", "merge"),
        ("P3.1", "untreated indication", "P1.3", "renumbered", "clash"),
        ("P2.1", "possible adverse event", "P2.1", "same", "same"),
        ("P3.2", "unnecessary drug use", "P3.2", "same", "same"),
        ("P3.3", "unclear, escape only", "P3.3", "same", "same"),
        ("—", "no counterpart", "P3.1", "cost-effectiveness", "clash2"),
    ]
    KIND_STYLE = {
        "same":   (NEU_L, "white"),
        "merge":  ("#FAC775", INK),
        "clash":  (RED, "white"),
        "clash2": ("#F09595", INK),
    }
    for k, (c1, e1, c2, e2, kind) in enumerate(sub):
        y = SUB - k * SUB_STEP
        fc, tc = KIND_STYLE[kind]
        ax.add_patch(Rectangle((0.01, y - 0.023), 0.48, 0.046,
                               facecolor=fc, alpha=0.30, edgecolor=fc,
                               lw=0.5, zorder=2))
        ax.add_patch(Rectangle((0.51, y - 0.023), 0.48, 0.046,
                               facecolor=fc, alpha=0.30, edgecolor=fc,
                               lw=0.5, zorder=2))
        ax.text(0.03, y, c1, ha="left", va="center", fontsize=4.2,
                color=INK, fontweight="bold", zorder=3)
        ax.text(0.10, y, e1, ha="left", va="center", fontsize=3.9,
                color=INK, zorder=3)
        ax.text(0.53, y, c2, ha="left", va="center", fontsize=4.2,
                color=INK, fontweight="bold", zorder=3)
        ax.text(0.60, y, e2, ha="left", va="center", fontsize=3.9,
                color=INK, zorder=3)
    # ---- 结论：一行一句；最后一行与 c 面板 x 轴标题 “P-segment …” 同高 ----
    ax.plot([0.01, 0.99], [DIV, DIV], color=NEU_M, lw=0.5, zorder=1)
    _lines = [
        ("Sub-level: P3.1 is renumbered.", INK),
        ("V1.0 = untreated indication · V9.1 = cost-effectiveness.", NEU_D),
        ("Collision not triggered: P3.1 = 0 / 90 outputs.", NEU_D),
        ("V9.1-only codes P1.2 / P1.3 used 21 times.", NEU_D),
        ("Codes compatible; top-level names identical; sub-levels not 1:1.", NEU_D),
    ]
    # 以 c 面板 xlabel 的垂直中心为目标，把最后一条结论对齐过去
    fig.canvas.draw()
    _r8c = fig.canvas.get_renderer()
    _clab = axC.xaxis.label.get_window_extent(_r8c)
    _dbb = ax.get_window_extent(_r8c)
    _target = ((_clab.y0 + _clab.y1) / 2 - _dbb.y0) / _dbb.height
    _TOP = DIV - 0.03
    _H = 0.040
    _STEP = (_TOP - (_target + _H / 2)) / (len(_lines) - 1)
    for _i, (_txt, _col) in enumerate(_lines):
        ax.text(0.02, _TOP - _i * _STEP, _txt, ha="left", va="top",
                fontsize=4.0, color=_col, clip_on=False)
        # 2026-11-02 用户裁定：d 面板标题取消，仅留标识（图注再解释）。
    # 2026-11-02 用户裁定（参照 Fig.7）：a/c 标识对齐各自
    #   "DeepSeek-V4.1-Flash" 的左端；b/d 标识对齐 b 面板 y 轴标题左缘。
    fig.canvas.draw()
    _r8 = fig.canvas.get_renderer()
    def _xof8(_art):
        return _art.get_window_extent(_r8).x0
    def _deepseek_x(_ax):
        return next(_xof8(t) for t in _ax.get_yticklabels()
                    if "DeepSeek-V4.1-Flash" in t.get_text())
    _x_dsA, _x_dsC = _deepseek_x(axA), _deepseek_x(axC)
    _x_byl = _xof8(axB.yaxis.label)
    def _dx8(_ax, _target):
        return (_target - _ax.get_window_extent(_r8).x0) / fig.dpi * 72
    panel_label(axA, "a", dx_pt=_dx8(axA, _x_dsA))
    panel_label(axB, "b", dx_pt=_dx8(axB, _x_byl))
    panel_label(axC, "c", dx_pt=_dx8(axC, _x_dsC))
    panel_label(axD, "d", dx_pt=_dx8(axD, _x_byl))

    save_pub(fig, "Fig8_pcne_transfer", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exclude_axes=[cax],
             exemptions=[{"panels": ["a", "b", "c", "d"],
                          "checks": ["panel-label"],
                          "reason": "a/c 对齐各自 DeepSeek 名称，b/d 对齐 b 的 y 轴标题（用户 2026-11-02）"}])


# ============================================================
# Fig.1 · 总scheme
# ============================================================
def fig1(M, D, P, save_pub, panel_label, clean_axes, chan_color, short, tiny_):
    """Fig.1 · 总 scheme：把整篇论文的结构前置。

    a 全流程（139 药师 → 逐格投票 → 98/7 切分 → 9 模型双条件 → 两把尺子）
    b 规模矩阵（9 模型 × 5 遍 × 15 例，两条件）
    c 三条核心发现的视觉索引
    d 九张正文图的导航
    """
    order = sorted(M.items(), key=lambda kv: -kv[1]["A"]["consensus"]["rate"])
    keys = [k for k, _ in order]
    fig = plt.figure(figsize=(COLW, 150 * MM_IN))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.50, wspace=0.24,
                          left=0.045, right=0.985, top=0.925, bottom=0.085)

    # ---------- a · 全流程 ----------
    axA = ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    # 🔴 改为**两行错落布局**（2026-10-07 修·首版 5 个框横排挤成重叠）：
    #   框内文字在 5pt 下单个框需约 19 个 x 单位才放得下"9 models × 5 repeats"
    #   这类长标题，5×19 = 95 已占满整幅 → 必须分 3 + 2 两行、上下错落。
    # 🔴 两行错落布局 + **框内只放标题**（2026-10-07 修·两次仍重叠）：
    #   首版把副标题也塞进框里 → 5pt 下需要约 19 个 x 单位才放得下
    #   "majority becomes the reference"，5 框横排必然重叠。
    #   现改为：框内**一行标题**，副标题一律移到框**下方**成图注，
    #   框宽收到 17–19 单位，五框横排/错落都不挤。
    steps = [
        ("two rating rounds", "139 & 46 raters, blind", BLUE),
        ("per-cell voting", "votes vs designated answers", BLUE),
        ("98 / 7 split", "consensus / disputed", BLUE),
        ("9 models × 5 repeats", "15 cases × 2 arms", TEAL),
        ("two criteria", "98 + 7 cells scored", NEU_D),
    ]
    rows_y = [(72.0, 26.0), (32.0, 26.0)]
    xs_plan = [[1.0, 21.5, 42.0], [8.0, 34.0]]
    box_w = [18.0, 18.0, 18.0, 21.0, 20.0]
    idx = 0
    for ri, (y0, hh) in enumerate(rows_y):
        for xi in xs_plan[ri]:
            t1, t2, c = steps[idx]
            w = box_w[idx]
            # 🔴 alpha 0.10 → 0.22（2026-10-07 修 FAIL×3）：
            #   审计器把「文字 bbox 落在半透明 fill 内」判为 text-stroke，
            #   除非该 fill 被归入 contained_fill_overlays 白名单。
            #   提高不透明度使其落入「同色底上的文字＝正常」判定。
            # 🔴 **边框 lw=0.8 → 0**（2026-10-07 修 FAIL 的决定性一步）：
            #   审计器判据是「文字 bbox 被某条 **stroke 线段**穿过」，
            #   而框的四条边正是 stroke —— 只要文字贴近框边（5pt 高、框高 26 单位），
            #   必然命中。改为**无边框纯底色块**：视觉上仍是分块，
            #   但不再产生任何 stroked path → 从根上断掉这条碰撞路径。
            ax.add_patch(Rectangle((xi, y0), w, hh, facecolor=c, alpha=0.22,
                                   edgecolor="none", lw=0, zorder=2))
            ax.text(xi + w / 2, y0 + hh / 2, t1, ha="center", va="center",
                    fontsize=5.0, color=c, fontweight="bold", zorder=3)
            ax.text(xi + w / 2, y0 - 3.0, t2, ha="center", va="top",
                    fontsize=4.1, color=NEU_D, zorder=3)
            idx += 1
    # 🔴 箭头改走**框外的空隙**（2026-10-07 修 FAIL×3）：
    #   首版箭头终点落在下一框的边框上（框宽 18，箭头指到 21.0/ 41.5，
    #   而框 2 起于 21.5 / 框 3 起于 42.0）→ 描边穿过框内文字。
    #   现改为：横向箭头**只画在两框之间的 2.5 单位空隙里**
    #   （起点 = 前框右缘 −0.3，终点 = 后框左缘 +0.3），
    #   纵向转折画在 x = 51.0（框 3 右缘 60 之外，无文字）。
    # 🔴 **不画流程箭头**（2026-10-07 修 FAIL×1 的最终解法）：
    #   箭头是5pt 高字号下唯一会稳定命中「短标题 + 副标题」双行文字的元素——
    #   横向箭头必然落在框 1/框 4 的标题行，纵向转折必然落在某框的中心线
    #   （实测与「consensus / disputed」副标题 PDF x 完全重合）。
    #   三次挪位（框外空隙 → 提高 alpha → 移出中心）都只把冲突转移到别的元素，
    #   **符合 R44「此类改动须换策略而非继续微调」的教训**。
    #   替代表达：流程顺序由**框的左右位置**交代（上排 1→2→3，折回下排 4→5），
    #   图注再补一句顺序说明；无 stroked path → 从根上断掉这条碰撞路径。
    ax.text(50, 99, "Study design (steps 1-3 top row, 4-5 bottom row): "
                    "from divergent human raters to a two-criterion benchmark",
            ha="center", va="center", fontsize=5.2, color=INK, fontweight="bold")
    ax.text(50, 20, "Raters never saw each other's answers or the reference; cells "
                    "whose votes did not support the designated answers form the "
                    "contested set",
            ha="center", va="center", fontsize=4.4, color=NEU_D, style="italic")
    panel_label(ax, "a", dx=-0.03, dy=1.02)

    # ---------- b · 规模矩阵 ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    nrows = len(keys)
    grid = np.zeros((nrows, 30))       # 15 例 × 2 条件 = 30 格
    for i in range(nrows):
        grid[i, :] = 5                   # 每格 = 5 遍
    # 🔴 首版整片纯蓝、看不出「格」的结构（2026-10-07 修）：
    #   加浅色格线把 30 个格子显式分开，并让每 5 格（=1 例）交替深浅，
    #   读者才能看出「15 例 × 5 遍」这两个维度。
    ax.imshow(grid, cmap=Blues_cmap(), vmin=0, vmax=5.4, aspect="auto")
    # 逐格白描边（不用 ax.grid：网格会越出色块边界去撞 y 轴标签）
    for i in range(nrows + 1):
        ax.plot([-0.5, 29.5], [i - 0.5, i - 0.5], color="white", lw=0.35, zorder=4)
    for j in range(31):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.35, zorder=4)
    # 每 5 格（= 1 个案例）用一条更亮的分隔线
    for j in range(0, 31, 5):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.9, zorder=5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([tiny_(M[k]["pretty"], 13) for k in keys], fontsize=4.6)
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(chan_color(k))
    ax.set_xticks([7, 22])
    ax.set_xticklabels(["arm A\n(manual + workflow + 20 worked cases)",
                        "arm B\n(codebook only)"], fontsize=4.6)
    ax.axvline(14.5, color="white", lw=1.4)   # A/B 两臂的分界
    ax.tick_params(which="both", length=0, pad=1.5)
    ax.set_xlabel("each cell = 5 repeats × 1 case", fontsize=4.8)
    ax.set_title("9 models × 15 cases × 5 repeats × 2 conditions = 1,350 sessions",
                 fontsize=5.0, color=NEU_D, pad=5, loc="left")
    panel_label(ax, "b", dx=-0.20)

    # ---------- c · 三条核心发现 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    finds = [
        ("1", "Accuracy has a ceiling",
         "69–89% on the 98 consensus cells;\n8 of 9 models inside an 8 pp band", BLUE),
        ("2", "Difficulty is stratified, not random",
         "A / O easy (89–100%), C and I hardest\n(41–64% / 51–83%)", NEU_D),
        ("3", "Structured output is the decisive constraint",
         "remove it and 9/9 models × 15 runs = 0/135\nproduced no parseable answer", RED),
    ]
    y = 92
    for num, t1, t2, c in finds:
        ax.add_patch(Rectangle((2, y - 15), 96, 15, facecolor=c, alpha=0.18,
                               edgecolor=c, lw=0.8, zorder=2))
        ax.text(7, y - 7.5, num, ha="center", va="center", fontsize=8,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 5.5, t1, ha="left", va="center", fontsize=5.4,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 11.5, t2, ha="left", va="center", fontsize=4.5,
                color=INK, zorder=3, linespacing=1.3)
        y -= 19
    ax.text(50, 97, "Three findings", ha="center", va="center", fontsize=5.6,
            color=INK, fontweight="bold")
    panel_label(ax, "c", dx=-0.03, dy=1.02)

    # ---------- d · 图导航 ----------
    axD = ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    figs = [
        ("Fig.2", "the reference is itself contested"),
        ("Fig.3", "nine models, 69–89%"),
        ("Fig.4", "output format is the decisive constraint"),
        ("Fig.5", "handouts also raise model agreement"),
        ("Fig.6", "where the wrong answers come from"),
        ("Fig.7", "errors are shared, not random"),
        ("Fig.8", "the pattern survives a codebook swap"),
        ("Fig.9", "does the manual actually help?"),
    ]
    y = 95
    for fno, cap in figs:
        ax.text(4, y, fno, ha="left", va="center", fontsize=5.2,
                color=BLUE, fontweight="bold")
        ax.text(20, y, cap, ha="left", va="center", fontsize=4.9, color=INK)
        y -= 10.5
    ax.text(50, 100, "Figures in this paper (Fig.1 is this scheme)",
            ha="center", va="center", fontsize=5.6, color=INK, fontweight="bold")
    panel_label(ax, "d", dx=-0.03, dy=1.02)

    save_pub(fig, "Fig1_overview_scheme", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exclude_axes=[cba := axB.inset_axes([0, 0, 0, 0])] if False else [])


def Blues_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pcne_blues", ["#E6F1FB", "#185FA5"])    # 🔴 **不画箭头**（2026-10-07 修 FAIL×2 的最终解法）：
    #   箭头是 5.0pt 高字号下唯一会稳定命中「短标题 + 副标题」双行文字的元素——
    #   横向箭头必然落在框 1/框 4 的标题行（实测 PDF x 命中 72.3–76.1 与 46.9–49.4），
    #   竖向转折必然落在某框的中心线上（实测与「consensus / disputed」完全重合）。
    #   三次挪位（框外空隙 → 提高 alpha → 移出中心）都只把冲突转移到别的元素，
    #   **符合 R44「批量改名类改动须换策略而非继续微调」的教训**。
    #   替代表达：给每个框**编号**（1→5），流程顺序由编号与图注交代，
    #   既干净又不会与任何文字碰撞。
    # 流程顺序改由**框本身的左右位置**交代（上排 1→2→3，折回下排 4→5），
    # 不再叠加编号数字——白字序号同样会与框内标题争夺那 2pt 的垂直余量。
    ax.text(50, 99, "Study design (steps 1-3 top row, 4-5 bottom row): "
                    "from divergent human raters to a two-criterion benchmark",
            ha="center", va="center", fontsize=5.2, color=INK, fontweight="bold")
    ax.text(50, 20, "Raters never saw each other's answers or the reference; cells "
                    "whose votes did not support the designated answers form the "
                    "contested set",
            ha="center", va="center", fontsize=4.4, color=NEU_D, style="italic")
    panel_label(ax, "a", dx=-0.03, dy=1.02)

    # ---------- b · 规模矩阵 ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    nrows = len(keys)
    grid = np.zeros((nrows, 30))       # 15 例 × 2 条件 = 30 格
    for i in range(nrows):
        grid[i, :] = 5                   # 每格 = 5 遍
    # 🔴 首版整片纯蓝、看不出「格」的结构（2026-10-07 修）：
    #   加浅色格线把 30 个格子显式分开，并让每 5 格（=1 例）交替深浅，
    #   读者才能看出「15 例 × 5 遍」这两个维度。
    ax.imshow(grid, cmap=Blues_cmap(), vmin=0, vmax=5.4, aspect="auto")
    # 逐格白描边（不用 ax.grid：网格会越出色块边界去撞 y 轴标签）
    for i in range(nrows + 1):
        ax.plot([-0.5, 29.5], [i - 0.5, i - 0.5], color="white", lw=0.35, zorder=4)
    for j in range(31):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.35, zorder=4)
    # 每 5 格（= 1 个案例）用一条更亮的分隔线
    for j in range(0, 31, 5):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.9, zorder=5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([tiny_(M[k]["pretty"], 13) for k in keys], fontsize=4.6)
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(chan_color(k))
    ax.set_xticks([7, 22])
    ax.set_xticklabels(["arm A\n(manual + workflow + 20 worked cases)",
                        "arm B\n(codebook only)"], fontsize=4.6)
    ax.axvline(14.5, color="white", lw=1.4)   # A/B 两臂的分界
    ax.tick_params(which="both", length=0, pad=1.5)
    ax.set_xlabel("each cell = 5 repeats × 1 case", fontsize=4.8)
    ax.set_title("9 models × 15 cases × 5 repeats × 2 conditions = 1,350 sessions",
                 fontsize=5.0, color=NEU_D, pad=5, loc="left")
    panel_label(ax, "b", dx=-0.20)

    # ---------- c · 三条核心发现 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    finds = [
        ("1", "Accuracy has a ceiling",
         "69–89% on the 98 consensus cells;\n8 of 9 models inside an 8 pp band", BLUE),
        ("2", "Difficulty is stratified, not random",
         "A / O easy (89–100%), C and I hardest\n(41–64% / 51–83%)", NEU_D),
        ("3", "Structured output is the decisive constraint",
         "remove it and 9/9 models × 15 runs = 0/135\nproduced no parseable answer", RED),
    ]
    y = 92
    for num, t1, t2, c in finds:
        ax.add_patch(Rectangle((2, y - 15), 96, 15, facecolor=c, alpha=0.18,
                               edgecolor=c, lw=0.8, zorder=2))
        ax.text(7, y - 7.5, num, ha="center", va="center", fontsize=8,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 5.5, t1, ha="left", va="center", fontsize=5.4,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 11.5, t2, ha="left", va="center", fontsize=4.5,
                color=INK, zorder=3, linespacing=1.3)
        y -= 19
    ax.text(50, 97, "Three findings", ha="center", va="center", fontsize=5.6,
            color=INK, fontweight="bold")
    panel_label(ax, "c", dx=-0.03, dy=1.02)

    # ---------- d · 图导航 ----------
    axD = ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    figs = [
        ("Fig.2", "the reference is itself contested"),
        ("Fig.3", "nine models, 69–89%"),
        ("Fig.4", "output format is the decisive constraint"),
        ("Fig.5", "handouts also raise model agreement"),
        ("Fig.6", "where the wrong answers come from"),
        ("Fig.7", "errors are shared, not random"),
        ("Fig.8", "the pattern survives a codebook swap"),
        ("Fig.9", "does the manual actually help?"),
    ]
    y = 95
    for fno, cap in figs:
        ax.text(4, y, fno, ha="left", va="center", fontsize=5.2,
                color=BLUE, fontweight="bold")
        ax.text(20, y, cap, ha="left", va="center", fontsize=4.9, color=INK)
        y -= 10.5
    ax.text(50, 100, "Figures in this paper (Fig.1 is this scheme)",
            ha="center", va="center", fontsize=5.6, color=INK, fontweight="bold")
    panel_label(ax, "d", dx=-0.03, dy=1.02)

    save_pub(fig, "Fig1_overview_scheme", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exclude_axes=[cba := axB.inset_axes([0, 0, 0, 0])] if False else [])


def Blues_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pcne_blues", ["#E6F1FB", "#185FA5"])    # 🔴 **不画箭头**（2026-10-07 修 FAIL×2 的最终解法）：
    #   箭头是 5.0pt 高字号下唯一会稳定命中「短标题 + 副标题」双行文字的元素——
    #   横向箭头必然落在框 1/框 4 的标题行（实测 PDF x 命中 72.3–76.1 与 46.9–49.4），
    #   竖向转折必然落在某框的中心线上（实测与「consensus / disputed」完全重合）。
    #   三次挪位（框外空隙 → 提高 alpha → 移出中心）都只把冲突转移到别的元素，
    #   **符合 R44「批量改名类改动须换策略而非继续微调」的教训**。
    #   替代表达：给每个框**编号**（1→5），流程顺序由编号与图注交代，
    #   既干净又不会与任何文字碰撞。
    for i2, xi in enumerate([1.0, 21.5, 42.0, 8.0, 34.0]):
        ri = 0 if i2 < 3 else 1
        y0, hh = rows_y[ri]
        w = box_w[i2]
        ax.text(xi + 2.0, y0 + hh - 2.2, str(i2 + 1), ha="left", va="top",
                fontsize=4.4, color="white", fontweight="bold", zorder=4)
    ax.text(50, 99, "Study design (steps 1-3 top row, 4-5 bottom row): "
                    "from divergent human raters to a two-criterion benchmark",
            ha="center", va="center", fontsize=5.2, color=INK, fontweight="bold")
    ax.text(50, 20, "Raters never saw each other's answers or the reference; cells "
                    "whose votes did not support the designated answers form the "
                    "contested set",
            ha="center", va="center", fontsize=4.4, color=NEU_D, style="italic")
    panel_label(ax, "a", dx=-0.03, dy=1.02)

    # ---------- b · 规模矩阵 ----------
    axB = ax = fig.add_subplot(gs[0, 1])
    nrows = len(keys)
    grid = np.zeros((nrows, 30))       # 15 例 × 2 条件 = 30 格
    for i in range(nrows):
        grid[i, :] = 5                   # 每格 = 5 遍
    # 🔴 首版整片纯蓝、看不出「格」的结构（2026-10-07 修）：
    #   加浅色格线把 30 个格子显式分开，并让每 5 格（=1 例）交替深浅，
    #   读者才能看出「15 例 × 5 遍」这两个维度。
    ax.imshow(grid, cmap=Blues_cmap(), vmin=0, vmax=5.4, aspect="auto")
    # 逐格白描边（不用 ax.grid：网格会越出色块边界去撞 y 轴标签）
    for i in range(nrows + 1):
        ax.plot([-0.5, 29.5], [i - 0.5, i - 0.5], color="white", lw=0.35, zorder=4)
    for j in range(31):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.35, zorder=4)
    # 每 5 格（= 1 个案例）用一条更亮的分隔线
    for j in range(0, 31, 5):
        ax.plot([j - 0.5, j - 0.5], [-0.5, nrows - 0.5], color="white", lw=0.9, zorder=5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([tiny_(M[k]["pretty"], 13) for k in keys], fontsize=4.6)
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(chan_color(k))
    ax.set_xticks([7, 22])
    ax.set_xticklabels(["arm A\n(manual + workflow + 20 worked cases)",
                        "arm B\n(codebook only)"], fontsize=4.6)
    ax.axvline(14.5, color="white", lw=1.4)   # A/B 两臂的分界
    ax.tick_params(which="both", length=0, pad=1.5)
    ax.set_xlabel("each cell = 5 repeats × 1 case", fontsize=4.8)
    ax.set_title("9 models × 15 cases × 5 repeats × 2 conditions = 1,350 sessions",
                 fontsize=5.0, color=NEU_D, pad=5, loc="left")
    panel_label(ax, "b", dx=-0.20)

    # ---------- c · 三条核心发现 ----------
    axC = ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    finds = [
        ("1", "Accuracy has a ceiling",
         "69–89% on the 98 consensus cells;\n8 of 9 models inside an 8 pp band", BLUE),
        ("2", "Difficulty is stratified, not random",
         "A / O easy (89–100%), C and I hardest\n(41–64% / 51–83%)", NEU_D),
        ("3", "Structured output is the decisive constraint",
         "remove it and 9/9 models × 15 runs = 0/135\nproduced no parseable answer", RED),
    ]
    y = 92
    for num, t1, t2, c in finds:
        ax.add_patch(Rectangle((2, y - 15), 96, 15, facecolor=c, alpha=0.18,
                               edgecolor=c, lw=0.8, zorder=2))
        ax.text(7, y - 7.5, num, ha="center", va="center", fontsize=8,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 5.5, t1, ha="left", va="center", fontsize=5.4,
                color=c, fontweight="bold", zorder=3)
        ax.text(14, y - 11.5, t2, ha="left", va="center", fontsize=4.5,
                color=INK, zorder=3, linespacing=1.3)
        y -= 19
    ax.text(50, 97, "Three findings", ha="center", va="center", fontsize=5.6,
            color=INK, fontweight="bold")
    panel_label(ax, "c", dx=-0.03, dy=1.02)

    # ---------- d · 图导航 ----------
    axD = ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    figs = [
        ("Fig.2", "the reference is itself contested"),
        ("Fig.3", "nine models, 69–89%"),
        ("Fig.4", "output format is the decisive constraint"),
        ("Fig.5", "handouts also raise model agreement"),
        ("Fig.6", "where the wrong answers come from"),
        ("Fig.7", "errors are shared, not random"),
        ("Fig.8", "the pattern survives a codebook swap"),
        ("Fig.9", "does the manual actually help?"),
    ]
    y = 95
    for fno, cap in figs:
        ax.text(4, y, fno, ha="left", va="center", fontsize=5.2,
                color=BLUE, fontweight="bold")
        ax.text(20, y, cap, ha="left", va="center", fontsize=4.9, color=INK)
        y -= 10.5
    ax.text(50, 100, "Figures in this paper (Fig.1 is this scheme)",
            ha="center", va="center", fontsize=5.6, color=INK, fontweight="bold")
    panel_label(ax, "d", dx=-0.03, dy=1.02)

    save_pub(fig, "Fig1_overview_scheme", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c", axD: "d"},
             exclude_axes=[cba := axB.inset_axes([0, 0, 0, 0])] if False else [])


def Blues_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pcne_blues", ["#E6F1FB", "#185FA5"])
