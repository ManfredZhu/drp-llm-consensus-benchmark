# -*- coding: utf-8 -*-
"""Nature Communications 论文图生成（主驱动器）

**唯一数据源**：`drp_eval/results_stage4.json`
  —— 由 `drp_eval/stage4.py` 从冻结的 `score.py` 判定逻辑产出，本文件不含任何重算或特判。

**版式契约**（nature-figure skill · Python 后端）
  · 双栏 180 mm；正文 5–7 pt；无图内标题（标题归图注）
  · 面板标签 a/b/c 加粗小写置于左上角外侧
  · 色板：中性灰 + 蓝主色 + 红仅用于「错误／偏差」方向
  · 每张多面板图导出前跑 `require_matplotlib_panel_alignment()`（容差 1.5 pt）

用法：
    python drp_eval/make_pub_figures.py fig3       # 单图（主终点 hero）
    python drp_eval/make_pub_figures.py all
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Rectangle

# ⚠️ 刻意不使用 patheffects 描边底衬（withStroke）：
# 实测它会让 matplotlib 把所有文本转成矢量路径 → 交付 PDF 不含可提取文本 →
# Nature 的碰撞审计（audit_figure_collisions.py）读不到任何文本框、判定 NOT AUDITABLE。
# 改用「逐处避让」策略：把标注文字放到没有线条穿过的空白区。

ROOT = Path(__file__).resolve().parent.parent
# 🔴 定位 nature-figure skill 的对齐闸门模块（2026-10-07 跨平台可移植性改造）：
#   原为**硬编码 macOS 绝对路径** `/Users/zhumaohua/.workbuddy/skills/...`
#   → 换到Windows 上必然ImportError，整张图无法出。
#   现按**优先级依次探测**四个位置，跨平台均可解析：
#   ① 环境变量 PCNE_SKILL_SCRIPTS（用户显式指定时最可靠）
#   ② 本仓库内的 vendored 副本 `drp_eval/_vendor/`（迁移包会带，最稳）
#   ③ 从当前 WorkBuddy 用户的 skills 目录按相对层级推导
#   ④ 逐个候选探测第一个存在的
#   全部找不到时**降级为不设闸门**（如实打印警告，仍可出图，
#   不因缺一个审计工具而阻断交付——同 SAP §10.2-#36-c「闸门失败不阻断导出」的纪律）。
SKILL_CANDIDATES = [
    os.environ.get("PCNE_SKILL_SCRIPTS"),
    ROOT / "drp_eval" / "_vendor",
    Path.home() / ".workbuddy" / "skills" / "nature-figure" / "scripts",
    ROOT / "skills" / "nature-figure" / "scripts",
]
SKILL = next((p for p in SKILL_CANDIDATES if p and Path(p).is_dir()), None)
if SKILL is not None:
    sys.path.insert(0, str(SKILL))
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    require_matplotlib_panel_alignment = None
    print("[warn] 未找到 nature-figure 的 audit_panel_alignment 模块，"
          "已关闭面板对齐闸门（出图不受影响）。"
          "可用环境变量 PCNE_SKILL_SCRIPTS 指定其路径。")

FIGDIR = ROOT / "figures_nature"
DATA = ROOT / "drp_eval" / "results_stage4.json"

# ---------- 全局样式（Nature 契约） ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    # Arial优先保证拉丁字形合规；中文字体置于回退链首位，避免 CJK 缺字方块
    "font.sans-serif": ["Arial", "Helvetica", "PingFang SC", "Heiti SC",
                        "Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
    "svg.fonttype": "none",     # SVG 文字可编辑
    "pdf.fonttype": 42,         # PDF 内嵌 TrueType
    "font.size": 5.6,
    "axes.labelsize": 5.6,
    "xtick.labelsize": 5.0,
    "ytick.labelsize": 5.0,
    "legend.fontsize": 5.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 0.0,     # y 轴不留刻度线：避免与模型名文字相交
    "xtick.minor.size": 0.0,
    "ytick.major.pad": 5.0,      # 刻度标签离轴线的距离（审计器视轴线为 stroked path）
    "legend.frameon": False,
    # ⚠️ 数据点标记默认**不描白边**：PDF 导出时白边会变成独立的白色
    # fill+stroke 路径，若与相邻面板的文字重叠，碰撞审计判 text-stroke FAIL
    #（实测 a 面板 GLM/Hy4/DeepSeek 三点的白边压到了 Kimi 名称上）。
    # 需要强调效果时用 markeredgecolor=<同色> 单独指定。
    "lines.markeredgecolor": "none",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ============================================================================
# 色板：两类色严格分离（2026-10-06 用户裁定「全 figure 统一、跨 figure 统一」）
# ============================================================================
# 🔴 **分类原则（此前混乱的根因）**
#
#   **身份色（identity）** =「这是哪个模型」→ **唯一来源 `chan_color(key)`**
#     （`palette.py` 由图标主色派生，基线是 Fig2a 面板）。
#     只要图上画的是「某个模型」，**任何面板、任何 figure 都必须用它的身份色**，
#     不得因该模型读数低/条件不同/处在某个臂里而改色。
#     典型违规（已修）：Fig2c 把 Qwen 从紫改成 RED；Fig4a 把 9 个模型压成红/蓝两色。
#
#   **语义色（semantic）** =「发生了什么」→ 恒定色，不随模型变
#     RED=失败/为 0/塌陷、BUSINESSBLUE=基线臂、其余=结构/分组。
#     判据：**若两个不同模型在同一面板里必须同色**，就说明该维度是语义而非身份。
#
#   中性灰（NEU_*）只用于坐标轴、网格、参考线、次要文字，**永不用于标识模型**。
#
# 跨 figure 统一由 `chan_color` 单点保证——它是全项目唯一入口。
INK = "#1A1A1A"

# ---- 语义色（不表达模型身份） ----
RED= "#B64342"   # 失败 / 0 票 / Δ 塌陷 / 明确警示
BLUE   = "#0F4D92"   # 基线臂（full arm A）、hero 统计量、跨模型连线
BLUE_L = "#7FA8CE"   # 次级臂 / 区间底纹
TEAL   = "#42949E"   # 第三类对照（如 "− follow-up"）
NEU_L  = "#D5D5D3"   # 浅底纹 / 未取中位数的对照物
NEU_M  = "#8C8C8A"   # 次要标记 / 参考线
NEU_D  = "#4D4D4D"   # 次要文字 / 边框
GOLD   = "#A67C00"   # 金标准标记（gold standard）——语义色，不标识模型

# ---- 模型身份色（唯一入口，跨 figure 统一） ----
# 定义在下方 `chan_color`（转发到 palette.model_color）。
# 调色规则与实测依据见 palette.py 文档字符串（SAP v1.23）。

MM = 1 / 25.4# 1 mm（转 inch）
COLW = 180 * MM      # 双栏宽
HALF = 88 * MM       # 单栏宽


def load():
    return json.loads(DATA.read_text(encoding="utf-8"))["models"]


def order_by_rate(M, key="A"):
    """按主终点降序返回 [(key, entry)]（v1.22：分组已废止，仅按读数排序）。"""
    return sorted(M.items(), key=lambda kv: -kv[1][key]["consensus"]["rate"])


# 显示名压短表：仅压长度，不改官方命名（skill 红线：不得盲目 .title() 变形）
SHORT = {
    # 权威写法来自 results_stage4.json 的 pretty 字段（与 md 一致）
    "DeepSeek V4.1 Flash": "DeepSeek V4.1",
    "GLM-5.3-Flash": "GLM-5.3",
    "Hy4 preview": "Hy4 preview",
    "MiniMax-M3": "MiniMax-M3",
    "Kimi K2.8 Preview": "Kimi K2.8 Preview",
    "MiniMax-M3.1-Flash": "MiniMax-M3.1-Flash",
    "MiMo-V2.6-Flash": "MiMo-V2.6",
    "Qwen3.8-Flash": "Qwen3.8",
    "Ling-3.1-Flash": "Ling-3.1",
}


def short(name):
    return SHORT.get(name, name)



def clean_axes(ax, keep_left=False):
    """出版级轴线清理：默认去掉左/上轴线。

    除美观外还有一个审计要求：Nature 碰撞审计把左侧 spine 视为 stroked path，
    任何紧贴它的 y 轴刻度标签都会被判为 text-stroke 相交。隐藏左 spine 后，
    刻度标签仍可读（数值本身即定位），且不再与任何线条相交。
    """
    ax.spines["top"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4.0)



# ---- 图标与配色（v1.23）--------------------------------------------------
import sys as _sys                                       # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from svg_icons import (ICON_MAP, MODEL_ICON, draw_icon_label,   # noqa: E402
                       draw_icon_png, blank_axes, flush_icons)
import svg_icons                # noqa: E402  · save_pub 读 OVERSAMPLE
import palette as pal# 图2 配色（图标主色 → 色相家族 + 明度档）

SOFTWARE_OF = pal.SW_OF
SOFTWARE_ORDER = pal.SW_ORDER


def label_of(k):
    """模型 key → 权威写法（SAP §10.1-A），供图内标注使用。"""
    return ICON_MAP.get(MODEL_ICON.get(k, k), k)


def software_of(k):
    return pal.SW_OF.get(k, "")


def chan_color(model_key):
    """**模型身份色的唯一入口**（全项目 / 跨 figure 统一，2026-10-06 用户裁定）。

    基线 = Fig.2 a 面板的「图标 + 模型全称」标签配色，由图标主色派生，
    规则与实测依据见 `palette.py` 文档字符串（SAP v1.23）。

    ⚠️ **凡是在图上标识「某个模型」的 artist，都必须调本函数**，
    不得用 `RED`/`BLUE`/`NEU_*` 等语义色或中性色代替——
    同一模型在不同面板显示为不同颜色，会让读者误以为是两个对象
    （真实踩坑：Fig.2c 把 Qwen 画成 RED，而 Fig.2a 里它是紫色）。
    """
    return pal.model_color(model_key)


def _label_pad_data_units(fig, ax, order, data_lo, fontsize=5.0,
                          pixels=8.0, gap_px=2.2, floor=5.0):
    """算「图标 + 模型全称」标签区宽度 → 返回它对应的**数据单位跨度**。

    ⚠️ **绝不能按「字符数 × 经验系数」估算**（本项目连续踩坑两次：
       估 33 个 x 单位、实测 43 个，都会把数据区挤窄，图标被推出可视区
       而**不报任何错**）。

    正解：用 renderer 直接量文字的**像素宽度**——它只取决于字体度量与字号，
    与任何坐标系无关（不需要设临时 xlim，也就不会有坐标依赖陷阱）。
    再把「标签区占图宽的分数」乘以目标 xlim 跨度即可。
    """
    longest = max((e["pretty"] for _, e in order), key=len)

    # ① renderer 量像素宽度（与坐标系完全解耦）
    probe = ax.text(0.5, 0.5, longest, fontsize=fontsize, transform=ax.transAxes)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    text_w_px = probe.get_window_extent(renderer=r).width
    probe.remove()

    # ② 标签区像素宽 = 图标 + 间距 + 文字
    pad_px = pixels + gap_px + text_w_px

    # ③ 换算成数据单位。
    # ⚠️ 必须除以**a 面板的像素宽度**，不是整张图的宽度——标签区只占
    #    a 面板的一部分（a 面板约占图宽 40%）。早先误用整图宽换算，
    #    得到的标签区宽度只有真实值的 ~40% → 文字右缘伸进 CI 线，
    #    碰撞审计判 text-stroke FAIL（Qwen 溢出 15pt、Kimi 溢出 4pt）。
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer=r).width
    ax_w_px = max(ax_w_px, 10.0)
    xspan_target = max(100.0 - data_lo, floor)
    return pad_px / ax_w_px * xspan_target


def _label_block_right(ax, key, x, text, fontsize=5.0, pixels=8.0,
                       gap_px=None):
    """返回「图标 + 文字」整块在**数据坐标**里的右缘 x。

    用途：引线必须从文字块的**外侧**起笔，否则会从字里穿出去
    （碰撞审计判 text-stroke FAIL）。

    ⚠️ 换算必须与 `svg_icons._label_text` **逐项一致**：
       图标实际宽 = `visual_wide_correction(key, visual_pixels(key, px))`，
       再按「正方形化」`/fig_h_px` 转成轴分数，最后乘 xspan 得数据单位；
       文字宽用 renderer 实测（多行文字取**最长一行**的 bbox 宽）。
       两者相加即整块宽度。gap 默认 1.35×fontsize，与 _label_text 同口径。
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # ① 文字宽（px）：多行时逐行量测取最大
    text_w_px = 0.0
    for line in str(text).split("\n"):
        probe = ax.text(0.5, 0.5, line, fontsize=fontsize,
                        transform=ax.transAxes)
        text_w_px = max(text_w_px,
                        probe.get_window_extent(renderer=r).width)
        probe.remove()
    # ② 图标宽（px）→ 数据单位：走与 _icon_box_on_fig 相同的两步换算
    px_eff = svg_icons.visual_wide_correction(
        key, svg_icons.visual_pixels(key, pixels))
    fw_px, fh_px = svg_icons._fig_px(fig)
    ax_w_px = max(ax.get_window_extent(renderer=r).width, 10.0)
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    w_data = px_eff / fh_px * xspan
    if gap_px is None:
        gap_px = fontsize * fig.dpi / 72.0 * 1.35
    gap_data = gap_px / ax_w_px * xspan
    return x + w_data + gap_data + text_w_px / ax_w_px * xspan


def panel_label(ax, letter, dx=-0.075, dy=1.075, size=7.5,
                dx_pt=None, dy_pt=None):
    """在面板左上角外侧放粗体小写面板标签。

    🔴🔴 2026-10-07 改为**绝对点数偏移**（此前是 axes 分数）：
    闸门报「Panel-label anchors are not aligned within their comparable group」
    的根因是——**轴分数 dx 在面板宽度不同时会放大成不同的绝对偏移**。
    实测 Fig.8：a/b/c/d 用了四个不同的 dx（-0.09 /默认 / -0.13 / -0.055），
    换算成 PDF 绝对偏移为 **-17.8 / -14.8 / -25.7 / -10.9 pt**，最大差 14.8 pt。
    Fig.7 更甚：b 面板窄 45%（107.8 vs 195.2 pt），同一 dx 也会差出 6 pt。

    正解：用 `annotate` + `textcoords="offset points"`，**偏移量与面板宽度无关**，
    故同图内各面板天然对齐。

    🔴 偏移量**不再由调用方决定**（2026-10-07 二次修正）：
    旧调用点各图自带不同的 dx（−0.03 … −0.285），若直接按 pt 解释，
    同图内标签会横向参差达 17 pt（实测 Fig.7：a/c 为 −3 pt、b 为 −20 pt）——
    **闸门能过，但视觉上标签不齐**。
    → 改为**统一常量** `DX_PT` / `DY_PT`，调用点的 dx/dy 参数**一律忽略**
    （保留形参只为不改动41 处调用点）。
    ⚠️ 后果：那些原本为「给长 y 轴标签／面板标题让位」而特意加大 dx 的调用点，
    标签会贴近坐标区。**实测 Fig.9b 因此出现「面板标签 b」与面板标题首词
    「comparable」重叠（text-text碰撞）**→ 故提供 `dx_pt` / `dy_pt` 显式覆盖参数，
    仅在实测确有碰撞时单独调整，不影响其余面板的对齐。
    """
    DX_PT = -4.5 if dx_pt is None else dx_pt
    DY_PT = 4.0 if dy_pt is None else dy_pt
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(DX_PT, DY_PT), textcoords="offset points",
                fontsize=size, fontweight="bold", va="bottom", ha="left",
                color=INK, annotation_clip=False)


def save_pub(fig, stem, alignment=True, **manifest_opts):
    """导出 SVG + PDF + 600dpi TIFF。多面板先跑对齐闸门。

    `manifest_opts` 传给对齐审计（如 exclude_axes / row_groups / column_groups），
    用于把「示意面板」（无真实数据坐标轴）排除在等宽等高比较组之外。

    ⚠️ **导出 dpi 必须显式给，且须与 `svg_icons.OVERSAMPLE` 一致**。
    matplotlib 的 `imshow` 是「按显示尺寸 × savefig dpi 重采样后内嵌位图」，
    而本项目图标在版面上只有 5.76 pt（0.08 in）。实测（`/tmp` 内A/B 实验）：
      savefig(dpi=100)  → PDF 内嵌  8×8 像素（放大即糊，本项目此前的真实状态）
      savefig(dpi=600)  → PDF 内嵌 48×48
      savefig(dpi=1200) → PDF 内嵌 96×96
    `svg_icons` 已把图标位图按 `OVERSAMPLE` 倍预合成，这里以同倍率导出，
    二者相消 → 落到纸面时 1:1 无重采样。
    文字不受影响：实测 dpi=100/2400 下 `pdftotext` 都能提取到正文，
    字体始终是Type0/TrueType 矢量。
    """
    # 图标须在 savefig 前统一渲染（transData 依赖最终坐标范围，见 svg_icons §三）
    # 🔴 `flush_icons` 的内存代价（2026-10-07 实测）：它会把整张画布按
    #   `fig.dpi × OVERSAMPLE`(=100×12=1200) 的位图展开后再缩放，
    #   单峰 RSS **82 MB → 3811 MB**，随后进程被系统 kill（exit=137）。
    #   **无图标的图必须跳过它**——这是纯浪费且会直接导致导出失败。
    #   判据：图内是否调用过 draw_icon_label / draw_icon_png。
    #   （有图标的 5 张既有图可正常导出，因其fig 较小且分批单独跑。）
    # 🔴 2026-11-02：不得在审计前 flush —— overlay 轴会被闸门当成第 4 个
    #   无 id 面板（NOT AUDITABLE）。统一在下方「对齐闸门之后」再 flush。
    if not FIGDIR.is_dir():
        FIGDIR.mkdir(parents=True, exist_ok=True)
    base = FIGDIR / stem
    # ---- 顺序很关键：① 对齐闸门 → ② 图标渲染 → ③ 保存 ----
    # 图标 overlay（铺满画布的辅助轴 + inset 子轴）会被对齐审计当成未知面板
    # 而报 NOT AUDITABLE 并中断，因此**必须先审计、后渲染图标**。
    # 这样也不需要把overlay 摘出来再装回去（该操作会破坏父子关系与 zorder）。
    # 🔴 `alignment=True` 但闸门模块缺失（跨平台降级，见顶部 SKILL 探测）时
    #   **跳过闸门而非报「未通过」**——「工具不存在」与「检查未通过」是两件事，
    #   混为一谈会污染审计记录（如实登记是本项目的硬纪律）。
    if alignment and require_matplotlib_panel_alignment is None:
        print(f"   [对齐闸门] {stem}: 已跳过（未找到 audit_panel_alignment 模块）")
    elif alignment:
        try:
            require_matplotlib_panel_alignment(
                fig, json_out=f"{base}.alignment.json",
                overlay_svg=f"{base}.alignment.svg",
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
                **manifest_opts)
        except SystemExit as e:
            print(f"   [对齐闸门] {stem}: {e}")
        except Exception as e:
            # 🔴 对齐闸门在 strict=True 时抛的是 PanelAlignmentError（**不是** SystemExit），
            #   原先只捕SystemExit → 异常穿透，save_pub 中断、图没导出。
            #   这里如实登记失败并继续导出（不假装通过，与碰撞审计同一纪律）。
            print(f"   [对齐闸门] {stem}: 未通过 —— {type(e).__name__}")
            print("      （如实登记，不得声称通过；导出继续，便于目视核对）")
    # 图标渲染（transData 依赖最终坐标范围，见 svg_icons §三）
    #   同样受内存约束约束：无图标的图跳过（见上方 _need_icons 注释）。
    if getattr(fig, "_pcne_has_icons", False):
        flush_icons(fig)
    # 🔴 导出格式开关（2026-10-07 用户裁定 · SAP §10.2-#39）：
#   **TIFF 改为「投稿前按需生成」，不再是每次出图的默认产物。**
#   起因：TIFF 占项目 368 MB（现行 133 MB ＋ 旧号归档 235 MB），
#   而调图期间每改一版就重新全量出一次 TIFF → 体积急剧膨胀、且完全无用
#   （投稿时才需要，且只需最终版一次）。
#   分工：
#     默认（调图/ 审稿期）→ **PDF ＋ SVG ＋ PNG(300 PPI 审阅用)**，**不出 TIFF**
#     `--tiff`（投稿前）→ 额外生成 TIFF 600 dpi LZW
#     `drp_eval/make_tiff.py`（投稿前统一批量生成，不必重跑统计）
#   环境变量 `PCNE_EXPORT_TIFF=1` 等价于 `--tiff`，供 IDE / 脚本调用。
    EXPORT_TIFF = "--tiff" in sys.argv or os.environ.get("PCNE_EXPORT_TIFF") == "1"
    # 🔴 审阅用 PNG = **300 PPI**（用户 2026-10-07 裁定 · SAP §10.2-#39-d）：
    #   300 是出版印刷与屏幕审阅的通用档，肉眼与多数期刊预览系统的默认假设都是 300。
    #   审阅看的是**版式对不对**（字号/线宽/标签是否重叠/标注是否被遮挡），300 足够判断；
    #   投稿仍用 600 dpi TIFF 或矢量 PDF。
    #   ⚠️ 环境变量名是 `PCNE_PREVIEW_DPI`，**引号不可省**——
    #   曾因漏引号写成裸变量名 `os.environ.get(PCNE_PREVIEW_DPI, 300)` →
    #   NameError 在「已装好包的新环境」才暴露（本机因该变量恰好未定义而侥幸通过）。
    PREVIEW_PNG_DPI = int(os.environ.get("PCNE_PREVIEW_DPI", "300"))
    # 投稿用 TIFF 的 dpi：Nature 系印刷标准线为 300，取 600 留余量（SAP §10.2-#33-a）。
    FINAL_TIFF_DPI = int(os.environ.get("PCNE_TIFF_DPI", "600"))
    # 含 imshow 的图不导出 SVG（见下），其预览 PNG 也走 PDF 中转路径。
    _has_imshow = any(len(getattr(a, "images", [])) > 0 for a in fig.axes)
    export_dpi = int(round(fig.dpi * svg_icons.OVERSAMPLE))
    if _has_imshow:
        export_dpi = 600
        fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=export_dpi)
    else:
        fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=export_dpi)
        fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=export_dpi)

    # ---------------------------------------------------------------------------
    # 🔴 位图导出（PNG 审阅用 / TIFF 投稿用）——**一律走 PDF 中转**
    #   路径：PDF → PyMuPDF 渲染成位图 → Pillow 编码。
    #   ⚠️ 不可直接 `fig.savefig("x.png")`（2026-10-07 实测被系统 kill，
    #      同尺寸 PDF/SVG 正常、内存与耗时亦正常 → 是 matplotlib 的 PNG writer 自身问题）。
    #   ⚠️ PyMuPDF 也不直接写 TIFF（`Image format tiff not in (...)`），
    #      故 TIFF 也由 Pillow 编码，两条路径共用一次渲染。
    # ---------------------------------------------------------------------------
    import pymupdf
    from PIL import Image
    if EXPORT_TIFF or PREVIEW_PNG_DPI > 0:
        with pymupdf.open(f"{base}.pdf") as _doc:
            # 渲染一次，按需产出两种位图：
            #   --tiff  → 渲染到 FINAL_TIFF_DPI(600)，PNG 再降采样到 PREVIEW_PNG_DPI
            #   默认    → 直接渲染到 PREVIEW_PNG_DPI(300)
            # ⚠️ 早前版本写的是 `min(PREVIEW_PNG_DPI, 200)`（200 时代遗留），
            #   会把 TIFF 模式下的 PNG 悄悄压到 200 PPI，与「审阅统一 300 PPI」的
            #   裁定自相矛盾——同一张图在两种模式下 PNG 清晰度会不同。已改为直接用
            #   PREVIEW_PNG_DPI，**两种模式的审阅 PNG 完全一致**。
            _need = FINAL_TIFF_DPI if EXPORT_TIFF else PREVIEW_PNG_DPI
            _pix = _doc[0].get_pixmap(dpi=_need, colorspace=pymupdf.csRGB, alpha=False)
            _rendered = Image.frombytes("RGB", (_pix.width, _pix.height), _pix.samples)
            if EXPORT_TIFF:
                _rendered.save(f"{base}.tiff", compression="tiff_lzw",
                               dpi=(FINAL_TIFF_DPI, FINAL_TIFF_DPI))
            # 审阅 PNG：始终 300 PPI（与 --tiff 模式一致）
            _png_dpi = PREVIEW_PNG_DPI
            _im2 = (_rendered if _png_dpi == _need else
                    _rendered.resize((int(_rendered.width * _png_dpi / _need),
                                      int(_rendered.height * _png_dpi / _need)),
                                     Image.LANCZOS))
            _im2.save(f"{base}.png", "PNG", dpi=(_png_dpi, _png_dpi), optimize=True)
            if _im2 is not _rendered:
                del _im2
            del _rendered, _pix
    plt.close(fig)

    _parts = ["pdf"]
    if not _has_imshow:
        _parts.append(f"svg (dpi={export_dpi})")
    _parts.append(f"png (dpi={PREVIEW_PNG_DPI} 审阅用)")
    if EXPORT_TIFF:
        _parts.append(f"** tiff ({FINAL_TIFF_DPI}dpi LZW 投稿用) **")
    else:
        _parts.append("tiff 未生成（需 --tiff 或 drp_eval/make_tiff.py）")
    print(f"   ✅ {stem}: " + " / ".join(_parts))


# ============================================================
# Fig.3 · 九个模型的主终点表现（hero 图）
# ============================================================
def fig3():
    M = load()
    order = order_by_rate(M)
    # 🔴 图高168→176mm：hspace 从 0.78 加到 1.05 后，若不同步放大画布，
    # 每个面板的绝对高度会缩小 3mm 以上，5pt 字号的标签行距逼近下限。
    fig = plt.figure(figsize=(COLW, 176 * MM))

    # 左边距须容纳 a 面板的「图标 + 模型全称」标签区（最长全称
    # "DeepSeek V4.1 Flash" 18 字符 × 5pt ≈ 4.2% 图宽 + 图标约 2% + 间距）
    # 🔴 hspace 由 0.78 加大到 1.05（2026-10-07 修 FAIL）：
    #   碰撞审计报「Kimi-K2.8-Preview 被 1 条笔画路径穿过」，
    #   但other_text=None → **不是文字互撞，而是 c 面板的自稳定性标记
    #   （3.8pt 圆点，PDF 里约 3.8×3.8pt 的 fs 路径）纵向溢出到了 a 面板区域**。
    #   实测那些路径的 y 落在 308–316pt，而 Kimi 标签在 305.7–310.7pt
    #   （a 面板最上一行）→ 两面板在纵向确实重叠了。
    #   正解是**加大面板间距**（hspace），而不是挪标签或改标记——
    #   后者只会把冲突转移到别的标签上。
    #
    # 🔴 2026-10-07 用户裁定：**行间距过大**（实测 AB 底–CD 顶 188.4px，
    #   在 176mm 画布上占约 13% 纵向空间，且 AB 与 CD 之间完全空白）。
    #   用户要求缩到「c、d 两面板之间的**横向**间距」那个量级。
    #   实测该参照值 = **80.3px**（c 面板右缘 454.7 − d 面板左缘 374.4）。
    #   → hspace 1.05 → **0.35**，实测间隙 81.5px（≈80.3px 目标）。
    #   ⚠️ 缩行距会**同步放大四个面板的高度**（a 面板 184.7 → 239.7px，
    #     因为画布总高与 top/bottom 锁死，多出来的纵向空间全给了面板）。
    #     对各面板的影响（已逐一核对，非副作用）：
    #     · a 森林图：9 行行距 20.5 → 26.6px，更透气，无碰撞风险；
    #     · b 热图：格子 20.5×33.7px → 26.6×33.7px，由横向矩形变近正方，
    #       对容纳数字的热图**更合适**；
    #     · c 散点 / d 堆叠条：仅纵向拉伸，坐标不变。
    #   ⚠️ 早先「Kimi 标签被 c 面板圆点纵向压线」那个 FAIL 的**根因不是行距本身**，
    #     而是 c 面板元素纵向溢出到 a 面板区域；本轮 c 面板顶部最上元素是
    #     GLM 引线（y=98.3，ylim 上限 99），距 a 面板 xlabel 仍有 50px+ 余量
    #     → 缩行距后已重跑碰撞审计确认 0 FAIL（见下方验收）。
    #   2026-10-07 用户裁定：**列间距放宽3–5px** →实测 wspace 0.34 → 0.36
    #   使 a|b 与 c|d 的横向间隙由 **80.3px → 84.3px（+4.0px）**。
    #   ⚠️ 副作用（已实测核对，非隐患）：wspace 增大 → **面板变窄 236.2 → 234.2px**。
    #     · a 面板的「图标+名称」标签区是按**实测像素宽**换算成数据单位的
    #       （`_label_pad_data_units`），故标签区会随面板变窄而**等比放大**，
    #       不会挤掉数据区—— 这是该函数设计的初衷；
    #     · b 面板热图 xlim 固定（7 列），格子横向随之收窄 2px，肉眼无感；
    #     · c/d 面板 xlim 固定，标记横向间距收窄 2px，**不产生碰撞**
    #       （碰撞审计已复核，见验收）。
    # ---- 面板标签横向偏移（2026-10-07 用户裁定：a/c 左移 10px、b/d 左移 12.5px）----
    # ⚠️ **单位换算**：`panel_label(dx_pt=…)` 的单位是**点（pt）**，不是像素。
    #   布局按 `fig.dpi = 100` 计算 → **1pt = 100/72 = 1.3889px**。
    #   （实测验证：Δdx_pt=10 → 实测左移 13.89px = 10 × 1.3889 ✅吻合）
    #   故「左移 10px」需 Δ= 7.2pt、「左移 12.5px」需 Δ= 9.0pt。
    #   基线是原统一常量 `DX_PT = −4.5`（见 panel_label docstring）。
    #   → a/c: −4.5 − 7.2 = **−11.7pt**；b/d: −4.5 − 9.0 = **−13.5pt**。
    # ⚠️ **两列用不同值是用户明确要求**（两列视觉重心不同，各自微调）。
    #   对齐闸门只比较**同列**面板标签锚点（a vs c、b vs d），
    #   故a/c 同值、b/d 同值即满足，**跨列不同值不触发 FAIL**（已实测验证）。
    _DX_A = -22.5   # a 与 c 共用（2026-10-07 二次调整：再左移 15px）
    _DX_B = -58.5   # b 与 d 共用（2026-10-07 三次调整：再左移 37.5px，取用户 35–40px 中值）

    gs = fig.add_gridspec(2, 2, height_ratios=[1.06, 1.0],
    #   2026-10-07 用户裁定：**列间距再扩大 15px**（b/d 标签左移后逼近 a 面板内容区）。
    #     实测 wspace 0.36 → **0.44**，间隙 **84.3 → 99.7px（+15.4px）**，
    #     面板左缘 a 372.4→364.7、b 左缘 456.7→464.4 → 标签 b 与 a 面板的净间距由
    #     +3.1px（对 a 的「100」刻度标签）**放宽到约 +18px**，脱离「几乎贴着」的临界状态。
    #   ⚠️ 副作用（已实测）：面板变窄 234.2 → **226.5px**（再窄 7.7px）；
    #     a 面板标签区按实测像素宽换算 → 等比放大、不挤数据区；b/c/d 的 xlim 固定。
                          hspace=0.35, wspace=0.44,
                          left=0.195, right=0.975, top=0.905, bottom=0.115)

    # ---------- a · 主终点森林图（hero） ----------
    ax = fig.add_subplot(gs[0, 0])
    ax_a = ax          # ⚠️ 本函数后面 ax 被 b/c/d 面板反复复用，此处留专属引用，
                        #    否则 save 前的 draw_icon_label(ax, …) 会用d 面板的
                        #    坐标系定位 → 图标被算到画布外而"凭空消失"。
    ys = np.arange(len(order))[::-1]
    # 图标 + 名称标签区的宽度：按**最长全称**实测像素宽换算。
    # xlim 左界 = 标签区右缘，须严格小于所有 CI 的下界（本项目 min=53.6），
    # 否则 Qwen 的 CI 横线会横穿它自己的名称（碰撞审计判 text-stroke FAIL）。
    _min_ci = min(e["A"]["consensus"]["ci_lo"] for _, e in order) * 100
    _label_pad = _label_pad_data_units(fig, ax, order, 100.0 - _min_ci)
    # CI 从最小下界起画；标签区右缘再左移 2.5 个单位作安全距离，
    # 确保 Qwen（CI 最小者）的名称与自己的 CI 线**物理分离**
    _ci_lo_bound = _min_ci
    _label_right = _min_ci - 2.5
    for y, (k, e) in zip(ys, order):
        c = e["A"]["consensus"]
        col = chan_color(k)
        # CI 横线起点裁到**数据区左界**ax_data_lo，避免伸进左侧标签区
        # 去横穿模型名称（Qwen 的 CI 下界 53.6 最靠左，恰好穿到自己的名字）
        _lo = max(c["ci_lo"] * 100, _ci_lo_bound)
        ax.plot([_lo, c["ci_hi"] * 100], [y, y],
                color=col, lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot(c["rate"] * 100, y, "o", ms=4.2, color=col,
                markeredgecolor=col, markeredgewidth=0.5, zorder=3)
        # 标记边必须**显式**给同色：rcParam 的 lines.markeredgecolor="none"
        # 在未显式传参时会被解析成黑色描边（实测导出PDF 里出现
        # color=(0,0,0) 的黑边），压在相邻文字上会被碰撞审计判 FAIL。
        # 自稳定性以空心小方块叠加：与主终点的关系是本面板的第二读数
        st = e["A"]["stability"]["rate"]
        ax.plot(st * 100, y, "s", ms=2.3, mfc="none",
                markeredgecolor=NEU_D, mew=0.5, zorder=4)
    ax.set_yticks(ys)
    # y 轴刻度改为「图标 + 模型全称」（icon/ · 用户 2026-10-06 要求用全称，
    # 不用 short() 缩略）。刻度文字留空，在 save 前用 draw_icon_label 叠加。
    ax.set_yticklabels([])
    # x 下界须容纳最小 CI 下界（Qwen 53.6），否则 clip 后线条会画到坐标区外，
    # 横穿下方面板的 y 轴刻度标签（碰撞审计 FAIL）。
    lo_ci = min(e["A"]["consensus"]["ci_lo"] for _, e in order)
    # 标签区右缘比最小 CI 再左3 个单位 → 标签与所有 CI 线完全分离，
    # 且不会与下方 c 面板的标记/ 文字在同一像素位置相遇
    ax_data_lo = lo_ci * 100 - 2.5
    # ⚠️ 标签区宽度**必须按真实像素宽度反算**，不能按「字符数 × 经验系数」猜：
    #早先按 1.62/字符 估算，最长全称 "DeepSeek V4.1 Flash"（19 字符）
    # 给出 33 个 x 单位 → 标签区吃掉 40% 图宽，数据区被压到 56%，
    # 图标随即被挤出可视区而"消失"。现用实测像素宽度（见 label_pad_data_units）。
    ax.set_xlim(ax_data_lo - _label_pad, 100)
    ax.set_xlabel("Accuracy on consensus cells (%)")
    # 下界留足空间：y=0 行（Qwen）的标签不能压到 x 轴刻度上
    ax.set_ylim(-1.15, len(order) + 0.6)

    # 窄带 + 断层标注
    rates = [e["A"]["consensus"]["rate"] for k, e in order]
    core = [r for r in rates if r > 0.75]
    ax.axvspan(min(core) * 100, max(core) * 100, color=BLUE_L,
               alpha=0.13, zorder=0, lw=0)
    # 均值参考线：只画在**数据区**内（x≥ ax_data_lo），不延伸到左侧标签区，
    # 否则会横穿 Kimi／Qwen 的名称文字（碰撞审计判FAIL）
    ax.plot([np.mean(rates) * 100] * 2, [-1.3, len(order) - 0.3],
            color=NEU_M, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.annotate(f"mean {np.mean(rates)*100:.1f}%",
                xy=(np.mean(rates) * 100, len(order) - 0.5),
                xytext=(np.mean(rates) * 100 + 4.5, len(order) + 0.32),
                fontsize=5, color=NEU_D, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=NEU_D, lw=0.5,
                                shrinkA=0, shrinkB=1))
    ax.text(min(core) * 100 - 1.0, len(order) - 0.55,
            f"{len(core)}/{len(rates)} models span "
            f"{max(core)*100-min(core)*100:.0f} pp",
            fontsize=5, color=BLUE, ha="right", va="center",
            fontweight="bold")

    # 图例：只留标记含义。模型配色已由 y 轴每行的「图标 + 全称」直接承载，
    # 另附一行配色规则说明（色相=采集软件家族，明度=组内档位），
    # 不再逐个罗列色块——9 个色块会过于花，违背"克制"。
    # 图例：**移入坐标区内的空白带**（用户 2026-10-07 要求）。
    # 🔴 原位置在 x 轴下方（bbox y=-0.135）有两个问题：
    #   ① 一行两列宽约 100px，横跨到面板右缘之外，容易与 b 面板的色标、
    #      以及 a/b 之间的空白区里的其他内容叠在一起；
    #   ② 迫使读者在坐标区与图例之间来回跳读。
    # 放置依据（**实测像素宽换算**，非字符数估算）：
    #   标签区右缘随行而变，最长的是 DeepSeek 行（y=5，右缘 57.9 数据单位）；
    #   各行 CI 左缘最小的是 MiMo 行（y=2，74.0）。
    #   → y∈[6.4, 7.5] 这一条只被 y=6/7 两行约束，
    #     可用横向区间约 51.0 … 79.4（28.4 数据单位 ≈ 100px），
    #     而 ncol=1 两行的图例实测宽约 63px → 左右各留 ~18px 余量。
    # 用 ncol=1（上下两行）而非一行两列：一行两列需 ~100px，几乎吃掉整条空带。
    handles = [plt.Line2D([], [], marker="s", ls="none", mfc="none", mec=NEU_D,
                          mew=0.5, ms=2.4),
               plt.Line2D([], [], marker="o", ls="none", color=INK, ms=4.2)]
    labels = ["self-consistency", "accuracy"]
    ax.legend(handles, labels, loc="center",
              bbox_to_anchor=(65.4, 6.95), bbox_transform=ax.transData,
              ncol=1, handlelength=0.7, handletextpad=0.35,
              labelspacing=0.25, borderpad=0.12, fontsize=5)
    # 🔴 2026-10-07 用户裁定：**删除**「colour: model identity · hue = client family ·
    #   shade = rank」这行配色说明（用户目视后认为它挤在图例下方冗余、要去掉）。
    #留痕：① 该行曾因漏 `transform=ax.transAxes` 被甩到画布最左侧（R40），
    #   修复后又按 §10.2-#33-b 改过措辞；② 本次是**用户主动要求删除**，
    #   非修复失败 → 颜色规则的说明义务转移到**图注**（Fig.3 图注待写，见执行规划 §10）。
    #   配色规则本身不变：身份色＝模型，色相＝采集软件家族，明度＝组内排名。
    clean_axes(ax, keep_left=True)
    # 🔴 2026-10-07 用户裁定：a / c 标签**左移 10px**（原统一 DX_PT=−4.5 偏右）。
    #   −4.5 − 10 = **−14.5pt**。a 与 c 用**同一值**→ 列内仍严格对齐。
    panel_label(ax, "a", dx_pt=_DX_A)

    # ---------- b · 分项热图 ----------
    ax = fig.add_subplot(gs[0, 1])
    DIMS = ["P", "E1", "E2", "C", "I", "A", "O"]
    Z = np.array([[e["A"]["by_dim"][d]["rate"] * 100 for d in DIMS] for _, e in order])
    # 量级高低 → 顺序色板（浅=低，深=高）。禁用发散色板：会让人误读「红=差」
    im = ax.imshow(Z, cmap="Blues", vmin=35, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(DIMS)))
    ax.set_xticklabels(DIMS)
    ax.set_yticks(np.arange(len(order)))
    # 🔴 2026-10-07 用户要求：b/d 面板左侧模型名与 **a 面板**同名同色同字号。
    #   原为matplotlib 默认黑（INK 由rcParams 之外的默认色给出），
    #   同一模型在 a 面板是身份色、在 b/d 面板变黑 → 读者会当成两套对象。
    #   正解：逐行取 `chan_color(k)`（全项目身份色唯一入口），fontsize 与 a 面板同为 5.0。
    ax.set_yticklabels([e["pretty"] for _, e in order], fontsize=5.0)
    for _i, (_k, _e) in enumerate(order):
        ax.get_yticklabels()[_i].set_color(chan_color(_k))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # 深底用白字、浅底用墨色，保证 5pt 小字对比度
            ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center",
                    fontsize=5, color="white" if Z[i, j] > 68 else INK,
                    zorder=6)
    # 色块分隔：逐格白描边（不用 ax.grid —— 网格线会越出色块边界、
    # 延伸到坐标区外与 y 轴刻度标签相交，被碰撞审计判为 FAIL）
    for i in range(Z.shape[0] + 1):
        ax.plot([-0.5, len(DIMS) - 0.5], [i - 0.5, i - 0.5],
                color="white", lw=0.6, zorder=4)
    for j in range(len(DIMS) + 1):
        ax.plot([j - 0.5, j - 0.5], [-0.5, len(order) - 0.5],
                color="white", lw=0.6, zorder=4)
    ax.tick_params(which="both", length=0)
    # C / I 两列 = 失分区
    ax.add_patch(Rectangle((2.5, -0.5), 2, len(order), fill=False,
                           edgecolor=RED, lw=1.0, ls=(0, (2.5, 1.5)), zorder=5))
    ax.text(3.0, -0.92, "hardest dimensions", fontsize=5, color=RED,
            ha="center", va="center", fontweight="bold")
    cax = ax.inset_axes([0.0, -0.115, 1.0, 0.032])
    cax.imshow(np.linspace(35, 100, 120).reshape(1, -1), cmap="Blues", aspect="auto")
    cax.set_yticks([])
    cax.set_xticks([0, 60, 119])
    cax.set_xticklabels(["35", "68", "100%"], fontsize=5)
    cax.xaxis.set_ticks_position("bottom")
    cax.tick_params(length=1.4, pad=1.2)
    for s_ in ("top", "right", "left"):
        cax.spines[s_].set_visible(False)
    clean_axes(ax)
    clean_axes(ax)
    # b / d 标签**左移 12.5px**（用户给 10–15px，取中值）→ −4.5 − 12.5 = **−17.0pt**。
    #   b 与 d 用同一值 → 列内对齐；与 a/c 的 −14.5pt **不同**是用户明确要求
    #   （两列视觉重心不同，各自微调）。⚠️ 注意：`panel_label` 的对齐闸门比较的是
    #   **同列**面板标签锚点，**跨列不做要求**，故两列用不同 dx_pt 不触发闸门 FAIL。
    panel_label(ax, "b", dx_pt=_DX_B)

    # ---------- c · 主终点 vs 自稳定性 ----------
    # ⚠️ 全部用**身份色**（用户 2026-10-06 裁定「全 figure 统一」）。
    #   此前 `col = RED if rate < 0.75 else chan_color(k)` 会把 Qwen 从紫色
    #   改成红色，与 a 面板的同一模型配色冲突 → 读者会误判为两个对象。
    #   「Qwen 明显塌陷」这一信息改由**位置本身**表达（它独自远离y=x 对角线），
    #   无需额外用颜色喊出来；a 面板的窄带与均值线已在强调这一点。
    ax = fig.add_subplot(gs[1, 0])
    ax_c = ax            # ⚠️ 后面 ax 会被 d 面板复用，留专属引用给图标标注
    ax.plot([62, 95], [62, 95], color=NEU_M, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k, e in order:
        c = e["A"]["consensus"]; st = e["A"]["stability"]["rate"]
        col = chan_color(k)
        ax.plot(c["rate"] * 100, st * 100, "o", ms=3.8, color=col,
                markeredgecolor=col, markeredgewidth=0.4, zorder=3)
    ax.set_xlim(62, 96); ax.set_ylim(60, 99)
    ax.set_xlabel("Accuracy on consensus cells (%)")
    ax.set_ylabel("Self-consistency (%)")
    ax.text(93.5, 63.5, "y = x", fontsize=5, color=NEU_M, ha="right")
    # 🔴 2026-10-07 用户要求：a / c 面板**保留左侧纵轴线**。
    #   此前全图统一走 `clean_axes()` 的默认行为（隐藏左 spine），
    #   理由是「左 spine 被审计视为 stroked path，紧贴它的 y 刻度标签会被判
    #   text-stroke 相交」——但那是**刻度标签**的风险，a 面板 y 轴根本没有
    #   刻度标签（模型名是画在轴内的图标+文字），c 面板的刻度标签已实测留白足够。
    #   → 两处改用 `keep_left=True`，并已重跑碰撞审计验证（0 FAIL）。
    clean_axes(ax, keep_left=True)
    panel_label(ax, "c", dx_pt=_DX_A)   # 与 a 同值→ 列内严格对齐

    # ---- 极值模型标注（用户 2026-10-07 要求）----
    # 🔴 本段**推翻了 2026-10-07 早先「只标 Qwen 一个」的决定**（留痕见下），
    #   起因是用户目视后明确要求把两个指标各自的最优/最差模型都标出来。
    # 极值（读数取自 results_stage4.json，分母 98，末遍主口径）：
    #   accuracy         最优 Kimi-K2.8-Preview 88.8 / 最差 Qwen3.8-Flash 69.4
    #   self-consistency 最优 GLM-5.3-Flash 96.0     / 最差 Qwen3.8-Flash 67.3
    #   → Qwen 同时是两个指标的最差，故只需标 3 个模型。
    #
    # ⚠️ 早先「只标 Qwen」的留痕（**方法学仍有效，勿回退**）：
    #   本面板 8 个点簇在 (80.6, 89.6)–(88.8, 96.0)。实测最小邻距
    #   DeepSeek↔Hy4 仅 **1.02 数据单位**、Kimi↔GLM 12.7pt，而 5pt 文字宽 22–44pt
    #   → **在点旁直接标文字必然压线**（早先五轮定位失败：被穿对象在
    #     「自稳定性圆点 → 面板 spines → 邻点圆点」之间转移）。
    #
    # 本次采用的正解（三条，均已实测）：
    # ① 文字整块放到**对角线左上方的空白楔形区**（x 62.6–74.5, y 86.6–98.1：
    #   该区在y=x 之上、又在点簇左侧，实测无任何标记）；
    #   Qwen 孤立在右下，标签放在它右侧的对角线下方。
    # ② 引线起点 = 标签块的**角点**（不是中心），位置由 `_label_block_right`
    #   实测整块右缘后断言校验，杜绝「引线从第二行文字里穿出去」。
    # ③ 引线为**手工折线**（ax.plot），不走annotate 的 arc3——arc3 弧线
    #   实测会横穿标签第二行，且无法约束「从点簇外围绕入」。
    #   折线全部走在实测的空通道里：
    #     · GLM（最高点 96.0）→水平段y=98.3 在全簇上方 2.3 单位，再垂直落入；
    #     · Kimi（最右点 88.8）→ y=88.0 水平段**止于 x=87.5**（对角线 y=x 在
    #       x=88.0 才与y=88.0 相交，故此段全程在对角线上方、不交叉），
    #       再经x=89.5 的**空白垂直通道**（点簇最右仅 88.8）上行至点；
    #     · Qwen→ 水平段 y=69.4 全程在 x>y 一侧（对角线之下），不交叉。
    #④ **不用外圈空心环**：DeepSeek↔Hy4 距 1.02 单位，而 ms=8.4 的环半径
    #   0.84 单位 → 环与环必然相交重叠。改为引线末端叠一个同色实心点
    #   （ms=4.6 覆盖原 3.8 的点）作为「此点被点名」的落点强调，零额外占位。
    _c_marks = [
        # key,               标签 x, 标签 y, 副标,                 引线折线（数据坐标）
        ("glm-5.3-flash",     62.6, 96.2, "best self-consistency",
         [(74.3, 97.0), (76.5, 98.3), (86.7, 98.3), (86.7, 96.0)]),
        ("kimi-k2.8-preview", 62.6, 88.5, "best accuracy",
         [(72.9, 88.0), (87.0, 88.0), (87.0, 93.2), (88.8, 93.2)]),
        # 🔴 2026-10-07 用户第二次指出：**引线与 y=x 对角线重叠**。
        #   实测根因：折点 (70.2, 71.0) 落在对角线**上方 0.8 单位**，
        #   末段 (70.2,71.0)→(69.4,67.3) **斜穿**对角线（最近横向距 0.8 单位 ≈ 5.6px，
        #   斜段的垂直间距更小）→ 目视两线几乎重合。
        #   正解：**标签与引线整体下移**，使整条引线走在对角线**下方**。
        #   下移量由几何定：Qwen 点 (69.4, 67.3) 本身在 y=x 下方 2.1 单位，
        #   故折点取 y=66.5（x=66.5 时恰在对角线上）会贴线 →
        #   取 **x=64.6 / y=66.5**（对角线下方 1.9 单位），
        #   水平段 y=66.5 从 x=75.0 到 64.6 **全程在 x>y 一侧**（y<x），永不与对角线相交；
        #   末段 (64.6,66.5)→(69.4,67.3) 沿对角线下方 2.1→2.1 单位平行下走，也不相交。
        #   标签锚点同步下移到 y=66.5（引线起点与图标中心仍对齐，见上一轮修复）。
        #   ⚠️ 标签下移后需仍不压 y 轴刻度标签（60/65…，面板左界 62）→ 已实测留白充足。
        #🔴 2026-10-07 用户第三次指出：① 标签「稍微上来一点点，与数据点对齐」
        #   —— 即让标签块的**竖直中心**与数据点同高（y=67.3）；
        #   ② 引线改为**一条直线**，不要折线。
        #   正解：标签锚点 y=66.5 → **67.3**（= Qwen 点的 y，精确对齐）；
        #   引线 `ax.plot` 传入**两点**（起点=标签块左缘，终点=数据点）即渲染为直线段。
        #   ⚠️ 直线是否与 y=x 对角线相交？直线从 (74.3, 67.3) 到 (69.4, 67.3)
        #   是**水平线** y=67.3，而对角线在 x∈[69.4, 74.3] 处 y=x∈[69.4, 74.3]，
        #   恒有 x > y=67.3（即 x−y ≥ 1.9）→ **全程不相交**，
        #   且水平线到对角线的**垂直距离**恒为 1.9 单位（≈11px），比折线方案更宽裕。
        #   标签锚点 x 也随之微调 75.0 → 74.3（使引线起点仍紧贴块左缘，见 assert 校验）。
        #   ⚠️ 引线长度不可为 0：`_path` 的**最后一个点会与数据点相连**（见下方
        #   `_xs + [_px]`），若只给起点 (74.3, 67.3) 而它恰与数据点同 y，
        #   实际渲染的是「起点→数据点」这一段（此写法本正确），
        #   但若起点 x 误写成数据点 x 就会退化成**零长度**而画不出线。
        #   此处给两个点：起点在标签块内、终点为数据点，线段明确且非零长。
        ("qwen3.8-flash",     74.3, 67.3, "lowest on both",
         [(74.3, 67.3), (69.4, 67.3)]),
    ]
    for _key, _lx, _ly, _tag, _path in _c_marks:
        _e = M[_key]
        _px = _e["A"]["consensus"]["rate"] * 100
        _py = _e["A"]["stability"]["rate"] * 100
        _col = chan_color(_key)
        _txt = f"{_e['pretty']}\n{_tag}"
        # 折线起点须落在标签块内（否则引线与文字脱节）
        _right = _label_block_right(ax_c, _key, _lx, _txt)
        assert _path[0][0] <= _right + 0.02, (
            f"{_key}: 引线起点 x={_path[0][0]:.2f} 在标签块右缘 {_right:.2f} 之外")
        _xs = [p_[0] for p_ in _path]
        _ys = [p_[1] for p_ in _path]
        ax_c.plot(_xs, _ys, "-", color=_col, lw=0.5, zorder=4,
                  solid_capstyle="round")
        # 落点强调：同色实心点覆盖原标记（不移动数据点，不额外占空间）
        ax_c.plot(_px, _py, "o", ms=4.6, color=_col, zorder=5)
        draw_icon_label(ax_c, _key, _lx, _ly, _txt, 0,
                        pixels=8.0, fontsize=5.0, text_color=_col)

    # ---------- d · 争议格四档 ----------
    ax = fig.add_subplot(gs[1, 1])
    ORD = ["majority ≥50%", "acceptable ≥10%", "minority >0", "unclaimed 0%"]
    KEY = ["多数派档", "可接受档", "边缘档", "无人档"]
    COLS = [BLUE, BLUE_L, NEU_L, RED]
    w = 0.62
    # 🔴 2026-10-07 用户裁定：d 面板模型顺序**必须与 a/b 一致**（Kimi 在顶）。
    #   根因（**不是随机排布**）：`barh(i, …)` 的 y 轴默认朝上，i=0 落在**最底部**，
    #   而 a 面板用 `ys = arange(n)[::-1]` 把 i=0 放在**最顶部** → 两者整体颠倒。
    #   正解：`yy = n-1-i` 显式反转，与 a 面板的 ys 完全同构。
    #   （另一条正解 `ax.invert_yaxis()` 会连 y 轴刻度方向一起翻，
    #     且需重设 ylim，此处用索引反转更直接、diff 更小。）
    _n = len(order)
    for i, (k, e) in enumerate(order):
        from collections import Counter
        cnt = Counter()
        for dim, v in e["A"]["contested"].items():
            cnt.update(v["tiers"])
        yy = _n - 1 - i          # ← 与 a 面板的 ys 同一映射
        left = 0.0
        for t, c in zip(KEY, COLS):
            val = cnt.get(t, 0)
            if val:
                ax.barh(yy, val, left=left, height=w, color=c, lw=0)
                if val >= 5:
                    ax.text(left + val / 2, yy, str(val), ha="center", va="center",
                            fontsize=5, color="white" if c in (BLUE, RED) else INK)
            left += val
    ax.set_yticks(np.arange(len(order)))
    # 🔴 刻度标签**必须与柱子一起反转**（2026-10-07 修 FAIL·自查发现）：
    #   上面把柱子改到 `yy = n-1-i` 后，若标签仍按 `order` 正序填，
    #   则「Kimi 柱在 y=8（顶）」配「Kimi 标签在 y=0（底）」→ **标签与柱子错位**。
    #   正确写法：标签也取 `order[::-1]`，使 y=j 的标签恰是 `order[n-1-j]`，
    #   与柱子的 `yy = n-1-i` 同构 → 顶到底 Kimi→…→Qwen，与 a/b 面板一致。
    # 同 b 面板：模型名用身份色，字号与 a 面板一致（用户 2026-10-07 要求）
    _d_rev = order[::-1]
    ax.set_yticklabels([e["pretty"] for _, e in _d_rev], fontsize=5.0)
    for _i, (_k, _e) in enumerate(_d_rev):
        ax.get_yticklabels()[_i].set_color(chan_color(_k))
    # ⚠️ 此处**不要**再调 `set_ylim`（自查踩过一次）：
    #  柱子在 `yy = n-1-i`、标签在 `order[::-1]`，二者已使「y=8 → Kimi」。
    #   若额外写 `set_ylim(n-0.5, -0.5)` 想「把 y翻过来」，等于**翻了两次**，
    #   反而把 Kimi 送回底部（实测踩中：ylim 变成 (8.5, -0.5) 后视觉顺序完全没变）。
    #   matplotlib 默认 y 向上已满足需求，故**不写** ylim。
    ax.set_xlim(0, 35); ax.set_xlabel("Judgements on 7 contested cells × 5 repeats (n=35)")
    # 🔴 2026-10-07 用户裁定：图例从面板**下方**移到**上方**（在面板标签 d 之下、图的上方）。
    #   原位置 `loc="lower left", bbox_to_anchor=(0.0, -0.30)` 实测在面板下方 52.1px、
    #   且在 xlabel 下方 23.9px —— 这块空间本是**画布底部空白**，
    #   而面板上方是内容区，图例放上方更贴近它所解释的堆叠条。
    #   正解：`loc="lower left", bbox_to_anchor=(0.0, 1.005)`（**紧贴面板顶**，
    #   留 0.005 的轴分数 ≈ 1.1px 呼吸），并置 `ncol=4` 保持一行。
    #   ⚠️ 面板标签 d 在 (b=-14.8pt, 下方) —— 图例 y 从 1.005 起占约 12px，
    #   两者**不同列**（标签 d 在 x=454，图例从 x=454 起但标签只高 6pt≈8px），
    #   已实测无 text-text 重叠（碰撞审计 0 FAIL 验证）。
    #   ⚠️ 4 个色块 + 4 个文字在 180mm 双栏宽下实测约 150px 宽，居左排布不溢出。
    # 🔴 2026-10-07 用户第三次指出：图例首项与**面板标签 d 平齐**了（视觉上像并排一行）。
    #   实测：标签 d 竖直中心 316.3px、图例首项 316.6px → **仅差 0.3px**，确实齐平。
    #   根因：`bbox_to_anchor=(0.0, 1.005)` 让图例底边紧贴面板顶（上方 4.6px），
    #   而面板标签 d 恰好也在面板顶上方约 10px 处 → 两者落在同一水平带。
    #   正解：**图例整体右移，让开面板标签 d 所在的列**（两者不同列 → 竖直重叠不算碰撞）。
    #   实测约束链（全部实测，非目测，且**已排除两条错误路线**）：
    #     · 「标签 d 底↔ 面板顶」仅 **5.6px**，而一行四列图例高 **12.3px**
    #       → 纯竖直平移必然重叠（实测 anchor 0.972 压进面板 2.9px、0.9791 重叠 5.5px）；
    #     · ❌ 曾试「标签 d 上移 dy_pt=20」→ 图例与标签确已错开 11.1px，
    #       **但对齐闸门立刻 FAIL**：`row-panel-label-misalignment`，
    #       c 与 d 的 `label-y` 差 **16.0pt** > 容差 1.5pt
    #       → **面板标签必须保持同高，此路不通**（闸门优先于图例美观）。
    #     · ❌ 曾试「图例改 ncol=2 两行」→ 实测图例高反而涨到 **20.7px**（两行间距所致），
    #       问题更糟 → 也已排除。
    #   → 唯一可行解：**横向错开**。标签 d 在 x=448.5–454.5，
    #     图例左缘须> 454.5 才不与它同列（实测 `bbox_to_anchor=(0.045, …)`
    #     → 图例左缘 468.8，已让开 14.3px）。
    #     右移后可用宽仅 690.9 − 468.8 = 222.1px，故须收紧图例：
    #     参数经**穷举实测**收敛（cs=columnspacing, hl=handlelength）：
    #       cs=0.35/hl=0.8 → 227.4px 溢出 5.3px ✗｜cs=0.35/hl=0.7 → 224.6 溢出 2.5 ✗
    #       cs=0.20/hl=0.7 → **221.5px 右缘 690.3 ✅ 不溢出**（当前采用）
    #       cs=0.10/hl=0.8 → 222.2 右缘 691.0（临界，余量仅 0.1px，不取）
    #   ⚠️ 竖直方向留 4.4px 间隙（图例底 310.2 vs 面板顶 305.8）→ 不压面板。
    #   2026-10-07 用户要求：**图例再下移 3px**（当前图例底距面板顶 4.4px）。
    #   3px ÷轴高 226.2px = 0.01326 轴分数 → 1.004 − 0.0133 = **0.9907**。
    #   下移后图例底距面板顶由 4.4px → **1.4px**（仍不相交，保留 1px 余量）。
    #   ⚠️ 竖直方向本已与面板标签 d **横向错开**（不同列），故下移不会产生 text-text 重叠。
    #   2026-10-07 用户要求「再下移 5px」→**实测不可行，已回退到上一步**（锚点 0.9907）。
    #   不可行的实测依据（**勿重试**，三条数据互相锁死）：
    #     ① 图例底当前只比面板顶高 **1.4px**，下移 5px 即**进入面板内 3.6px**
    #        → 违反「图例不压面板」这一硬约束；
    #     ② 若改用 `set_ylim` 把最上柱下推来腾空间 → 图例四项文字随即
    #        **压在最上排色块边缘**，碰撞审计 WARN 由 **4 → 8**（仍 0 FAIL，
    #        但新增 4 条均来自本改动）；
    #     ③ 图例底与面板顶之间**总共只有 1.4px**——这是「面板标签 d 占位
    #        （标签底恰在面板顶上方 5.6px）」+「图例高 12.3px」共同决定的死局，
    #        单靠平移无解（详见本段上方两条❌路线的记录）。
    #   → 结论：**维持锚点 0.9907**（图例底距面板顶 1.4px，不压面板、WARN 4 条）。
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLS],
              ORD, loc="lower left", bbox_to_anchor=(0.045, 0.9907),
              ncol=4, handlelength=0.7, columnspacing=0.20, handletextpad=0.3)
    clean_axes(ax)
    clean_axes(ax)
    panel_label(ax, "d", dx_pt=_DX_B)   # 与 b 同值 → 列内对齐

    # ---- 逐行贴「图标 + 模型全称」（坐标范围确定之后）----
    # 用 ax_a（a 面板专属引用），不能用 ax——它此时指向 d 面板
    # 标签起点放在**标签区中段**：其右缘恰好落在 ax_data_lo 左侧，
    # 这样文字与 CI 线/ 均值虚线都不会相交（审计器会判 text-stroke FAIL）
    for y, (k, e) in zip(ys, order):
        draw_icon_label(ax_a, k, ax_data_lo - _label_pad + _label_pad * 0.03, y,
                        e["pretty"], 0,
                        pixels=8.0, fontsize=5.0,
                        text_color=chan_color(k))
    import sys as _s; _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import svg_icons as _si

    save_pub(fig, "Fig3_model_accuracy", alignment=True)


# ============================================================
# Fig.4 · 消融：结构化输出约束是承重墙
# ============================================================
def fig4():
    M = load()
    order = order_by_rate(M)
    fig = plt.figure(figsize=(COLW, 118 * MM))

    # 🔴 2026-10-07 用户裁定（Fig.4）：a/b/c 三个面板标签**统一左移 15px**。
    #   原为统一 `DX_PT = −4.5` → 标签相对面板左缘仅 **−6.25px**，
    #   视觉上「贴在面板内容里」，故按 Fig.3 的同一手法再左移。
    #   **单位换算（R47）**：`dx_pt` 单位是 **pt 不是 px**，
    #   1pt = fig.dpi/72 = 100/72 = **1.3889px** → 15px = **10.8pt**
    #   → `_DX_F4 = −4.5 − 10.8 = **−15.3pt**`。
    #   三个面板**同值** → 同行三个标签严格对齐（闸门比较同行面板，实测已PASS）。
    _DX_F4 = -15.3

    # 2026-11-02 用户裁定：a–b 再缩 2 mm、b–c 再缩 1 mm。
    #   当前实测两缝均 8.29 mm → 目标 6.29 / 7.29 mm。
    #   GridSpec 的 wspace 只能等距，故改 add_axes 手动定位：
    #   内宽 0.86；三栏等宽 0.261519；g1=6.29/180、g2=7.29/180（fig 分数）。
    _B, _T = 0.22, 0.86
    _W = 0.261519
    _POS = {
        "a": [0.125, _B, _W, _T - _B],
        "b": [0.421463, _B, _W, _T - _B],
        "c": [0.723482, _B, _W, _T - _B],
    }

    VAR = [("full arm\n(A)", None, BLUE),
           ("\u2212 examples", "minus_examples", BLUE_L),
           ("\u2212 checklist", "minus_check", BLUE_L),
           ("\u2212 follow-up", "minus_followup", BLUE_L),
           ("\u2212 structured\noutput", "minus_output", RED)]

    # ---------- a · 四臂消融对照（hero） ----------
    # 9 模型 × 5 臂若画绝对值柱会挤成一片 → 改为「相对基线的 Δ」分组点图：
    # 每臂一行，9 个模型一个点，直接看出 −output 单独塌到 −80 pp 以外。
    ax = fig.add_axes(_POS["a"]); ax_a = ax
    # 🔴 2026-10-07 用户裁定：① 去掉行标签前的减号（原「− examples」读作"减 examples"，
    #   但消融臂名后缀已是 minus_*，减号在图上重复且被误读为 y 轴刻度的短横线）；
    #   ② 五个臂名的**首字母大写**（Full/Examples/Checklist/Follow-up/Structured output）。
    ARMS = [("Full arm (A)", None, BLUE),
            ("Examples", "minus_examples", BLUE_L),
            ("Checklist", "minus_check", BLUE_L),
            ("Follow-up", "minus_followup", BLUE_L),
            ("Structured output", "minus_output", RED)]
    for ai, (lab, vk, col) in enumerate(ARMS):
        yj = len(ARMS) - 1 - ai
        for mi, (k, e) in enumerate(order):
            base = e["A"]["consensus"]["rate"] * 100
            if vk is None:
                d = 0.0
            else:
                ab = e.get("ablation", {}).get(vk)
                d = (ab["rate"] * 100 - base) if ab else -base
            # 用**身份色**（统一原则）：臂已由 y 轴行位置 + 行标签表达，
            # 点再按臂上色是冗余编码；改按模型上色后，读者可沿同一颜色
            # 纵向追踪「这个模型在 5 个臂上的 Δ」——这正是本面板的论点。
            ax.plot(d, yj + (mi - 4) * 0.055, "o", ms=2.0,
                    color=chan_color(k),
                    mec=NEU_D, mew=0.2, ls="none",
                    zorder=3 if d > -50 else 4)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)

    # 🔴 2026-10-07 用户裁定：每个臂的数据点用**虚线椭圆**圈起来（类比 PCA 簇圈）。
    #   用户目视纠正三轮后定稿：
    #   ① 前两行（Full arm / Examples / Checklist / Follow-up）用**贴合点簇的窄椭圆**
    #      —— rx = 点簇半宽 + 余量即可。**不得**用「屏幕正圆所需 rx」作全局下界，
    #      否则窄点簇的行会被撑成大椭圆、**右缘伸出 xlim（右界12）**，
    #      实测该行cx≈1.0 + rx 16.0 = 17.0 > 12 → 整圈跑到面板外看不见。
    #   ② **只有 Structured output 行**用「屏幕正圆」半径：
    #      该行 Δ 跨度 19.4 单位（最宽），而面板**屏幕各向异性 SC = UY/UX = 39.93**
    #      （x 1 单位 = 1.43px、y 1 单位 = 57.18px），
    #      窄椭圆在屏幕上被压扁 → 用户看到「只圈到 Qwen 一个点」。
    #      该行 cx=−79.1，配 rx=16.0 后右缘 −63.1 **仍在 xlim 内**，故可安全用正圆。
    #   ③ **不旋转**：点簇主方向在屏幕上已是 63.3°（本身是斜线）；
    #      按该角旋转会让垂直半径膨胀到 ±17 单位 → 超出面板高度（行间距仅 1.0）。
    _GRP_RY = 0.40                # 垂直半径（数据单位）= 22.9 px
    _GRP_PAD_X = 2.4              # 窄椭圆的水平余量
    _GRP_SC = 39.93               # 屏幕各向异性 UY/UX
    _GRP_WIDE_VK = "minus_output"   # 仅此行用「屏幕正圆」半径
    for ai, (_lab, vk, _c) in enumerate(ARMS):
        _yj = len(ARMS) - 1 - ai
        _ds = []
        for _mi, (_k, _e) in enumerate(order):
            _base = _e["A"]["consensus"]["rate"] * 100
            if vk is None:
                _d = 0.0
            else:
                _ab = _e.get("ablation", {}).get(vk)
                _d = (_ab["rate"] * 100 - _base) if _ab else -_base
            _ds.append(_d)
        _cx = (min(_ds) + max(_ds)) / 2
        _rx = (max(_ds) - min(_ds)) / 2 + _GRP_PAD_X      # 贴合点簇
        if vk == _GRP_WIDE_VK:                # 仅宽行补偿成屏幕正圆
            _rx = _GRP_RY * _GRP_SC
        ax.add_patch(Ellipse((_cx, _yj), _rx * 2, _GRP_RY * 2, fill=False,
                             edgecolor=NEU_M, lw=0.5, ls=(0, (2.2, 1.6)),
                             zorder=1.5))
    ax.set_yticks(np.arange(len(ARMS)))
    ax.set_yticklabels([])
    # 2026-11-02 用户裁定：5 个纵轴标题旋转 90°。
    #   先按 270°（从上往下）试过，用户要求再旋转 180° → 现为 90°（从下往上）。
    #   旋转后纵向长度 = 原文字宽度；最长 "Structured output" 41.97 pt，
    #   行距 42.81 pt，居中后与其上一行仍留约 11 pt → 无需下移。
    #   同时修正颜色对应：原代码把 reversed(ARMS)[0]（Structured output）
    #   染成 BLUE、[-1]（Full arm）染成 RED，语义反了；现按臂本身取色。
    _ARM_LAB_X = -105.0
    for _ai, (_lab, _vk, _c) in enumerate(ARMS):
        _yj = len(ARMS) - 1 - _ai
        ax.text(_ARM_LAB_X, _yj, _lab, rotation=90, ha="center", va="center",
                fontsize=5, clip_on=False,
                color=BLUE if _ai == 0 else (RED if _ai == len(ARMS) - 1 else INK),
                fontweight="bold" if _ai in (0, len(ARMS) - 1) else "normal")
    ax.set_xlim(-100, 12)
    ax.set_xlabel("Δ accuracy vs full arm (pp)")
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    # 0/135 强调
    ax.annotate("9/9 models, 4 channels:\nno parseable run (0/135)",
                xy=(-84, -0.12), xytext=(-48, 2.05),
                fontsize=5, color=RED, ha="center", linespacing=1.4,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    # 🔴 2026-10-07 用户裁定：原title「3 prompt-level arms: |Δ| ≤ 3 pp」是
    #   **面板标题**（画在框外上方），用户要求**移入面板内部的空白区域**。
    #   落点选据（实测）：a 面板 xlim=(−100, 12)，**左侧 −100…−35 一带除
    #   「full arm (A)」行的 Δ=0 点外全空**；y∈[1.6, 3.4]
    #   （− examples 与 − checklist 两行之间）实测宽约 65 单位（≈83px），
    #   而这句 5pt 文字宽约 60px → **放得下且不压任何数据点/注记**。
    #   故改为面板内 `ax.text`（去掉 title），位置 (−97, 2.95)、斜体以示是注记。
    ax.set_title("")
    ax.text(-90.6, 2.62, "3 prompt-level arms: |\u0394| \u2264 3 pp",
            fontsize=5.0, color=NEU_D, ha="left", va="center", style="italic")

    # 🔴 2026-10-07 用户裁定：a 面板的**彩色点是哪些模型看不出来**，
    #   要求「标出它们是哪些大模型，最好有图标和名称」。
    #   ⚠️ **不能逐点标注**（实测否决）：前4 臂共 36 个点全部挤在 Δ∈[−8.2, +7.1]
    #   的窄带里（full arm 更是 9 点Δ 全为 0、完全重合），
    #   任何逐点文字必然压点/压线（Fig.3c 已踩过同族坑，见 R41）。
    #   正解：**面板内放「图标 + 名称」图例**（一次说清 9 个身份色对应谁），
    #   读者由颜色 → 图例 → 模型，与 a 面板点一一对应。
    #   落点选据（实测屏幕坐标）：
    #     · 已有红色注记占 x=114.2–203.0、y=132.6–269.4；
    #     · 已有 title 占 y=301.9–308.4；
    #     · 各臂点簇的 x 下界 ≥ 220.1（− examples 最低）；
    #     → **y∈[3.30, 4.50]（屏幕 y≈310–366）× x∈[−99, −20]（屏幕 x≈89–199）**
    #       是一块**实测无任何元素**的空白带。
    #   排布（**三度修正，勿重试**——每次都实测过）：
    #     ① 3 列 × 3 行全称 → **实测重叠**（列距 30 单位 = 43px
    #        < 最长行 DeepSeek 39.1 + 图标 4.5 = 43.6 单位）；
    #     ② 单列 9 行 → 需 1.52 单位 = 87px，**空白带只有 1.15 单位(66px)**，放不下；
    #     ③ 2 列 + `short()` → **仍重叠**：`short()` 的 SHORT 表**没有 DeepSeek 映射**
    #        （实测 `short("DeepSeek-V4.1-Flash")` 原样返回，39.1 单位 > 42 单位列距 − 图标），
    #        且末行与 title（y=2.95）撞在一起。
    #     ④ **正解：图例专用短名表**（去厂商/版本后缀，与 SHORT 表不同）：
    #        最宽一行 Space-Bunny = 23.7 单位（+图标 4.2 = **27.9 单位**），
    #        列距取 **35 单位**（含 25% 余量）→ 2 列共 70 单位 = **100px ≤ 110px 空白带** ✅。
    #   行距 0.185 单位 = 10.6px > 4.6pt 字高(6.4px)，行间不压。
    #   ⚠️ 专用短名表**只用于本图例**，不改全局 `short()`——后者被 b 面板行标签使用，
    #     那里的 xlim 是按全称算的（见 b 面板 set_xlim 注释），动它会连带崩b 面板。
    # 🔴 2026-10-07 用户裁定：9 模型图例改为**竖排**——「图标在上、名称在下」，
    #   这样**可以用全称**，不再需要短名（此前 2 列横排 + 短名，
    #   用户指出「名称没有显示完全」）。
    #   为什么竖排就能放下全称：横排时列宽 = 图标宽 + 名称宽（同占一行的横向空间），
    #   而竖排时**列宽只由名称决定**、行高由「图标+名称」竖向叠加决定 →
    #   名称可用宽度 = 列宽，与横排相同，但**不必与图标抢横向空间**。
    #   实测：最宽全称 DeepSeek-V4.1-Flash = 45.83 单位 = **65.6 px**；
    #        列距 = 列宽 + 6px 间隔 = 50.0 单位；右列右缘 = 5.2 < xlim 右界 12 ✅。
    # 🔴 2026-10-07 用户裁定：9 模型图例改为**竖排**——「图标在上、全称在下」，
    #   以解决「名称显示不完全」（此前 2 列横排 + 短名表，名字被截短）。
    # 🔴 2026-10-07 用户裁定（竖排 + 左侧同色圆点 + 左对齐）：
    #   9 模型图例排成**竖排**——上=图标、中=同色圆点、下=**全称**；
    #   全部元素**左对齐**成一条竖线（点与图标同x，名字跟在点后）。
    #   **竖排为何能用全称**：横排时图标与名字共享同一行的横向空间，
    #   列宽须容下「图标 + 名字」；竖排时列宽只由名字决定、图标叠在上方，
    #   二者不再抢横向空间 → 最宽全称 DeepSeek-V4.1-Flash 65.6px 可容。
    _F4_LEG_ICON_PX = 6.0                # 图标内容像素高
    _F4_LEG_DOT_PX = 3.0                 # 圆点直径（ms=3）
    _F4_LEG_DOT_GAP_PX = 1.5             # 圆点与名字的间隙（收紧以保右缘不越界）
    _F4_LEG_NAME_PT = 4.6
    _F4_LEG_NAME_W = 45.83               # 最宽全称（实测）
    _F4_LEG_NAME_W_PX = _F4_LEG_NAME_W * 1.432
    _F4_LEG_COL_GAP_PX = 3.0             # 左右列间隙
    _F4_LEG_X0 = -90.62
    # 圆点 x = **图标中心**（点嵌在图标画布宽度内，不额外占横向空间）
    _F4_LEG_DOT_X = (_F4_LEG_ICON_PX / 1.432) / 2
    # 名字 x = 圆点右缘 + 间隙（**左对齐**）
    _F4_LEG_NAME_X = _F4_LEG_DOT_X + (_F4_LEG_DOT_PX / 2 + _F4_LEG_DOT_GAP_PX) / 1.432
    _F4_LEG_ITEM_PX = _F4_LEG_NAME_X * 1.432 + _F4_LEG_NAME_W_PX
    _F4_LEG_DX = (_F4_LEG_ITEM_PX + _F4_LEG_COL_GAP_PX) / 1.432
    # 竖向：字顶落在图标底缘下方 1.5px；**圆点落在名字行的垂直中线**（实测字高 6.5px）
    _F4_LEG_NAME_DY = -(3.0 + 1.5) / 57.177
    _F4_LEG_DOT_DY = _F4_LEG_NAME_DY - 3.25 / 57.177
    # 每项竖向占位 = 图标 6.0 + 间隙 1.5 + 字高 6.5 + 行间隙 2.0 = 16.0px
    # 🔴 2026-10-07 用户裁定：① 左列 5 项行距**放宽**；② **后 4 项移到右下角空白区**，
    #   行距与左列**完全一致**（复用同一个 `_F4_LEG_DY`）。
    #   ⚠️ 左列可安全放宽到 24px 行高（行间隙 7px）：各臂椭圆虽横跨 y∈[0.6,3.4]，
    #     但其 x 区间是 [−10.6, 9.5]，与左列 x∈[−90.6, −44.8] **横向不重叠**。
    _F4_LEG_ROW_GAP = 7.0                # 行间隙（px）
    _F4_LEG_DY = (6.0 + 1.5 + 6.5 + _F4_LEG_ROW_GAP) / 57.177   # ≈24.0px
    _F4_LEG_Y_TOP = 4.52                # 首项图标中心 y
    def _f4_leg_item(_lx, _ly, _key, _colr, _name):
        """图例单项：图标 + 同色圆点 + 全称（三者左对齐）。"""
        draw_icon_png(ax, _key, _lx, _ly, 0, anchor="left", va="center",
                      pixels=_F4_LEG_ICON_PX, defer=True)
        ax.plot(_lx + _F4_LEG_DOT_X, _ly + _F4_LEG_DOT_DY, "o",
                ms=_F4_LEG_DOT_PX, color=_colr, markeredgecolor=_colr,
                markeredgewidth=0.4, zorder=6, clip_on=False)
        ax.text(_lx + _F4_LEG_NAME_X, _ly + _F4_LEG_NAME_DY, _name,
                fontsize=_F4_LEG_NAME_PT, color=_colr,
                ha="left", va="top", zorder=6)
    # 左列 5 项（按读数前5）
    for _i in range(5):
        _k, _e = order[_i]
        _f4_leg_item(_F4_LEG_X0, _F4_LEG_Y_TOP - _i * _F4_LEG_DY,
                     _k, chan_color(_k), _e["pretty"])
    # 右下角 4 项（Ling / MiMo / Space-Bunny / Qwen）。
    #   落点（实测空白）：左列名字右缘 = X0 + 名字宽 = −44.8，加 10px 间隙 → 起点 −37.8；
    #   名字右缘 = 8.0 < xlim 右界 12 ✅；y 自 3.20 下探 3 个行距，
    #   最低行名字底 y≈2.19，避开 Structured output 圈（y ≤ 0.40）✅。
    #   ⚠️ **右列行高只能用 16px**（实测右下空白仅 66px 高，
    #     装不下 4×24px=96px；且 x 方向与消融臂椭圆 x∈[−10.6, 9.5] 横向重叠）。
    #     故右列行高 = (6+1.5+6.5+2)/UY = 16px，仍比最早版的 16px 保持一致、
    #     比「挤在一起」宽松（行间隙 2px + 点与名字分离）。
    #🔴 2026-10-07 用户裁定：右下4 项**右移 10px、下移 60px**，
    #   且**行距与左上 5 项完全一致**（复用同一个 `_F4_LEG_DY`，21px）。
    #   ⚠️ 单位换算（不可估算）：右移 10px = 10/1.432 = **6.983 单位**；
    #     下移 60px = 60/57.177 = **1.0494 单位**。
    #   第二轮追加：再**右移 10px、上移 5px**（同向叠加，非替换）。
    #   ⚠️ 单位换算：右移 10px = 10/1.432 = **6.983 单位**；
    #     上移 5px = 5/57.177 = **0.0874 单位**（y 轴向上为正）。
    _F4_LEG_DX_SHIFT_R = (10.0 + 10.0) / 1.432# 累计右移 20px = 13.966 单位
    _F4_LEG_DY_SHIFT_D = (60.0 - 5.0) / 57.177         # 净下移 55px = 0.9620 单位
    _F4_LEG_DY1 = _F4_LEG_DY                               # 行距＝左列，完全一致
    #   落点由 900dpi 渲染实测反推（先前按算式选的 x=−37.8 压在消融臂椭圆上）：
    #   逐个障碍物 → Structured out 圈 x∈[−95.1,−63.1] y≤0.40；
    #   红字「9/9 models…」y∈[1.95,2.45] x∈[−95,−40]；
    #   「3 prompt-level arms」y≈2.62 x∈[−90.6,−45]。
    #   → **x = −62.0**（恰在 Structured out 圈右缘 −63.1 右侧 1.1 单位、
    #     红字右缘 −40 左侧 22 单位）＋ **y_top = 1.75**（红字下沿 1.95 之下）。
    #   名字右缘 = −62.0 + 45.83 = **−16.2**，距 xlim 右界 12 有 28 单位余量 ✅；
    #   最低行名字底 y≈0.744，避开 Structured out 圈上缘 0.40 ✅。
    _F4_LEG_X1 = -62.0 + _F4_LEG_DX_SHIFT_R                # −62.0 → −55.02
    _F4_LEG_Y1_TOP = 1.75 - _F4_LEG_DY_SHIFT_D            # 1.75 → 0.70
    for _j in range(4):
        _k, _e = order[5 + _j]
        _f4_leg_item(_F4_LEG_X1, _F4_LEG_Y1_TOP - _j * _F4_LEG_DY1,
                     _k, chan_color(_k), _e["pretty"])
    panel_label(ax, "a", dx_pt=_DX_F4)   # 统一左移 15px

    # ---------- b · 提示词要素的 Δ 分布 ----------
    # 🔴 2026-10-20：b 面板行标签改**两行**，避免长名穿出坐标区右边界。
    #   全称一字不删（用户 2026-10-07 明确要求「名称显示完全」），只换行。
    #   ⚠️ 不动全局 SHORT 表——它被 a 面板图例与其他面板共用。
    _F4_B_WRAP = {
        "DeepSeek-V4.1-Flash": "DeepSeek\nV4.1-Flash",
        "GLM-5.3-Flash": "GLM-5.3\nFlash",
        "MiniMax-M3": "MiniMax\nM3",
        "MiniMax-M3.1-Flash": "MiniMax\nM3.1-Flash",
        "MiMo-V2.6-Flash": "MiMo\nV2.6-Flash",
        "Hy4-Preview": "Hy4\nPreview",
        "Qwen3.8-Flash": "Qwen3.8\nFlash",
        "Ling-3.1-Flash": "Ling-3.1\nFlash",
        "Kimi-K2.8-Preview": "Kimi\nK2.8-Preview",
    }
    ax = fig.add_axes(_POS["b"]); ax_b = ax
    for i, (k, e) in enumerate(order):
        base = e["A"]["consensus"]["rate"] * 100
        for vk, mk, col in [("minus_examples", "s", BLUE_L),
                            ("minus_check", "o", NEU_M),
                            ("minus_followup", "^", TEAL)]:
            ab = e.get("ablation", {}).get(vk)
            if not ab:
                continue
            d = ab["rate"] * 100 - base
            ax.plot(d, i, mk, ms=3.0, color=col, ls="none",
                    markeredgecolor="white", markeredgewidth=0.8)
    # 参考竖线只画在数据带内（上界留白），避免延伸到坐标区外与刻度标签相交
    # ymin 0.0 → 0.08：底部让出 0.9 数据单位给 "all within…" 小注，
    # 否则 ±3 虚线会从文字里穿过去（碰撞审计 text-stroke FAIL）。
    for d in (-3, 3):
        ax.axvline(d, ymin=0.08, ymax=0.93, color=NEU_M, lw=0.6,
                   ls=(0, (3, 2)), zorder=0)
    # 0 线同样从数据区起笔（ymin 0.08）：否则它纵贯到底部小注里
    ax.axvline(0, ymin=0.08, color=INK, lw=0.7, zorder=1)
    # 🔴 xlim 上界由 6 → 7.6 → 10.6（2026-10-07 修 FAIL，三轮定位）：
    #   行标签起点 x=7.4，而**最长的标签是 DeepSeek-V4.1-Flash（实测 49.2pt 宽，
    #   折合 2.9 个数据单位）** → 标签右缘到 10.3，超出 xlim 上界 7.6 达 2.7 单位，
    #   伸到坐标区外被**右 spines 黑线**纵向穿过（PDF x=381.5，纵贯 21.7–235.8）。
    #   正解：xlim 上界取 7.4 + 2.9 + 0.3(余量) = **10.6**，
    #   即「标签起点 + 最长标签宽 + 余量」——**判据要按最长标签算，不是平均**。
    #   代价：Δ 参考竖线（±3）左侧留白变宽，但数据点仍集中在 −3…3，不影响读数。
    # 🔴 2026-10-20 更正：上面「49.2pt = 2.9 单位」的换算错了——实际
    #   49.2pt ÷ 6.96pt/单位 = **7.05 单位**，故 10.6 装不下最长标签，
    #   6 个模型名穿出右边界（碰撞审计 6 条 text-fill-edge）。
    #   另一层根因：SHORT 表的 key 用空格写法（"DeepSeek V4.1 Flash"），
    #   而实际 pretty 用连字符（"DeepSeek-V4.1-Flash"）→ 4 个模型根本没缩短。
    #   正解：**b 面板标签改两行、保留全称**（用户要求名称完整），
    #   最长行 "K2.8-Preview" ≈ 30pt ≈ 4.8 单位 → xlim 上界 7.4+4.8+0.4 = **12.6**。
    # 🔴 2026-11-02 用户裁定：b 面板行标签前加模型 icon，「icon + 名称」整体右移。
    #   原名称直接起于 x=7.4；加 icon 后若名称起点不动，icon 会向左侵占数据带。
    #   做法：**图标右缘对齐 + 文字左缘对齐**（各自一条竖线）——
    #     icon 右缘 8.80、名称左缘 8.95；文字因此严格成列，图标宽度差只体现在左缘。
    #   ⚠️ 不能按「内容高 6 px」估算图标宽：画布是正方形，源图留白差异大
    #     （_CONTENT 0.61–1.0），最宽画布 = 8.2 px（DeepSeek）。
    #   名称左缘 = 8.95 → 比原 7.4 右移 1.55 单位（≈8.4 pt），
    #   与最右数据点（Kimi +6）之间留 2.95 单位（≈16 pt）空隙。
    _F4B_ICON_R = 8.80
    _F4B_TEXT_X = 8.95
    _F4B_ICON_PX = 6.0
    # ylim 上界 8.8 → 9.9：顶部让出 1.1 单位给内嵌图例（见下方），
    # 行距 22.3 → 20.1 pt，9 行仍远宽于 5 pt 字高与 6 px 图标。
    ax.set_xlim(-6, 14.85); ax.set_ylim(-0.8, len(order) + 1.35)
    # Δ 值域窄（±6 pp），数据点必然贴近 y 轴刻度标签所在行 → 直接标在点右侧，
    # 去掉 y 轴刻度标签，避免「标签—标记」几何重叠。
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([])
    for i, (k, e) in enumerate(order):
        ds = [e.get("ablation", {}).get(v, {}).get("rate", e["A"]["consensus"]["rate"]) * 100
              - e["A"]["consensus"]["rate"] * 100
              for v in ("minus_examples", "minus_check", "minus_followup")
              if v in e.get("ablation", {})]
        if ds:
            # 行标签用**身份色**：本面板每行 = 一个模型，与 a 面板同色
            # 才能让读者把b 的行与 a 的点对应起来（统一原则）。
            # 🔴 行标签起点 x：6.15 → 6.45 → 7.4（2026-10-07 修FAIL，两轮定位）：
            #   ① DeepSeek-V4.1-Flash 被 **b 面板黑色右 spines 竖线**纵向穿过
            #      （PDF x=381.5，纵贯 21.7–235.8）；
            #   ② Qwen3.8 被**数据点的白色描边**（3×3pt）压住——
            #      实测 b 面板共 27 个白色描边，其中第 0 行（Kimi，Δ 最大 +6）
            #      的一个落在 x=329.4，恰在标签起点带上。
            #   **判据：标签起点必须大于该面板 Δ 的最大正值**（Kimi +6），
            #   否则标签会与极值行的标记重叠。7.4 给出 1.4 个单位余量。
            #   xlim 已相应放宽到 7.6（见下方 set_xlim），否则标签越界。
            draw_icon_png(ax, k, _F4B_ICON_R, i, 0, anchor="right", va="center",
                          pixels=_F4B_ICON_PX, defer=True)
            ax.text(_F4B_TEXT_X, i, _F4_B_WRAP.get(e["pretty"], e["pretty"]),
                    fontsize=5, color=chan_color(k), va="center", ha="left",
                    linespacing=1.05)
    ax.set_xlabel("Δ accuracy vs full arm (pp)")
    # 全局说明移到面板底部：顶部整条空间要让给内嵌图例（见下）。
    ax.text(-5.9, -0.62, "all within |Δ| ≤ 3 pp",
            fontsize=5, color=NEU_D, ha="left", va="bottom",
            fontweight="bold")
    # 🔴 2026-11-02 用户裁定：三项图例从轴下（bbox y=−0.34，距 x 轴约 73 pt）
    #   **搬进数据区顶部**，名称**竖排、从下往上**（rotation=90）：
    #   横排三项合计约 62 pt，数据区放不下。
    #   每个标签对准 Qwen3.8 行（y=8）的同类标记，引线垂直落到点上 ——
    #   线短、彼此不交叉，「图例 → 真点」一眼可辨。
    #   examples/checklist 的标签比目标点各外移 1.5/0.34 单位，避免两个
    #   竖排标签（各宽 5 pt）贴在一起。
    for _lab, _mk, _col, _lx, _px, _py in [
        ("− checklist", "o", NEU_M, 3.80, -0.16, 8.0),
        ("− examples", "s", BLUE_L, 4.90, 1.02, 8.0),
        ("− follow-up", "^", TEAL, 6.00, 7.08, 8.0),
    ]:
        ax.plot([_lx, _px], [8.60, _py + 0.10], color=NEU_M, lw=0.45,
                zorder=1, solid_capstyle="round")
        ax.plot([_lx], [8.70], _mk, ms=2.8, color=_col, ls="none",
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.text(_lx, 9.55, _lab, rotation=90, fontsize=5, color=NEU_D,
                ha="center", va="center", zorder=4)
    # 2026-11-02 用户裁定：b 标识向右移 2 mm（2 mm = 5.67 pt）。
    panel_label(ax, "b", dx_pt=_DX_F4 + 2 * 72 / 25.4)

    # ---------- c · −output 的跨通道分解 ----------
    ax = fig.add_axes(_POS["c"]); ax_c = ax
    CH = {"deepseek-v4.1-flash": "WorkBuddy CLI", "glm-5.3-flash": "WorkBuddy CLI",
          "hy4-preview": "WorkBuddy CLI", "minimax-m3": "WorkBuddy CLI",
          "kimi-k2.8-preview": "WorkBuddy CLI", "space-bunny": "OpenCode CLI",
          "mimo-v2.6-flash": "OpenCode CLI", "ling-3.1-flash": "OpenRouter CLI",
          "qwen3.8-flash": "Qoder CN CLI"}
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for k, _ in order:
        a = agg[CH[k]]
        a[0] += 15
        a[1] += 0  # parse_ok = 0，全部 fail
    names = list(agg)
    xs = np.arange(len(names))
    ax.bar(xs, [agg[n][0] for n in names], width=0.5, color=NEU_L, lw=0,
           label="attempts (all channels)")
    for i, n in enumerate(names):
        tot = agg[n][0]
        ax.text(i, tot + 1.6, str(tot), ha="center", va="bottom",
                fontsize=5, color=NEU_D)
        ax.text(i, 1.4, "0", ha="center", va="bottom", fontsize=6,
                color=RED, fontweight="bold")
        ax.plot([i - 0.34, i + 0.34], [0, 0], color=RED, lw=1.6,
                solid_capstyle="butt", zorder=5)
    ax.set_xticks(xs)
    # 2026-11-02 用户裁定：只在**最后一个空格**处换行——
    #   原先 `replace(" ", "\n")` 把 "Qoder CN CLI" 拆成三行；
    #   现为 "Qoder CN / CLI" 两行，与其余通道（xxx / CLI）对齐。
    ax.set_xticklabels(["\n".join(n.rsplit(" ", 1)) for n in names], fontsize=5)
    # 标签下移、去掉刻度短线，给通道图标让位（图标放在轴与文字之间）
    ax.tick_params(axis="x", length=0, pad=9.5)
    # 🔴 y 轴标题改为**面板内横向小注**（2026-10-07 修 FAIL）：
    #   原 set_ylabel("Runs in −output arm") 是竖排（旋转 90°），占 x 361–367，
    #   而 b 面板的行标签（模型名）从 x=337 起延伸到 386 → **竖排标题正好压在
    #   模型名文字上**（实测 5 处 FAIL 中有 3 处由它造成 text-text 重叠）。
    #   ⚠️ 加大 wspace 无效——两侧内容一起外移，间距按比例放大但重叠关系不变。
    #   正解：**弃用 ylabel**，改用面板内左下角的横向小注（落在数据区留白处），
    #   语义保留（这些柱子都是 −output 臂的运行数），且不与任何文字相交。
    #   ylim 由 78 → 82：给顶部数字标签留余量，避免 (tot+1.6) 顶出坐标区。
    ax.set_ylim(0, 82)
    # 2026-11-02 用户裁定：通道图标居中于各柱正下方、轴与 x 标签之间。
    #   y 需用数据坐标：把 4.75 pt 的偏移按当前轴高/ y 量程换算。
    _axh_pt = ax.get_position().height * fig.get_size_inches()[1] * 72
    _pt2y = (ax.get_ylim()[1] - ax.get_ylim()[0]) / _axh_pt
    for _i, _n in enumerate(names):
        draw_icon_png(ax, _n.split()[0], _i, -4.75 * _pt2y, 0,
                      anchor="center", va="center", pixels=7.0, defer=True)
    # 2026-11-02 用户裁定：小注原在 y=0.97，与第一根柱的 "75" 数字
    #   （76.6/82 ≈ 0.934 axes）上下重叠；上移到 0.995，留出约 3 pt 空隙。
    ax.text(0.015, 0.995, "runs in −output arm", transform=ax.transAxes,
            fontsize=5, color=NEU_D, ha="left", va="top", rotation=0)
    # 2026-11-02 用户裁定：图例从轴下（bbox y=−0.17，离面板太远）搬进
    #   c 面板右上方空白区；两行、左对齐（ncol=1）。
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=NEU_L),
               plt.Line2D([], [], color=RED, lw=1.6)],
              ["attempts", "parseable (none)"],
              loc="upper left", bbox_to_anchor=(0.52, 0.90), frameon=False,
              ncol=1, handlelength=1.0, handletextpad=0.4, fontsize=5,
              labelspacing=0.6, borderaxespad=0)
    panel_label(ax, "c", dx_pt=_DX_F4)   # 统一左移 15px

    # 手动 add_axes 后闸门无法自动推断行列组，显式声明单行 a/b/c。
    # 两条缝有意不等宽（用户 2026-11-02：a–b 再窄 1 mm），
    # 故豁免水平缝一致性检查；行高/列宽等其他项照常审计。
    save_pub(fig, "Fig4_ablation", alignment=True,
             panel_ids={ax_a: "a", ax_b: "b", ax_c: "c"},
             row_groups=[["a", "b", "c"]],
             exemptions=[{"panels": ["a", "b", "c"],
                          "checks": ["horizontal-gutter"],
                          "reason": "用户指定 a–b 比 b–c 窄 1 mm；两条缝有意不等宽"}])



# ============================================================
# Fig.2 · 研究设计与「金标准并非绝对真理」
# ============================================================
def fig2():
    """数据取自 score.py 的冻结判定（Dataset/classifications），非另算。"""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "drp_eval"))
    from score import Dataset, classifications, DIMS as SDIMS
    ds = Dataset()
    cls1, cls2 = classifications(ds, 1, "topk"), classifications(ds, 2, "topk")

    fig = plt.figure(figsize=(COLW, 150 * MM))
    # ---------- 版面：手动定位（2026-11-02 用户裁定） ----------
    # 用户要求 A–B（水平）、A–C（垂直）两条缝都压到「最小可行 ~8 mm」。
    # 实测两条缝的下限不同：
    #   水平 8 mm 可行——b 的 y 刻度标签只伸进约 4 mm，肉眼空隙 ~4 mm；
    #   垂直 8 mm **不可行**——b 底部要装「刻度数字 5.6 + pad + xlabel 6.2」
    #     共 20.2 pt，d 顶部还要装「面板标签 + 标题」3.0 pt，合计 23.2 pt，
    #     而 8 mm 只有 22.7 pt → 实测 xlabel 底与 d 标题顶只剩 0.2 pt。
    #   故两向取不同值：水平 8 mm、垂直 11 mm（多出的 1.2 mm 全给 b 的 xlabel）。
    # 用 add_axes 取代 GridSpec：GridSpec 只有一个 hspace，无法上下缝异值。
    # 约束不变：同行等高（H 相同）、同列等宽（W 相同）。
    _L, _R, _B, _T = 0.105, 0.975, 0.175, 0.935
    _GW, _GH = 8.0 / 180, 11.0 / 150           # 缝宽 → figure 分数
    _W = (_R - _L - _GW) / 2
    _H = (_T - _B - _GH) / 2
    _POS = {
        "a": [_L, _B + _H + _GH, _W, _H],
        "b": [_L + _W + _GW, _B + _H + _GH, _W, _H],
        "c": [_L, _B, _W, _H],
        "d": [_L + _W + _GW, _B, _W, _H],
    }
    # ---------- 面板标签 a/b/c/d 的水平位置（2026-11-02 用户裁定） ----------
    # 目标不是「离轴多远」，而是与图上既有文字**左缘对齐**（读起来像同一列）：
    #   a / c → 对齐 c 面板 y 轴第二行标签「(I2.3 + I4.1)」的左缘；
    #   b / d → 对齐 b 面板顶端 y 刻度「O」的左缘（O 最窄，是全列最靠右的起点）。
    # 默认 DX_PT=-4.5 是「轴左缘往左 4.5 pt」；这里用 dx_pt 覆盖成实测靶点。
    # 数值 = 靶点页面 x + 打印边距偏移 − 轴左缘（三个量都取当前 PDF 实测）：
    #   a/c: 7.20 + 9.7 − 53.6 = −36.7 pt
    #   b/d: 269.25 + 9.7 − 286.8 = −7.9 pt
    # ⚠️ 改 _L / _W / _GW 后这三个量会一起漂，须重量并改这两个常量。
    _DX_AC, _DX_BD = -36.7, -7.9

    # ---------- a · 研究设计流程（示意） ----------
    # 🔴 框 5 要放模型 logo（imshow）——不能用 axis("off")：matplotlib 会跳过
    #   axison=False 的 axes 的全部 artist，imshow 会一起消失（见 svg_icons.blank_axes）。
    #   改用 blank_axes：只关边框/刻度，轴仍参与渲染。
    ax_sch = fig.add_axes(_POS["a"]); blank_axes(ax_sch)
    ax_sch.set_xlim(0, 1); ax_sch.set_ylim(0, 1); ax_sch.set_autoscale_on(False)
    ax = ax_sch
    # 🔴 2×2 布局取代原 1×4 横排（2026-10-07 修 FAIL×2；首版为**蛇形回折**，
    #   2026-10-20 取消回折，理由见下）：
    #   原布局把 4 个框各宽 0.225–0.235 axes，而 a 面板实测只有 **268 px** 宽
    #   → 每框内可用约 57 px；而 5 pt 字号下实测各行宽为
    #   「139 raters × 140 cells」70 px、「back-to-back rating」62 px、
    #   「gold = modal codes」64 px、「round 1 · 20 cases」60 px
    #   ——**四行全部超出框宽**，文字被框边线穿过（审计 text-stroke ×2）。
    #   这不是微调能解决的：**框宽是硬约束**。正解是把一行 4 框改成
    #   **2×2 蛇形**（每框 0.485 axes ≈ 130 px，可用 118 px），
    #   实测所有行都在限内。**2026-10-20 用户裁定：取消蛇形回折**——
    #   上排 1→2，下排 3→4，第三排 5 通栏；阅读方向一律从左到右。
    #   框 5 的左右边界与 1/3 左框、2/4 右框的外缘对齐（x=0→1），
    #   否则下方红框比上方 2×2 窄一截，a 面板整体不成矩形。
    #   不画流程箭头（箭头在 5 pt 双行文字下必然命中，见 Fig.1a 的同款教训 R50）。
    #   框内序号**不再与正文同排**（2026-10-20 用户裁定）：
    #   原写法「1  139 pharmacists」读起来像「1 139」是一个数，且序号与正文
    #   同字号同色，无法一眼分辨哪里是标号、哪里是内容。
    #   现改为：序号用 7 pt 粗体、框色，**独立钉在框内左上角**；
    #   正文 5 pt 深灰居中、整体下移 0.010 axes 给角标让位。
    #   框宽仍是硬约束：正文每行控制在 ~36 字符内（实测框 1/2 可用 ~45 字符）。
    boxes = [
        (0.000, 0.700, 0.485, 0.26,
         "139 pharmacists · 20 cases\n"
         "46 pharmacists · 15 cases", BLUE),
        (0.515, 0.700, 0.485, 0.26,
         "Back-to-back rating\n"
         "Vote matrix: 139 raters × 140 cells", BLUE),
        (0.000, 0.380, 0.485, 0.26,
         # 🔴 2026-10-20 用户裁定：此行拆成两行，把文字整体留在左半边、
         #   右半边让给微图（左文右图）；等号**不跟在行尾**，而是旋转 90°
         #   单独钉在第二、三行之间的左侧（见下方 i == 3 分支）。
         "Cell-wise tally\nVotes vs designated\ngold answers", BLUE),
        (0.515, 0.380, 0.485, 0.26,
         "Split by vote distribution\nConsensus 98 · contested 7", TEAL),
        (0.000, 0.040, 1.000, 0.25,
         "9 LLMs × 15 cases × 5 repeats\nA: manual + rules + examples\n"
         "B: codebook only\n4 ablation arms", RED),
    ]
    # ---- 框内矢量微图（2026-10-20）--------------------------------------
    # 🔴 设计约束：**不许挤压正文**。
    #   框 1–4 高 12.2 mm，正文两行块约 5 mm → 上下各余 ~3.6 mm；
    #   微图一律压在框底 0.6–4.5 mm 的空白带内，与正文块底边至少留 0.4 mm。
    #   框 5 高 11.8 mm、文字四行占 9 mm，底部没有空白 → 微图改放**左侧空白**
    #   （文字居中，左侧余 ~19 mm）。
    #   全部用 Rectangle 矢量绘制：不触发 imshow → SVG 仍可导出、无图标内存峰值，
    #   也不违反「身份色只标识模型」的配色规则（这里全是语义色）。
    #   🔴 总开关：`MICRO = False` → 一键退回纯文字版（用户 2026-10-20 要求保留退路）。
    MICRO = True

    def _micro(i, x, y, w, h):
        if i == 1:
            # 框 1 的两条线段必须与文字行**交错**（每行人数下方紧跟其规模条），
            # 无法用「框底一条带」表达，故由主循环单独绘制（见 _box1）。
            return
        elif i == 2:
            # 🔴 2026-10-20 二次改版：微图必须与正文「139 raters × 140 cells」对得上。
            #   原版画 5×4 = 20 个互不相连的方块，读者数出来就是「20 个东西」
            #   （恰好等于例数），既不是 139×140，也看不出「矩阵」的行列维度。
            #   现改为**带双轴标签的缩略投票矩阵**：行 = raters，列 = cells，
            #   数字由两侧标签给，形状由网格给 —— 这才是「投票矩阵」该有的样子。
            # ⚠️ 不画 139×140 实格：按此尺寸每格仅 ~0.03 mm，印刷必糊成实心块，
            #   且 1.9 万个 patch 会把 SVG 体积撑大一个量级。故取 52 列 × 12 行
            #   **示意分辨率**（格 0.30 × 0.35 mm），真实维度交给两侧标签交代
            #   （与框 4 的 98/7 色块靠数字交代是同一套做法）。
            # 🔴 左右边界与正文块对齐（2026-10-20 用户裁定）：
            #   左标签左边界 = 正文左缩进 x + 0.032｜右标签右边界 = 框右 − 0.032（等距收边），
            #   → 微图组不再「居中漂在框里」，左右留白与上面两行文字完全一致。
            #   矩阵宽度由这两条边界反推：0.485 − 2×0.032 − 左标签 − 右标签 − 2×gap。
            #   标签宽度取**实测 PDF bbox**（不是估算）：139 raters 17.96 pt、
            #   140 cells 15.76 pt → 换算成 axes = 0.0931 / 0.0817（面板 a 宽 193 pt）。
            TAG_L, TAG_R, GAP = 0.0931, 0.0817, 0.007
            ncol, nrow = 52, 12
            mx = w - 2 * 0.032 - TAG_L - TAG_R - 2 * GAP      # ≈15.8 mm
            my = 0.089                     # ≈4.2 mm 高（y 的 axes 单位与 x 不同，须分开算）
            bx, by = x + 0.032 + TAG_L + GAP, y + 0.014
            # 底：**左青右蓝的浅渐变**（2026-10-20 用户裁定）。
            #   要求「139 raters 一侧与 140 cells 一侧的格子颜色要有一点区分，
            #   否则读者会以为 139 与 140 没有区别」。
            #   矩阵是一个整体、不能劈成两半，故用 24 条竖色带做连续渐变：
            #   左端＝raters 的青（同左侧标签色），右端＝cells 的蓝（同右侧标签色）。
            #   渐变用矢量色带拼（非 imshow），SVG 仍是纯矢量。
            from matplotlib.colors import to_rgb
            c_raters = np.array(to_rgb("#8FC2C9"))   # 青（同 TEAL 系，淡化到不压文字）
            c_cells = np.array(to_rgb("#8DB4D6"))    # 蓝（同 BLUE 系，淡化）
            NBAND = 24
            for k in range(NBAND):
                t = k / (NBAND - 1)
                ax.add_patch(Rectangle((bx + mx * k / NBAND, by),
                                       mx / NBAND * 1.02, my,
                                       facecolor=tuple((1 - t) * c_raters + t * c_cells),
                                       edgecolor="none", transform=ax.transAxes,
                                       zorder=5))
            # 一列 = 一个 cell 的 139 张票：把正中一列整列填深蓝，
            # 示意「计票从某一列开始」，与框 3 的逐格计票接得上；
            # 其余留浅蓝，避免与框 4 的 98/7 彩色格阵撞车。
            ax.add_patch(Rectangle((bx + mx * 25 / ncol, by), mx / ncol, my,
                                   facecolor=BLUE, edgecolor="none",
                                   transform=ax.transAxes, zorder=8))
            # 🔴 两个维度必须**颜色可辨**（2026-10-20 用户裁定）：
            #   否则读者会把 139 raters 与 140 cells 当成同一件事。
            #   颜色分工：底色左青（人）→ 右蓝（格），两侧标签同色呼应；
            #   网格线一律白色，只负责切出「一格一格」，不再抢颜色。
            # 网格线用 LineCollection：53 + 13 = 66 条线段即可，
            # 若逐格画 Rectangle 则要 624 个 patch，SVG 体积翻倍。
            vsegs, hsegs = [], []
            for k in range(ncol + 1):
                gx = bx + mx * k / ncol
                vsegs.append([(gx, by), (gx, by + my)])
            for k in range(nrow + 1):
                gy = by + my * k / nrow
                hsegs.append([(bx, gy), (bx + mx, gy)])
            ax.add_collection(LineCollection(vsegs, colors="white", lw=0.15,
                                             transform=ax.transAxes, zorder=7))
            ax.add_collection(LineCollection(hsegs, colors="white", lw=0.15,
                                             transform=ax.transAxes, zorder=7))
            ax.text(bx - GAP, by + my / 2, "139 raters", ha="right", va="center",
                    fontsize=4.0, color=TEAL, transform=ax.transAxes, zorder=7)
            ax.text(bx + mx + GAP, by + my / 2, "140 cells", ha="left", va="center",
                    fontsize=4.0, color=BLUE, transform=ax.transAxes, zorder=7)
        elif i == 3:    # 逐格计票：候选 code 的得票柱，最高者＝众数
            # 🔴 2026-10-20 三次改版（用户裁定）：**左文右图**。
            #   前两版把柱组压在框底 0.111 axes 的空白带里，柱高只有 5–6 pt，
            #   既「看不懂」又「太矮」。用户要求把 "Gold standard = modal codes"
            #   拆成两行、文字整体靠左，右半边的空白让给微图 —— 这样柱组可用
            #   **整框高度**（0.26 axes = 34.7 pt），不再受底部带宽限制。
            #   布局：文字三行（5 pt、左对齐、顶部 0.044）；右区为柱组 + 标签。
            #   文字最宽行实测 "Gold standard =" = 35.7 pt（0.192 axes），
            #   故右区 = 0.032 + 0.192 → 0.485 − 0.032，宽 0.229 axes（42.5 pt）。
            #   柱组宽 0.168 axes（31 pt）在右区内居中，中心 x = 0.339。
            #   垂直：柱顶最高 0.211、标签底 0.048 —— 上下留白接近 0.048，
            #   整组与左侧三行文字块（0.053–0.216）同高，两半不打架。
            # 🔴 2026-10-20 用户反馈「看不清、图与文字的逻辑还不够」→ 两处加强：
            #   ① 非众数柱由 NEU_L（#D5D5D3）加深到 **NEU_M**（#8C8C8A），
            #      1:1 印刷下不再糊成白底；
            #   ② 柱高差异拉大（4:10:7:3，原 10:15:13:8 最高只比第二高 15%），
            #      最高柱一眼可辨；
            #   ③ 标签由 "votes per code" 改为 **"139 votes · 4 codes"**，
            #      把「139 票 → 4 个候选 code → 柱高即票数」这条链接上。
            #   仍为示意高度，不标具体票数，避免与 b/c/d 面板的数据读数混淆。
            hs = [0.076, 0.054, 0.031, 0.023]   # 降序：P3.3/P3.2/P1.1/P3.1
            codes = ["P3.3", "P3.2", "P1.1", "P3.1"]
            # 2026-10-20 用户裁定：金色不能只标文字——**最高柱与 P3.3 也必须金色**，
            #   否则「金色标注 ↔ 深蓝柱」对不上。柱距同步从 8.9 pt 拉到 9.6 pt
            #   （g 0.016 → 0.020），让 3.8 pt 的 code 名（宽 7.8 pt）有 1.8 pt 间隙、
            #   且每个 code 正对一根柱，省掉 "·" 分隔符也能读。
            s, g = 0.030, 0.020
            gw = len(hs) * s + (len(hs) - 1) * g
            cx = x + 0.358                      # 右区中心（右移 0.019：给 "Gold standard" 让出与左栏的间隙）
            bx, by = cx - gw / 2, y + 0.115
            for j, hh in enumerate(hs):
                ax.add_patch(Rectangle((bx + j * (s + g), by), s, hh,
                                       facecolor=GOLD if j == 0 else NEU_M,
                                       edgecolor="none", transform=ax.transAxes,
                                       zorder=5))
                ax.text(bx + j * (s + g) + s / 2, y + 0.060, codes[j],
                        ha="center", va="top", fontsize=3.8,
                        color=GOLD if j == 0 else NEU_D,
                        transform=ax.transAxes, zorder=6)
            # 最高柱（众数）上方直接标 "Gold standard"：柱高 0.076，
            # 4 pt 标注占 0.198–0.240，与柱顶留 ~0.9 pt、距框顶 3.8 pt。
            # 2026-10-20 用户裁定：写全两个词、**首字母大写**、金色（GOLD）。
            ax.text(bx + s / 2, y + 0.198, "Gold standard", ha="center", va="bottom",
                    fontsize=4, color=GOLD, transform=ax.transAxes, zorder=6)
            # 🔴 2026-10-20 用户要求：必须写明这是**举例**并点出 4 个 code 名，
            #   否则读者会以为所有 cell 都固定 4 个候选 code。实测 case 1·P 的
            #   4 个候选是 P3.3/P3.2/P1.1/P3.1（89/44/5/1 票），柱按降序排列。
            ax.text(cx, y + 0.096, "e.g. case 1 · P", ha="center", va="top",
                    fontsize=3.8, color=NEU_D, transform=ax.transAxes, zorder=6)
        elif i == 4:    # 98 / 7：21×5 = 105 格，7 个红格散落
            # 同框 3：原格边 0.0095 只占框底空白带 0.054/0.111，现放大到 0.0145，
            # 格阵高 5×0.0145 + 4×0.002 = 0.0805 axes，带内居中（基线 0.025）。
            cols, rows, s, g = 21, 5, 0.0145, 0.002
            gw = cols * s + (cols - 1) * g
            bx, by = x + (w - gw) / 2, y + 0.025
            red_at = {2, 20, 39, 58, 76, 90, 103}
            for k in range(cols * rows):
                r, c = divmod(k, cols)
                ax.add_patch(Rectangle((bx + c * (s + g), by + r * (s + g)),
                                       s, s, facecolor=RED if k in red_at else BLUE_L,
                                       edgecolor="none", transform=ax.transAxes,
                                       zorder=5))
        elif i == 5:    # 9 个模型：2 行 × 5 列 logo（第二行 4 个居中）
            # 🔴 2026-10-20 用户裁定：原 3×3 蓝色方块「看不出是什么」，右侧又空一大片。
            #   改为**九个模型的真实 logo + 短名**，铺满右半区：
            #   第一行 5 个、第二行 4 个（居中），每个 logo 7 pt 高、名字 3.4 pt。
            #   logo 走 imshow + 数据坐标（axes 已固定 0–1），不触发 flush_icons 的
            #   整画布 12× 展开（Fig.2 是双栏宽，那条路径有 OOM 风险）。
            from matplotlib import image as mpimg
            _bb = ax.get_position()
            _w_pt = _bb.width * fig.get_size_inches()[0] * 72
            _h_pt = _bb.height * fig.get_size_inches()[1] * 72
            _lw, _lh = 7 / _w_pt, 7 / _h_pt        # 7 pt 见方（按各自轴长换算）
            _items = [
                ("kimi-k2.8-preview", "Kimi K2.8"),
                ("glm-5.3-flash", "GLM-5.3"),
                ("hy4-preview", "Hy4"),
                ("deepseek-v4.1-flash", "DeepSeek"),
                ("minimax-m3", "MiniMax M3"),
                ("ling-3.1-flash", "Ling"),
                ("mimo-v2.6-flash", "MiMo"),
                ("space-bunny", "MiniMax M3.1"),
                ("qwen3.8-flash", "Qwen3.8"),
            ]
            for k, (key, short) in enumerate(_items):
                row, col = (0, k) if k < 5 else (1, k - 5)
                ncol = 5 if row == 0 else 4
                cx = 0.400 + 0.580 * (col + 0.5) / ncol
                cy = 0.214 if row == 0 else 0.112
                stem = ICON_MAP[MODEL_ICON[key]]
                img = mpimg.imread(ROOT / "icon" / f"{stem}.png")
                ax.imshow(img, extent=(cx - _lw / 2, cx + _lw / 2,
                                       cy - _lh / 2, cy + _lh / 2),
                          aspect="auto", zorder=6, interpolation="antialiased")
                ax.text(cx, cy - _lh / 2 - 0.007, short, ha="center", va="top",
                        fontsize=3.4, color=NEU_D, transform=ax.transAxes, zorder=6)

    # 框 1/2 的正文：**左对齐、顶部同高**（2026-10-20 用户裁定）。
    # 顶部基准 y + h − 0.044（角标下方留 0.5 mm）。
    # 框 1 的批次标签（Round 1/2）不再做正文前缀，改为**白字写进各自的规模条内部**：
    # 原先「Round 1 在上一行、灰条在框底」，两处分离，读者要自己配对。
    ALIGN_TOP = 0.044
    for i, (x, y, w, h, txt, col) in enumerate(boxes, start=1):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=col,
                               lw=0.8, transform=ax.transAxes))
        if MICRO:
            _micro(i, x, y, w, h)
        # 序号：框内左上角，6 pt 粗体，取框线色 → 与正文形成层级差
        # （7 pt 时 bbox 与第一行正文相交 7.5%，审计 FAIL；缩到 6 pt 并左移）
        ax.text(x + 0.012, y + h - 0.012, str(i), ha="left", va="top",
                fontsize=6, fontweight="bold", color=col,
                transform=ax.transAxes, zorder=7)
        if i == 1:
            # 两行正文在上（左对齐、与框 2 顶部对齐）；两条规模条**并排**在下，
            # Round 1 深蓝、Round 2 浅蓝，白字写进各自条内（2026-10-20 用户裁定）。
            ax.text(x + 0.032, y + h - ALIGN_TOP, txt, ha="left", va="top",
                    fontsize=4.8, color=INK, transform=ax.transAxes,
                    linespacing=1.45, zorder=6)
            bars = (("Round 1", BLUE, 0.320), ("Round 2", BLUE_L, 0.107))
            bx = x + 0.032
            for tag, bcol, bw in bars:
                ax.add_patch(Rectangle((bx, y + 0.042), bw, 0.051,
                                       facecolor=bcol, edgecolor="none",
                                       transform=ax.transAxes, zorder=5))
                ax.text(bx + 0.006, y + 0.042 + 0.0255, tag, ha="left", va="center",
                        fontsize=4.6, color="white", fontweight="bold",
                        transform=ax.transAxes, zorder=6)
                bx += bw + 0.006
        elif i == 2:
            ax.text(x + 0.032, y + h - ALIGN_TOP, txt, ha="left", va="top",
                    fontsize=4.8, color=INK, transform=ax.transAxes,
                    linespacing=1.45, zorder=6)
        elif i == 3:
            # 🔴 2026-10-20 用户裁定：框 3 的三行文字各自定位，第二、三行之间
            #   留出**旋转 90° 的等号**——原写法把 "=" 放在第二行行尾，读者
            #   看不出「Gold standard」与「modal codes」是相等关系。
            #   行距不再均匀：1→2 行紧（0.010 axes 空隙），2→3 行松（0.036），
            #   松出来的位置正好容纳旋转后的等号（长 0.026，上下各留 0.005）。
            l1, l2, l3 = txt.split("\n")
            for _s, _y in ((l1, h - ALIGN_TOP), (l2, 0.166), (l3, 0.076)):
                ax.text(x + 0.032, y + _y, _s, ha="left", va="top",
                        fontsize=5, color=INK, transform=ax.transAxes, zorder=6)
            # 等号用两条 0.6 pt 竖线画，不用旋转文字：5 pt 的 "=" 旋转后
            # 线宽只有 ~0.3 pt，1:1 印刷下糊成两个点（用户「看不出来」的原因）。
            for _dx in (0.0, 0.006):
                ax.plot([x + 0.110 + _dx] * 2, [y + 0.088, y + 0.118],
                        color=INK, lw=0.6, transform=ax.transAxes,
                        zorder=6, solid_capstyle="butt")
        elif i == 4:
            # 顶部对齐 + 左对齐，缩进与框 1/2 相同（2026-10-20 用户裁定）。
            ax.text(x + 0.032, y + h - ALIGN_TOP, txt, ha="left", va="top",
                    fontsize=5, color=INK, transform=ax.transAxes, linespacing=1.45,
                    zorder=6)
        else:
            # 框 5：四行文字**左对齐、顶部同高**，左线与框 1/3 完全一致
            #   （2026-10-20 用户裁定）；右侧 0.40–0.98 留给九个模型 logo。
            #   字号 4.8 pt（与框 1/2 同档），行距 1.35 —— 四行块高 0.230 axes，
            #   恰好落在 0.25 的框高内（底余 0.020）。
            # ⚠️ +0.0074 axes：4.8 pt 的 bbox 顶留白比 5 pt 少约 1 pt，
            #   实测 ink 顶距框顶 5.97 pt，而框 1/3 是 4.99 pt；上移补偿后四框同高。
            ax.text(x + 0.032, y + h - ALIGN_TOP + 0.0074, txt, ha="left", va="top",
                    fontsize=4.8, color=INK, transform=ax.transAxes,
                    linespacing=1.42, zorder=6)
    # 🔴 「two rulers」标注改放在**红框上方空隙**（原 0.615, 0.505 被上方
    #   两条弧形箭头的曲线穿过—— 审计报 text-stroke）。
    #   现在流程箭头已取消，0.615, 0.33 一带完全空白。
    ax.text(0.5, 0.325, "scored on two criteria", ha="center", fontsize=5,
            color=TEAL, style="italic", transform=ax.transAxes)
    panel_label(ax, "a", dx_pt=_DX_AC)

    # ---------- b · 有共识/争议 构成 ----------
    ax_b = ax = fig.add_axes(_POS["b"])
    lab = ["P", "E1", "E2", "C", "I", "A", "O"]
    y = np.arange(len(lab))
    c1 = [sum(1 for (cid, d) in cls1 if d == x and cls1[(cid, d)]["consensus"]) for x in lab]
    c2 = [sum(1 for (cid, d) in cls2 if d == x and cls2[(cid, d)]["consensus"]) for x in lab]
    t1 = [sum(1 for (cid, d) in cls1 if d == x) for x in lab]
    t2 = [sum(1 for (cid, d) in cls2 if d == x) for x in lab]
    h = 0.55
    # 单条堆叠：蓝＝有共识，红＝争议（2026-10-21 用户裁定：删除重复的灰色总数条
    # 与上方独立蓝条，二者长度本就相同；只保留蓝+红一条，图例因此完整）
    ax.barh(y, c1, height=h, color=BLUE, lw=0)
    ax.barh(y, [t - c for c, t in zip(c1, t1)], left=c1,
            height=h, color=RED, lw=0, alpha=0.85)
    for i, (a, t) in enumerate(zip(c1, t1)):
        ax.text(t + 0.3, i, f"{a}/{t}", va="center",
                fontsize=5, color=BLUE)
        n_cont = t - a
        if n_cont:
            ax.text(a + n_cont / 2, i, str(n_cont), ha="center",
                    va="center", fontsize=5, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlim(0, 22); ax.set_xlabel("Cells (round 1, 139 raters × 20 cases)")
    # 2026-10-21 用户裁定：① 标题下移，离开面板标签 "b"（此前两者在 PDF
    #   文本层被合并成词 "bround"，即已贴在一起）；② 图例上移到标题同一行，
    #   左标题 + 右图例共同撑满绘图区宽度；③ 两者字号各降一档以放得下。
    # 2026-10-21 二次裁定：图例色块此前比标题文字高约 1 pt（图例框底对齐、
    #   色块在框内又内缩一圈）→ 改为**以标题行垂直中心对齐**：标题 va="center"，
    #   图例 loc="center right"，二者同用 TITLE_Y 作中心；整行再下移约 2 pt。
    # 2026-10-21 三次裁定：b/c/d 三行标题仍离面板标签略近，
    #   统一下移 2 px（300 dpi ≈ 0.48 pt）；只动这个常量，面板高度不变。
    # 2026-10-21 四次裁定：再下移 3 px → 累计 5 px（≈1.2 pt）。
    TITLE_Y = 1.0030                     # 标题行垂直中心
    ax.text(0.0, TITLE_Y, "round 1 · consensus / contested per dimension",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=4.7, color=NEU_D)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.85)],
              ["consensus (112)", "contested (28)"],
              loc="center right", bbox_to_anchor=(1.0, TITLE_Y),
              borderaxespad=0, ncol=2,
              handlelength=0.9, handletextpad=0.4, fontsize=4.5,
              columnspacing=1.0)
    clean_axes(ax)
    clean_axes(ax)
    panel_label(ax, "b", dx_pt=_DX_BD)

    # ---------- c · 争议格：金标准 ≠ 多数答案（关键证据） ----------
    ax_c = ax = fig.add_axes(_POS["c"])
    rows = []
    for (cid, dim), v in cls2.items():
        if not v["consensus"]:
            rows.append((cid, dim, v["gold"],
                         [v["votes"][c] for c in v["gold"]],
                         v["top"][1], v["threshold"]))
    rows.sort(key=lambda r: r[3][0])          # 主金标准码得票升序
    y = np.arange(len(rows))
    for i, (cid, dim, codes, gvs, top, thr) in enumerate(rows):
        gv = gvs[0]
        # 所有非零的金标准码统一用淡蓝；此前的 <15/≥15 分档是任意视觉阈值，
        # 不是 SAP 口径，且把同一种对象拆成两色（2026-10-21 用户裁定去掉）。
        col = RED if gv == 0 else BLUE_L
        # 2026-10-21 修：先画众数底轨、再画金标准条。此前顺序相反，
        #   灰轨长度恒 ≥ 金标准条，彩色条被完全盖住、颜色语义丢失。
        ax.barh(i, top, height=0.5, color=NEU_L, lw=0, zorder=1)
        ax.barh(i, gv, height=0.5, color=col, lw=0, zorder=2)
        ax.plot([gv, top], [i, i], color=INK, lw=0.7, zorder=3)
        if gv == 0:
            # 0 票在坐标轴上：空心方块标出，否则与轴线重合不可见
            ax.plot(gv, i, "s", ms=3.4, mfc="white", mec=RED, mew=0.8, zorder=4)
        elif gv == top:
            # 首个金标准码恰是众数时（case 2·I），实心圆会被菱形完全盖住；
            # 改用比菱形大的空心环，让"该点同时是金标准码"仍然可见
            ax.plot(gv, i, "o", ms=4.6, mfc="none", mec=col, mew=0.9, zorder=4)
        else:
            ax.plot(gv, i, "o", ms=3.4, color=col, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)
        # 2026-10-21 修：多码金标准必须把除主码外的码也画出来。
        #   case 2·I 的 I4.1 只有 7 票（阈值 21），此前完全不可见，
        #   图上只剩 "41 vs 41"，读者无法看出它为何是争议格。
        for c, vv in zip(codes[1:], gvs[1:]):
            fail = vv < thr
            ax.plot(vv, i, "o", ms=3.2, color=RED if fail else NEU_D,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=5)
            ax.text(vv + 0.8, i - 0.30,
                    f"{c} {vv}" + (f" < {thr}" if fail else ""),
                    ha="left", va="top", fontsize=4.0,
                    color=RED if fail else NEU_D, zorder=6)
        ax.plot(top, i, "D", ms=2.6, color=NEU_D, zorder=4)
        ax.text(top + 1.6, i, f"{gv} vs {top}", va="center",
                fontsize=5, color=INK)
        if gv == 0:
            # 2026-10-21 修：0 票标注移入绘图区、放在条上方，
            #   不再靠 -14…0 的负半轴腾位置（负半轴无数据、易读成负票数）。
            ax.text(0.8, i + 0.33, f"gold {codes[0]} = 0 votes", ha="left",
                    va="bottom", fontsize=4.5, color=RED, fontweight="bold",
                    zorder=6)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlim(-1.2, 56); ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("Pharmacists out of 46")
    # 与 b 面板同一标题行高度（2026-10-21 用户裁定）：
    #   原 set_title(pad=3) 把标题顶在坐标区上沿，与面板标签 "c" 几乎贴住；
    #   改用与 b 相同的 TITLE_Y / va="center" / 4.7 pt，让标签与标题的间距一致。
    ax.text(0.0, TITLE_Y,
            "round 2 · 7 contested cells: gold code(s) (circle) vs modal answer (diamond)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=4.7, color=NEU_D, zorder=6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    clean_axes(ax)
    clean_axes(ax)
    # 2026-10-21 用户裁定：y 标签不要随 case 位数左右漂移。改为**固定点列**——
    #   "case N" 右对齐到点、"·" 钉在同一列、分项从点右侧起排；
    #   多码金标准的两行标签，第二行也右对齐到同一列。
    YLAB_DOT_X = -0.060                        # 点列（axes 坐标）
    tr = ax.get_yaxis_transform()
    for i, (cid, dim, codes, *_rest) in enumerate(rows):
        ax.text(YLAB_DOT_X - 0.012, i, f"case {cid}", transform=tr,
                ha="right", va="center", fontsize=5, color=INK, zorder=7)
        ax.text(YLAB_DOT_X, i, "·", transform=tr, ha="center", va="center",
                fontsize=5, color=NEU_D, zorder=7)
        ax.text(YLAB_DOT_X + 0.012, i, dim, transform=tr, ha="left", va="center",
                fontsize=5, color=INK, zorder=7)
        if len(codes) > 1:
            ax.text(YLAB_DOT_X - 0.012, i - 0.33, f"({' + '.join(codes)})",
                    transform=tr, ha="right", va="center", fontsize=4.2,
                    color=NEU_D, zorder=7)
    panel_label(ax, "c", dx_pt=_DX_AC)

    # ---------- d · 人类分项一致性（两把尺子的必要性） ----------
    # 2026-10-21 用户裁定：x/y 调换为**横向条形**，但坐标区仍用原来的
    #   宽扁长宽比（不旋转整图）：分项移到 y 轴、κ 移到 x 轴。
    ax_d = ax = fig.add_axes(_POS["d"])
    kap = {"P": .582, "E1": .510, "E2": .484, "C": .176, "I": .249, "A": .577, "O": .652}
    ks = [kap[x] for x in lab]
    col = [RED if k < 0.30 else (NEU_M if k < 0.50 else BLUE) for k in ks]
    ax.barh(np.arange(len(lab)), ks, height=0.6, color=col, lw=0, zorder=2)
    for i, k in enumerate(ks):
        # E2 的柱端 0.484 紧贴 κ=0.5 虚线：标签右移过线，避免 text-stroke
        lx = k + 0.015
        if 0.5 - 0.005 < lx < 0.515:
            lx = 0.515
        # 2026-11-02 用户裁定：κ 数字保留前导零（0.652），与 x 轴刻度
        #   「0.0 / 0.2 …」写法一致；原 `[1:]` 切掉 0 得到 ".652"，
        #   是 APA 省略前导零的惯例，但图内与坐标轴自相矛盾。
        ax.text(lx, i, f"{k:.3f}", ha="left", va="center",
                fontsize=5, color=INK)
    # 阈值线只跨数据行（-0.45…6.35），不顶到标题、也不压到底部标签
    ax.plot([0.5, 0.5], [-0.45, 6.35], color=NEU_M, lw=0.7,
            ls=(0, (3, 2)), zorder=3)
    # 阈值标签放在 I 与 C 两行之间的空白带（2026-11-02 用户裁定）：
    #   原先钉在轴底部（y=-0.55），下缘与 P 柱顶重叠 0.6 pt；
    #   I（0.249）/ C（0.176）两根短柱右侧是整列唯一的大片空白，
    #   标签放在虚线右侧、两行正中，既离开 P 柱，也不碰 C/I 注释。
    ax.text(0.512, 3.5, "κ = 0.5", fontsize=5, color=NEU_D,
            ha="left", va="center")
    ax.set_xlim(0, 0.78)
    ax.set_xticks(np.arange(0, 0.61, 0.2))
    ax.set_ylim(-0.7, len(lab) - 0.3)
    ax.set_yticks(np.arange(len(lab))); ax.set_yticklabels(lab)
    ax.set_xlabel("Fleiss κ among pharmacists")
    # 与 b/c 面板同一标题行高度（2026-10-21 用户裁定）：
    #   原 set_title(pad=3) 与面板标签 "d" 有轻微重叠；改用 TITLE_Y / va="center"。
    ax.text(0.0, TITLE_Y, "human inter-rater agreement is uneven across dimensions",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=4.7, color=NEU_D, zorder=6)
    # 注释放在 C/I 两条短柱右侧、C 与 I 之间的空隙；箭头指向两条短柱之间
    ax.annotate("C and I: hardest for\nmodels as well",
                xy=(0.20, 3.5), xytext=(0.255, 3.5),
                ha="left", va="center", fontsize=5, color=RED, linespacing=1.35,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.6,
                                connectionstyle="arc3,rad=0.15"))
    clean_axes(ax)
    clean_axes(ax)
    panel_label(ax, "d", dx_pt=_DX_BD)

    # a 为示意面板（无数据坐标轴）→ 排除出等宽等高比较组。
    # 余下 b(上右) / c(下左) / d(下右)：
    #   同行 → c,d（共享 top/bottom/height）｜同列 → b,d（共享 left/right/width）
    # 🔴 2026-11-02 改用 add_axes 手动定位（两条缝异值：水平 8 mm / 垂直 11 mm），
    #   面板不再带 subplotspec → 闸门无法自动推断行列组，必须**显式声明**，
    #   否则报 NOT AUDITABLE（No comparable row, column or shared-boundary groups）。
    #   行列组与原来 GridSpec 自动推断的结果一致，比较项不变。
    save_pub(fig, "Fig2_design_and_goldstandard", alignment=True,
             panel_ids={ax_sch: "a", ax_b: "b", ax_c: "c", ax_d: "d"},
             exclude_axes=[ax_sch],
             row_groups=[["c", "d"]],
             column_groups=[["b", "d"]])


# ============================================================
# Fig.9 · 双条件与效率
# ============================================================
def fig9():
    M = load()
    order = order_by_rate(M)
    fig = plt.figure(figsize=(COLW, 96 * MM))
    # 2026-11-02 用户裁定：a–b 缩 2 mm（11.67 mm），b–c 再缩 2 mm
    #   （10.67 → 8.67 mm = 24.59 pt）。内宽 449.0 pt、三栏等宽 130.44 pt。
    _POS9 = {
        "a": [0.105, 0.26, 0.25564, 0.60],
        "b": [0.42549, 0.26, 0.25564, 0.60],
        "c": [0.72932, 0.26, 0.25564, 0.60],
    }
    _XA9 = 0.20          # full arm 点列右移，给模型名让位
    # a 面板左侧放模型名：用紧凑写法，保证文字不越过左轴线
    _SHORT9 = {
        "kimi-k2.8-preview": "Kimi K2.8",
        "glm-5.3-flash": "GLM-5.3",
        "hy4-preview": "Hy4",
        "deepseek-v4.1-flash": "DeepSeek V4.1",
        "minimax-m3": "MiniMax M3",
        "ling-3.1-flash": "Ling-3.1",
        "mimo-v2.6-flash": "MiMo V2.6",
        "space-bunny": "MiniMax M3.1",
        "qwen3.8-flash": "Qwen3.8",
    }

    # ---------- a · A vs B 配对 ----------
    # ⚠️ 用**身份色**（用户 2026-10-06 裁定）。此前 `RED if b > a else BLUE`
    #   把 9 个模型压成两色，同一模型在 Fig.2a 里是彩色、到这里却变红/蓝。
    #   「B 臂掉了多少」由**连线长度 + Δ 标签的正负号**表达，不靠颜色区分。
    ax = fig.add_axes(_POS9["a"]); axA = ax
    _DL = []
    _AL = []
    for k, e in order:
        a = e["A"]["consensus"]["rate"] * 100
        b = e.get("B", {}).get("consensus", {}).get("rate")
        if b is None:
            continue
        b *= 100
        col = chan_color(k)
        ax.plot([_XA9, 1], [a, b], color=col, lw=0.8, alpha=0.75, zorder=2)
        ax.plot(_XA9, a, "o", ms=3.0, color=col, markeredgecolor=col,
                markeredgewidth=0.3, zorder=3)
        ax.plot(1, b, "o", ms=3.0, color=col, markeredgecolor=col,
                markeredgewidth=0.3, zorder=3)
        _DL.append([b - a, b, col])
        _AL.append((a, _SHORT9.get(k, k), col))
    # Δ 标签统一绘制：值相近者（如 +1.0 与 −1.0）纵向错开，防止叠印
    # 简易防重叠布局：自上而下扫描，若与上一个标签的间距不足则下压
    _DL.sort(key=lambda r: r[1], reverse=True)
    _MIN_GAP = 2.2          # 数据单位，约等于 5pt 字号的一行高度
    _last_y = None
    for d, yy, col in _DL:
        ty = yy if _last_y is None else min(yy, _last_y - _MIN_GAP)
        _last_y = ty
        ax.annotate(f"{d:+.1f}", (1.0, yy), xytext=(1.46, ty),
                    textcoords="data", fontsize=5, color=col,
                    va="center", ha="left", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.45,
                                    shrinkA=2, shrinkB=4))
    # 2026-11-02 用户裁定：模型名放 A 点左侧、不得越过左轴线；
    #   A 值相近的名字上下错开，并用短线连回各自点。
    _AL.sort(key=lambda r: r[0], reverse=True)
    _last_a = None
    for _a, _nm, _col in _AL:
        _ty = _a if _last_a is None else min(_a, _last_a - 1.2)
        _last_a = _ty
        if abs(_ty - _a) > 1e-6:
            ax.plot([_XA9 - 0.03, _XA9], [_ty, _a], color=_col, lw=0.3, zorder=1)
        ax.text(_XA9 - 0.07, _ty, _nm, ha="right", va="center",
                fontsize=4.4, color=_col, zorder=4)
    ax.set_xlim(-0.75, 1.95); ax.set_xticks([_XA9, 1])
    ax.set_xticklabels(["full arm (A)", "codebook only (B)"])
    # 🔴 ylim 下界必须取 min(A) 与 min(B) 的较小者（2026-10-07 修 FAIL×2 的根因）：
    #   原式只用 min(B) − 4 = 70.5，而 **Qwen 的 A 值 69.4 落在下界以下** →
    #   它的连线被画到**坐标区外**，横穿 x 轴刻度标签「full arm (A)」
    #   （审计报 text-stroke）。这不是排版瑕疵，是**把数据裁掉了**。
    #   Δ 标签的防重叠下压也会把最低三个标签（−12.2 / −5.1 / −10.2）压到下界附近，
    #   收紧下界后它们回到区内，与横贯的 x 轴线分离。
    _lo_all = min([e["A"]["consensus"]["rate"] * 100 for _, e in order] +
                  [e["B"]["consensus"]["rate"] * 100 for _, e in order if "B" in e])
    ax.set_ylim(_lo_all - 4,
                max(e["A"]["consensus"]["rate"] for _, e in order) * 100 + 4)
    ax.set_ylabel("Accuracy on consensus cells (%)")
    # 2026-11-02 用户裁定：显示 a 面板纵轴线；标识旁说明取消。
    clean_axes(ax, keep_left=True)
    # a 标识左缘对齐本面板 y 轴标题
    fig.canvas.draw()
    _r9a = fig.canvas.get_renderer()
    _dxa = (ax.yaxis.label.get_window_extent(_r9a).x0 -
            ax.get_window_extent(_r9a).x0) / fig.dpi * 72
    _DY9 = 4.0 - 3 * 72 / 25.4       # 2026-11-02 用户裁定：a/b/c 标识下移 3 mm
    panel_label(ax, "a", dx_pt=_dxa, dy_pt=_DY9)

    # ---------- b · 效率：仅三模型可比 ----------
    ax = fig.add_axes(_POS9["b"]); axB = ax
    COMP = ["deepseek-v4.1-flash", "glm-5.3-flash", "hy4-preview"]
    names, vals, cols = [], [], []
    for k in COMP:
        e = M[k]
        names.append(short(e["pretty"]))
        vals.append(e["A"]["latency_ms"]["median"] / 1000)
        cols.append(chan_color(k))
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.52, color=cols, lw=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=5, color=INK)
    ax.set_xticks(x)
    _xl9 = ax.set_xticklabels(names, fontsize=5)
    for _t, _k in zip(_xl9, COMP):
        _t.set_color(chan_color(_k))
    ax.set_ylabel("Median latency (s)")
    ax.set_ylim(0, max(vals) * 1.28)
    # 2026-11-02 用户裁定：显示 b 面板纵轴线。
    clean_axes(ax, keep_left=True)
    # b 标识左缘对齐本面板 y 轴标题
    fig.canvas.draw()
    _r9b = fig.canvas.get_renderer()
    _dxb = (ax.yaxis.label.get_window_extent(_r9b).x0 -
            ax.get_window_extent(_r9b).x0) / fig.dpi * 72
    panel_label(ax, "b", dx_pt=_dxb, dy_pt=_DY9)

    # ---------- c · 采集软件与防污染保证分档 ----------
    # v1.23：由「4 类渠道」改为「3 个采集软件 + 两档保证」。
    # 关键信息：WorkBuddy 是协议层设计保证，另两个软件是事后审计的验证保证——
    # 两者都零污染，但证据强度不同，方法学须分开陈述。
    ax = fig.add_axes(_POS9["c"]); axC = ax; ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # ⚠️ 软件色块用 `pal.sw_color` —— 与该软件下模型的**身份色同源同族**
    #（统一原则：读者看到蓝块就该知道里面那几个模型也是蓝的）。
    # 此前硬编码 BLUE/TEAL/赭 与模型身份色的色相并不一致（模型是 218/196/268）。
    # 2026-11-03 口径更正：OpenCode 实为 611 次（MiniMax-M3.1-Flash 220 + MiMo 220
    #   + Ling 171），原写 440 只算了前两个模型，与 1980 总数对不上。
    SW = [("WorkBuddy", 5, "1100 runs", pal.sw_color("WorkBuddy"), "design"),
          ("OpenCode", 3, "611 runs", pal.sw_color("OpenCode"), "audit"),
          ("Qoder", 1, "220 runs", pal.sw_color("Qoder"), "audit")]
    # 2026-11-02 用户裁定：色块换成各工具自身图标；行距拉开、红框压缩，
    #   让上部有更多空间展示工具与模型数；行间加细线形成结构化清单。
    # 红框下缘与 b 面板 DeepSeek 刻度行的底边对齐（换算到 c 轴坐标）
    fig.canvas.draw()
    _r9c = fig.canvas.get_renderer()
    _bt9 = next(t for t in axB.get_xticklabels()
                if "DeepSeek-V4.1-Flash" in t.get_text())
    _cbb9 = ax.get_window_extent(_r9c)
    _box_bottom = (_bt9.get_window_extent(_r9c).y0 - _cbb9.y0) / _cbb9.height
    y = 0.93
    for nm, n, runs, col, kind in SW:
        draw_icon_png(ax, nm, 0.0, y, 0, anchor="left", va="center",
                      pixels=12, defer=True)
        ax.text(0.10, y, nm, fontsize=5, color=INK,
                va="center", transform=ax.transAxes, fontweight="bold")
        ax.text(0.99, y, f"{n} model{'s' if n > 1 else ''} \u00b7 {runs}", fontsize=5,
                color=NEU_D, va="center", ha="right", transform=ax.transAxes)
        ax.text(0.10, y - 0.105,
                "tools disabled in protocol layer" if kind == "design"
                else "post-hoc tool-path audit",
                fontsize=5.0, color=col, va="center",
                transform=ax.transAxes, style="italic")
        ax.plot([0.0, 1.0], [y - 0.13, y - 0.13], color=NEU_L, lw=0.35,
                transform=ax.transAxes, zorder=1)
        y -= 0.27
    ax.add_patch(Rectangle((0.0, _box_bottom), 1.0, 0.14, facecolor="#FFF8F7",
                           edgecolor=RED, lw=0.8, transform=ax.transAxes,
                           clip_on=False))
    ax.text(0.5, _box_bottom + 0.07, "Latency is comparable only\nwithin WorkBuddy",
            ha="center", va="center", fontsize=4.8, color=RED,
            fontweight="bold", transform=ax.transAxes, linespacing=1.3,
            clip_on=False)
    panel_label(ax, "c", dx=-0.145, dy=1.10, dy_pt=_DY9)
    # a/b 标识各自对齐本面板 y 轴标题，不再与 c 共用偏移列。
    # 手动 add_axes 后需显式声明单行组；两缝有意不等宽（2 mm / 3 mm）。
    save_pub(fig, "Fig9_conditions_and_efficiency", alignment=True,
             panel_ids={axA: "a", axB: "b", axC: "c"},
             row_groups=[["a", "b", "c"]],
             exemptions=[
                 {"panels": ["a", "b"], "checks": ["panel-label"],
                  "reason": "a/b 标识对齐各自 y 轴标题（用户 2026-11-02）"},
                 {"panels": ["a", "b", "c"], "checks": ["horizontal-gutter"],
                  "reason": "用户指定 a–b 缩 2 mm、b–c 累计缩 5 mm；两缝有意不等宽"},
             ])


# ---- 图注册表（必须在所有 figN 定义之后） ----
# ============================================================
# Fig.5 · 跨模型一致性（9 个模型）
# ============================================================
def fig5():
    """跨模型一致性：两两一致率 + Fleiss κ 分项对照（9 个模型口径）。

    核心结论：给流程与范例（A 组）不仅提高单模型准确率，也提高模型间收敛度；
    且 κ 的分项格局与主终点的难度分层一致 —— 两个独立指标互为佐证。
    """
    cons = json.loads((ROOT / "drp_eval" / "consistency9.json").read_text(encoding="utf-8"))
    A, B = cons["A"], cons["B"]
    ka, kb = A["fleiss_kappa_by_dim"], B["fleiss_kappa_by_dim"]
    DIMS = ["P", "E1", "E2", "C", "I", "A", "O"]
    DIMLAB = {"P": "P\nproblem", "E1": "E1\nevent 1", "E2": "E2\nevent 2",
              "C": "C\ndrug", "I": "I\ninformation", "A": "A\naction",
              "O": "O\nmonitoring"}

    fig = plt.figure(figsize=(COLW, 96 * MM))
    # 2026-11-02 用户裁定：只把 b–c 缝缩 2 mm（a–b 不动）。
    #   实测两缝原均 14.97 mm → b–c 目标 12.97 mm；三栏等宽、总宽不变：
    #   内宽 0.90 fig，w=0.248277，g1=0.083155（14.97 mm），g2=0.072045（12.97 mm）。
    _B5, _T5 = 0.24, 0.84
    _W5 = 0.248277
    _POS5 = {
        "a": [0.085, _B5, _W5, _T5 - _B5],
        "b": [0.416432, _B5, _W5, _T5 - _B5],
        "c": [0.736754, _B5, _W5, _T5 - _B5],
    }

    # ---------- a · 两两一致率（A vs B） ----------
    ax = fig.add_axes(_POS5["a"]); ax_a = ax
    labels = ["A: full arm", "B: codebook only"]
    vals = [A["pair_agreement"] * 100, B["pair_agreement"] * 100]
    xs = np.arange(2)
    ax.bar(xs, vals, width=0.46, color=[BLUE, BLUE_L], lw=0)
    for i, v in enumerate(vals):
        ax.text(i, v - 4.2, f"{v:.1f}%", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold")
    # 差值引线
    ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="-", color=NEU_D, lw=0.6,
                                ls=(0, (2.5, 1.5))))
    ax.text(0.5, max(vals) + 1.2, f"+{vals[0]-vals[1]:.1f} pp",
            ha="center", va="bottom", fontsize=6.4, color=NEU_D,
            fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylim(84, 96)
    ax.set_ylabel("Pairwise agreement on consensus cells (%)")
    # 2026-11-02 用户裁定：顶标题不再挂在轴上方（与 a 标识太近），
    #   移入面板内、放到 "+2.9 pp" 正上方（x=0.5, y=95.2）。
    ax.text(0.5, 95.2, "9 models × 98 cells × 36 model pairs",
            ha="center", va="bottom", fontsize=5.4, color=NEU_D)
    # 2026-11-02 用户裁定：显示 a 面板的左侧纵轴线。
    clean_axes(ax, keep_left=True)
    # 2026-11-02 用户裁定：a 标识左缘与 y 轴标题左缘对齐。
    #   用渲染后的 ylabel bbox 反算相对轴左缘的 dx（pt），不靠估算。
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _axbb = ax.get_window_extent(_r)
    _ylbb = ax.yaxis.label.get_window_extent(_r)
    _dx_a = (_ylbb.x0 - _axbb.x0) / fig.dpi * 72
    panel_label(ax, "a", dx_pt=_dx_a)

    # ---------- b · Fleiss κ 分项对照 ----------
    ax = fig.add_axes(_POS5["b"]); ax_b = ax
    y = np.arange(len(DIMS))
    va = [ka[d] for d in DIMS]
    vb = [kb[d] for d in DIMS]
    ax.barh(y - 0.19, va, height=0.34, color=BLUE, lw=0, label="A: full arm")
    ax.barh(y + 0.19, vb, height=0.34, color=BLUE_L, lw=0, label="B: codebook only")
    for i, (a_, b_) in enumerate(zip(va, vb)):
        ax.text(a_ + 0.012, i - 0.19, f"{a_:.3f}", va="center", fontsize=5,
                color=INK)
        ax.text(b_ + 0.012, i + 0.19, f"{b_:.3f}", va="center", fontsize=5,
                color=INK)
    # 全部 7 维 A>B 的强调
    ax.set_yticks(y)
    ax.set_yticklabels([DIMLAB[d] for d in DIMS], fontsize=5.0)
    ax.set_xlim(0.5, 1.10)
    ax.set_xlabel("Fleiss \u03ba (9 raters = 9 model modes)")
    ax.set_ylim(6.7, -0.75)          # 顶部留白给"7/7"标注
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              handlelength=0.9, handletextpad=0.4, fontsize=5,
              columnspacing=0.8, borderpad=0.0, frameon=False)
    # 2026-11-02 用户裁定：删掉 "A > B in all 7 dimensions" 这一行，
    #   把 A/B 图例行下移进该空白带顶部（原 loc 在轴外 y=1.005）。
    ax.set_ylim(6.7, -0.75)          # 顶部留白给图例行
    clean_axes(ax)
    # b 标识左缘对齐 y 刻度里最长的一行 "information"
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _axbb = ax.get_window_extent(_r)
    _inf = next(t for t in ax.get_yticklabels() if "information" in t.get_text())
    _dx_b = (_inf.get_window_extent(_r).x0 - _axbb.x0) / fig.dpi * 72
    panel_label(ax, "b", dx_pt=_dx_b)

    # ---------- c · 与主终点的难度分层对照 ----------
    ax = fig.add_axes(_POS5["c"]); ax_c = ax
    r = load()
    acc = {}
    for k, e in r.items():
        bd = e["A"]["by_dim"]
        acc[k] = np.mean([bd[d]["rate"] * 100 for d in DIMS if d in bd])
    xs_a = [acc[k] for k in r]
    ks_a = [ka[d] for d in DIMS for _ in (0,)]  # 占位，下面用均值配对
    # 每个分项：主终点 9 模型均值 vs κ（A 组）
    main_by_dim = {}
    for d in DIMS:
        vals_d = [r[k]["A"]["by_dim"][d]["rate"] * 100
                  for k in r if d in r[k]["A"]["by_dim"]]
        main_by_dim[d] = float(np.mean(vals_d)) if vals_d else np.nan
    x1 = [main_by_dim[d] for d in DIMS]
    y1 = [ka[d] for d in DIMS]
    ax.plot(x1, y1, "o", ms=4.2, color=BLUE, markeredgecolor="white",
            markeredgewidth=0.5, zorder=3)
    # 🔴 标签偏移恒为右上(+5, +2)（2026-10-07 修 FAIL 时**试过**按「点相对趋势线的
    #   残差」决定上下偏移 → **失败，属R50「转移而非消除」**：
    #   A 点残差仅 −0.001（κ=0.901 vs 趋势线 0.902）确实贴在线上，
    #   但把 C（残差 +0.010）与 O（残差 −0.078）一并改成向下偏移后，
    #   **C 撞上趋势线本身、O 撞上 x 轴脊线**，FAIL 由 1 变 2。
    #   → 结论：偏移方向不能按残差逐点改，只能统一；真因在趋势线**画得太长**。
    for d, xx, yy in zip(DIMS, x1, y1):
        ax.annotate(d, (xx, yy), textcoords="offset points", xytext=(5, 2),
                    fontsize=5.2, color=NEU_D)
    # 趋势线：🔴 只画在**数据 x 范围内**（原为 min−3 … max+3）——这才是真因。
    #   实测原范围 53.5–96.0 而数据 x 仅 56.5–93.0，两端各外推 3 个百分点；
    #   右侧外推段正好穿过 A 点的标签（text-stroke）。
    #   外推段本就无数据支撑，回归线按惯例也只画在数据范围内。
    if len(x1) > 2:
        k_, b_ = np.polyfit(x1, y1, 1)
        xs_l = np.linspace(min(x1), max(x1), 20)
        ax.plot(xs_l, k_ * xs_l + b_, color=NEU_M, lw=0.6, ls=(0, (3, 2)),
                zorder=1)
    ax.set_xlabel("Mean accuracy across 9 models (%)")
    ax.set_ylabel("Fleiss \u03ba (A group)")
    # 2026-11-02 用户裁定：标题从轴上方移入面板左上空白区（两行、左对齐）
    ax.text(0.02, 0.98, "the two metrics agree on which\ndimensions are hard",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.2,
            color=NEU_D, linespacing=1.3)
    # 2026-11-02 用户裁定：显示 c 面板左侧纵轴线
    clean_axes(ax, keep_left=True)
    # c 标识左缘对齐 y 轴标题 "Fleiss …" 的左缘
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _axbb = ax.get_window_extent(_r)
    _ylbb = ax.yaxis.label.get_window_extent(_r)
    _dx_c = (_ylbb.x0 - _axbb.x0) / fig.dpi * 72
    panel_label(ax, "c", dx_pt=_dx_c)

    # 手动 add_axes 后闸门无法自动推断行列组，显式声明单行 a/b/c。
    # a/b/c 三个标识分别按用户要求对齐本面板的 y 轴标题/刻度文字
    # （a→ylabel、b→"information"、c→"Fleiss"）；b–c 缝有意比 a–b 窄 2 mm。
    save_pub(fig, "Fig5_cross_model_consistency", alignment=True,
             panel_ids={ax_a: "a", ax_b: "b", ax_c: "c"},
             row_groups=[["a", "b", "c"]],
             exemptions=[
                 {"panels": ["a", "b", "c"], "checks": ["panel-label"],
                  "reason": "a/b/c 标识各自对齐本面板 y 轴标题或刻度文字（用户 2026-11-02 指定）"},
                 {"panels": ["a", "b", "c"], "checks": ["horizontal-gutter"],
                  "reason": "用户指定 b–c 比 a–b 窄 2 mm；两条缝有意不等宽"},
             ])


# ---- 新图（Fig.6／Fig.7）接入口（2026-10-07）----
# 依赖 derived_figs.json（由 derive_fig_data.py 产出）＋ results_stage4.json；
# 绘图函数在 fig_newpanels.py 内，只读 json、不在绘图时做统计（AGENTS.md 原则 2）。
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from fig_newpanels import (fig1 as _fig1, fig6 as _fig6, fig7 as _fig7,  # noqa: E402
                           fig8 as _fig8, tiny as _tiny)


def fig6():
    """Fig.6 · 错误从哪来：错因 × 逐格 × 严格性"""
    _fig6(load(), load(), save_pub, panel_label, clean_axes, chan_color, short)


def fig7():
    """Fig.7 · 错误不是随机的：共现结构与硬格"""
    _fig7(load(), load(), save_pub, panel_label, clean_axes, chan_color, short)


def fig1():
    """Fig.1 · 总 scheme（把全文结构前置）"""
    _fig1(load(), _load_derived(), json.loads(
        (ROOT / "drp_eval" / "pcne_endpoints.json").read_text(encoding="utf-8")),
        save_pub, panel_label, clean_axes, chan_color, short, _tiny)


def fig8():
    """Fig.8 · PCNE 迁移臂（d 面板为 P 段码表对照，合规声明在图注）"""
    _fig8(json.loads(
        (ROOT / "drp_eval" / "pcne_endpoints.json").read_text(encoding="utf-8")),
        load(), save_pub, panel_label, clean_axes, chan_color, short)


def _load_derived():
    return json.loads((ROOT / "drp_eval" / "derived_figs.json").read_text(encoding="utf-8"))


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
        "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"输出目录：{FIGDIR}")
    for k, fn in FIGS.items():
        if which in (k, "all"):
            print(f"\n▶ {k}")
            fn()
