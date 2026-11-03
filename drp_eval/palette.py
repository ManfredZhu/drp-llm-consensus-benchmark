"""图 2 配色方案（SAP v1.23 · 2026-10-06）

## 设计目标（用户要求）
1. 子图配色**与模型图标颜色保持一致**；
2. 几个模型若同色，须**另设颜色区分**；
3. **总体克制**，颜色从已有图标取，不要五颜六色。

## 依据：图标实测主色（`measure_icon_colors`）
对 `drp_eval/icons_png/` 逐个取「不透明像素的众数色，过滤黑白灰」：

| 图标 | 主色 | 色相 | 判定 |
|---|---|---|---|
| DeepSeek V4.1 Flash | `#4868F8` | 229 | 彩色 |
| Hy4 preview | `#0050E8` | 219 | 彩色 |
| MiniMax-M3 | `#F04050` | 355 | 彩色 |
| Ling-3.1-Flash | `#1078F8` | 213 | 彩色 |
| WorkBuddy | `#6848F8` | 251 | 彩色 |
| Qoder | `#28D858` | 136 | 彩色 |
| OpenRouter | `#C8F800` | 72 | 彩色 |
| MiniMax-M3.1-Flash | 同 MiniMax-M3（品牌图标复用） | 355 | 彩色 |
| **GLM-5.3-Flash** | — | — | **纯黑白徽标** |
| **Kimi K2.8 Preview** | — | — | **纯黑K** |
| **MiMo-V2.6-Flash** | — | — | **黑白文字标** |
| **Qwen3.8-Flash** | — | — | **紫黑六角** |
| **OpenCode** | — | — | **纯黑白方框** |

→ **9 个模型里只有 4 个有彩色图标**，故"与图标一致"只能对彩色者成立；
黑白图标（5 个）取**同组邻近色相**，这是"克制"而非另起炉灶。

## 规则（三条，同时满足用户的三个要求）
1. **色相跟图标**：彩色图标 → 直接沿用其色相；
   黑白图标 → 取所在采集软件色相的**邻近值**（±10~16°），视觉上同族。
2. **同组用同色相不同明度**：同一采集软件的多个模型共用一个色相，
   用 **明度三档（0.0 / −0.14 / −0.24）** 拉开 → 既区分模型，
   又保留"同组"线索，比给 9 个不同色相克制得多。
3. **全局色相预算 = 4 个家族**（不是 9 个）：
   - `WorkBuddy` 组 → **蓝 218**（来自 DeepSeek／Hy4 的蓝）
   - `OpenCode` 组 → **青蓝 196**（来自 Ling 的蓝青）
   - `Qoder`（仅 Qwen）→ **紫 268**（来自 Qwen 的紫黑六角）
   - `MiniMax-M3` → **砖红 355**（来自其红色图标；**大幅降饱和**，
     否则纯红 #F04050 在出版版面过于刺眼，破坏"克制"）

   全部**饱和度压到 0.34–0.52、明度 0.30–0.55**（印刷友好区间，
   浅底上可读，且不与黑色文字争夺注意力）。

## 读取
```python
from svg_icons import model_color          # 单个模型
from svg_icons import MODEL_COLOR, SW_COLOR, SW_ORDER# 查表
```
"""

from __future__ import annotations

import colorsys


# ---------------------------------------------------------------- 色相家族
# 每个采集软件一个色相家族（SAP v1.23：3 个采集软件 + MiniMax 单列）
_FAMILY = {
    "WorkBuddy": 218,      # 蓝——来自 DeepSeek #4868F8 / Hy4 #0050E8
    "OpenCode": 196,       # 青蓝——来自 Ling #1078F8
    "Qoder": 268,          # 紫——来自 Qwen 的紫黑六角
}

# 模型 → 采集软件（与 SOFTWARE_OF 同源，见 svg_icons）
SW_OF = {
    "deepseek-v4.1-flash": "WorkBuddy",
    "glm-5.3-flash": "WorkBuddy",
    "hy4-preview": "WorkBuddy",
    "minimax-m3": "WorkBuddy",
    "kimi-k2.8-preview": "WorkBuddy",
    "ling-3.1-flash": "OpenCode",
    "space-bunny": "OpenCode",
    "mimo-v2.6-flash": "OpenCode",
    "qwen3.8-flash": "Qoder",
}

# 模型 → (色相微调, 明度偏移, 饱和偏移)
#明度档按「同组内读数由高到低」分配：读数最高用最深，
# 使「颜色深浅」与「排名先后」同向，读者无需查图例。
# ⚠️ 明度偏移取**负值**＝更浅。基准明度见 _BASE_SV。
# 档差设计为 0.00 / −0.10 / −0.20三档 + 邻近色相 ±10~16°，
# 保证同组内任意两个模型在色相或明度上**至少有一项明显不同**。
_SLOT = {
    # ── WorkBuddy（蓝 218）：4 档
    "deepseek-v4.1-flash": (0, 0.00, 0.00),# 蓝原色相，最深
    "glm-5.3-flash": (-16, -0.09, 0.00),        # 偏蓝紫，稍浅
    "hy4-preview": (0, -0.19, 0.00),            # 同蓝，更浅
    "kimi-k2.8-preview": (14, -0.28, 0.00),     # 更浅的蓝紫
    # ── MiniMax 同厂两代（砖红 355）：同色系，色相/明度/饱和度三向区分
    #   2026-10-20：Space-Bunny 揭秘为 MiniMax-M3.1-Flash，故并入本族。
    "minimax-m3": (0, 0.00, -0.18),
    "space-bunny": (16, -0.12, -0.04),
    # ── OpenCode（青蓝 196）：2 档
    "ling-3.1-flash": (0, 0.00, 0.00),
    "mimo-v2.6-flash": (-12, -0.22, 0.00),
    # ── Qoder（紫 268）：单模型
    "qwen3.8-flash": (0, 0.00, 0.00),
}

# 家族基准饱和度/明度（印刷友好区间：浅底上可读、不抢黑色文字）
# 饱和度 / 明度基准。饱和度 0.38–0.58（避免纯色刺眼）、
# 明度 0.72–0.86（保证白底版面上可读，且不与黑色正文争视觉权重）。
_BASE_SV = {
    "WorkBuddy": (0.54, 0.82),
    "OpenCode": (0.50, 0.80),
    "Qoder": (0.42, 0.74),
    "MiniMax": (0.58, 0.76),     # 砖红：饱和度略高
}
_MINIMAX_FAMILY = "MiniMax"


def _hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _rgb2hex(c):
    return "#%02X%02X%02X" % tuple(int(round(max(0.0, min(1.0, x)) * 255)) for x in c)


def _hsv_hex(h, s, v):
    return _rgb2hex(colorsys.hsv_to_rgb((h % 360) / 360.0,
                                        max(0.0, min(1.0, s)),
                                        max(0.0, min(1.0, v))))


def _build():
    out = {}
    for key, (dh, dl, ds) in _SLOT.items():
        sw = SW_OF[key]
        fam = _MINIMAX_FAMILY if key in ("minimax-m3", "space-bunny") else sw
        s0, v0 = _BASE_SV[fam]
        h = _FAMILY.get(sw, 218) + dh if fam != _MINIMAX_FAMILY else 355 + dh
        out[key] = _hsv_hex(h, s0 + ds, v0 + dl)
    return out


MODEL_COLOR = _build()

# 采集软件的主色（用于图例/ 面板标识）——取该组内最深的一档
SW_COLOR = {sw: MODEL_COLOR[min((k for k in MODEL_COLOR if SW_OF[k] == sw),
                                key=lambda k: _SLOT[k][1])]   # 最浅=最深色
            for sw in _FAMILY}
SW_ORDER = ["WorkBuddy", "OpenCode", "Qoder"]


def model_color(key):
    """模型 key → 配色（未知 key 回退中性灰）。"""
    return MODEL_COLOR.get(key, "#6E6E6E")


def sw_color(sw):
    return SW_COLOR.get(sw, "#6E6E6E")


# 便于审计：把「图标主色 → 最终配色」的映射打印出来核对
if __name__ == "__main__":
    NAMES = {
        "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        "glm-5.3-flash": "GLM-5.3-Flash",
        "hy4-preview": "Hy4 preview",
        "minimax-m3": "MiniMax-M3",
        "kimi-k2.8-preview": "Kimi K2.8 Preview",
        "ling-3.1-flash": "Ling-3.1-Flash",
        "space-bunny": "MiniMax-M3.1-Flash",
        "mimo-v2.6-flash": "MiMo-V2.6-Flash",
        "qwen3.8-flash": "Qwen3.8-Flash",
    }
    ICON_MAIN = {
        "deepseek-v4.1-flash": "#4868F8", "hy4-preview": "#0050E8",
        "minimax-m3": "#F04050", "ling-3.1-flash": "#1078F8",
    }
    print(f"{'模型':<22}{'采集软件':<11}{'图标主色':>10}{'图内配色':>10}"
          f"{'色相':>6}{'S':>6}{'V':>6}")
    print("-" * 71)
    for k in sorted(NAMES, key=lambda x: (SW_OF[x], _SLOT[x][1])):
        c = MODEL_COLOR[k]
        r, g, b = _hex2rgb(c)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        print(f"{NAMES[k]:<22}{SW_OF[k]:<11}"
              f"{ICON_MAIN.get(k, '—'):>10}{c:>10}"
              f"{h * 360:>6.0f}{s:>6.2f}{v:>6.2f}")
    fams = {SW_OF[k] for k in NAMES} | {"MiniMax"}
    print(f"\n色相家族 {len(fams)} 个：{sorted(fams)}（3 个采集软件 + MiniMax 因红色图标单列）")
    print("采集软件主色：", SW_COLOR)