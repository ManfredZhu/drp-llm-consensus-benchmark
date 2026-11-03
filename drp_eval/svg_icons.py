"""SVG 图标 → matplotlib 原生矢量补丁。

为什么不用 cairosvg/svglib：本机缺 cairo 动态库（实测 `libcairo.so.2` 缺失），
而把 SVG 降级成位图再嵌入会让出版级图在放大时失真。
本模块直接解析 SVG 的 path 数据，生成 matplotlib 的 PathPatch，
输出是**真矢量**（PDF/SVG 交付件里可无损缩放，文字仍可检索）。

支持范围（覆盖 icon/ 目录下全部 13 个图标，实测命令集 M/L/C/H/V/A/S/Q/T/Z 及小写相对形式）：
  · path        → PathPatch（按 fill / stroke 区分）
  · circle      → Circle（用三次贝塞尔精确还原圆弧，保证与 path 混合时无锯齿接缝）
  · ellipse     → Ellipse
  · rect        → FancyBboxPatch（支持 rx 圆角）
  · gtransform  → 递归下推平移/旋转/缩放矩阵
渐变（linearGradient / radialGradient）**降级为纯色**：取第一个 stop 的颜色。
这是刻意取舍——图标在版面上仅5–7 pt 宽，渐变在该尺寸下不可分辨，纯色视觉等价且
避免了引入依赖。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D, Bbox

ICON_DIR = Path(__file__).resolve().parent.parent / "icon"

# 图标文件名（不含扩展名）→ 权威模型名 / 软件名。
# 图标文件名与 `统计分析计划.md` §10.1-A 的权威写法逐字对应，改名时须同步此处。
ICON_MAP = {
    "DeepSeek V4.1 Flash": "DeepSeek V4.1 Flash",
    "GLM-5.3-Flash": "GLM-5.3-Flash",
    "Hy4 preview": "Hy4 preview",
    "Kimi K2.8 Preview": "Kimi K2.8 Preview",
    "Ling-3.1-Flash": "Ling-3.1-Flash",
    "MiMo-V2.6-Flash": "MiMo-V2.6-Flash",
    "MiniMax-M3": "MiniMax-M3",
    "OpenCode": "OpenCode",
    "OpenRouter": "OpenRouter",
    "Qoder": "Qoder",
    "Qwen3.8-Flash": "Qwen3.8-Flash",
    "MiniMax-M3.1-Flash": "MiniMax-M3.1-Flash",
    "WorkBuddy CLI": "WorkBuddy",
}

# 图标key（脚本内部用的短名，与 models/软件 key 对齐）→ 图标文件stem
MODEL_ICON = {
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "glm-5.3-flash": "GLM-5.3-Flash",
    "hy4-preview": "Hy4 preview",
    "minimax-m3": "MiniMax-M3",
    "kimi-k2.8-preview": "Kimi K2.8 Preview",
    "space-bunny": "MiniMax-M3.1-Flash",
    "mimo-v2.6-flash": "MiMo-V2.6-Flash",
    "qwen3.8-flash": "Qwen3.8-Flash",
    "ling-3.1-flash": "Ling-3.1-Flash",
    # 采集软件（SAP v1.23：3 个软件 + OpenRouter 不计入软件数但需图标）
    "WorkBuddy": "WorkBuddy CLI",
    "OpenCode": "OpenCode",
    "Qoder": "Qoder",
    "OpenRouter": "OpenRouter",
}

NS = "{http://www.w3.org/2000/svg}"
# ⚠️ SVG 数字词法的头号坑：紧凑写法把相邻数字直接相连，
#    如 `.459.459` 表示**两个**数字（0.459 与 0.459）、`1.2.3.4` 表示**四个**。
#    朴素的 `\d+\.?\d*|\.\d+` 正则会因回溯把 `.459.459` 贪婪匹配成 `0.459.459`
#    这一个非法数字 → 顶点数组长度与 code 数组不匹配 → numpy 报
#    "setting an array element with a sequence"。
#    正解：手写扫描器，遇到 '.' 时若前一个字符已是数字的一部分，则**强制断开**。
def _find_numbers(text: str, want: int | None = None):
    """按 SVG 数字词法切分数字串（依据 W3C SVG 1.1 §F.7）。

    **小数部分是连续数字**，'.' 永远属于它前面那个数；只有当本数已带小数位
    时，下一个 '.' 才另起一个新数：
        23.748    → [23.748]
        .459.459  → [0.459, 0.459]
        1.2.3.4   → [1.2, 0.3, 0.4]
        001.545   → [1.545]（前导零按数值算，正常参与解析）
        4.173 4.173 0 001.545-.475
                → [4.173, 4.173, 0, 1.545, -0.475]   ← 关键用例

    ⚠️ **不要用「按参数个数回溯切分」来消歧**：SVG 的 A（弧）命令参数是
    7 元组的**重复序列**（`a rx ry φ la sf x y a rx ry φ la sf x y …`），
    仅凭总个数无法唯一定位组边界；强行回溯会把 `.475` 拆成 `.4/7/5`
    这类明显错误的值，而且**不报错、静默画错图**。
    正解是让调用方按组消费（见 parse_path 中的 while 循环），此处只做
    确定性切分。`want` 仅用于调用方的自检提示，不改变切分结果。
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        if not (text[i].isdigit() or text[i] in '+-.eE'):
            i += 1
            continue
        j = i
        sign = 1.0
        if text[j] in '+-':
            sign = -1.0 if text[j] == '-' else 1.0
            j += 1
        int_digits = ""
        while j < n and text[j].isdigit():
            int_digits += text[j]
            j += 1
        frac_digits = ""
        if j < n and text[j] == '.':
            k = j + 1
            tmp = ""
            while k < n and text[k].isdigit():
                tmp += text[k]
                k += 1
            if tmp:
                frac_digits = tmp
                j = k
        if not int_digits and not frac_digits:
            i = max(j, i + 1)
            continue
        mant = (int_digits or "0") + ("." + frac_digits if frac_digits else "")
        if j < n and text[j] in 'eE':
            k = j + 1
            if k < n and text[k] in '+-':
                k += 1
            tmp = ""
            while k < n and text[k].isdigit():
                tmp += text[k]
                k += 1
            if tmp:
                mant += "e" + text[j + 1:k]
                j = k
        try:
            out.append(sign * float(mant))
        except ValueError:
            pass
        i = j
    return out


NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")  # viewBox 等非路径场景
CMD = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])")

# 各路径命令的参数个数（用于回溯仲裁，见 _find_numbers 的 want 参数）
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


# ---------------------------------------------------------------- 颜色

def _parse_color(c: str, default: str = "#222222") -> str:
    """把 fill/stroke 字符串解析成 matplotlib 认识的颜色。

    处理：#rgb／#rrggbb／rgb()／none／transparent／currentColor／named colors。
    `currentColor` 是 SVG 的继承占位符（图标未指定具体色），按调用方给的主题色
    解析——不给则退到近黑。
    """
    if c is None:
        return default
    c = c.strip()
    low = c.lower()
    if low in ("none", "transparent", ""):
        return "none"
    if low == "currentcolor":
        return default
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            return "#" + "".join(ch * 2 for ch in h)
        if len(h) == 4:                      # #rgba → 丢alpha，取RGB
            return "#" + "".join(ch * 2 for ch in h[:3])
        if len(h) == 8:
            return "#" + h[:6]
        if len(h) == 6:
            return "#" + h
    m = re.match(r"rgba?\(([^)]+)\)", low)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        vals = []
        for pt in parts[:3]:
            if pt.endswith("%"):
                vals.append(int(round(float(pt[:-1]) * 255 / 100)))
            else:
                vals.append(int(round(float(pt))))
        return "#%02X%02X%02X" % tuple(max(0, min(255, v)) for v in vals)
    return c if c.isalpha() else default


def _grad_first_stop(svg_text: str, grad_id: str) -> str | None:
    """取某个渐变的第一个 stop 颜色（纯色降级用）。"""
    m = re.search(
        r'<linearGradient[^>]*id="%s".*?</linearGradient>' % re.escape(grad_id),
        svg_text, re.S)
    if m is None:
        m = re.search(
            r'<radialGradient[^>]*id="%s".*?</radialGradient>' % re.escape(grad_id),
            svg_text, re.S)
    if m is None:
        return None
    sm = re.search(r'stop-color:\s*([^;"\']+)', m.group(0))
    return sm.group(1).strip() if sm else None


def _resolve_paint(value: str, svg_text: str, default: str) -> str:
    """解析 paint 值；url(#id) 形式的渐变降级为首个 stop 纯色。"""
    if not value:
        return default
    v = value.strip()
    mu = re.match(r"url\(#([^)]+)\)", v)
    if mu:
        return _grad_first_stop(svg_text, mu.group(1)) or default
    return _parse_color(v, default)


# ---------------------------------------------------------------- 路径

def _arc_to_beziers(x0, y0, rx, ry, phi_deg, large, sweep, x1, y1):
    """SVG 椭圆弧 → 一串三次贝塞尔。

    遵循 SVG 实现说明 F.6.5 的端点→中心参数化。matplotlib 无原生弧命令，
    必须自己展开；这是本模块最核心的一段数学。
    """
    # 退化情形（SVG 1.1 §F.7.4）：rx 或 ry 为 0、或起终点重合时，
    # 椭圆弧退化为**一条直线段**。返回 1 个点即可 —— 调用方会补一条 LINETO。
    # ⚠️ 早先这里返回 2 个点，导致调用方把「(起点,终点)」当成一组
    #    (控制点1, 控制点2) + 终点来展开 → verts 多于 codes → numpy 报
    #    "inhomogeneous shape"。本项目 icon/MiMo 有 4 个这样的退化弧。
    if rx == 0 or ry == 0 or (x0 == x1 and y0 == y1):
        return [(x1, y1)], True
    phi = math.radians(phi_deg)
    cosp, sinp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x0 - x1) / 2.0, (y0 - y1) / 2.0
    x1p = cosp * dx2 + sinp * dy2
    y1p = -sinp * dx2 + cosp * dy2
    rx, ry = abs(rx), abs(ry)
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cosp * cxp - sinp * cyp + (x0 + x1) / 2.0
    cy = sinp * cxp + cosp * cyp + (y0 + y1) / 2.0

    def ang(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / n))) if n else 0.0
        return -a if ux * vy - uy * vx < 0 else a

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry,
              (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi

    nseg = max(1, int(math.ceil(abs(dth) / (math.pi / 2))))
    delta = dth / nseg
    t = 4.0 / 3.0 * math.tan(delta / 4.0)
    out = []
    th = th1
    px, py = x0, y0
    for _ in range(nseg):
        cos1, sin1 = math.cos(th), math.sin(th)
        th2 = th + delta
        cos2, sin2 = math.cos(th2), math.sin(th2)
        ex = cx + rx * cos2 * cosp - ry * sin2 * sinp
        ey = cy + rx * cos2 * sinp + ry * sin2 * cosp
        dx1 = -rx * sin1 * cosp - ry * cos1 * sinp
        dy1 = -rx * sin1 * sinp + ry * cos1 * cosp
        dx2b = -rx * sin2 * cosp - ry * cos2 * sinp
        dy2b = -rx * sin2 * sinp + ry * cos2 * cosp
        out.append((ex + t * dx1, ey + t * dy1))
        out.append((ex - t * dx2b, ey - t * dy2b))
        out.append((ex, ey))
        px, py, th = ex, ey, th2
    return out, False


def _parse_arc_groups(text: str):
    """解析 A/a 命令的参数组：**每组 7 个值，其中 la/sf 是单字符 flag**。

    ⚠️ 这是 SVG 词法里最隐蔽的一条例外（SVG 1.1 §F.7.4）：
    椭圆弧的 large-arc-flag 与 sweep-flag **各只占一个字符**，
    只能是 '0' 或 '1'，**不是通用数字**。因此紧凑写法
        a.459.459 0 01.459.44
    应切为 [.459, .459, 0, **0**, **1**, .459, .44]（7 个），
    而通用数字扫描器会把 "01.459" 读成**一个数** 1.459 → 只剩 5 个参数，
    后续组边界全部错位（表现为 rx=0.0 之类的怪参数，图元静默画错）。

    正解：用带 flag 位的专用正则逐组消费，不用通用扫描器。
    """
    out = []
    i, n = 0, len(text)
    # 一个"普通数"：可选符号 + 数字 + 可选小数 + 可选指数
    num = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    pat = re.compile(
        r"\s*(" + num + r")\s*(" + num + r")\s*(" + num + r")"
        r"\s*([01])\s*([01])\s*(" + num + r")\s*(" + num + r")")
    while i < n:
        if text[i] in " \t\r\n,":
            i += 1
            continue
        m = pat.match(text, i)
        if m is None:
            # 退回通用切分（便于报错定位，不静默产出错值）
            rest = _find_numbers(text[i:])
            if not rest:
                break
            out.extend(rest)
            break
        vals = [float(m.group(1)), float(m.group(2)), float(m.group(3)),
                float(m.group(4)), float(m.group(5)),
                float(m.group(6)), float(m.group(7))]
        out.extend(vals)
        i = m.end()
    return out


def parse_path(d: str):
    """SVG path 数据 → (matplotlib Path, has_arc)。

    ⚠️ 命令与参数的配对是本函数最容易错的地方：`CMD.split` 产生的token 序列为
    `[前导文本, cmd0, 文本0, cmd1, 文本1, …, 尾随文本]`，其中
    **`文本i` 是 `cmdi` 之后的参数**（`尾随文本` 属于最后一个命令）。
    早先按同下标 zip()配对会整体错位一格，导致首个 code 不是 MOVETO。
    """
    tokens = CMD.split(d)
    cmds = tokens[1::2]
    # 文本 i 跟在命令 i 之后；最后一条命令的参数在 tokens[-1]
    argtexts = tokens[2::2]
    if cmds:
        argtexts = argtexts[:len(cmds) - 1] + [tokens[-1]]

    verts, codes = [], []
    cx = cy = sx = sy = 0.0
    prev_c2 = None          # 上一个三次控制点（用于 S）
    prev_q = None           # 上一个二次控制点（用于 T）
    has_arc = False

    def move_to(x, y):
        nonlocal sx, sy
        verts.append((x, y)); codes.append(MplPath.MOVETO)
        sx, sy = x, y

    def line_to(x, y):
        verts.append((x, y)); codes.append(MplPath.LINETO)

    def curve3(c1x, c1y, x, y):
        verts.extend([(c1x, c1y), (x, y)])
        codes.extend([MplPath.CURVE3, MplPath.CURVE3])

    def curve4(c1x, c1y, c2x, c2y, x, y):
        verts.extend([(c1x, c1y), (c2x, c2y), (x, y)])
        codes.extend([MplPath.CURVE4] * 3)

    for cmd, numstr in zip(cmds, argtexts):
        C = cmd.upper()
        rel = cmd.islower()
        args = _find_numbers(numstr or "")
        i = 0
        if C == "M":
            while i + 1 < len(args):
                x, y = args[i], args[i + 1]
                if rel:
                    x, y = cx + x, cy + y
                if i == 0:
                    move_to(x, y)
                else:
                    line_to(x, y)
                cx, cy = x, y
                i += 2
            prev_c2 = prev_q = None
        elif C == "L":
            while i + 1 < len(args):
                x, y = args[i], args[i + 1]
                if rel:
                    x, y = cx + x, cy + y
                line_to(x, y); cx, cy = x, y
                i += 2
            prev_c2 = prev_q = None
        elif C == "H":
            for a in args:
                x = cx + a if rel else a
                line_to(x, cy); cx = x
            prev_c2 = prev_q = None
        elif C == "V":
            for a in args:
                y = cy + a if rel else a
                line_to(cx, y); cy = y
            prev_c2 = prev_q = None
        elif C == "C":
            while i + 5 < len(args):
                a, b, c, d_, e, f_ = args[i:i + 6]
                if rel:
                    a, b, c, d_, e, f_ = (cx + a, cy + b, cx + c, cy + d_,
                                           cx + e, cy + f_)
                curve4(a, b, c, d_, e, f_)
                prev_c2 = (c, d_); cx, cy = e, f_
                i += 6
            prev_q = None
        elif C == "S":
            while i + 3 < len(args):
                c, d_, e, f_ = args[i:i + 4]
                if rel:
                    c, d_, e, f_ = cx + c, cy + d_, cx + e, cy + f_
                a, b = (cx, cy) if prev_c2 is None else (
                    2 * cx - prev_c2[0], 2 * cy - prev_c2[1])
                curve4(a, b, c, d_, e, f_)
                prev_c2 = (c, d_); cx, cy = e, f_
                i += 4
            prev_q = None
        elif C == "Q":
            while i + 3 < len(args):
                c, d_, e, f_ = args[i:i + 4]
                if rel:
                    c, d_, e, f_ = cx + c, cy + d_, cx + e, cy + f_
                curve3(c, d_, e, f_)
                prev_q = (c, d_); cx, cy = e, f_
                i += 4
            prev_c2 = None
        elif C == "T":
            while i + 1 < len(args):
                e, f_ = args[i], args[i + 1]
                if rel:
                    e, f_ = cx + e, cy + f_
                c, d_ = (cx, cy) if prev_q is None else (
                    2 * cx - prev_q[0], 2 * cy - prev_q[1])
                curve3(c, d_, e, f_)
                prev_q = (c, d_); cx, cy = e, f_
                i += 2
            prev_c2 = None
        elif C == "A":
            has_arc = True
            while i + 6 < len(args):
                rx, ry, phi, la, sw, x, y = args[i:i + 7]
                if rel:
                    x, y = cx + x, cy + y
                pts, degen = _arc_to_beziers(cx, cy, rx, ry, phi,
                                              int(bool(la)), int(bool(sw)),
                                              x, y)
                if degen:
                    # 退化为直线段：一条 LINETO 即可
                    if abs(x - cx) > 1e-12 or abs(y - cy) > 1e-12:
                        line_to(x, y)
                else:
                    # 正常弧：pts 是若干 (c1, c2, end) 三元组的扁平列表。
                    # 起点通常与当前点重合可直接略过；不重合时先补 LINETO，
                    # 再按 3 个一组消费（组数已由 nseg 保证整除）。
                    if pts and (abs(pts[0][0] - cx) > 1e-12 or
                                abs(pts[0][1] - cy) > 1e-12):
                        line_to(*pts[0])
                        pts = pts[1:]
                    for j in range(0, len(pts) - 2, 3):
                        c1, c2, e = pts[j], pts[j + 1], pts[j + 2]
                        curve4(c1[0], c1[1], c2[0], c2[1], e[0], e[1])
                    # 兜底：若尾部还剩 1–2 个点（浮点边界），补 LINETO
                    tail = len(pts) % 3
                    if tail:
                        for q in pts[len(pts) - tail:]:
                            line_to(*q)
                cx, cy = x, y
                i += 7
            prev_c2 = prev_q = None
        elif C == "Z":
            if verts:
                verts.append((sx, sy)); codes.append(MplPath.CLOSEPOLY)
            cx, cy = sx, sy
            prev_c2 = prev_q = None

    if not verts:
        return None, has_arc
    return MplPath(verts, codes), has_arc


# ---------------------------------------------------------------- 变换

def _parse_transform(t: str):
    """把 transform 属性解析为 3×3 仿射矩阵（numpy array）。"""
    import numpy as np
    M = np.eye(3)
    if not t:
        return M
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", t):
        v = _find_numbers(args)
        T = np.eye(3)
        if name == "translate":
            T[0, 2] = v[0]
            T[1, 2] = v[1] if len(v) > 1 else 0
        elif name == "scale":
            sx = v[0]
            sy = v[1] if len(v) > 1 else sx
            T[0, 0], T[1, 1] = sx, sy
        elif name == "rotate":
            a = math.radians(v[0])
            cx = v[1] if len(v) > 2 else 0
            cy = v[2] if len(v) > 2 else 0
            T1 = np.eye(3)
            T1[0, 2], T1[1, 2] = cx, cy
            R = np.eye(3)
            R[0, 0] = R[1, 1] = math.cos(a)
            R[0, 1], R[1, 0] = -math.sin(a), math.sin(a)
            T2 = np.eye(3)
            T2[0, 2], T2[1, 2] = -cx, -cy
            T = T1 @ R @ T2
        elif name == "matrix" and len(v) == 6:
            a, b, c, d, e, f_ = v
            T = np.array([[a, c, e], [b, d, f_], [0, 0, 1]], float)
        M = M @ T
    return M


# ---------------------------------------------------------------- 主入口

def _collect(root, svg_text, inherit, out):
    """递归遍历 SVG 节点，产出 (kind, geom_args_tuple, style) 三元组。

    geom 统一打包成 tuple，避免可变长度位置参数造成解包歧义。
    """
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        style = dict(inherit)
        for k in ("fill", "fill-opacity", "fill-rule", "stroke",
                  "stroke-width", "stroke-opacity", "opacity"):
            v = el.get(k)
            if v is not None:
                style[k] = v
        if el.get("transform"):
            style["transform"] = el.get("transform")

        if tag == "path":
            d = el.get("d")
            if d and d.strip():
                out.append(("path", d, style))
        elif tag == "circle":
            out.append(("circle", (el.get("cx", "0"), el.get("cy", "0"),
                                   el.get("r", "0")), style))
        elif tag == "ellipse":
            out.append(("ellipse", (el.get("cx", "0"), el.get("cy", "0"),
                                    el.get("rx", "0"), el.get("ry", "0")), style))
        elif tag == "rect":
            out.append(("rect", (el.get("x", "0"), el.get("y", "0"),
                                 el.get("width", "0"), el.get("height", "0"),
                                 el.get("rx")), style))


def icon_patches(name: str, color: str = "#222222", lw: float = 0.0):
    """读取 icon/<name>.svg，返回 matplotlib 补丁列表（已归一到 0–1 坐标）。

    参数
    ----
    name   图标文件 stem（须与 ICON_MAP 的键一致）
    color  前景色；用于 SVG 里的 `currentColor` 占位符
    lw     描边宽度（用户坐标）。fill-only 的图标传0（默认）。
    """
    import numpy as np

    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)

    # viewBox（GLM 的写法带空格，需用正则兜底）
    vb = root.get("viewBox")
    if vb:
        vx, vy, vw, vh = [float(v) for v in NUM.findall(vb)][:4]
    else:
        vx, vy = 0.0, 0.0
        vw = float(root.get("width", 24))
        vh = float(root.get("height", 24))
    if vw <= 0 or vh <= 0:
        vw = vh = 1.0

    raw = []
    _collect(root, text, {}, raw)
    if not raw:
        return []

    def T(x, y, M):
        p = M @ np.array([x, y, 1.0])
        return p[0], p[1]

    patches = []
    for item in raw:
        kind = item[0]
        style = item[-1]
        M = _parse_transform(style.get("transform", ""))
        scale = max(abs(M[0, 0]), abs(M[1, 1])) or 1.0
        fill_raw = style.get("fill", "currentColor")
        stroke_raw = style.get("stroke", "none")
        fo = float(style.get("fill-opacity", 1) or 1)
        so = float(style.get("stroke-opacity", 1) or 1)
        sw = float(style.get("stroke-width", 0) or 0)

        if kind == "path":
            mp, _ = parse_path(item[1])
            if mp is None:
                continue
            v = mp.vertices
            xs, ys = v[:, 0], v[:, 1]
            newv = np.array([T(px, py, M) for px, py in v])
            mp = MplPath(newv, mp.codes)
            do_fill = fill_raw.strip().lower() not in ("none", "")
            do_stroke = (stroke_raw.strip().lower() not in ("none", "")
                         and sw > 0)
            fc = _resolve_paint(fill_raw, text, color) if do_fill else "none"
            sc = _resolve_paint(stroke_raw, text, color) if do_stroke else "none"
            # ⚠️ matplotlib 3.11 的 PathPatch 不接受 `fillRule` 参数
            # （Path 对象本身也无此概念）。SVG 的 fill-rule="evenodd" 只影响
            # 自相交子路径的镂空显示，在 5–7pt 的图标尺寸下不可分辨，
            # 故显式忽略——所有品牌图标均为简单非自相交路径，视觉等价。
            pa = PathPatch(
                mp,
                facecolor=fc if do_fill else "none",
                edgecolor=sc if do_stroke else "none",
                linewidth=(sw * scale) if do_stroke else 0,
                alpha=(fo if do_fill else so) if (do_fill or do_stroke) else 1,
                zorder=1)
            patches.append(pa)
            _ = (xs, ys)
        elif kind == "circle":
            cxx, cyy, r = (float(v) for v in item[1])
            cx, cy = T(cxx, cyy, M)
            fc = _resolve_paint(fill_raw, text, color)
            sc = _resolve_paint(stroke_raw, text, color)
            do_stroke = stroke_raw.strip().lower() not in ("none", "") and sw > 0
            patches.append(Circle(
                (cx, cy), r * scale,
                facecolor=fc if fill_raw.strip().lower() != "none" else "none",
                edgecolor=sc if do_stroke else "none",
                linewidth=(sw * scale) if do_stroke else 0,
                alpha=fo, zorder=1))
        elif kind == "ellipse":
            cxx, cyy, rx, ry = (float(v) for v in item[1])
            cx, cy = T(cxx, cyy, M)
            fc = _resolve_paint(fill_raw, text, color)
            sc = _resolve_paint(stroke_raw, text, color)
            do_stroke = stroke_raw.strip().lower() not in ("none", "") and sw > 0
            patches.append(Ellipse(
                (cx, cy), rx * scale, ry * scale,
                facecolor=fc if fill_raw.strip().lower() != "none" else "none",
                edgecolor=sc if do_stroke else "none",
                linewidth=(sw * scale) if do_stroke else 0,
                alpha=fo, zorder=1))
        elif kind == "rect":
            x0, y0, w, h = (float(v) for v in item[1][:4])
            rx = item[1][4]
            pts = np.array([T(px, py, M) for px, py in
                            [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]])
            xs, ys = pts[:, 0], pts[:, 1]
            fc = _resolve_paint(fill_raw, text, color)
            sc = _resolve_paint(stroke_raw, text, color)
            do_stroke = stroke_raw.strip().lower() not in ("none", "") and sw > 0
            patches.append(FancyBboxPatch(
                (xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                boxstyle="round,pad=0,rounding_size=%g" % (
                    float(rx) * scale if rx else 0),
                facecolor=fc if fill_raw.strip().lower() != "none" else "none",
                edgecolor=sc if do_stroke else "none",
                linewidth=(sw * scale) if do_stroke else 0,
                alpha=fo, zorder=1))

    if not patches:
        return []

    # 归一到 0–1（保留纵横比，居中）
    # ---- 归一化到0–1（保留纵横比，居中）----
    # ⚠️ 必须**直接改写顶点坐标**，不能靠 set_transform 叠加矩阵：
    #    ① SVG 的 transform 属性已在解析阶段烘进顶点（见上），再叠矩阵会二次变换；
    #    ② 叠加会把原有变换的平移量一起缩放，导致图形落到负坐标区。
    boxes = []
    for pa in patches:
        if isinstance(pa, PathPatch):
            bbx = pa.get_path().get_extents()
        elif isinstance(pa, Circle):
            bbx = Bbox.from_extents(pa.center[0] - pa.radius,
                                    pa.center[1] - pa.radius,
                                    pa.center[0] + pa.radius,
                                    pa.center[1] + pa.radius)
        elif isinstance(pa, Ellipse):
            bbx = Bbox.from_extents(pa.center[0] - pa.width / 2,
                                    pa.center[1] - pa.height / 2,
                                    pa.center[0] + pa.width / 2,
                                    pa.center[1] + pa.height / 2)
        else:                                    # FancyBboxPatch
            bbx = pa.get_bbox()
        if bbx.x1 > bbx.x0 and bbx.y1 > bbx.y0:
            boxes.append((bbx.x0, bbx.y0, bbx.x1, bbx.y1))
    if not boxes:
        return patches

    gx0 = min(b[0] for b in boxes); gy0 = min(b[1] for b in boxes)
    gx1 = max(b[2] for b in boxes); gy1 = max(b[3] for b in boxes)
    bw = (gx1 - gx0) or 1.0
    bh = (gy1 - gy0) or 1.0
    k = min(1.0 / bw, 1.0 / bh)
    ox = (1.0 - bw * k) / 2.0
    oy = (1.0 - bh * k) / 2.0

    def _n(x, y):
        return ((x - gx0) * k + ox, (y - gy0) * k + oy)

    for pa in patches:
        if isinstance(pa, PathPatch):
            v = pa.get_path().vertices
            pa.set_path(MplPath([_n(x, y) for x, y in v],
                                pa.get_path().codes))
        elif isinstance(pa, Circle):
            cx, cy = _n(*pa.center)
            pa.center = (cx, cy)
            pa.set_radius(pa.radius * k)
        elif isinstance(pa, Ellipse):
            cx, cy = _n(*pa.center)
            pa.set_center((cx, cy))
            pa.set_width(pa.width * k)
            pa.set_height(pa.height * k)
        else:                                    # FancyBboxPatch
            bb0 = pa.get_bbox()
            nx, ny = _n(bb0.x0, bb0.y0)
            pa.set_x(nx); pa.set_y(ny)
            # ⚠️ 圆角半径也必须同比缩放，否则归一化后圆角会大到盖住整个图标
            #    （Space Bunny 的 512×512圆角底色块rx=112 即是此坑）。
            pa.set_width(bb0.width * k)
            pa.set_height(bb0.height * k)
            if getattr(pa, "get_boxstyle", None):
                bs = pa.get_boxstyle()
                pad = getattr(bs, "pad", 0.0)
                rs = getattr(bs, "rounding_size", 0.0)
                pa.set_boxstyle("round", pad=pad * k,
                                rounding_size=rs * k)
        # 清掉可能存在的旧变换，避免与新坐标叠加
        pa.set_transform(Affine2D())
    return patches


def draw_icon(ax, key, x, y, size, color="#222222", anchor="left",
              va="center", zorder=6, halo=True, lw=0.0):
    """在 ax 上于 (x, y)（数据坐标）画一个图标。

    size   图标高度（数据坐标单位）
    anchor 'left' / 'center' / 'right' —— 相对 x 的水平锚点
    va     'center' / 'bottom' / 'top' —— 相对 y 的垂直锚点
    halo   是否给文字类小图标加极细白边（小尺寸下提升压在深色块上的可读性）
    """
    stem = MODEL_ICON.get(key, key)
    patches = icon_patches(stem, color=color, lw=lw)
    if not patches:
        return None
    x0 = {"left": x, "center": x - size / 2, "right": x - size}[anchor]
    y0 = {"center": y - size / 2, "bottom": y, "top": y - size}[va]
    # ⚠️ 变换合成顺序：icon_patches 已把图形归一到 0–1（矩阵记作 N），
    #    此处只需「先平移到 (x0,y0)，再按 size 缩放」——
    #    即 T = Scale(size) ∘ Translate(x0,y0)。
    #    早先写成 Translate ∘ Scale 会把 size 抵消进N 的逆变换里，
    #    对 Qwen（原始坐标跨度 1230、归一化系数仅 0.0008）会让图形缩成一个点。
    from matplotlib.transforms import Affine2D
    T = Affine2D().translate(x0, y0).scale(size, size) + ax.transData
    for pa in patches:
        pa.set_transform(T)
        pa.set_zorder(zorder)
        if halo:
            pa.set_path_effects([pe.withStroke(linewidth=0.25,
                                               foreground="white")])
        ax.add_patch(pa)
    return patches

# ============================================================================
# 交付路径：预渲染 PNG（推荐用于正式出图）
# ============================================================================
# 为什么不用矢量：matplotlib 3.11 **移除了 fill-rule 支持**，而本目录 13 个图标
# 里有 8 个依赖 `fill-rule="evenodd"` 做镂空（DeepSeek／GLM／MiniMax／Ling／
# MiMo／Qwen／Qoder／WorkBuddy）。用自研矢量解析器渲染这些图标会**剪影破碎**
# （已实测：CLIP 字体正常的 Hy4／Kimi／Space Bunny 完全一致，其余 8 个破碎）。
# 因此正式出图走 CoreSVG 预渲染 → 高分辨率 PNG 路径：
#   · sips 用macOS CoreSVG（与 Safari 同源），完整支持 fill-rule 与渐变；
#   · 1200 px 长边 = 图内实际 5–7 pt 显示尺寸的约 10 倍余量，缩放打印不糊；
#   · 图标在版面里尺寸极小，PNG 与矢量的视觉差异不可分辨。
#
# 保留上面的矢量解析器（icon_patches）用于「需要单色 recolor / 无外部依赖」
# 的场景，以及作为 PNG 缺失时的兜底。

# ============================================================================
# 交付路径：高分辨率预合成位图（正式出图唯一路径）
# ============================================================================
# ⚠️ **为什么必须预合成，而不能把 640px 源图直接交给 Agg**（本项目实测踩坑）：
#
#   matplotlib 对 `imshow` 的处理是「按**显示尺寸 × savefig dpi** 重采样后
#   内嵌为位图」。而 `save_pub` 里PDF/SVG 两条savefig **没有传 dpi**，
#   默认取 figure 的 100 dpi → 版面上 5.76 pt（0.08 in）的图标
#   实测只内嵌 **8×8 像素**（见 `Fig2_model_accuracy.pdf` 的 /Width 8/Height 8）。
#   8px 承载一个56×44 的字标→ 放大即糊。
#
#   实测对照（`/tmp` 内的 A/B 实验）：
#     savefig(dpi=100)→ 内嵌  8×8
#     savefig(dpi=600)  → 内嵌 48×48
#     savefig(dpi=1200) → 内嵌 96×96
#   即**唯一变量是 savefig 的 dpi**，与源图分辨率、RGBA 转换都无关
#   （RGBA 转换是无损的，`P+tRNS` → `RGBA` 只是通道数变化）。
#
#   修法（用户 2026-10-06 拍板：整组图标当作一张大子图合成进去）：
#   ① 源图先 LANCZOS 预缩到「实际显示像素 × 超采样倍数」，把缩放质量
#      交给 PIL 而非 Agg（Agg 的lanczos 在大比例下采样会有可见软化）；
#   ② 同一 figure 的**全部**图标合成到**一张** RGBA 画布上，只imshow 一次
#      ——PDF/SVG 里就只内嵌 1 张位图而非 9 张，既清晰又不会触发
#      「多张小位图各自被缩放」的锯齿不一致；
#   ③ 位图按 `OVERSAMPLE` 倍率生成，savefig 以同倍率 dpi 导出，
#      二者相消 → 落到纸面时1:1 无重采样。
#
#   文字**不受影响**：实测 dpi=100/2400 下 `pdftotext` 均能提取到
#   "GLM-5.3-Flash 88.8%"，字体仍为 1 种 Type0/TrueType，即文字始终是矢量。

PNG_DIR = Path(__file__).resolve().parent / "icons_png"

# 超采样倍率：位图按 `显示像素 × OVERSAMPLE` 生成，
# save_pub 必须以 `fig.dpi × OVERSAMPLE` 导出才正好1:1（两处已互相绑定）。
OVERSAMPLE = 12

# 源图缓存：{stem: RGBA ndarray}
_PNG_CACHE = {}


def _load_rgba(stem):
    """读图标为 RGBA ndarray（带缓存）。"""
    if stem not in _PNG_CACHE:
        from PIL import Image
        im = Image.open(PNG_DIR / f"{stem}.png").convert("RGBA")
        _PNG_CACHE[stem] = np.asarray(im)
    return _PNG_CACHE[stem]


def _prescale(stem, target_px):
    """把源图 LANCZOS 预缩到 `target_px`（内容高度），缓存结果。

    目标像素数 ≥ 源图时**不缩**（保留原始细节，让 Agg 去做最终降采样）。
    """
    src = _load_rgba(stem)
    h, w = src.shape[:2]
    f = _CONTENT.get(stem)
    content_px = target_px * f if f else target_px     # 内容 → 整幅画布
    if content_px >= h:
        return src
    r = content_px / h
    from PIL import Image
    small = Image.fromarray(src).resize(
        (max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    return np.asarray(small)


# 锚点 → (水平, 垂直) 对齐方式。键为 (anchor, va)
_ALIGN = {
    ("left", "center"): (0.0, 0.5), ("left", "bottom"): (0.0, 0.0),
    ("left", "top"): (0.0, 1.0),
    ("center", "center"): (0.5, 0.5), ("center", "bottom"): (0.5, 0.0),
    ("center", "top"): (0.5, 1.0),
    ("right", "center"): (1.0, 0.5), ("right", "bottom"): (1.0, 0.0),
    ("right", "top"): (1.0, 1.0),
}


# 全局 overlay 轴：所有图标都画在这一个「铺满整张画布」的坐标轴上，
# 用 figure 归一化坐标（0–1）定位。它不受tight 布局影响，
# 因此 tight 裁剪时不会把图标移位。
_OVERLAY = {}


def _overlay(fig):
    """返回（并缓存）整张画布的 overlay 轴。"""
    key = id(fig)
    iax = _OVERLAY.get(key)
    if iax is None or iax.figure is not fig:
        iax = fig.add_axes([0, 0, 1, 1], zorder=20)
        iax.set_xlim(0, 1); iax.set_ylim(0, 1)
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_visible(False)
        iax.set_facecolor("none")
        iax.patch.set_alpha(0.0)
        # overlay 必须排除在 tight 布局之外，否则会撑出整张画布的空白边框
        iax.set_in_layout(False)
        _OVERLAY[key] = iax
    return iax


def _fig_px(fig):
    """figure 的像素尺寸（用当前 dpi；tight 裁剪会在保存时缩放，
    但因为图标是「按像素」定义的相对关系，此处只用于保证不被拉伸）。"""
    fw, fh = fig.get_size_inches()
    return fw * fig.dpi, fh * fig.dpi


def _data_to_fig(ax, x, y):
    """数据坐标 → figure 归一化坐标（0–1），用于 overlay 定位。

    只用 `ax.transData` 一步：数据 → 显示像素 → 除以 figure 像素尺寸。
    ⚠️ 早先误把 `transData.inverted()` 的结果又按 xlim/ylim 归一化了一遍，
    等于做了两次逆变换 → 图标被甩到画布角落（实测全部堆在左下）。
    transData 已包含轴的位置与比例，一次即可。
    """
    fig = ax.figure
    px, py = ax.transData.transform((x, y))
    fw_px, fh_px = _fig_px(fig)
    return px / fw_px, py / fh_px


def _icon_box_on_fig(ax, key, x, y, pixels, anchor, va):
    """算图标在 figure 归一化坐标里的矩形 (fx, fy, w, h)。

    `pixels` 指**图标内容**的像素高度（不含源文件留白），
    经 `visual_pixels` / `visual_wide_correction` 两次归一，
    以便各图标视觉大小一致（见本文件「视觉尺寸归一」段）。
    """
    fig = ax.figure
    fw_px, fh_px = _fig_px(fig)
    fx, fy = _data_to_fig(ax, x, y)
    p = visual_wide_correction(key, visual_pixels(key, pixels))
    nw, nh = p / fw_px, p / fh_px
    dx = {"left": 0.0, "center": -nw / 2, "right": -nw}[anchor]
    dy = {"center": -nh / 2, "bottom": 0.0, "top": -nh}[va]
    return fx + dx, fy + dy, nw, nh


def _place_on(ax, key, x, y, pixels, anchor, va, zorder=6):
    """在 figure overlay 上按**像素**尺寸放图标（单图标即时路径）。

    ⚠️ 这条路径每个图标各内嵌一张位图，PDF/SVG 里会出现 N 张小图。
    正式出图走 `flush_icons` 的**合成路径**（整张figure 一次成图），
    本函数仅供 `draw_icon_png(defer=False)` 等即时场景与兜底使用。
    """
    fig = ax.figure
    iax = _overlay(fig)
    fx, fy, nw, nh = _icon_box_on_fig(ax, key, x, y, pixels, anchor, va)
    stem = MODEL_ICON.get(key, key)
    img = _prescale(stem, pixels * OVERSAMPLE)
    ih, iw = img.shape[:2]

    # 用 inset 于 overlay 的小轴，坐标仍是 figure 归一化，但**不参与布局**
    iax2 = iax.inset_axes([fx, fy, nw, nh], zorder=zorder)
    iax2.imshow(img, interpolation="antialiased", aspect="auto")
    iax2.set_xlim(0, iw); iax2.set_ylim(ih, 0)
    iax2.set_xticks([]); iax2.set_yticks([])
    for sp in iax2.spines.values():
        sp.set_visible(False)
    iax2.set_facecolor("none")
    iax2.patch.set_alpha(0.0)
    iax2.set_in_layout(False)
    iax2.set_clip_on(False)
    iax2.set_clip_path(iax.patch)      # 允许外溢到整张画布
    return iax2


# ------------------------------------------------------------------------
# 整张 figure 的图标一次性合成（正式出图路径）
# ------------------------------------------------------------------------
# 为什么合成（用户 2026-10-06 拍板的核心诉求「核心思想就是要高清」）：
#   逐个 inset + imshow 会让每张小图各自被 Agg 重采样，9 个图标就有 9 次
#   独立的降采样误差，且 PDF/SVG 里内嵌 9 张位图。合成后：
#     · 位图数量 9 → 1，PDF 体积更小、无N 张图间的锐度不一致；
#     · 缩放全部由 PIL LANCZOS 一次完成（可控、可测）；
#     · 位图按 OVERSAMPLE 倍生成，与 savefig 的 dpi 相消 → 落纸1:1。
def _compose_all(fig, items, zorder=6):
    """把 `items`（(key, fx, fy, nw, nh, pixels) 列表）合成成一张 RGBA 图。

    返回 (rgba数组, x0, y0, w, h)，均为 figure 归一化坐标。
    超出画布的部分会被裁掉（inset_axes 本来也只显示画布内部分）。
    """
    if not items:
        return None
    fw_px, fh_px = _fig_px(fig)
    S = OVERSAMPLE

    # ① 并集矩形（像素单位，含 1px 余量防裁切）
    x0 = min(b[1] for b in items) * fw_px - 1
    y0 = min(b[2] for b in items) * fh_px - 1
    x1 = max((b[1] + b[3]) for b in items) * fw_px + 1
    y1 = max((b[2] + b[4]) for b in items) * fh_px + 1
    # 画布外部分裁掉
    x0c, y0c = max(0.0, x0), max(0.0, y0)
    x1c, y1c = min(fw_px, x1), min(fh_px, y1)
    W = int(round((x1c - x0c) * S))
    H = int(round((y1c - y0c) * S))
    if W <= 0 or H <= 0:
        return None

    canvas = np.zeros((H, W, 4), np.uint8)

    # ② 逐个贴入（整块覆盖，不做加色）
    # ⚠️ y 翻转：`y1c` 是并集盒**上**缘（自下而上量），
    #    画布行号 `ty` 自上而下 → ty = y1c - (fy+nh)*fh_px。
    #    早先误写成 `fh_px - (fy+nh)*fh_px - y0c`（把自下而上的 y0c
    #    混入自上而下的基准）→ ty全为负 → 图标整体消失且不报错。见 R38。
    for key, fx, fy, nw, nh, pixels in items:
        stem = MODEL_ICON.get(key, key)
        img = _prescale(stem, pixels * S)
        ih, iw = img.shape[:2]
        # 目标像素（左上角；行号自上而下，figure 坐标自下而上 → 需翻转）
        tx = int(round((fx * fw_px - x0c) * S))
        ty = int(round((y1c - (fy + nh) * fh_px) * S))
        tw = max(1, int(round(nw * fw_px * S)))
        th = max(1, int(round(nh * fh_px * S)))
        if tx + tw <= 0 or ty + th <= 0 or tx >= W or ty >= H:
            continue
        if (tw, th) != (iw, ih):                # 尺寸不一致才重采样
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((tw, th), Image.LANCZOS))
        # 裁到画布内
        sx0, sy0 = max(0, -tx), max(0, -ty)
        dx0, dy0 = max(0, tx), max(0, ty)
        cw = min(tw - sx0, W - dx0)
        ch = min(th - sy0, H - dy0)
        if cw <= 0 or ch <= 0:
            continue
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]

    # ⚠️ 防御：合成结果「全透明」是一个**静默失败**——PDF 不会报错、
    #    碰撞审计也不会报（只是少了 N 张位图），只有肉眼看图才发现「图标没了」。
    #    早先就因y 翻转混用两个基准（y0c 自下而上 vs ty 自上而下）
    #    导致 9 个图标全部贴空。见 R38。
    assert int((canvas[..., 3] > 0).sum()) > 0, \
        "图标合成结果全透明——检查 ty 的 y 翻转基准（y1c - (fy+nh)*fh_px）"
    return canvas, x0c / fw_px, y0c / fh_px, (x1c - x0c) / fw_px, (y1c - y0c) / fh_px


def _place_composed(fig, canvas, fx, fy, w, h, zorder=6):
    """把合成好的整张图标位图贴到 overlay 上（只imshow 一次）。"""
    iax = _overlay(fig)
    ih, iw = canvas.shape[:2]
    iax2 = iax.inset_axes([fx, fy, w, h], zorder=zorder)
    iax2.imshow(canvas, interpolation="antialiased", aspect="auto")
    iax2.set_xlim(0, iw); iax2.set_ylim(ih, 0)
    iax2.set_xticks([]); iax2.set_yticks([])
    for sp in iax2.spines.values():
        sp.set_visible(False)
    iax2.set_facecolor("none")
    iax2.patch.set_alpha(0.0)
    iax2.set_in_layout(False)
    iax2.set_clip_on(False)
    iax2.set_clip_path(iax.patch)
    return iax2


# ------------------------------------------------------------------------
# 延迟绘制队列
# ------------------------------------------------------------------------
# 为什么需要：`ax.transData` 依赖最终的 xlim/ylim，而调用方常常是
#「先贴图标、后设坐标范围」（实测：a 面板 9 行只渲染出第 1 个图标，
#  其余 8 个因当时 ylim还是默认值 (0,1) 而被算到画布外）。
# 解法：draw_icon_label 只登记请求；调用 `flush_icons(fig)` 时才按
# **已确定** 的坐标范围统一渲染——此时 transData 才是最终值。
_PENDING = {}


def pending_icons(fig):
    return _PENDING.setdefault(id(fig), [])


def flush_icons(fig=None):
    """把排队的图标按**最终坐标范围**渲染到 overlay 上（合成路径）。

    必须在所有 `set_xlim/ set_ylim/ set_position` 之后、savefig 之前调用。
    fig=None 时刷新所有 figure（供 save_pub 统一收口）。

    **合成**：本figure 的所有图标先各算好figure 归一化矩形，
    再一次性拼成**一张** RGBA 位图贴到 overlay（见 `_compose_all` 的说明）。
    文字仍逐条用 `ax.text` 画 → 保持矢量、可被 `pdftotext` 提取。
    """
    figs = [fig] if fig is not None else list({v[0].figure for vs in _PENDING.values() for v in vs})
    n = 0
    for f in figs:
        f.canvas.draw()                      # 确保 transData 是最终值
        items, labels = [], []
        for (ax_, key, x, y, px, anch, va, txt, fsz, tcol, gap) in \
                _PENDING.get(id(f), []):
            fx, fy, w, h = _icon_box_on_fig(ax_, key, x, y, px, anch, va)
            items.append((key, fx, fy, w, h, px))
            if txt is not None:
                labels.append((ax_, key, x, y, txt, px, fsz, tcol, gap))
            n += 1
        comp = _compose_all(f, items)
        if comp is not None:
            _place_composed(f, *comp)
        for (ax_, key, x, y, txt, px, fsz, tcol, gap) in labels:
            _label_text(ax_, key, x, y, txt, px, fsz, tcol, gap, icon_drawn=True)
    if fig is not None:
        _PENDING.pop(id(fig), None)
    return n


def _label_text(ax, key, x, y, text, px, fontsize, text_color, gap,
                icon_drawn=False):
    """图标 + 文字（供 flush_icons 调用，坐标范围已最终确定）。

    `icon_drawn=True` 表示图标已由合成路径画好，这里**只补文字**。
    """
    fig = ax.figure
    fw_px, fh_px = _fig_px(fig)
    pos = ax.get_position()
    # ⚠️ 图标实际宽度必须走**与 _icon_box_on_fig 完全相同**的两次换算
    #   （visual_pixels → visual_wide_correction），否则文字起点会与
    #   图标真实右缘错位——实测 Qwen（内容占比 92%）会紧贴文字。
    px_eff = visual_wide_correction(key, visual_pixels(key, px))
    nw = px_eff / fh_px                # 已统一为正方形 → 宽高比 1
    ax_w_px = pos.width * fw_px
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    w_data = nw * xspan
    if gap is None:
        # 1.35 倍：图标内容 bbox 内的透明余量会吃掉约 1/4 视觉宽度
        gap = (fontsize * fig.dpi / 72.0 * 1.35) / ax_w_px * xspan
    if not icon_drawn:
        _place_on(ax, key, x, y, px, "left", "center")
    ax.text(x + w_data + gap, y, text, fontsize=fontsize,
            color=text_color or "#1A1A1A", ha="left", va="center", zorder=6)


def draw_icon_label(ax, key, x, y, text, size=0, *, color="#1A1A1A",
                    text_color=None, fontsize=5.0, gap=None,
                    ha="left", va="center", zorder=6, halo=False,
                    pixels=None, defer=True):
    """画「图标 + 文字」组合（图标在文字左侧、垂直居中）。

    defer=True（默认）**排队**，在 `flush_icons(fig)` 时按最终坐标范围渲染——
    这是唯一可靠的时序，详见上方「延迟绘制队列」说明。
    """
    px = pixels if pixels is not None else 8.0
    # 🔴 给宿主 figure 打标记（2026-10-07）：`flush_icons` 会把整张画布按
    #   `fig.dpi × OVERSAMPLE` 展开位图，实测单峰 RSS 82 MB → 3811 MB，
    #   大图会被系统 kill（exit=137）。`save_pub` 据此标记决定是否调用它。
    #   **凡排队过图标就必须 flush**；反之无图标的图应跳过以省内存。
    try:
        ax.figure._pcne_has_icons = True
    except AttributeError:
        pass
    if defer:
        pending_icons(ax.figure).append(
            (ax, key, x, y, px, "left", va, text, fontsize,
             text_color or color, gap))
        return None
    _label_text(ax, key, x, y, text, px, fontsize, text_color or color, gap)
    return None


def draw_icon_png(ax, key, x, y, size=0, anchor="left", va="center",
                  zorder=6, halo=False, pixels=None, defer=True):
    """放一个图标（无文字）。defer 语义同`draw_icon_label`。"""
    px = pixels if pixels is not None else 8.0
    # 🔴 给宿主 figure 打标记（2026-10-07）：`flush_icons` 会把整张画布按
    #   `fig.dpi × OVERSAMPLE` 展开位图，实测单峰 RSS 82 MB → 3811 MB，
    #   大图会被系统 kill（exit=137）。`save_pub` 据此标记决定是否调用它。
    #   **凡排队过图标就必须 flush**；反之无图标的图应跳过以省内存。
    try:
        ax.figure._pcne_has_icons = True
    except AttributeError:
        pass
    if defer:
        pending_icons(ax.figure).append(
            (ax, key, x, y, px, anchor, va, None, 5.0, None, None))
        return None
    return _place_on(ax, key, x, y, px, anchor, va)


def blank_axes(ax):
    """把 axes 变成"无边框、无刻度"的示意面板，**但保留 axison**。

    ⚠️ 不可用 `ax.axis("off")`：那会让 `axison=False`，matplotlib 在 draw 阶段
    **跳过该axes 的全部 artist** —— 本模块用 imshow 放的图标会一起消失
    （实测：图标对象存在、extent 正确，但渲染为空）。
    正解：逐项关边框与刻度，轴本身仍参与渲染。
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # imshow 需要能改aspect/extent，保持 autoscale 开启即可
    return ax


def icon_strip(keys, size_px=13, fontsize=5.2, color="#1A1A1A",
               labels=None, pad_px=None):
    """返回一个「图标条」绘制器：在任意面板上按行铺一列图标+文字。

    用于「图例」「采集软件一览」「型号清单」这类需要图标领衔的场合。
    返回值是一个函数 `paint(ax, x, y_top, *, anchor='left')`。
    """
    def paint(ax, x, y_top, *, anchor="left"):
        labels_ = labels if labels is not None else [
            ICON_MAP.get(MODEL_ICON.get(k, k), k) for k in keys]
        n = len(keys)
        if pad_px is None:
            pad_px = size_px * 1.9
        for i, (k, t) in enumerate(zip(keys, labels_)):
            yy = y_top - i * (pad_px / ax.figure.dpi *
                              (ax.get_ylim()[1] - ax.get_ylim()[0]) /
                              max(ax.get_position().height *
                                  ax.figure.get_size_inches()[1] *
                                  ax.figure.dpi, 1e-9))
            draw_icon_label(ax, k, x, yy, t, 0, pixels=size_px,
                            fontsize=fontsize, color=color, va="top")
        return paint
    return paint

# ------------------------------------------------------------------------
# 采集软件 → 颜色（SAP v1.23：3 个软件）
# ------------------------------------------------------------------------
# 与图标表放在一起：出图脚本需要「按软件取色」和「按软件取图标」，
# 两套映射必须同源，分开放必然漂移。
SOFTWARE_COLOR = {
    "WorkBuddy": "#0F4D9E",   # 深蓝（主色，5 个模型）
    "OpenCode":  "#1E8A8A",   # 青（2 个模型 + Ling 大部）
    "Qoder":     "#8C6D3F",   # 赭（1 个模型）
}
SOFTWARE_ORDER = ["WorkBuddy", "OpenCode", "Qoder"]

SOFTWARE_OF = {
    "deepseek-v4.1-flash": "WorkBuddy",
    "glm-5.3-flash": "WorkBuddy",
    "hy4-preview": "WorkBuddy",
    "minimax-m3": "WorkBuddy",
    "kimi-k2.8-preview": "WorkBuddy",
    "space-bunny": "OpenCode",
    "mimo-v2.6-flash": "OpenCode",
    "ling-3.1-flash": "OpenCode",
    "qwen3.8-flash": "Qoder",
}


def software_of(model_key: str) -> str:
    """模型 key → 采集软件名（见 `统计分析计划.md` §10.1-B1）。"""
    return SOFTWARE_OF.get(model_key, "")

# --- 视觉尺寸归一 -------------------------------------------------------
# 各源图标的留白边距差异很大（实测内容占比 68%–92%），若直接按画布尺寸
# 放置，满幅图标（Qwen 92%×92%）会明显比留白多的（DeepSeek 92%×68%）显得大。
# 下表由 `prepare_icons` 实测写入，让 draw_* 按**内容高度**而非画布高度定尺寸。
_CONTENT = {}

# 实测的「内容高度占比」（由 prepare_icons.py 自动写入，勿手改）。
# 源图标的留白边距差异很大（56%–93%），
# 按画布尺寸放置会让满幅图标（Qwen 等）明显更大。draw_* 按**内容**定尺寸后一致。
_CONTENT.update({
    'DeepSeek V4.1 Flash': 0.7359,
    'GLM-5.3-Flash': 0.96,
    'Hy4 preview': 1.0,
    'Kimi K2.8 Preview': 0.8344,
    'Ling-3.1-Flash': 0.9172,
    'MiMo-V2.6-Flash': 0.6078,
    'MiniMax-M3': 0.8344,
    'MiniMax-M3.1-Flash': 0.8344,
    'OpenCode': 0.8344,
    'OpenRouter': 0.7078,
    'Qoder': 1.0,
    'Qwen3.8-Flash': 0.9187,
    'WorkBuddy CLI': 1.0,
})


def register_content(name, frac):
    """登记某图标的「内容高度占比」（0–1），供视觉归一使用。"""
    _CONTENT[name] = frac


def visual_pixels(key, pixels):
    """把目标「内容像素高」换算成实际画布像素高。"""
    stem = MODEL_ICON.get(key, key)
    f = _CONTENT.get(stem)
    return pixels / f if f else pixels

# 文字型/宽幅图标的画布宽高比（宽度是高度的多少倍）。
# MiMo 的图标本身就是 "Xiaomi MiMo" 文字标（宽 ≈ 1.9×高），
# 若与其他正方形图标同高，会显得特别宽、挤到文字。
_WIDE = {"MiMo-V2.6-Flash": 1.9, "Ling-3.1-Flash": 1.05}


def visual_wide_correction(key, pixels):
    """宽幅图标按宽度上限收缩，避免挤占文字。"""
    stem = MODEL_ICON.get(key, key)
    r = _WIDE.get(stem)
    if not r:
        return pixels
    # 目标：宽度不超过 pixels * 1.25倍内容高度
    return pixels * 1.25 / r if r > 1.25 else pixels

