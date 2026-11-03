#!/usr/bin/env python
"""图标预处理：把 `icon/` 的原始 PNG 归一为出图可直接使用的 RGBA 图标。

**为什么需要这一步**（三个实测问题）：
1. **模式不统一**：12 个是调色板（P + tRNS），1 个（Space Bunny）是 RGBA。
   matplotlib 的 `imshow` 对调色板支持有限 → 统一转 RGBA。
2. **GLM 是黑底徽标**：四角不透明且 81% 纯黑。白底版面上一个纯黑方块很突兀
   → 保留黑底（那是它的品牌设计），但加一圈浅灰描边以便识别。
3. **留白边距差异极大**：实测内容高度占比 56%（MiMo）– 93%（Hy4）。
   若按画布尺寸放置，满幅图标会明显更大 → **裁到内容 bbox + 4% 余量，
   补齐为正方形**，再把实测占比写回 `svg_icons._CONTENT`，让 `draw_*`
   按**内容高度**定尺寸，视觉大小才一致。

用法（源图更新后重跑）：
```bash
/Users/zhumaohua/.workbuddy/binaries/python/envs/default/bin/python \
    drp_eval/prepare_icons.py
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "icon"
DST = Path(__file__).resolve().parent / "icons_png"
PAD = 0.04                     # 裁切后四周余量（占内容边长）
MAX_PX = 1200                  # 输出上限（600dpi 印刷下约需 170px，留 7 倍余量）

# 图标键（须与 svg_icons.MODEL_ICON 的值一致）→ 源文件名
# 唯一例外：图标文件叫 `WorkBuddy.png`，但键是 `WorkBuddy CLI`
KEYS = {
    "DeepSeek V4.1 Flash": "DeepSeek V4.1 Flash",
    "GLM-5.3-Flash": "GLM-5.3-Flash",
    "Hy4 preview": "Hy4 preview",
    "Kimi K2.8 Preview": "Kimi K2.8 Preview",
    "MiniMax-M3": "MiniMax-M3",
    "Ling-3.1-Flash": "Ling-3.1-Flash",
    "MiMo-V2.6-Flash": "MiMo-V2.6-Flash",
    "Qwen3.8-Flash": "Qwen3.8-Flash",
    "MiniMax-M3.1-Flash": "MiniMax-M3.1-Flash",
    "WorkBuddy CLI": "WorkBuddy",
    "OpenCode": "OpenCode",
    "Qoder": "Qoder",
    "OpenRouter": "OpenRouter",
}


def _square(im, pad=PAD):
    """裁掉留白并补齐为正方形。"""
    bb = im.getchannel("A").getbbox()
    if bb:
        im = im.crop(bb)
    w, h = im.size
    side = max(w, h)
    p = round(side * pad)
    canvas = Image.new("RGBA", (side + 2 * p, side + 2 * p), (0, 0, 0, 0))
    canvas.paste(im, ((canvas.width - w) // 2, (canvas.height - h) // 2), im)
    return canvas


def _content_frac(im):
    """内容高度占画布的比例。"""
    bb = im.getchannel("A").getbbox()
    return round((bb[3] - bb[1]) / im.height, 4) if bb else 1.0


def prepare():
    DST.mkdir(exist_ok=True, parents=True)
    for f in DST.glob("*.png"):
        f.unlink()

    info = {}
    print(f"{'图标键':<22}{'源文件':<24}{'内容占比':>9}{'输出':>12}  处理")
    print("-" * 78)
    for key, fname in KEYS.items():
        src = SRC / f"{fname}.png"
        if not src.exists():
            print(f"{key:<22}❌ 源缺失：{fname}.png")
            continue
        im = Image.open(src).convert("RGBA")
        a = np.array(im)
        al = a[:, :, 3]
        corners = [al[0, 0], al[0, -1], al[-1, 0], al[-1, -1]]
        note = ""
        if all(c >= 250 for c in corners):            # 四角不透明 → 有底色
            rgb = a[:, :, :3].astype(np.int16)
            if (rgb.max(axis=2) <= 40).mean() > 0.5:  # 黑底徽标（GLM）
                w0 = max(2, int(al.shape[0] * 0.02))
                edge = np.zeros(al.shape, bool)
                edge[:w0, :] = edge[-w0:, :] = edge[:, :w0] = edge[:, -w0:] = True
                a[:, :, 3] = np.where(edge, 0, al)
                note = "黑底·去外黑边"
            else:                                     # 白底
                a[:, :, 3] = np.where(rgb.min(axis=2) >= 245, 0, al)
                note = "白底·抠除"
        im = Image.fromarray(a, "RGBA")
        info[key] = _content_frac(im)
        canvas = _square(im)

        if "黑底" in note:                            # 补浅灰描边
            ring = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ring)
            d.rounded_rectangle(
                [0, 0, canvas.width - 1, canvas.height - 1],
                radius=canvas.width // 9,
                outline=(206, 212, 220, 255),
                width=max(2, canvas.width // 60))
            ring.alpha_composite(canvas)
            canvas = ring
            note += "+描边"

        if max(canvas.size) > MAX_PX:
            r = MAX_PX / max(canvas.size)
            canvas = canvas.resize(
                (round(canvas.width * r), round(canvas.height * r)),
                Image.LANCZOS)
        canvas.save(DST / f"{key}.png", optimize=True)
        print(f"{key:<22}{fname + '.png':<24}"
              f"{info[key] * 100:>8.1f}%{f'{canvas.width}x{canvas.height}':>12}  "
              f"{note or '透明底·裁切'}")

    # 把实测占比写回svg_icons._CONTENT，供视觉尺寸归一
    _write_content(info)
    print(f"\n✅ {len(info)}/13 已归一→ {DST}")
    print(f"   内容占比范围 {min(info.values()):.1%} – {max(info.values()):.1%}"
          f"（视觉归一已启用）")


def _write_content(info):
    """把实测占比写进 svg_icons.py（保持模块与产物同步，勿手改占比）。"""
    mod = Path(__file__).resolve().parent / "svg_icons.py"
    s = mod.read_text(encoding="utf-8")
    start = s.index("_CONTENT.update({")
    end = s.index("})", start) + 2
    block = ("_CONTENT.update({\n"
             + "\n".join(f"    {k!r}: {v}," for k, v in sorted(info.items()))
             + "\n})")
    s = s[:start] + block + s[end:]
    mod.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    prepare()