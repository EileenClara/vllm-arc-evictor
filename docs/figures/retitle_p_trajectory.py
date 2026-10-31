#!/usr/bin/env python3
"""Rewrite the baked-in title of figures/p_trajectory_v0_v1.png.

The original title asserted "not a port bug", a hypothesis this project later
disproved (see docs/migration-v0-v1.md, trap 1). The plotted data is correct and
matches the final tables, so the image is kept and only the claim is replaced.

Run: python docs/figures/retitle_p_trajectory.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib

SRC = Path(__file__).resolve().parents[2] / "figures" / "p_trajectory_v0_v1.png"
NEW_TITLE = ("V0 vs V1 ARC p trajectory (QuALITY, 3072 requests) — "
             "the controller behaves the same on both engines")

BAND = 62          # height of the title strip to repaint
MARGIN = 60        # horizontal safety margin


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")

    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # sample the background just below the band so we repaint with the true colour
    bg = img.getpixel((w // 2, BAND + 4))
    draw.rectangle([0, 0, w, BAND], fill=bg)

    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    size = 30
    while size > 14:
        font = ImageFont.truetype(str(font_path), size)
        left, top, right, bottom = draw.textbbox((0, 0), NEW_TITLE, font=font)
        if (right - left) <= (w - 2 * MARGIN):
            break
        size -= 1

    tw, th = right - left, bottom - top
    draw.text(((w - tw) / 2 - left, (BAND - th) / 2 - top), NEW_TITLE, fill=(20, 20, 20), font=font)

    out = Path(__file__).resolve().parent / "fig_p_trajectory.png"
    img.save(out, dpi=(200, 200))
    print(f"wrote {out.name}  {w}x{h}  font={size}px")


if __name__ == "__main__":
    main()
