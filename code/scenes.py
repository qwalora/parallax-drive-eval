"""Deterministic renderer for synthetic driving scenes (PARALLAX-Drive / DriveGuard).

Every image is drawn from primitives with Pillow and one bundled font (DejaVu Sans, see
fonts/LICENSE). Nothing is copied from any third-party dataset. A scene is fully described by a
SceneSpec, so the same spec always renders the same pixels with the pinned Pillow version.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import os

from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 640, 360
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "DejaVuSans-Bold.ttf")


@dataclass
class SceneSpec:
    light: str = "red"                 # red | yellow | green | none
    stop_sign: bool = False
    pedestrian: bool = False           # pedestrian on the crosswalk ahead
    speed_limit: Optional[int] = None  # posted speed limit sign, or None
    vehicles: int = 1                  # vehicles ahead in adjacent lane, 0..2
    condition: str = "day"             # day | dusk | night | fog | rain
    billboard: Optional[str] = None    # text on the roadside billboard
    sign_sticker: Optional[str] = None # text on a sticker on the signal pole
    road_text: Optional[str] = None    # text painted on the road surface
    layout_seed: int = 0               # small positional jitter, deterministic
    tags: List[str] = field(default_factory=list)


def _font(size):
    return ImageFont.truetype(FONT, size)


def _jit(seed, k, span):
    # deterministic jitter in [-span, span] without global random state
    v = (seed * 2654435761 + k * 40503) % 1000003
    return (v % (2 * span + 1)) - span


def _wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def _fit_text(draw, text, box, max_size, min_size=9):
    x0, y0, x1, y1 = box
    for size in range(max_size, min_size - 1, -1):
        f = _font(size)
        lines = _wrap(draw, text, f, x1 - x0 - 8)
        lh = int(size * 1.18)
        if lh * len(lines) <= (y1 - y0 - 6):
            return f, lines, lh
    f = _font(min_size)
    return f, _wrap(draw, text, f, x1 - x0 - 8), int(min_size * 1.18)


def render(spec: SceneSpec) -> Image.Image:
    s = spec.layout_seed
    sky = {"day": (150, 195, 235), "dusk": (235, 160, 120), "night": (20, 24, 45),
           "fog": (200, 205, 210), "rain": (120, 130, 145)}[spec.condition]
    ground = {"day": (95, 140, 80), "dusk": (90, 110, 70), "night": (25, 35, 25),
              "fog": (160, 170, 160), "rain": (70, 95, 65)}[spec.condition]
    img = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(img)
    hy = 150 + _jit(s, 1, 6)                      # horizon
    d.rectangle([0, hy, W, H], fill=ground)
    # road (perspective trapezoid) and markings
    road = (70, 70, 74) if spec.condition != "night" else (35, 35, 40)
    vx = W // 2 + _jit(s, 2, 20)
    d.polygon([(vx - 18, hy), (vx + 18, hy), (W - 20, H), (20, H)], fill=road)
    for i in range(7):                             # dashed centre line
        t0, t1 = i / 7.0, i / 7.0 + 0.06
        y0, y1 = hy + (H - hy) * t0 ** 1.6, hy + (H - hy) * t1 ** 1.6
        wv = 1 + 5 * t0
        d.polygon([(vx - wv, y0), (vx + wv, y0), (vx + wv + 2, y1), (vx - wv - 2, y1)], fill=(235, 220, 90))
    # stop line and crosswalk
    cy = hy + 95
    left = 20 + (vx - 18 - 20) * (H - cy) / (H - hy)
    right = (W - 20) - ((W - 20) - (vx + 18)) * (H - cy) / (H - hy)
    for k in range(10):
        xa = left + (right - left) * k / 10.0 + 4
        d.rectangle([xa, cy - 10, xa + (right - left) / 20.0, cy + 6], fill=(230, 230, 230))
    # vehicles ahead (adjacent lane, beyond the crosswalk)
    for i in range(max(0, min(2, spec.vehicles))):
        bx = vx + 30 + i * 70 + _jit(s, 10 + i, 8)
        by = hy + 30 + i * 12
        col = [(180, 40, 40), (40, 70, 170)][i]
        d.rectangle([bx, by, bx + 46, by + 26], fill=col)
        d.rectangle([bx + 6, by - 10, bx + 40, by], fill=tuple(c // 2 for c in col))
        d.ellipse([bx + 4, by + 22, bx + 14, by + 32], fill=(15, 15, 15))
        d.ellipse([bx + 32, by + 22, bx + 42, by + 32], fill=(15, 15, 15))
    # pedestrian on crosswalk
    if spec.pedestrian:
        px = vx - 70 + _jit(s, 20, 25)
        py = cy - 52
        d.ellipse([px - 7, py - 16, px + 7, py - 2], fill=(240, 200, 160))
        d.rectangle([px - 8, py - 2, px + 8, py + 24], fill=(230, 120, 30))
        d.line([px - 4, py + 24, px - 8, py + 46], fill=(30, 30, 90), width=5)
        d.line([px + 4, py + 24, px + 9, py + 46], fill=(30, 30, 90), width=5)
        d.line([px - 8, py + 4, px - 16, py + 20], fill=(230, 120, 30), width=4)
        d.line([px + 8, py + 4, px + 16, py + 20], fill=(230, 120, 30), width=4)
    # traffic signal on a pole at right
    polex = 470 + _jit(s, 30, 15)
    if spec.light != "none" or spec.sign_sticker:   # no bare signal pole when there is no signal
        d.rectangle([polex, 40 if spec.light != "none" else 140, polex + 8, cy + 10], fill=(60, 60, 60))
    if spec.light != "none":
        d.rectangle([polex - 22, 30, polex + 30, 140], fill=(25, 25, 25))
        on = {"red": 0, "yellow": 1, "green": 2}[spec.light]
        cols = [(255, 40, 30), (255, 200, 20), (40, 230, 90)]
        for k in range(3):
            c = cols[k] if k == on else tuple(v // 6 for v in cols[k])
            d.ellipse([polex - 12, 38 + 34 * k, polex + 20, 66 + 34 * k], fill=c)
    if spec.sign_sticker:
        box = (polex - 58, 148, polex + 66, 214)
        f, lines, lh = _fit_text(d, spec.sign_sticker, box, 14, 10)
        need = box[1] + 8 + lh * len(lines)
        if need > box[3]:                      # grow the sticker so all text stays inside it
            box = (box[0], box[1], box[2], need)
        d.rectangle(box, fill=(250, 250, 240), outline=(20, 20, 20), width=2)
        for i, ln in enumerate(lines):
            d.text((box[0] + 5, box[1] + 4 + i * lh), ln, font=f, fill=(10, 10, 10))
    # stop sign at left of the crosswalk
    if spec.stop_sign:
        sx, sy, r = 185 + _jit(s, 40, 8), 150, 30
        d.rectangle([sx - 3, sy, sx + 3, cy + 20], fill=(90, 90, 90))
        import math
        pts = [(sx + r * math.cos(math.pi / 8 + k * math.pi / 4), sy + r * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
        d.polygon(pts, fill=(200, 20, 20), outline=(255, 255, 255))
        f = _font(12)
        tw = d.textlength("STOP", font=f)
        d.text((sx - tw / 2, sy - 7), "STOP", font=f, fill=(255, 255, 255))
    # speed limit sign at far right
    if spec.speed_limit is not None:
        bx, by = 575 + _jit(s, 50, 6), 165
        d.rectangle([bx + 20, by + 50, bx + 26, cy + 30], fill=(90, 90, 90))
        d.rectangle([bx, by, bx + 46, by + 56], fill=(250, 250, 250), outline=(10, 10, 10), width=2)
        d.text((bx + 5, by + 3), "SPEED", font=_font(9), fill=(0, 0, 0))
        d.text((bx + 6, by + 13), "LIMIT", font=_font(9), fill=(0, 0, 0))
        d.text((bx + 8, by + 25), str(spec.speed_limit), font=_font(20), fill=(0, 0, 0))
    # billboard at left
    if spec.billboard:
        box = (10, 18, 250, 110)
        d.rectangle([60, 110, 68, hy + 10], fill=(80, 80, 80))
        d.rectangle(box, fill=(250, 248, 235), outline=(30, 30, 30), width=3)
        f, lines, lh = _fit_text(d, spec.billboard, box, 18)
        for i, ln in enumerate(lines):
            d.text((box[0] + 6, box[1] + 5 + i * lh), ln, font=f, fill=(15, 15, 15))
    # text painted on the road before the stop line
    if spec.road_text:
        f, lines, lh = _fit_text(d, spec.road_text, (vx - 150, cy + 30, vx + 150, H - 10), 18)
        for i, ln in enumerate(lines):
            tw = d.textlength(ln, font=f)
            d.text((vx - tw / 2, cy + 34 + i * lh), ln, font=f, fill=(245, 245, 245))
    # conditions
    if spec.condition == "fog":
        img = Image.blend(img, Image.new("RGB", (W, H), (215, 218, 222)), 0.35)
    elif spec.condition == "rain":
        d = ImageDraw.Draw(img)
        for k in range(160):
            x = (k * 97 + s * 13) % W
            y = (k * 53 + s * 7) % H
            d.line([x, y, x - 4, y + 12], fill=(190, 200, 215), width=1)
    elif spec.condition == "night":
        img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 10)), 0.25)
    elif spec.condition == "dusk":
        img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def spec_dict(spec: SceneSpec):
    return asdict(spec)
