"""Optional poster panels; reuses app map rendering read-only."""
import io
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app import core
from app.pictures import LIFT, LUMA, floor, render, png

DIFFERS = (200, 40, 160)
SURE_RAMP = ((241, 243, 249), (31, 58, 147))

def confidence_picture(conf):
    """The model's confidence in one light-to-dark blue: pale is a toss-up, dark is
    sure. White where there is no terrain."""
    t = np.clip((conf - floor()) / (1 - floor()), 0, 1)[:, :, None]
    lo, hi = (np.array(c, np.float32) for c in SURE_RAMP)
    rgb = lo + (hi - lo) * np.nan_to_num(t, nan=0)
    rgb[np.isnan(conf)] = 255
    return Image.fromarray(rgb.astype(np.uint8))


def differences(base, key, pred):
    """Where `pred` gives another unit than `key`, in magenta over the background;
    where they agree, or either says nothing, the plain background."""
    rgb = np.repeat(base[:, :, None], 3, 2).astype(np.float32)
    differ = (key != 255) & (pred != 255) & (key != pred)
    shade = (LIFT + (1 - LIFT) * base)[differ][:, None]
    colour = np.array(DIFFERS) / 255.0
    rgb[differ] = 0.25 * rgb[differ] + 0.75 * np.clip(shade * colour / (colour @ LUMA), 0, 1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def units(base, codes, strength=0.65):
    """A unit map tinted over the background; 255 (no unit) shows the background."""
    return render(base, [(np.where(codes == 255, 0, codes).astype(np.uint8), strength)])


def font(size):
    return ImageFont.load_default(size=size)


def figure(title, panels, px_m, note=""):
    """Panels -- (label, picture) pairs, all one size -- side by side under a title, with
    a legend of the units, the difference colour and the confidence scale, a note, and a
    1 km scale bar at `px_m` metres per pixel. Returns a PIL image."""
    w, h = panels[0][1].size
    pad, gap, top, label, ramp = 60, 40, 130, 70, 380
    width = 2 * pad + len(panels) * w + (len(panels) - 1) * gap
    small, tiny = font(38), font(30)
    keys = [(core.CMAP[code][:3], unit.replace("_", " ")) for code, unit in core.CODES.items()]
    keys.append((DIFFERS, "model differs"))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    spans = [60 + int(probe.textlength(text, font=small)) + 70 for _, text in keys]
    wrap = sum(spans) + 20 + ramp > width - 2 * pad       # the confidence scale on its own row
    rows = 2 if wrap else 1
    out = Image.new("RGB", (width, top + label + h + 45 + rows * 100 + 70), "white")
    ink = ImageDraw.Draw(out)
    dark, grey = (20, 20, 20), (80, 80, 80)

    size = 64
    while size > 28 and ink.textlength(title, font=font(size)) > width - 2 * pad:
        size -= 4                                    # a long title shrinks to fit
    ink.text((pad, 35 + (64 - size) // 2), title, fill=dark, font=font(size))
    for i, (name, pic) in enumerate(panels):
        x = pad + i * (w + gap)
        ink.text((x, top + 10), name, fill=dark, font=font(44))
        out.paste(pic, (x, top + label))

    x, y = pad, top + label + h + 45
    for (colour, text), span in zip(keys, spans):
        ink.rectangle((x, y, x + 44, y + 44), fill=colour)
        ink.text((x + 60, y + 2), text, fill=dark, font=small)
        x += span
    x, y = (pad, y + 100) if wrap else (x + 20, y)
    lo, hi = (np.array(c) for c in SURE_RAMP)
    for j in range(ramp):
        ink.line((x + j, y, x + j, y + 44),
                 fill=tuple(int(v) for v in lo + (hi - lo) * j / (ramp - 1)))
    ink.text((x, y + 52), f"{floor():.0%}", fill=grey, font=tiny)
    ink.text((x + ramp / 2, y + 52), "confidence", fill=grey, font=tiny, anchor="ma")
    ink.text((x + ramp, y + 52), "100%", fill=grey, font=tiny, anchor="ra")

    base = out.height - 45                               # the bottom line: note, scale bar
    if note:
        ink.text((pad, base), note, fill=(110, 110, 110), font=tiny, anchor="ls")
    bar, right = 1000 / px_m, width - pad                # 1 km, in pixels
    ink.rectangle((right - bar, base - 16, right, base - 2), fill=dark)
    ink.text((right - bar - 16, base), "1 km", fill=dark, font=small, anchor="rs")
    return out


def zipped(pictures):
    """{file name: PIL image} as the bytes of a zip of PNGs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, pic in pictures.items():
            z.writestr(name, png(pic))
    return buf.getvalue()
