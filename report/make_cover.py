"""Render the designed A4 cover page (report/cover_page.png) with matplotlib.

Numbers on the cover (best BPC etc.) are read from results/summary.json and the run tables.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.image as mpimg
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONTS = Path("/root/msc_work/tools/lo/opt/libreoffice25.8/share/fonts/truetype")
for f in FONTS.glob("*.ttf"):
    if f.name.startswith(("Rubik", "NotoSerif-", "NotoSans-", "LiberationMono", "DejaVuSansMono")):
        fm.fontManager.addfont(str(f))
SANS, SERIF, MONO = "Rubik", "Noto Serif", "DejaVu Sans Mono"

NAVY, NAVY2, TEAL, GOLD, INK, MUTED, PANEL = "#0b2545", "#13315c", "#00798c", "#e3a82b", "#1b1f2a", "#5b6475", "#f3f6fa"

summary = json.loads((ROOT / "results/summary.json").read_text())
runs = list(csv.DictReader(open(ROOT / "results/tables/all_runs.csv")))
meta = json.loads((ROOT / "data/processed/meta.json").read_text())

MEMBERS = [("Shafin Ahmed", "262-25-035", "262-25-035@diu.edu.bd"),
           ("Zohir Rayhan", "262-25-036", "262-25-036@diu.edu.bd"),
           ("Sheikh Ajwad", "262-25-037", "262-25-037@diu.edu.bd")]

W, H = 8.27, 11.69
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec="none", r=0.12, lw=1, z=1, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, alpha=alpha))


def t(x, y, s, size, color=INK, font=SANS, weight="normal", ha="center", va="center", style="normal", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, family=font, weight=weight, ha=ha, va=va, style=style, zorder=z, **kw)


# ---------------------------------------------------------------- header band
band_y = 8.35
ax.add_patch(Rectangle((0, band_y), W, H - band_y, fc=NAVY, zorder=0))
# subtle gradient strip + character "rain" texture
for i in range(40):
    ax.add_patch(Rectangle((0, band_y + i * (H - band_y) / 40), W, (H - band_y) / 40, fc=NAVY2, alpha=i / 40 * 0.55, lw=0, zorder=0))
rng = np.random.default_rng(7)
text_src = "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife "
for row in range(11):
    yy = band_y + 0.18 + row * 0.29
    off = rng.integers(0, 60)
    t(0.1, yy, (text_src * 3)[off:off + 70], 7.2, color="white", font="Liberation Mono", ha="left", alpha=0.055 + 0.01 * (row % 3), z=1)
# recurrent network motif (nodes + links) on the right side of the band
nodes = [(6.0 + 0.55 * c, band_y + 0.55 + 0.55 * r) for c in range(4) for r in range(5)]
for (x1, y1) in nodes:
    for (x2, y2) in nodes:
        if 0 < x2 - x1 < 0.6 and abs(y2 - y1) < 1.2 and rng.random() < 0.55:
            ax.plot([x1, x2], [y1, y2], color="#4f7cac", lw=0.5, alpha=0.35, zorder=1)
for (x, y) in nodes:
    ax.add_patch(Circle((x, y), 0.045, fc="#8fb8de", ec="none", alpha=0.45, zorder=2))

# logo card
logo = mpimg.imread(str(HERE / "diu_logo.png"))
lw_, lh_ = 3.0, 3.0 * logo.shape[0] / logo.shape[1]
lx, ly = (W - lw_) / 2, H - 0.42 - lh_
box(lx - 0.22, ly - 0.14, lw_ + 0.44, lh_ + 0.28, "white", r=0.14, z=3)
ax.imshow(logo, extent=(lx, lx + lw_, ly, ly + lh_), zorder=4, interpolation="lanczos")

t(W / 2, ly - 0.48, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", 9.2, color="#b9d3ee", weight="medium",
  **{"fontstretch": "normal"})
t(W / 2, ly - 0.93, "CSE638  ·  DEEP LEARNING", 21, color="white", weight="bold")
t(W / 2, ly - 1.32, "Group Assignment 1 (CT1)  —  Group Report", 11.5, color="#d7e6f5")
ax.add_patch(Rectangle((0, band_y - 0.07), W, 0.07, fc=GOLD, zorder=3))

# ---------------------------------------------------------------- title block
chip_w = 3.7
box((W - chip_w) / 2, 7.80, chip_w, 0.36, TEAL, r=0.18, z=3)
t(W / 2, 7.98, "TASK 3   ·   RNN / SEQUENCE MODELING", 9.5, color="white", weight="bold")
t(W / 2, 7.36, "Character-Level Language Modeling", 23, color=NAVY, font=SERIF, weight="bold")
t(W / 2, 6.92, "of Jane Austen's Novels", 23, color=NAVY, font=SERIF, weight="bold")
t(W / 2, 6.46, "with Vanilla RNN, GRU and LSTM", 17, color=TEAL, font=SERIF, style="italic")
t(W / 2, 6.10, f"Dataset: six public-domain novels by Jane Austen (Project Gutenberg) · "
  f"{sum(meta['split_chars'].values()) / 1e6:.2f} M characters · {meta['vocab_size']}-symbol vocabulary",
  8.6, color=MUTED, style="italic")

# ---------------------------------------------------------------- unrolled RNN illustration
inp = list("Pride_and")
out = list("ride_and_")
n = len(inp); cw, gap = 0.46, 0.21
x0 = (W - (n * cw + (n - 1) * gap)) / 2 + 0.35
yc = 4.92
for i in range(n):
    x = x0 + i * (cw + gap)
    shade = 0.35 + 0.65 * i / (n - 1)
    box(x, yc - 0.21, cw, 0.42, TEAL, r=0.07, z=3, alpha=shade)
    t(x + cw / 2, yc, "h", 9, color="white", weight="bold", z=6)
    # input char
    box(x + 0.09, yc - 0.86, cw - 0.18, 0.32, "white", ec="#c5cfdb", r=0.06, lw=0.8, z=3)
    t(x + cw / 2, yc - 0.70, inp[i].replace("_", "␣"), 10.5, color=INK, font=MONO, weight="bold")
    ax.add_patch(FancyArrowPatch((x + cw / 2, yc - 0.53), (x + cw / 2, yc - 0.23), arrowstyle="-|>", mutation_scale=8,
                                 color="#8a96a8", lw=0.9, zorder=2))
    # output char
    box(x + 0.09, yc + 0.54, cw - 0.18, 0.32, "#fff6e0", ec=GOLD, r=0.06, lw=0.8, z=3)
    t(x + cw / 2, yc + 0.70, out[i].replace("_", "␣"), 10.5, color="#8a5a00", font=MONO, weight="bold")
    ax.add_patch(FancyArrowPatch((x + cw / 2, yc + 0.23), (x + cw / 2, yc + 0.53), arrowstyle="-|>", mutation_scale=8,
                                 color="#8a96a8", lw=0.9, zorder=2))
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + cw + 0.01, yc), (x + cw + gap - 0.01, yc), arrowstyle="-|>", mutation_scale=9,
                                     color=GOLD, lw=1.6, zorder=4))
for lab, yy in ((r"next char  $\hat{y}_t$", yc + 0.70), (r"hidden state  $h_t$", yc), (r"input char  $x_t$", yc - 0.70)):
    t(x0 - 0.12, yy, lab, 7.2, color=MUTED, ha="right")

# ---------------------------------------------------------------- key numbers
best = min(runs, key=lambda r: float(r["Val BPC"]))
stats = [(f"{len(meta['books'])}", "novels"), (f"{len(runs)}", "models trained"),
         (f"{summary['best_test_bpc']:.3f}", f"best test BPC ({best['Cell']})"),
         (f"{summary['best_ngram_test_bpc']:.3f}", "best n-gram BPC")]
tw, tg = 1.72, 0.14
tx0 = (W - (4 * tw + 3 * tg)) / 2
for i, (num, lab) in enumerate(stats):
    x = tx0 + i * (tw + tg)
    box(x, 3.22, tw, 0.64, PANEL, ec="#dbe3ec", r=0.10, lw=0.8, z=2)
    ax.add_patch(Rectangle((x + 0.14, 3.22 + 0.11), 0.045, 0.42, fc=[TEAL, NAVY2, GOLD, MUTED][i], zorder=3))
    t(x + 0.32, 3.62, num, 15, color=NAVY, weight="bold", ha="left")
    t(x + 0.32, 3.38, lab, 7.6, color=MUTED, ha="left")

# ---------------------------------------------------------------- group card
gy, gh = 1.44, 1.64
box(0.62, gy, W - 1.24, gh, "white", ec="#d3dce6", r=0.12, lw=1.0, z=2)
ax.add_patch(Rectangle((0.62, gy + 0.12), 0.07, gh - 0.24, fc=NAVY, zorder=3))
t(0.95, gy + gh - 0.22, "GROUP", 8, color=TEAL, weight="bold", ha="left")
t(0.95, gy + gh - 0.47, "Recurrent Minds", 16, color=NAVY, weight="bold", ha="left", font=SERIF)
cols = [(0.97, "#"), (1.35, "Student Name"), (3.75, "Student ID"), (5.2, "Email")]
hy = gy + gh - 0.76
for cx, cl in cols:
    t(cx, hy, cl.upper(), 7.2, color=MUTED, weight="bold", ha="left")
ax.plot([0.95, W - 0.85], [hy - 0.12, hy - 0.12], color="#d3dce6", lw=0.8, zorder=3)
for i, (nm, sid, em) in enumerate(MEMBERS):
    ry = hy - 0.30 - i * 0.21
    for (cx, _), val, wgt in zip(cols, (str(i + 1), nm, sid, em), ("normal", "medium", "normal", "normal")):
        t(cx, ry, val, 9.4, color=INK, weight=wgt, ha="left")

# ---------------------------------------------------------------- submitted to / on
cy, ch = 0.62, 0.70
cwid = (W - 1.24 - 0.2) / 2
for k, (head, l1, l2) in enumerate((("SUBMITTED TO", "Sadat Hasan", "Adjunct Faculty, Department of CSE, DIU"),
                                    ("SUBMISSION DATE", "17 November 2026", "CSE638 · Masters in CSE"))):
    x = 0.62 + k * (cwid + 0.2)
    box(x, cy, cwid, ch, PANEL, ec="#dbe3ec", r=0.10, lw=0.8, z=2)
    t(x + 0.22, cy + ch - 0.18, head, 7.6, color=TEAL, weight="bold", ha="left")
    t(x + 0.22, cy + 0.34, l1, 12.5, color=NAVY, weight="bold", ha="left")
    t(x + 0.22, cy + 0.14, l2, 8.2, color=MUTED, ha="left")

# ---------------------------------------------------------------- footer band
ax.add_patch(Rectangle((0, 0), W, 0.46, fc=NAVY, zorder=0))
ax.add_patch(Rectangle((0, 0.46), W, 0.035, fc=GOLD, zorder=1))
t(0.62, 0.23, "Codebase:  github.com/Th3X-Zohir/ct1-rnn-language-modeling", 8.6, color="white", ha="left")
t(W - 0.62, 0.23, "Daffodil International University  ·  2026", 8.6, color="#b9d3ee", ha="right")

fig.savefig(HERE / "cover_page.png", dpi=250)
print("cover written")
