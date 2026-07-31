"""
make_acorn_figure.py
Renders the ACORN system schematic as a print-resolution vector figure.

ACORN = Automated CNMFe Of Recording-Networks.

Outputs (next to this script):
  acorn_schematic.pdf   -- vector, drop straight into the manuscript
  acorn_schematic.png   -- 300 dpi raster preview

Run:
  C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe make_acorn_figure.py
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# ---------------------------------------------------------------- palette
INK        = "#16211F"
MUTED      = "#566965"
LINE       = "#DBE3E0"
TEAL       = "#0E7C73"
TEAL_STR   = "#0A5F58"
TEAL_TINT  = "#E1F0EE"
AMBER      = "#BE7A1C"
AMBER_STR  = "#9C6413"
AMBER_TINT = "#F6EAD4"
BAD        = "#B8492E"
GOOD       = "#2E7D5B"
WHITE      = "#FFFFFF"

MONO = {"family": "monospace"}
SANS = {"family": "sans-serif"}

# ---------------------------------------------------------------- canvas
# 14x8 inches, 10 data-units per inch -> square units (round corners stay round).
FIG_W, FIG_H = 14, 7
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 140)
ax.set_ylim(0, 70)
ax.axis("off")
ax.set_facecolor(WHITE)
fig.patch.set_facecolor(WHITE)


def card(x, y, w, h, accent, fill=WHITE, lw=1.1, dashed=False, rounding=1.4, z=2):
    """Rounded rectangle. Returns nothing; draws in place."""
    style = "round,pad=0,rounding_size=%.2f" % rounding
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        linewidth=lw, edgecolor=accent, facecolor=fill,
        mutation_aspect=1.0, zorder=z,
    )
    if dashed:
        p.set_linestyle((0, (4, 3)))
    ax.add_patch(p)


def toptab(x, y, w, accent, z=3, thick=0.7):
    """A short accent bar sitting on the top edge of a card (color-coded)."""
    ax.add_patch(FancyBboxPatch(
        (x + 1.0, y - thick / 2), w - 2.0, thick,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=0, facecolor=accent, zorder=z))


def tag(cx, y, text, fg, bg):
    ax.text(cx, y, text, ha="center", va="center", fontsize=6.4, fontweight="bold",
            color=fg, zorder=5, **MONO,
            bbox=dict(boxstyle="round,pad=0.32", fc=bg, ec="none"))


def stage(x, w, num, title, desc, accent, tagtext, tagfg, tagbg,
          y=38.0, h=18.0, title_color=None):
    card(x, y, w, h, LINE, fill=WHITE, lw=1.1, z=2)
    toptab(x, y + h, w, accent)
    cx = x + w / 2
    # number (left) + tag (right) on the top line
    ax.text(x + 2.0, y + h - 2.6, num, ha="left", va="center",
            fontsize=8, fontweight="bold", color=MUTED, zorder=5, **MONO)
    tag(x + w - 4.6, y + h - 2.6, tagtext, tagfg, tagbg)
    # title
    ax.text(cx, y + h - 6.6, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=title_color or INK, zorder=5, **SANS)
    # description (wrapped)
    body = "\n".join(textwrap.wrap(desc, width=26))
    ax.text(cx, y + h - 9.4, body, ha="center", va="top",
            fontsize=6.6, color=MUTED, zorder=5, linespacing=1.35, **SANS)


def connector(x0, x1, y=47.0, color=MUTED):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.4, color=color, zorder=4,
        shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- geometry
N = 7
BOX_W = 16.5
STEP = 19.0
START = 4.75
lefts = [START + i * STEP for i in range(N)]
rights = [l + BOX_W for l in lefts]
Y0, H = 38.0, 18.0

# ---------------------------------------------------------------- loop panel (behind)
lp_left = lefts[4] - 2.4
lp_right = rights[6] + 2.2
LP_BOT, LP_H = 29.5, 29.0
card(lp_left, LP_BOT, lp_right - lp_left, LP_H, AMBER, fill=AMBER_TINT + "00",
     lw=1.3, dashed=True, rounding=2.2, z=1)
# tint fill drawn separately (very light) so the dashed edge stays crisp
ax.add_patch(FancyBboxPatch(
    (lp_left, LP_BOT), lp_right - lp_left, LP_H,
    boxstyle="round,pad=0,rounding_size=2.2",
    linewidth=0, facecolor=AMBER_TINT, alpha=0.45, zorder=0))
# panel header chip
ax.text(lp_left + 2.4, 58.0, "SELF-IMPROVING LOOP", ha="left", va="center",
        fontsize=6.6, fontweight="bold", color=WHITE, zorder=6, **MONO,
        bbox=dict(boxstyle="round,pad=0.34", fc=AMBER, ec="none"))

# ---------------------------------------------------------------- stages
stages = [
    ("01", "Ingest", "Watches for each new recording; waits for a safe, complete transfer.",
     TEAL, "AUTO", TEAL_STR, TEAL_TINT),
    ("02", "Auto-tune", "Sets gSig/gSiz, min_corr, min_pnr and bd automatically -- no hand-tuning.",
     TEAL, "AUTO", TEAL_STR, TEAL_TINT),
    ("03", "Extract", "Headless CNMF-E factorizes the video into hundreds of candidates.",
     TEAL, "AUTO", TEAL_STR, TEAL_TINT),
    ("04", "Measure", "13 features per candidate: shape, trace quality, motion, Cn fit.",
     TEAL, "AUTO", TEAL_STR, TEAL_TINT),
    ("05", "Auto-curate", "Classifier scores candidates, drops junk, flags motion & merges.",
     TEAL_STR, "AI", WHITE, TEAL),
    ("06", "Review", "You keep / delete / merge -- survivors only, with a PDF pre-read.",
     AMBER, "YOU", WHITE, AMBER),
    ("07", "Capture labels", "Your decisions are saved as labeled training examples.",
     AMBER, "AUTO", TEAL_STR, TEAL_TINT),
]
for i, (num, title, desc, accent, tt, tfg, tbg) in enumerate(stages):
    tcol = AMBER_STR if i == 5 else (TEAL_STR if i == 4 else INK)
    stage(lefts[i], BOX_W, num, title, desc, accent, tt, tfg, tbg, title_color=tcol)

# forward connectors between all consecutive boxes
for i in range(N - 1):
    col = TEAL_STR if i == 3 else MUTED   # the 04 -> 05 hand-off is emphasized
    connector(rights[i] + 0.6, lefts[i + 1] - 0.6, color=col)

# ---------------------------------------------------------------- return / online-retraining loop
box7_cx = (lefts[6] + rights[6]) / 2   # tail: under "Capture labels"
box5_cx = (lefts[4] + rights[4]) / 2   # head: under "Auto-curate" (the classifier)
Y_BOX  = Y0            # 38.0 = box bottoms; verticals start here so they touch the boxes
Y_LANE = 33.0         # the horizontal return lane
HEAD_B = 36.2         # where the left vertical stops and the arrowhead begins
# One continuous U: down from Capture-labels, left along the lane, up into Auto-curate.
ax.plot([box7_cx, box7_cx, box5_cx, box5_cx],
        [Y_BOX, Y_LANE, Y_LANE, HEAD_B],
        color=AMBER, lw=2.4, solid_capstyle="round",
        solid_joinstyle="round", zorder=3)
ax.add_patch(Polygon(
    [(box5_cx - 1.25, HEAD_B), (box5_cx + 1.25, HEAD_B), (box5_cx, Y_BOX + 0.5)],
    closed=True, facecolor=AMBER, edgecolor="none", zorder=4))
ax.text((box5_cx + box7_cx) / 2, 31.0,
        "online retraining  ·  new labels sharpen the classifier",
        ha="center", va="center", fontsize=7.6, fontstyle="italic",
        color=AMBER_STR, zorder=6, **SANS)

# ---------------------------------------------------------------- hero / wordmark
ax.text(4.5, 65.4, "ACORN", ha="left", va="center", fontsize=31,
        fontweight="bold", color=INK, **SANS)
ax.text(4.5, 60.4,
        "Automated CNMFe Of Recording-Networks",
        ha="left", va="center", fontsize=11, fontweight="bold",
        color=TEAL_STR, **MONO)
ax.text(135.5, 66.2,
        "A self-improving curation layer on CNMF-E.",
        ha="right", va="center", fontsize=8.5, color=MUTED, **SANS)
ax.text(135.5, 62.6,
        "Drop in a recording -> get curated neurons -> the model learns from your review.",
        ha="right", va="center", fontsize=8.5, color=MUTED, **SANS)

# ---------------------------------------------------------------- stats strip
ax.plot([4.5, 135.5], [26.5, 26.5], color=LINE, lw=1.0, zorder=1)
ax.text(4.5, 25.0, "WHAT THE AUTOMATION BUYS YOU", ha="left", va="center",
        fontsize=7, fontweight="bold", color=TEAL_STR, **MONO)

tiles = [
    ("5 → 0", "Parameters you set by hand. Estimated per session from the animal and the images.", AMBER, AMBER_STR),
    ("13", "Features scored per candidate: shape, trace quality, motion, correlation-map fit.", TEAL, TEAL_STR),
    ("0.88", "Real-vs-junk skill: ranks a real neuron above a junk one ~88% of the time. (AUC: 1.0 = perfect, 0.5 = a coin flip.)", TEAL, TEAL_STR),
    ("~18%", "Of the junk, auto-removed before you ever look at it.", AMBER, AMBER_STR),
    ("<1%", "Real neurons ever auto-rejected -- tuned to almost never discard a good cell.", AMBER, AMBER_STR),
]
NT = len(tiles)
T_W = 23.5
T_GAP = 3.25
T_START = 4.75
ty, th = 5.6, 16.4
for i, (big, cap, accent, big_col) in enumerate(tiles):
    tx = T_START + i * (T_W + T_GAP)
    card(tx, ty, T_W, th, LINE, fill=WHITE, lw=1.0, rounding=1.2, z=2)
    # left accent rail
    ax.add_patch(FancyBboxPatch(
        (tx, ty + 1.0), 0.7, th - 2.0, boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=0, facecolor=accent, zorder=3))
    ax.text(tx + 2.8, ty + th - 4.6, big, ha="left", va="center",
            fontsize=19, fontweight="bold", color=big_col, zorder=4, **MONO)
    body = "\n".join(textwrap.wrap(cap, width=34))
    ax.text(tx + 2.8, ty + th - 8.6, body, ha="left", va="top",
            fontsize=6.4, color=MUTED, zorder=4, linespacing=1.35, **SANS)

# ---------------------------------------------------------------- footer touchpoints
ax.text(72.5, 2.6,
        "Human touchpoints per session    "
        "by hand: every step, every session      "
        "•      ACORN: one review, shrinking each cycle",
        ha="center", va="center", fontsize=7.6, color=INK, **SANS)
ax.text(4.5, 2.6, "Kheirbek Lab  ·  UCSF", ha="left", va="center",
        fontsize=7.4, fontweight="bold", color=TEAL_STR, **MONO)

# ---------------------------------------------------------------- save
out = Path(__file__).parent
fig.savefig(out / "acorn_schematic.pdf", facecolor=WHITE, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out / "acorn_schematic.png", dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.15)
print("wrote acorn_schematic.pdf and acorn_schematic.png to", out)
