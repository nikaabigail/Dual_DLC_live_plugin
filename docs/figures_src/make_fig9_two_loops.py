# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
GREEN, ORANGE, RED, GREY, BLUE = "#1d9e75", "#d85a30", "#e02828", "#8a8a8a", "#2f6fbd"
PURPLE = "#7a4fbf"
FAST_BG, SLOW_BG, LIM_BG = "#eaf2fb", "#f3eefb", "#fdecec"
OUT = "../media/report"


def box(ax, x, y, w, h, title, sub, edge, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.005,rounding_size=0.015",
                                fc=fc, ec=edge, lw=1.7, zorder=3))
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", zorder=4, linespacing=1.2)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
            fontsize=7.6, color="#444", zorder=4, linespacing=1.35)


def poly(ax, pts, col, lw=1.8, ls="-"):
    """Ортогональная трасса: линии + одна стрелка на последнем сегменте."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=col, lw=lw,
                           linestyle=ls, zorder=1, solid_capstyle="round"))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=15, lw=lw, color=col,
                                 linestyle=ls, zorder=1))


fig, ax = plt.subplots(figsize=(13.4, 6.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
W, H, GAP = 0.115, 0.145, 0.033
L = 0.075                                   # левое поле под вертикальную связь

yf, ys = 0.665, 0.215
xs = [L + 0.030 + i * (W + GAP) for i in range(4)]

# панели
ax.add_patch(Rectangle((L + 0.012, yf - 0.050), xs[3] + W + 0.020 - L - 0.012,
                       H + 0.100, fc=FAST_BG, ec="none", zorder=0))
ax.add_patch(Rectangle((L + 0.012, ys - 0.050), xs[3] + W + 0.020 - L - 0.012,
                       H + 0.100, fc=SLOW_BG, ec="none", zorder=0))

ax.text(L + 0.020, yf + H + 0.068, "FAST LOOP  —  WHEN to stimulate",
        fontsize=11, fontweight="bold", color=BLUE)
ax.text(L + 0.020, yf + H + 0.026,
        "52 Hz, deterministic, no learning. This is the safety path.",
        fontsize=8.5, color="#555")
ax.text(L + 0.020, ys + H + 0.068, "SLOW LOOP  —  WHAT to deliver",
        fontsize=11, fontweight="bold", color=PURPLE)
ax.text(L + 0.020, ys + H + 0.026,
        "once per trial (~10 steps), ~60 trials per session. Here the optimiser lives.",
        fontsize=8.5, color="#555")

box(ax, xs[0], yf, W, H, "Camera", "1920x220\n100 Hz", GREY)
box(ax, xs[1], yf, W, H, "DLCLive", "ResNet-50\nROI 448 px", RED)
box(ax, xs[2], yf, W, H, "Event\ndetector", "touch-down\nlift-off", GREEN)
box(ax, xs[3], yf, W, H, "Phase\npercent", "projection onto\nrolling reference", BLUE)
box(ax, xs[0], ys, W, H, "Gait score", "clearance, step,\nsymmetry, interlimb", GREEN)
box(ax, xs[1], ys, W, H, "GP-BO", "Gaussian process\n+ expected improv.", PURPLE)
box(ax, xs[2], ys, W, H, "Safety\nlimits", "hard gate,\nnot a penalty", RED, LIM_BG)
box(ax, xs[3], ys, W, H, "Parameter\nvector", "4 continuous\n(next figure)", ORANGE)

for row in (yf, ys):
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + W + 0.007, row + H / 2),
                                     (b - 0.007, row + H / 2), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.8, color="#555", zorder=2))

# связь: кинематика питает счёт. Идёт по левому полю, ничего не пересекая.
poly(ax, [(xs[0] - 0.010, yf + H / 2), (L - 0.010, yf + H / 2),
          (L - 0.010, ys + H / 2), (xs[0] - 0.008, ys + H / 2)], GREEN)
ax.text(L - 0.020, (yf + ys) / 2 + H / 2, "same DLC\nkeypoints", ha="right",
        va="center", fontsize=8.3, color=GREEN, fontweight="bold", linespacing=1.35)

# стимулятор
xst, yst, wst, hst = 0.820, 0.395, 0.150, 0.210
ax.add_patch(FancyBboxPatch((xst, yst), wst, hst,
                            boxstyle="round,pad=0.006,rounding_size=0.016",
                            fc="white", ec=GREY, lw=2.0, zorder=3))
ax.text(xst + wst / 2, yst + hst * 0.70, "STIMULATOR", ha="center", va="center",
        fontsize=11.5, fontweight="bold", zorder=4)
ax.text(xst + wst / 2, yst + hst * 0.33, "trigger  =  WHEN\nprogram  =  WHAT",
        ha="center", va="center", fontsize=8.5, color="#444", linespacing=1.6, zorder=4)

# триггер и канал управления - обе трассы ортогональные, метки в стороне
poly(ax, [(xs[3] + W + 0.007, yf + H / 2), (xst - 0.045, yf + H / 2),
          (xst - 0.045, yst + hst - 0.035), (xst - 0.006, yst + hst - 0.035)], BLUE)
ax.text((xs[3] + W + xst - 0.045) / 2 + 0.004, yf + H / 2 + 0.030,
        "TTL trigger\n145% of phase", ha="center", va="bottom", fontsize=8.4,
        color=BLUE, fontweight="bold", linespacing=1.35)

poly(ax, [(xs[3] + W + 0.007, ys + H / 2), (xst - 0.045, ys + H / 2),
          (xst - 0.045, yst + 0.035), (xst - 0.006, yst + 0.035)], ORANGE)
ax.text((xs[3] + W + xst - 0.045) / 2 + 0.004, ys + H / 2 - 0.030,
        "control channel\n(interface TBD)", ha="center", va="top", fontsize=8.4,
        color=ORANGE, fontweight="bold", linespacing=1.35)

# животное замыкает контур - по нижнему полю
poly(ax, [(xst + wst / 2, yst - 0.006), (xst + wst / 2, 0.105),
          (xs[0] + W / 2, 0.105), (xs[0] + W / 2, yf - 0.052)],
     "#aaaaaa", lw=1.6, ls=(0, (6, 4)))
ax.text((xst + xs[0]) / 2 + 0.02, 0.078,
        "animal closes the loop:  stimulation changes gait,  gait changes the score",
        ha="center", fontsize=9, color="#888", style="italic")

ax.text(0.5, 0.962, "Two loops, deliberately separated", ha="center",
        fontsize=14.5, fontweight="bold")
ax.text(0.5, 0.912, "Mixing them would cost both reproducibility and any claim "
                    "about phase specificity", ha="center", fontsize=9, color="#555")
fig.savefig(f"{OUT}/fig9_two_loops.png"); plt.close(fig)
print("fig9 перерисована")
