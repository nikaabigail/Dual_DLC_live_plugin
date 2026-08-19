# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
GREEN, ORANGE, RED, GREY, BLUE = "#1d9e75", "#d85a30", "#e02828", "#8a8a8a", "#2f6fbd"
PY, CPP = "#eaf2fb", "#fdeee7"
OUT = "../media/report"

W, GAP = .102, .052
fig, ax = plt.subplots(figsize=(12.2, 5.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

xs = [.048 + i * (W + GAP) for i in range(4)]
xc = [.690, .690 + (W + GAP)]
y, h = .50, .20

ax.add_patch(Rectangle((.030, .435), xs[3] + W + .022 - .030, .315, fc=PY, ec="none", zorder=0))
ax.text(.038, .765, "PYTHON   single_rt_dlc_live_bridge.py", fontsize=9.5,
        fontweight="bold", color=BLUE)
ax.add_patch(Rectangle((.672, .435), xc[1] + W + .022 - .672, .315, fc=CPP, ec="none", zorder=0))
ax.text(.680, .765, "C++   Open Ephys plugin", fontsize=9.5, fontweight="bold", color=ORANGE)


def box(x, title, sub, edge):
    ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc="white", ec=edge, lw=1.7, zorder=3))
    ax.text(x + W / 2, y + h * .74, title, ha="center", va="center", fontsize=9,
            fontweight="bold", zorder=4, linespacing=1.2)
    ax.text(x + W / 2, y + h * .26, sub, ha="center", va="center", fontsize=7.6,
            color="#444", zorder=4, linespacing=1.3)


box(xs[0], "Camera", "1920x220\n100 Hz", GREY)
box(xs[1], "DLCLive", "ResNet-50\nROI 448 px", RED)
box(xs[2], "Event\ndetector", "touch-down\nlift-off", GREEN)
box(xs[3], "Phase\npercent", "projection onto\nrolling reference", BLUE)
box(xc[0], "DDLP / UDP", "TTL word\n+ watchdog", ORANGE)
box(xc[1], "Stimulator", "TTL edge\nto pulse", GREY)

for a, b in [(xs[0], xs[1]), (xs[1], xs[2]), (xs[2], xs[3]), (xs[3], xc[0]), (xc[0], xc[1])]:
    ax.add_patch(FancyArrowPatch((a + W + .009, y + h / 2), (b - .009, y + h / 2),
                                 arrowstyle="-|>", mutation_scale=15, lw=1.8,
                                 color="#555", zorder=2))

ax.add_patch(FancyArrowPatch((xs[2] + W * .5, y), (xs[3] + W * .5, y),
                             arrowstyle="-|>", mutation_scale=14, lw=1.6, color=GREEN,
                             connectionstyle="arc3,rad=0.5", zorder=2))
ax.text((xs[2] + xs[3]) / 2 + W / 2, .355,
        "cycle boundaries: rebuild the reference every 10 steps",
        ha="center", fontsize=8.2, color=GREEN)

ly = .245
ax.add_patch(FancyArrowPatch((xs[0], ly), (xs[3] + W, ly), arrowstyle="<->",
                             mutation_scale=15, lw=2.0, color=RED))
ax.text((xs[0] + xs[3] + W) / 2, ly - .068, "MEASURED:  29 ms  movement to decision",
        ha="center", fontsize=10, color=RED, fontweight="bold")
ax.text((xs[0] + xs[3] + W) / 2, ly - .140,
        "frame 10.0  +  read 1.3  +  inference 18.6  +  detector 0.18  +  geometry 0.33",
        ha="center", fontsize=8, color="#555")

ax.add_patch(FancyArrowPatch((xc[0], ly), (xc[1] + W, ly), arrowstyle="<->",
                             mutation_scale=15, lw=2.0, color=GREY, linestyle="--"))
ax.text((xc[0] + xc[1] + W) / 2, ly - .068, "NOT MEASURED", ha="center",
        fontsize=10, color="#666", fontweight="bold")
ax.text((xc[0] + xc[1] + W) / 2, ly - .140,
        "decision to current at the electrode\nsoftware 0.18 ms + block quantisation 2-34 ms",
        ha="center", fontsize=8, color="#555", linespacing=1.4)

ax.text(.5, .955, "Closed-loop path: what is measured and what is not",
        ha="center", fontsize=14, fontweight="bold")
ax.text(.5, .880, "Phase estimation at runtime needs no events. Events are used only to cut "
                  "cycles when the reference is rebuilt.",
        ha="center", fontsize=9, color="#555")
fig.savefig(f"{OUT}/fig7_architecture.png"); plt.close(fig)
print("fig7 ok")
