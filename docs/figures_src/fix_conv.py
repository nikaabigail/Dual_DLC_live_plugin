# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
GREEN, ORANGE, RED, BLUE = "#1d9e75", "#d85a30", "#e02828", "#2f6fbd"
OUT = "../media/report"

fig, ax = plt.subplots(figsize=(11.0, 3.9))
ax.set_xlim(-8, 268); ax.set_ylim(-1.15, 1.05); ax.axis("off")

# основная полоса цикла
ax.add_patch(Rectangle((0, .30), 100, .30, fc=GREEN, ec="white", lw=1.5))
ax.add_patch(Rectangle((100, .30), 100, .30, fc=ORANGE, ec="white", lw=1.5))
ax.text(50, .45, "STANCE  0-100%   (295 ms)", ha="center", va="center",
        color="white", fontweight="bold", fontsize=10.5)
ax.text(150, .45, "SWING  100-200%   (105 ms)", ha="center", va="center",
        color="white", fontweight="bold", fontsize=10.5)
for v, lab in ((0, "foot strike"), (100, "foot off"), (200, "foot strike")):
    ax.plot([v, v], [.26, .64], color="#333", lw=1.4)
    ax.text(v, .72, lab, ha="center", fontsize=9, color="#333")

# полосы окон - подписи СПРАВА, не поверх
ax.add_patch(Rectangle((100, .05), 50, .15, fc=BLUE, alpha=.32, ec=BLUE, lw=1.2))
ax.text(206, .125, "canonical flexion window 100-150%  (Wenger 2016)",
        va="center", fontsize=9, color=BLUE, fontweight="bold")
ax.add_patch(Rectangle((100, -.22), 30, .15, fc=RED, alpha=.32, ec=RED, lw=1.2))
ax.text(206, -.145, "unreachable: first 29 ms of swing",
        va="center", fontsize=9, color=RED, fontweight="bold")

# рабочая точка
ax.plot([145, 145], [-.30, .64], color=GREEN, lw=2.4, ls="--")
ax.plot(145, .45, marker="v", color=GREEN, ms=9)
ax.text(145, -.42, "145%  working point", ha="center", fontsize=10.5,
        color=GREEN, fontweight="bold")
ax.text(145, -.58, "45% into swing = 47 ms after foot off", ha="center",
        fontsize=8.5, color=GREEN)

# масштаб процента
ax.text(103, -.88, "1% of stance = 2.95 ms        1% of swing = 1.05 ms        "
                   "the same 1% is 3x tighter in swing",
        ha="center", fontsize=9, color="#333")
ax.text(103, .95, "Phase convention: percent is normalised WITHIN each phase",
        ha="center", fontsize=13.5, fontweight="bold")
fig.savefig(f"{OUT}/fig8_convention.png"); plt.close(fig)
print("fig8 ok")
