# Запускать из docs/figures_src/. Рис. 6 требует report_data.json - он готовится
# отдельным скриптом, см. раздел 5 отчёта.
# -*- coding: utf-8 -*-
"""Figures for the report. All numbers are measured; sources cited in the text."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
GREEN, ORANGE, RED, GREY, BLUE = "#1d9e75", "#d85a30", "#e02828", "#8a8a8a", "#2f6fbd"
LGREY = "#c9d2db"
OUT = "../media/report"
D = json.load(open("report_data.json"))

# ---------- 1. Loop latency ------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 2.3), constrained_layout=True)
stages = [("frame period (100 Hz)", 10.0, GREY), ("frame read", 1.3, LGREY),
          ("DLC inference", 18.6, RED), ("event detector", 0.18, GREEN),
          ("phase geometry", 0.33, BLUE)]
left = 0
for name, val, col in stages:
    ax.barh(0, val, left=left, color=col, edgecolor="white", height=.5)
    if val > 1.2:
        ax.text(left + val / 2, 0, f"{val:.1f}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=9)
    left += val
ax.set_xlim(0, left * 1.02); ax.set_ylim(-1.0, 1.15); ax.set_yticks([])
ax.set_xlabel("ms")
ax.set_title(f"Sensing latency: movement to decision = {left:.1f} ms",
             loc="left", fontweight="bold")
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in stages]
ax.legend(hs, [n for n, _, _ in stages], ncol=5, frameon=False,
          loc="upper center", bbox_to_anchor=(.5, 1.32), fontsize=8, handlelength=1.2)
ax.text(left / 2, -.68, "inference = 91% of the compute (18.6 of 20.4 ms)",
        ha="center", fontsize=8.5, color=RED)
fig.savefig(f"{OUT}/fig1_latency.png"); plt.close(fig)

# ---------- 2. ROI width vs accuracy --------------------------------------
w = ["256\n(before)", "320", "448\n(training crop)", "640", "full\nframe"]
err = [0.99, 0.94, 0.86, 0.85, 0.90]
like = [.841, .865, .892, .906, .910]
ms = [18.3, 18.3, 18.3, 18.9, 20.3]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
b = a1.bar(w, err, color=[GREY, GREY, GREEN, GREY, GREY], width=.6)
a1.bar_label(b, fmt="%.2f", fontsize=8.5, padding=2)
a1.set_ylabel("pose error vs manual labels, px"); a1.set_ylim(0, 1.2)
a1.set_title("Accuracy on 68 held-out frames", loc="left", fontsize=10.5, fontweight="bold")
a1.text(2, 0.10, "-13%\np=5e-11", ha="center", fontsize=8.5, color="white", fontweight="bold")
a2.plot(w, like, "o-", color=BLUE, lw=2)
a2.set_ylabel("likelihood", color=BLUE); a2.tick_params(axis="y", labelcolor=BLUE)
a2.set_ylim(.82, .935)
a3 = a2.twinx(); a3.plot(w, ms, "s--", color=ORANGE, lw=2)
a3.set_ylabel("inference, ms", color=ORANGE); a3.tick_params(axis="y", labelcolor=ORANGE)
a3.set_ylim(17.5, 21.5); a3.grid(False)
a2.set_title("Confidence (blue) and time cost (orange)", loc="left", fontsize=10.5, fontweight="bold")
for ax in (a1, a2):
    ax.axvspan(1.5, 2.5, color=GREEN, alpha=.08)
fig.savefig(f"{OUT}/fig2_roi.png"); plt.close(fig)

# ---------- 3. Reachable phase targets ------------------------------------
tg = np.array([105, 110, 120, 125, 130, 135, 140, 145, 150, 160, 180, 195])
bias = np.abs([31.8, 26.8, 16.8, 11.8, 6.8, 1.8, -.9, -2.1, -2.2, -2.1, -1.1, 5.0])
p90 = np.array([42.4, 37.4, 27.4, 22.4, 17.4, 12.4, 12.9, 12.9, 14.7, 16.5, 20.0, 14.0])
fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
ax.axvspan(100, 150, color=BLUE, alpha=.09)
ax.axvspan(100, 130, color=RED, alpha=.14)
ax.plot(tg, bias, "o-", color=ORANGE, lw=2, label="|bias|")
ax.plot(tg, p90, "s-", color=BLUE, lw=2, label="scatter p90")
ax.axvline(145, color=GREEN, lw=2.2, ls="--")
ax.annotate("working point\n145%", xy=(145, 3), xytext=(160, 26),
            color=GREEN, fontsize=9.5, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
ax.text(114, 39, "unreachable\n(29 ms latency)", color=RED, fontsize=9,
        ha="center", fontweight="bold")
ax.text(140, 44.5, "canonical flexion window 100-150%", color=BLUE, fontsize=8.5, ha="center")
ax.set_xlabel("target phase, % of cycle (0-100 stance, 100-200 swing)")
ax.set_ylabel("error, % of phase")
ax.set_ylim(-2, 48)
ax.set_title("Which phase targets the loop can actually hit", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="center right")
fig.savefig(f"{OUT}/fig3_reachable.png"); plt.close(fig)

# ---------- 4. Robustness to event loss -----------------------------------
loss = [0, 10, 25, 50, 75]
fig, ax = plt.subplots(figsize=(5.8, 3.2), constrained_layout=True)
ax.plot(loss, [100, 91, 74, 51, 27], "o-", color=ORANGE, lw=2.2,
        label="timer (needs an event every cycle)")
ax.plot(loss, [98] * 5, "s-", color=GREEN, lw=2.2,
        label="geometry (no events at runtime)")
ax.set_xlabel("detector event loss, %"); ax.set_ylabel("cycles with a pulse, %")
ax.set_ylim(0, 108)
ax.set_title("Robustness to losing gait events", loc="left", fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
fig.savefig(f"{OUT}/fig4_robustness.png"); plt.close(fig)

# ---------- 5. Reference convergence --------------------------------------
n = [3, 5, 10, 20, 50, 269]
fig, ax = plt.subplots(figsize=(5.4, 3.0), constrained_layout=True)
ax.plot(range(len(n)), [5.9, 5.8, 5.3, 5.3, 5.5, 5.0], "o-", color=BLUE, lw=2.2)
ax.set_xticks(range(len(n))); ax.set_xticklabels([str(x) for x in n])
ax.axvline(2, color=GREEN, ls="--", lw=1.6)
ax.annotate("10 cycles ~ 4 s of walking", xy=(2, 5.3), xytext=(2.35, 5.85),
            color=GREEN, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
ax.set_xlabel("cycles averaged into the reference"); ax.set_ylabel("phase error, %")
ax.set_ylim(4.6, 6.2)
ax.set_title("How many steps the reference needs", loc="left", fontweight="bold")
fig.savefig(f"{OUT}/fig5_reference.png"); plt.close(fig)

# ---------- 6. Landing accuracy vs animal variability ---------------------
err_pct = np.array(D["landing_err_pct"]); sw = np.median(D["swing_ms"])
err_ms = err_pct * sw / 100.0
sw_all = np.array(D["swing_ms"])
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
b1.hist(err_ms, bins=28, color=BLUE, alpha=.85, edgecolor="white")
b1.axvline(0, color=RED, lw=2.2)
b1.axvline(np.median(err_ms), color=GREEN, lw=2, ls="--")
b1.set_xlabel("miss relative to target, ms"); b1.set_ylabel("pulses")
b1.set_title(f"Where the pulse lands (n={len(err_ms)})\n"
             f"median {np.median(err_ms):+.1f} ms, p90 {np.percentile(np.abs(err_ms),90):.1f} ms",
             loc="left", fontsize=10, fontweight="bold")
b2.hist(sw_all, bins=25, color=ORANGE, alpha=.85, edgecolor="white")
b2.set_xlabel("swing duration, ms"); b2.set_ylabel("steps")
b2.set_title(f"The animal's own variability\n"
             f"median {np.median(sw_all):.0f} ms, SD {sw_all.std():.1f} ms",
             loc="left", fontsize=10, fontweight="bold")
fig.savefig(f"{OUT}/fig6_landing.png"); plt.close(fig)
print("figures 1-6 done")
