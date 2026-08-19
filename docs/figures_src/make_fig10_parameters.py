# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
GREEN, ORANGE, RED, GREY, BLUE = "#1d9e75", "#d85a30", "#e02828", "#8a8a8a", "#2f6fbd"
PURPLE = "#7a4fbf"
OUT = "../media/report"
MONO = "DejaVu Sans Mono"

fig, ax = plt.subplots(figsize=(13.6, 8.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.972, "Stimulation parameters: fixed, searched, derived, limited",
        ha="center", fontsize=15, fontweight="bold")

CW, GAPC = 0.225, 0.023
X0 = (1.0 - (4 * CW + 3 * GAPC)) / 2
CY, CH = 0.535, 0.365
cols = [
    ("FIXED before the session", GREY, "#f2f2f2",
     ["pulse shape", "   biphasic, charge-balanced", "   cathodic-first",
      "   phase width     200 us", "   interphase        50 us", "",
      "montage", "   segment  L2 / S1", "   cathode / anode contact",
      "   mono- or bipolar", "   contact area, impedance"]),
    ("SEARCHED by the optimiser", PURPLE, "#f3eefb",
     ["amplitude        100-600 uA", "frequency          20-80 Hz",
      "train duration    50-200 ms", "interchannel        0-60 ms", "",
      "four continuous parameters,", "same count as Drainville 2025.", "",
      "Timing is NOT here: it is set", "by the fast loop, as a percent",
      "of the step cycle."]),
    ("DERIVED, never stored", BLUE, "#eaf2fb",
     ["interpulse   = 1000 / freq", "pulses/train = train / interpulse", "",
      "charge/phase = I x phase width", "charge density", "   = charge / contact area",
      "duty cycle", "   = train / (train + intertrain)", "",
      "computed as properties, so they", "cannot drift from their inputs"]),
    ("HARD LIMITS", RED, "#fdecec",
     ["max amplitude", "max charge per phase", "max charge density",
      "max duty cycle", "charge balance required", "",
      "checked BEFORE a point is even", "proposed.", "",
      "Constraints, not penalties: the", "optimiser cannot trade safety",
      "for score."]),
]
for i, (title, edge, fc, lines) in enumerate(cols):
    x = X0 + i * (CW + GAPC)
    ax.add_patch(FancyBboxPatch((x, CY), CW, CH,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                fc=fc, ec=edge, lw=1.8, zorder=3))
    ax.text(x + CW / 2, CY + CH - 0.028, title, ha="center", fontsize=10,
            fontweight="bold", color=edge, zorder=4)
    for j, ln in enumerate(lines):
        ax.text(x + 0.013, CY + CH - 0.068 - j * 0.0255, ln, ha="left", va="top",
                fontsize=7.3, color="#333", zorder=4, family=MONO)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + CW + 0.003, CY + CH / 2),
                                     (x + CW + GAPC - 0.003, CY + CH / 2),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.4, color="#999", zorder=2))

# ---------- пять интервалов ------------------------------------------------
ax.text(0.5, 0.462, 'The five different gaps that a single field "interval" '
                    'would confuse',
        ha="center", fontsize=12.5, fontweight="bold")

PY0, PH = 0.115, 0.300
ax.add_patch(Rectangle((0.030, PY0), 0.940, PH, fc="#fafafa", ec="#e0e0e0",
                       lw=1.2, zorder=0))
GY = PY0 + 0.195                                    # уровень глифов
LY = PY0 + 0.085                                    # уровень подписей
centres = [0.135, 0.325, 0.505, 0.690, 0.875]


def bar(x, w, h, col, y=None):
    ax.add_patch(Rectangle((x, (y if y is not None else GY) - h / 2), w, h,
                           fc=col, ec="none", zorder=3))


def span(x1, x2, y):
    ax.annotate("", xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))


def label(cx, n, name, col, sub):
    ax.text(cx, LY, f"{n}   {name}", ha="center", fontsize=9, color=col,
            fontweight="bold")
    ax.text(cx, LY - 0.038, sub, ha="center", fontsize=8, color="#555")


# 1 interphase: две фазы одного импульса
c = centres[0]
bar(c - 0.026, 0.011, 0.070, RED, GY + 0.020)
bar(c + 0.013, 0.011, 0.070, RED, GY - 0.020)
span(c - 0.015, c + 0.013, GY)
label(c, "1", "interphase", RED, "inside ONE pulse,  50 us")

# 2 interpulse
c = centres[1]
for k in range(4):
    bar(c - 0.046 + k * 0.030, 0.008, 0.095, PURPLE)
span(c - 0.038, c - 0.016, GY + 0.062)
label(c, "2", "interpulse", PURPLE, "= 1000 / frequency")

# 3 interchannel
c = centres[2]
for k in range(3):
    bar(c - 0.045 + k * 0.024, 0.007, 0.055, GREEN, GY + 0.030)
    bar(c - 0.021 + k * 0.024, 0.007, 0.055, ORANGE, GY - 0.030)
span(c - 0.045, c - 0.021, GY)
label(c, "3", "interchannel", GREEN, "L2 leads,  S1 follows")

# 4 intertrain
c = centres[3]
for grp in (-0.050, 0.016):
    for k in range(3):
        bar(c + grp + k * 0.010, 0.006, 0.095, BLUE)
span(c - 0.030, c + 0.016, GY + 0.062)
label(c, "4", "intertrain", BLUE, "between bursts,  ms")

# 5 intertrial
c = centres[4]
for grp in (-0.055, 0.020):
    bar(c + grp, 0.035, 0.075, GREY)
span(c - 0.020, c + 0.020, GY + 0.052)
label(c, "5", "intertrial", "#555", "rest between trials,  s")

ax.text(0.5, 0.040, 'A single field named "interval" loses which of these five was '
                    'changed. Six months later the protocol cannot be reconstructed.',
        ha="center", fontsize=9, color="#777", style="italic")
fig.savefig(f"{OUT}/fig10_parameters.png"); plt.close(fig)
print("fig10 перерисована")
