"""paper_problem.py - Fig. 1: the forecasting problem and its data regime (built from the real panels)."""
import json, sys
from pathlib import Path
from paths import FIG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load

ROOT = Path(__file__).parent; FIG = FIG
plt.rcParams.update({"font.size": 7.6, "axes.titlesize": 8, "axes.labelsize": 7.4,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
                     "axes.spines.top": False, "axes.spines.right": False})

w = 0.185, 0.361, 0.258, 0.135, 0.061
fig = plt.figure(figsize=(3.45, 2.85))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.78, wspace=0.42)

# ---------------- (a) region graph + forecast ----------------
ax = fig.add_subplot(gs[0, :])
z, meta = load("WA2014"); inc = z["inc"]; co = z["coords"]; W = z["W"]
tot = inc.sum(0); top = np.argsort(-tot)[:16]
for i in top:
    for j in top:
        if i < j and W[i, j] > 0:
            ax.plot([co[i, 1], co[j, 1]], [co[i, 0], co[j, 0]], color="k", alpha=0.18, lw=0.6, zorder=1)
sc = ax.scatter(co[top, 1], co[top, 0], s=10 + 90 * tot[top] / tot.max(), c=np.log1p(tot[top]),
                cmap="YlOrRd", edgecolors="k", linewidths=0.4, zorder=3)
ax.set_title("(a) Weekly incidence on a region graph", fontweight="bold", fontsize=8)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.margins(x=0.06, y=0.16)
ax.annotate("forecast 1 to 4 weeks ahead,\nevery region", xy=(0.015, 0.12), xycoords="axes fraction", fontsize=6.6)

# ---------------- (b) three structures ----------------
ax = fig.add_subplot(gs[1, 0])
ax.bar(range(1, 6), w, color="#d9480f")
ax.set_title("(b1) Local renewal", fontweight="bold", fontsize=8)
ax.set_xticks(range(1, 6)); ax.set_xlabel("lag (weeks)"); ax.set_ylabel("$w_s$")

ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
ax.set_title("(b2) Spread and response", fontweight="bold", fontsize=8)
# the three mechanisms, laid out on one row so each label sits beside its own arrow
pts = {"j": (0.06, 0.66), "i": (0.50, 0.66), "k": (0.94, 0.66)}
for src in ("j", "k"):
    ax.annotate("", xy=pts["i"], xytext=pts[src],
                arrowprops=dict(arrowstyle="-|>", color="#2b8a3e", lw=1.6,
                                shrinkA=7, shrinkB=8, mutation_scale=11))
ax.annotate("", xy=(0.565, 0.76), xytext=(0.435, 0.76),
            arrowprops=dict(arrowstyle="-|>", color="#d9480f", lw=1.4,
                            connectionstyle="arc3,rad=-1.4"))
ax.annotate("", xy=(0.50, 0.34), xytext=(0.50, 0.58),
            arrowprops=dict(arrowstyle="-|>", color="#1c7ed6", lw=1.5, shrinkA=0, shrinkB=0))
for key, (x, y) in pts.items():
    ax.scatter([x], [y], s=190, c="#e9ecef", edgecolors="k", linewidths=0.8, zorder=3)
    ax.text(x, y, f"${key}$", ha="center", va="center", fontsize=8, zorder=4)
ax.text(0.50, 1.02, r"$1-\rho_i$", ha="center", va="center", fontsize=7.5, color="#d9480f")
ax.text(0.280, 0.545, r"$\rho_i$", ha="center", va="top", fontsize=7.5, color="#2b8a3e")
ax.text(0.720, 0.545, r"$\rho_i$", ha="center", va="top", fontsize=7.5, color="#2b8a3e")
ax.text(0.50, 0.315, r"$\downarrow R_i$", ha="center", va="top", fontsize=7.5, color="#1c7ed6")
ax.text(0.50, 0.045, "renewal, importation, response", ha="center", va="center",
        fontsize=6.4, color="black")
ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.03, 1.15)

rows = []
for dn in ("WA2014", "DRC1820"):
    zz, mm = load(dn); x = zz["inc"]
    rows.append((dn, 100 * (x == 0).mean(), x.shape[0], x.shape[1]))
fig.text(0.5, -0.115, "data regime:  WA2014 37 weeks / 61 regions / {:.0f}% zero region-weeks\n"
                      "DRC1820 99 weeks / 29 regions / {:.0f}% zero".format(rows[0][1], rows[1][1]),
         ha="center", fontsize=6.6, color="#343a40")
fig.savefig(FIG / "problem.pdf", bbox_inches="tight", pad_inches=0.015)
print("ok", rows)
