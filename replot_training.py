"""Re-draw the training-dynamics figure from the saved curves (no retraining)."""
import sys; sys.path.insert(0, ".")
from pathlib import Path
from paths import FIG
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from panels import label_axes
from baselines import BASELINES
from run_experiments import FOLDS, ROOT
FIG = FIG; OURS = "RIG-Net-GI"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
H = pd.read_csv(ROOT / "results/training_curves.csv"); S = pd.read_csv(ROOT / "results/training_summary.csv")
models = list(BASELINES) + [OURS]
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
for col, dn in enumerate(FOLDS):
    Hd = H[H.dataset == dn]; Sd = S[S.dataset == dn].sort_values("best_val")
    top5 = [m for m in Sd.model.tolist() if m != OURS][:5]
    if 'MAST-Net' not in top5: top5 = top5[:4] + ['MAST-Net']   # keep the newest baseline in view
    ax = axes[0, col]
    for m in models:
        g = Hd[Hd.model == m]
        if not len(g): continue
        ours = m == OURS
        ax.plot(g.epoch, g.val_MAE, lw=1.6 if ours else 0.7,
                color="#d9480f" if ours else ("#1c7ed6" if m in top5 else "#ced4da"),
                zorder=3 if ours else 2)
    ax.set_yscale("log"); ax.set_title(f"{dn}: all 17 learned models", fontweight="bold")
    ax.set_xlabel("epoch")
    if col == 0: ax.set_ylabel("validation MAE")
    ax.plot([], [], color="#d9480f", lw=1.6, label="RIG-Net-GI")
    ax.plot([], [], color="#1c7ed6", lw=0.9, label="top-5 by val. MAE")
    ax.plot([], [], color="#ced4da", lw=0.9, label="other baselines")
    ax.legend(frameon=False)
    ax2 = axes[1, col]
    pal = ["#1c7ed6", "#2f9e44", "#862e9c", "#f59f00", "#495057"]
    cols = {m: pal[i] for i, m in enumerate(top5)}; cols[OURS] = "#d9480f"
    for m in dict.fromkeys([OURS] + top5):
        g = Hd[Hd.model == m]
        if not len(g): continue
        c = cols.get(m, "#495057")
        ax2.plot(g.epoch, g.val_MAE, lw=1.5 if m == OURS else 1.0, color=c, label=m)
        ax2.plot(g.epoch, g.train_MAE, lw=0.8, ls=":", color=c, alpha=0.7)
        b = Sd[Sd.model == m]
        if len(b): ax2.scatter([b.best_epoch.iloc[0]], [b.best_val.iloc[0]], s=16, color=c, zorder=5, marker="v")
    ax2.set_yscale("log"); ax2.set_xlabel("epoch")
    if col == 0: ax2.set_ylabel("MAE (solid val., dotted train)")
    ax2.set_title(f"{dn}: ours vs top-5 baselines", fontweight="bold")
    ax2.legend(frameon=False, ncol=2, fontsize=5.8)
label_axes(fig, axes)
fig.tight_layout(); fig.savefig(FIG / "training.pdf", bbox_inches="tight", pad_inches=0.015)
print("ok")
