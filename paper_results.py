"""paper_results.py - every number and figure used in the IEEE paper."""
import json, sys
from pathlib import Path
from paths import PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from panels import label_axes
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval
from analyze import pooled, scores, dm_test, PHASE

ROOT = Path(__file__).parent; RES = ROOT / "results"; OUT = PAPER; (OUT / "figs").mkdir(parents=True, exist_ok=True)
BASE = list(BASELINES); REF = ["Naive", "MA3"]
OURS = ["RIG-Net-GI", "RIG-Net", "RIG-Net-S (post-hoc)"]
ABL = ["RIG-Net-GI", "RIG-Net-GI w/o genomic prior", "RIG-Net-GI w/o genomic KL", "RIG-Net-GI w/o intervention",
       "RIG-Net-S (post-hoc)", "RIG-Net"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    out = {"main": {}, "dm": {}, "phase": {}, "horizon": {}, "ablation": {}}
    for dn in FOLDS:
        cache = {}
        for m in REF + BASE + list(dict.fromkeys(OURS + ABL)):
            pl = pooled(dn, m, SEEDS)
            if pl is None: print("missing", dn, m); continue
            cache[m] = pl
            sc = np.array([scores(p, y) for p, y, _, _ in pl])
            out["main"].setdefault(dn, {})[m] = {
                "MAE": sc[:, 0].mean(), "MAE_sd": sc[:, 0].std(),
                "RMSE": sc[:, 1].mean(), "RMSE_sd": sc[:, 1].std(),
                "PCC": sc[:, 2].mean(), "PCC_sd": sc[:, 2].std(), "n_seeds": int(len(sc))}
            pm = np.mean([p for p, _, _, _ in pl], 0); y, hh, fd = pl[0][1:]
            out["phase"].setdefault(dn, {})[m] = [float(np.abs(pm - y)[fd == f].mean()) for f in np.unique(fd)]
            out["horizon"].setdefault(dn, {})[m] = [float(np.abs(pm - y)[hh == h].mean()) for h in range(4)]
        y = cache["Naive"][0][1]
        for o in OURS:
            po = np.mean([p for p, _, _, _ in cache[o]], 0); res = {}
            for b in REF + BASE + [x for x in OURS if x != o]:
                pb = np.mean([p for p, _, _, _ in cache[b]], 0); st, p = dm_test(pb - y, po - y); res[b] = (float(st), float(p))
            out["dm"].setdefault(dn, {})[o] = res
    # ---- interpretability for RIG-Net-GI (seed 0, every fold): beta, gate shares
    interp = {}
    for dn in FOLDS:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}
        ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy"); gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
        betas, gates, rhos = [], [], []
        for f, s in enumerate(FOLDS[dn][0]):
            ds["G"] = gen[f"fold{f}"]; data = splits(ds["inc"], s, FOLDS[dn][1])
            for seed in SEEDS:
                _, _, w = train_eval("RIG-Net-GI", ds, data, seed, ABLATIONS["RIG-Net-GI"])
                w.model.eval()
                with torch.no_grad():
                    o = w.model(w.to_x(data[2][0]), w.Uw["te"], return_diag=True)
                betas.append(w.model.beta_raw.detach().clamp_min(0).numpy()); gates.append(o["pi"].mean((0, 1)).numpy()); rhos.append(float(o["rho"].mean()))
        b = np.array(betas)
        interp[dn] = {"beta_mean": b.mean(0).tolist(), "beta_sd": b.std(0).tolist(),
                      "R_reduction_per_SD_pct": (100 * (1 - np.exp(-b.mean(0)))).tolist(),
                      "gate_share_geo_adp_dyn_gen": np.mean(gates, 0).tolist(), "rho_mean": float(np.mean(rhos))}
        print(dn, interp[dn])
    out["interp"] = interp
    out["genomic"] = {dn: json.load(open(ROOT / "data" / f"{dn}_genomic_summary.json")) for dn in FOLDS}
    json.dump(out, open(OUT / "paper_numbers.json", "w"), indent=1, default=float)

    # ---------------- figures (single column, large fonts)
    cols = {"RIG-Net-GI": "#d9480f", "RIG-Net": "#f59f00", "MTGNN": "#1c7ed6", "N-BEATS": "#2f9e44", "EpiGNN": "#7048e8", "Naive": "#868e96"}
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
        for ax, dn in zip(axes, FOLDS):
            for m, c in cols.items():
                v = out["horizon"][dn][m]
                ax.plot(range(1, 5), v, marker="o", ms=3, lw=1.9 if m == "RIG-Net-GI" else 1.0, color=c, label=m)
            ax.set_title(dn, fontsize=8); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("horizon (wk)")
            ax.margins(y=0.14)
        axes[0].set_ylabel("MAE")
        axes[1].legend(frameon=False, fontsize=5.8, labelspacing=0.25, handlelength=1.3, borderpad=0.1)
        label_axes(fig, axes)
        fig.tight_layout(); fig.savefig(OUT / "figs" / "horizon.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), gridspec_kw={"width_ratios": [3, 4]})
        for ax, dn in zip(axes, FOLDS):
            ph = PHASE[dn]; best = [min(out["phase"][dn][b][i] for b in BASE) for i in range(len(ph))]
            x = np.arange(len(ph)); wdt = 0.27
            for k, (m, c) in enumerate([("RIG-Net-GI", "#d9480f"), ("RIG-Net", "#f59f00")]):
                ax.bar(x + (k - 1) * wdt, [out["phase"][dn][m][i] / best[i] for i in x], wdt, color=c, label=m)
            ax.bar(x + wdt, [1] * len(x), wdt, color="#1c7ed6", label="best baseline (per block)")
            ax.axhline(1, color="k", lw=0.7); ax.set_xticks(x)
            ax.set_xticklabels([p.split(" ")[0] for p in ph], fontsize=5.8)
            ax.set_title(dn, fontsize=8); ax.set_ylim(0, 2.0)
        axes[0].set_ylabel("MAE / best base.", fontsize=7.5)
        axes[1].legend(frameon=False, fontsize=5.8, labelspacing=0.25, handlelength=1.0, borderpad=0.1, loc="upper right")
        label_axes(fig, axes)
        fig.tight_layout(); fig.savefig(OUT / "figs" / "phase.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)


if __name__ == "__main__":
    main()
