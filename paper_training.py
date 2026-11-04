"""paper_training.py - per-epoch training curves for all 17 learned models.

Re-runs the exact training loop of run_experiments.train_eval with per-epoch logging of
training loss, validation MAE and test MAE (test recorded for diagnosis only; selection
still uses validation, as in the main protocol).
"""
import json, sys, time
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from run_experiments import (FOLDS, ABLATIONS, load, splits, metrics, Wrapper, ROOT,
                             BASELINE_LRS, L_IN, H_OUT)

OUT = PAPER; FIG = OUT / "figs"; RES = ROOT / "results"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
OURS = "RIG-Net-GI"; EPOCHS, PATIENCE = 300, 40


def curve(name, ds, data, seed=0, lr=1e-3, variant=None):
    trw = ds["inc"][: data[3][0]]; lw = np.log1p(trw); scaler = (float(lw.mean()), float(lw.std()))
    w = Wrapper(name, ds, scaler, seed, variant)
    if w.use_int:
        U = ds["U"]; v0 = data[3][0]; act = U[:v0, :, 2] > 0
        m_ = U[:v0][act].mean(0); s_ = U[:v0][act].std(0) + 1e-6
        Us = np.where(U[..., 2:3] > 0, (U - m_) / s_, 0.0).astype(np.float32)
        for k, part in (("tr", data[0]), ("va", data[1]), ("te", data[2])):
            w.Uw[k] = torch.tensor(Us[part[3] - 1])
    (Xtr, Ytr, Mtr, _), (Xva, Yva, Mva, _), (Xte, Yte, Mte, _), _ = data
    opt = torch.optim.Adam(w.model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []; best = 1e18; bad = 0; best_ep = 0; t0 = time.time()
    for ep in range(EPOCHS):
        w.model.train(); opt.zero_grad()
        loss = w.loss(Xtr, Ytr, Mtr)
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(w.model.parameters(), 5.0); opt.step()
        if hasattr(w.model, "project"): w.model.project()
        v = metrics(w.predict(Xva, "va"), Yva, Mva)["MAE"]
        t = metrics(w.predict(Xte, "te"), Yte, Mte)["MAE"]
        tr = metrics(w.predict(Xtr, "tr"), Ytr, Mtr)["MAE"]
        hist.append((ep, float(loss.item()), tr, v, t))
        if v < best - 1e-6: best, bad, best_ep = v, 0, ep
        else:
            bad += 1
            if bad > PATIENCE: break
    h = pd.DataFrame(hist, columns=["epoch", "loss", "train_MAE", "val_MAE", "test_MAE"])
    h["model"] = name; h["sec"] = time.time() - t0
    return h, dict(best_val=best, best_epoch=best_ep, epochs_run=len(hist), sec=time.time() - t0,
                   params=sum(p.numel() for p in w.model.parameters()))


def main():
    models = list(BASELINES) + [OURS]
    allh = []; summ = []
    for dn in FOLDS:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}
        ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy")
        gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
        fold = 1; st = FOLDS[dn][0][fold]; ds["G"] = gen[f"fold{fold}"]
        for tv in (1, 4):
            p = ROOT / "data" / f"{dn}_genomic_tau{tv}.npz"
            if p.exists(): ds[f"G_tau{tv}"] = np.load(p)[f"fold{fold}"]
        data = splits(ds["inc"], st, FOLDS[dn][1])
        for m in models:
            variant = ABLATIONS.get(m); lr = (variant or {}).get("lr", 1e-3)
            if m not in ABLATIONS:                    # baselines: the lr their protocol would pick
                best = None
                for cand in BASELINE_LRS:
                    h, s = curve(m, ds, data, 0, cand, None)
                    if best is None or s["best_val"] < best[1]["best_val"]: best = (h, s, cand)
                h, s, lr = best
            else:
                h, s = curve(m, ds, data, 0, lr, variant)
            h["dataset"] = dn; allh.append(h)
            summ.append({"dataset": dn, "model": m, "lr": lr, **s})
            print(f"{dn:8s} {m:14s} best_val={s['best_val']:.3f} @ep{s['best_epoch']:3d} "
                  f"({s['epochs_run']} run, {s['sec']:.1f}s)", flush=True)
    H = pd.concat(allh); H.to_csv(RES / "training_curves.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(RES / "training_summary.csv", index=False)

    # ---------------- figure: all models + top-5 zoom ----------------
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
    for col, dn in enumerate(FOLDS):
        Hd = H[H.dataset == dn]; Sd = S[S.dataset == dn].sort_values("best_val")
        top5 = [m for m in Sd.model.tolist() if m != OURS][:5]
        ax = axes[0, col]
        for m in models:
            g = Hd[Hd.model == m]
            if not len(g): continue
            ours = m == OURS
            ax.plot(g.epoch, g.val_MAE, lw=1.6 if ours else 0.7,
                    color="#d9480f" if ours else ("#1c7ed6" if m in top5 else "#ced4da"),
                    alpha=1.0 if (ours or m in top5) else 0.75, zorder=3 if ours else 2)
        ax.set_yscale("log"); ax.set_title(f"{dn}: all 17 learned models", fontweight="bold")
        ax.set_xlabel("epoch"); 
        if col == 0: ax.set_ylabel("validation MAE")
        ax.plot([], [], color="#d9480f", lw=1.6, label="RIG-Net-GI")
        ax.plot([], [], color="#1c7ed6", lw=0.9, label="top-5 by val. MAE")
        ax.plot([], [], color="#ced4da", lw=0.9, label="other baselines")
        ax.legend(frameon=False)
        # top-5 zoom with train/val gap
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
        ax2.legend(frameon=False, ncol=2, fontsize=7.2)
    fig.tight_layout(); fig.savefig(FIG / "training.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ---------------- convergence table ----------------
    piv = S.pivot_table(index="model", columns="dataset", values=["best_val", "best_epoch", "sec"])
    print(piv.round(2).to_string())
    json.dump({"summary": S.to_dict("records")}, open(OUT / "paper_numbers_train.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
