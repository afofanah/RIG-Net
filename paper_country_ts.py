"""paper_country_ts.py - one figure: national time-step predictions for all four countries.

Region-level forecasts are aggregated to the national level. The predictive interval is a
normal approximation to the sum of the regional negative binomials (variance mu + alpha*mu^2),
which is stated in the caption rather than passed off as the exact NB interval.
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT, ROOT
from paper_analysis import field

OUT = PAPER; FIG = OUT / "figs"
OURS = "RIG-Net-GI"
CC = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone", "COD": "DR Congo"}
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})


def collect(dn):
    z, meta = load(dn); ds = {k: z[k] for k in z.files}
    ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy"); gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
    T, N = ds["inc"].shape
    MU = np.full((H_OUT, T, N), np.nan); AL = np.zeros(N)
    for f, s in enumerate(FOLDS[dn][0]):
        ds["G"] = gen[f"fold{f}"]
        for tag, pat in (("G_tau%d", "{dn}_genomic_tau{v}.npz"), ("G_f%d", "{dn}_genomic_f{v}.npz")):
            for v in ((1, 4) if "tau" in pat else (25, 50, 75)):
                p = ROOT / "data" / pat.format(dn=dn, v=v)
                if p.exists(): ds[tag % v] = np.load(p)[f"fold{f}"]
        data = splits(ds["inc"], s, FOLDS[dn][1])
        _, _, w = train_eval(OURS, ds, data, 0, ABLATIONS[OURS])
        w.model.eval(); Xte, Yte, Mte, O = data[2]
        with torch.no_grad(): o = w.model(w.to_x(Xte), w.Uw["te"], return_diag=True)
        AL = o["phi"].numpy()
        for b, t0 in enumerate(O):
            for h in range(H_OUT):
                if Mte[b, h].any() and t0 + h < T: MU[h, t0 + h] = o["mu"][b, h].numpy()
    return ds["inc"], MU, AL, meta


def main():
    inc_w, MUw, ALw, mw = collect("WA2014")
    inc_d, MUd, ALd, md = collect("DRC1820")
    Pb_w, _, _ = field("WA2014", "MTGNN"); Pb_d, _, _ = field("DRC1820", "MTGNN")
    cc = np.array([n.split(":")[0] for n in mw["nodes"]])

    panels = []
    for code in ("GIN", "LBR", "SLE"):
        m = cc == code
        panels.append(dict(name=CC[code], weeks=pd.to_datetime(mw["weeks"]), obs=inc_w[:, m].sum(1),
                           mu1=MUw[0][:, m].sum(1), mu4=MUw[3][:, m].sum(1),
                           var=np.nansum(MUw[0][:, m] + ALw[None, m] * MUw[0][:, m] ** 2, 1),
                           base=Pb_w[0][:, m].sum(1), starts=FOLDS["WA2014"][0], blk=FOLDS["WA2014"][1]))
    panels.append(dict(name=CC["COD"], weeks=pd.to_datetime(md["weeks"]), obs=inc_d.sum(1),
                       mu1=MUd[0].sum(1), mu4=MUd[3].sum(1),
                       var=np.nansum(MUd[0] + ALd[None, :] * MUd[0] ** 2, 1),
                       base=Pb_d[0].sum(1), starts=FOLDS["DRC1820"][0], blk=FOLDS["DRC1820"][1]))

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
    for ax, p in zip(axes.ravel(), panels):
        wk = p["weeks"]
        for s in p["starts"]:
            ax.axvspan(wk[s], wk[min(s + p["blk"], len(wk)) - 1], color="#fff4e6", zorder=0)
        mu = np.where(p["mu1"] > 0, p["mu1"], np.nan)
        sd = np.sqrt(np.where(p["var"] > 0, p["var"], np.nan))
        ax.fill_between(wk, np.maximum(mu - norm.ppf(0.95) * sd, 0), mu + norm.ppf(0.95) * sd,
                        color="#d9480f", alpha=0.15, lw=0, label="90% interval (normal approx.)")
        ax.plot(wk, p["obs"], color="k", lw=1.5, label="observed")
        ax.plot(wk, mu, color="#d9480f", lw=1.6, label="RIG-Net-GI $h{=}1$")
        ax.plot(wk, np.where(p["mu4"] > 0, p["mu4"], np.nan), color="#f59f00", lw=1.3, label="RIG-Net-GI $h{=}4$")
        ax.plot(wk, np.where(p["base"] > 0, p["base"], np.nan), color="#1c7ed6", lw=1.1, ls="--", label="MTGNN $h{=}1$")
        ax.set_title(p["name"]); ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]: ax.set_ylabel("weekly cases")
    for ax in axes[1, :]: ax.set_xlabel("week")
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index(x) for x in ["observed", "RIG-Net-GI $h{=}1$", "RIG-Net-GI $h{=}4$",
                                  "MTGNN $h{=}1$", "90% interval (normal approx.)"] if x in l]
    fig.legend([h[i] for i in order], [l[i] for i in order], frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.045), fontsize=8.5,
               columnspacing=1.6, handletextpad=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(hspace=0.62)             # room for the row-1 letters
    label_axes(fig, axes, relayout=False)        # after layout, so the letters land correctly
    fig.savefig(FIG / "timestep_countries.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    summ = {}
    for p in panels:
        m = np.isfinite(p["mu1"]) & (p["mu1"] > 0)
        summ[p["name"]] = {"national_MAE_h1": float(np.mean(np.abs(p["mu1"][m] - p["obs"][m]))),
                           "national_MAE_h1_MTGNN": float(np.mean(np.abs(p["base"][m] - p["obs"][m]))),
                           "weeks_scored": int(m.sum()),
                           "coverage90_normal_approx": float(np.mean(
                               (p["obs"][m] >= p["mu1"][m] - 1.645 * np.sqrt(p["var"][m])) &
                               (p["obs"][m] <= p["mu1"][m] + 1.645 * np.sqrt(p["var"][m]))))}
    json.dump(summ, open(OUT / "paper_numbers_countryts.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
