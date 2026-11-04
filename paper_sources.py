"""paper_sources.py - what the WHO 2014-11-12 data package adds.

Each file carries the same country-week counted twice, once from situation reports and once
from the patient database. That gives (i) an independent check on our cleaned panel and
(ii) a measured observation-error floor: how far two official sources disagree about the
same week, which is a bound on what any forecaster can be expected to achieve.
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER, UPLOADS
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load

UP = UPLOADS; OUT = PAPER; FIG = OUT / "figs"
NAT = {"guinea": ("GIN", "Guinea"), "liberia": ("LBR", "Liberia"), "sierraleone": ("SLE", "Sierra Leone")}
CAP = {"conakry": ("GIN", "Conakry"), "montserrado": ("LBR", "Montserrado"), "freetown": ("SLE", "Freetown")}
COL = {"GIN": "#1c7ed6", "LBR": "#2f9e44", "SLE": "#862e9c"}
# national MAE of RIG-Net-GI, from paper_numbers_countryts.json
MODEL_MAE = {"GIN": 44.4, "LBR": 176.1, "SLE": 168.6}
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})


def series(fname):
    d = pd.read_csv(UP / f"{fname}.csv"); d["w"] = pd.to_datetime(d["Epi week"])
    return d.pivot_table(index="w", columns="Ebola data source", values="Numeric").sort_index()


def main():
    out = {}
    z, meta = load("WA2014"); inc = z["inc"]; wk = pd.to_datetime(meta["weeks"])
    cc = np.array([n.split(":")[0] for n in meta["nodes"]])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))

    # (a) our cleaned panel against the independent WHO patient database
    ax = axes[0]
    for f, (c, name) in NAT.items():
        p = series(f); pdb = p["Patient database"].dropna()
        ours = pd.Series(inc[:, cc == c].sum(1), index=wk)
        ax.plot(pdb.index, pdb.values, color=COL[c], lw=1.5, label=f"{name}, WHO")
        ax.plot(ours.index, ours.values, color=COL[c], lw=1.0, ls="--", alpha=0.8)
        j = pd.concat([ours.rename("o"), pdb.rename("w")], axis=1).dropna()
        out.setdefault("validation", {})[c] = {
            "overlap_weeks": int(len(j)), "pearson": float(j.o.corr(j.w)),
            "our_total": float(j.o.sum()), "who_total": float(j.w.sum()),
            "ratio": float(j.o.sum() / max(j.w.sum(), 1))}
    ax.set_yscale("symlog"); ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("weekly cases"); ax.set_title("Our series vs the WHO patient database", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.4)

    # (b) the two WHO sources against each other on the weeks both report
    ax = axes[1]
    disc = {}
    for f, (c, name) in NAT.items():
        b = series(f).dropna()
        ax.scatter(b["Patient database"], b["Situation report"], s=22, color=COL[c], label=name, zorder=3)
        d = np.abs(b["Situation report"] - b["Patient database"])
        rel = d / b[["Situation report", "Patient database"]].mean(1)
        disc[c] = {"weeks": int(len(b)), "mean_abs_diff": float(d.mean()),
                   "mean_rel_diff_pct": float(rel.mean() * 100)}
    lim = [0, 800]; ax.plot(lim, lim, "k--", lw=0.9); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("patient database"); ax.set_ylabel("situation report")
    ax.set_title("(b) Two WHO sources, same week", fontsize=8.5); ax.legend(frameon=False, fontsize=6.6)
    out["source_discrepancy"] = disc

    # (c) the observation-error floor against our forecast error
    ax = axes[2]
    xs = np.arange(3); w_ = 0.36
    ks = ["GIN", "LBR", "SLE"]
    ax.bar(xs - w_ / 2, [disc[k]["mean_abs_diff"] for k in ks], w_, color="#adb5bd",
           label="disagreement between\nthe two WHO sources")
    ax.bar(xs + w_ / 2, [MODEL_MAE[k] for k in ks], w_, color="#d9480f", label="RIG-Net-GI national MAE")
    for i, k in enumerate(ks):
        ax.text(i - w_ / 2, disc[k]["mean_abs_diff"] + 4, f"{disc[k]['mean_abs_diff']:.0f}", ha="center", fontsize=7)
        ax.text(i + w_ / 2, MODEL_MAE[k] + 4, f"{MODEL_MAE[k]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([NAT[f][1] for f in NAT], fontsize=7.5)
    ax.set_ylabel("weekly cases"); ax.set_title("(c) Forecast error vs noise floor", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.4)
    out["model_vs_floor"] = {k: {"model_MAE": MODEL_MAE[k], "source_disagreement": disc[k]["mean_abs_diff"],
                                 "ratio": MODEL_MAE[k] / max(disc[k]["mean_abs_diff"], 1e-9)} for k in ks}
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "sources.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # capital concentration, reported in text only
    cap = {}
    for f, (c, name) in CAP.items():
        a = series(f)["Patient database"].dropna()
        nat = series([k for k, v in NAT.items() if v[0] == c][0])["Patient database"].dropna()
        j = pd.concat([a.rename("cap"), nat.rename("nat")], axis=1).dropna()
        cap[name] = {"weeks": int(len(j)), "share_pct": float(100 * j.cap.sum() / max(j.nat.sum(), 1))}
    out["capital_share"] = cap
    json.dump(out, open(OUT / "paper_numbers_sources.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
