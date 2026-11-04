"""paper_nexus.py - health-access nexus graphs, cross-country relationships, genomic sequencing.

Data provenance per country (stated in the captions, not smoothed over):
  Sierra Leone : 1,099 geolocated health facilities (DHIS national list, 2010) -> true distance to nearest facility
  DRC          : healthsites.io facility counts + WorldPop per health zone (uploaded) -> facilities per 100k
  Liberia, Guinea : no geolocated facility list was obtainable; we fall back to the
                    laboratory-confirmation share, a surveillance-access proxy derived from the case data.
"""
import json, sys, re, unicodedata
from pathlib import Path
from paths import FIG, PAPER, UPLOADS
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import FOLDS, load
from genomic_prior import load_wa, load_drc, norm

ROOT = Path(__file__).parent; EXT = ROOT / "external"; UP = UPLOADS
OUT = PAPER; FIG = OUT / "figs"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
CC = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone", "COD": "DR Congo"}
COL = {"GIN": "#1c7ed6", "LBR": "#2f9e44", "SLE": "#862e9c", "COD": "#d9480f"}


def haversine(a, b):
    a = np.radians(np.atleast_2d(a)); b = np.radians(np.atleast_2d(b))
    dlat = b[None, :, 0] - a[:, None, 0]; dlon = b[None, :, 1] - a[:, None, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def main():
    out = {}
    wa = load("WA2014"); wm = json.load(open(ROOT / "data" / "WA2014.json"))
    dr = load("DRC1820"); dm = json.load(open(ROOT / "data" / "DRC1820.json"))
    wc = wa[0]["coords"]; dc = dr[0]["coords"]
    wnodes = wm["nodes"]; dnodes = dm["nodes"]
    cc = np.array([n.split(":")[0] for n in wnodes])
    Uw = np.load(ROOT / "data" / "WA2014_interventions.npy")
    Ud = np.load(ROOT / "data" / "DRC1820_interventions.npy")

    # ---------- access indicator per region ----------
    fac = pd.read_csv(EXT / "sle_facilities.csv")                 # NOTE: LAT column holds longitude
    hosp = fac[fac.FACILITY_T.str.contains("HOSPITAL", na=False)]
    fxy = np.column_stack([hosp.LONG.values, hosp.LAT.values])    # (lat, lon); LAT column holds longitude
    acc = {}; kind = {}
    sle = cc == "SLE"
    acc["SLE"] = haversine(wc[sle], fxy).min(1); kind["SLE"] = "km to nearest hospital"
    hs = pd.read_csv(UP / "healthsites_io__healthsite_count__static.csv").set_index("nom").healthsite_count
    pop = pd.read_csv(UP / "worldpop__pop_count__static.csv").set_index("nom").pop_count
    cnt = hs.reindex(dnodes).fillna(0).values; pp = pop.reindex(dnodes).fillna(pop.median()).values
    acc["COD"] = 1e5 * cnt / np.maximum(pp, 1); kind["COD"] = "facilities per 100k"
    for c in ("LBR", "GIN"):
        m = cc == c
        with np.errstate(invalid="ignore"):
            lab = np.nanmean(np.where(Uw[..., 2] > 0, Uw[..., 0], np.nan), 0)[m]
        lab = np.where(np.isfinite(lab), lab, np.nanmedian(lab))
        acc[c] = lab; kind[c] = "lab-confirmation share"

    # ================= FIG : nexus graphs =================
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
    Wgeo = wa[0]["W"]; Dgeo = dr[0]["W"]
    for ax, c in zip(axes.ravel(), ["SLE", "LBR", "GIN", "COD"]):
        if c == "COD":
            co = dc; A = Dgeo; v = acc[c]; names = [n for n in dnodes]
        else:
            m = cc == c; co = wc[m]; A = Wgeo[np.ix_(m, m)]; v = acc[c]
            names = [n.split(":")[1] for n, k in zip(wnodes, m) if k]
        for i in range(len(co)):
            for j in range(i + 1, len(co)):
                if A[i, j] > 0:
                    ax.plot([co[i, 1], co[j, 1]], [co[i, 0], co[j, 0]], color="black", alpha=0.2, lw=0.7, zorder=1)
        rev = c in ("SLE",)                                        # for distance, larger = worse
        sc = ax.scatter(co[:, 1], co[:, 0], c=v, s=78, cmap="RdYlGn" + ("_r" if rev else ""),
                        edgecolors="black", linewidths=0.5, zorder=3)
        worst = int(np.argmax(v) if rev else np.argmin(v))
        ha = "right" if co[worst, 1] > np.median(co[:, 1]) else "left"
        ax.annotate(names[worst], (co[worst, 1], co[worst, 0]), fontsize=8.5,
                    xytext=(-5 if ha == "right" else 5, 5), textcoords="offset points", ha=ha)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=8)
        cb.set_label(kind[c], fontsize=8.2)
        TITLE = {"SLE": "Sierra Leone: distance to a hospital",
                 "LBR": "Liberia: laboratory confirmation",
                 "GIN": "Guinea: laboratory confirmation",
                 "COD": "DR Congo: facility density"}
        ax.set_title(TITLE[c], fontweight="bold", color=COL[c], fontsize=9)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.tick_params(labelsize=8); ax.margins(0.13)
        out.setdefault("nexus", {})[c] = {"indicator": kind[c], "n_regions": int(len(v)),
                                          "median": float(np.nanmedian(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor="#74c476",
                      markeredgecolor="k", ms=8, label="region, better served"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#d73027",
                      markeredgecolor="k", ms=8, label="region, worse served"),
               Line2D([], [], color="black", alpha=0.25, lw=1.4, label="geographic adjacency"),
               Line2D([], [], marker="o", color="none", markerfacecolor="none",
                      markeredgecolor="none", ms=0, label="label: least-served region")]
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.015), fontsize=8.4, handletextpad=0.5, columnspacing=1.6)
    label_axes(fig, axes, relayout=False)
    fig.savefig(FIG / "nexus.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # correlation of access with model error / R, within SLE and DRC
    from paper_analysis import field
    Pw, Yw, _ = field("WA2014", "RIG-Net-GI"); Pd, Yd, _ = field("DRC1820", "RIG-Net-GI")
    ew = np.nanmean(np.abs(Pw[1] - Yw[1]), 0); ed = np.nanmean(np.abs(Pd[1] - Yd[1]), 0)
    out["nexus_corr"] = {
        "SLE_dist_vs_error": float(spearmanr(acc["SLE"], ew[cc == "SLE"], nan_policy="omit").statistic),
        "COD_facpc_vs_error": float(spearmanr(acc["COD"], ed, nan_policy="omit").statistic)}

    # ================= FIG : country relationships =================
    G = np.load(ROOT / "data" / "WA2014_genomic.npz")[f"fold{len(FOLDS['WA2014'][0])-1}"]
    inc = wa[0]["inc"]
    codes = ["GIN", "LBR", "SLE"]
    M = np.zeros((3, 3)); Cm = np.zeros((3, 3))
    for a, ca in enumerate(codes):
        for b, cb in enumerate(codes):
            M[a, b] = G[np.ix_(cc == ca, cc == cb)].sum()
            ia = inc[:, cc == ca].sum(1); ib = inc[:, cc == cb].sum(1)
            Cm[a, b] = np.corrcoef(ia, ib)[0, 1]
    fig = plt.figure(figsize=(7.16, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.42)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    im = axes[0].imshow(np.log1p(M), cmap="Purples", aspect="auto")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels([CC[c] for c in codes], rotation=25, ha="right")
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels([CC[c] for c in codes])
    for a in range(3):
        for b in range(3):
            axes[0].text(b, a, f"{M[a,b]:.0f}", ha="center", va="center", fontsize=7.4,
                         color="white" if np.log1p(M[a, b]) > 0.6 * np.log1p(M).max() else "black")
    axes[0].set_title("(a) Genomic linkage mass", fontweight="bold")
    cb1 = plt.colorbar(im, ax=axes[0], fraction=0.034, pad=0.015)
    cb1.set_label("$\\log(1+G)$", fontsize=6, labelpad=1); cb1.ax.tick_params(labelsize=5.8)
    im2 = axes[1].imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels([CC[c] for c in codes], rotation=25, ha="right")
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels([CC[c] for c in codes])
    for a in range(3):
        for b in range(3):
            axes[1].text(b, a, f"{Cm[a,b]:.2f}", ha="center", va="center", fontsize=6)
    axes[1].set_title("(b) National epidemic-curve corr.", fontweight="bold")
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.034, pad=0.015)
    cb2.set_label("Pearson $r$", fontsize=6, labelpad=1); cb2.ax.tick_params(labelsize=5.8)
    wa_inc = inc.sum(1); wa_start = int(np.argmax(wa_inc > 0))
    for c in codes:
        y = inc[:, cc == c].sum(1)
        axes[2].plot(np.arange(len(y)) - wa_start, y, color=COL[c], lw=1.4, label=CC[c])
    d_inc = dr[0]["inc"].sum(1); d_start = int(np.argmax(d_inc > 0))
    axes[2].plot(np.arange(len(d_inc)) - d_start, d_inc, color=COL["COD"], lw=1.4, ls="--",
                 label="DR Congo (2018\u201320)")
    axes[2].set_yscale("symlog"); axes[2].legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    axes[2].set_ylabel("weekly cases"); axes[2].set_xlabel("weeks since that outbreak's first case")
    axes[2].set_title("(c) National epidemic curves, aligned on outbreak start", fontweight="bold")
    axes[2].set_xlim(-2, None); axes[2].set_ylim(0, 2e4)
    label_axes(fig, axes, relayout=False)
    fig.savefig(FIG / "countries.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["country_rel"] = {"genomic_mass": M.tolist(), "curve_corr": Cm.tolist(),
                          "cross_border_fraction": float((M.sum() - np.trace(M)) / M.sum())}

    # ================= FIG : genomic sequencing by country =================
    Aw, mw = load_wa(wnodes); Ad, md = load_drc(dnodes)
    mw = mw[mw.node >= 0].copy(); mw["cc"] = [wnodes[n].split(":")[0] for n in mw.node]
    md = md[md.node >= 0].copy(); md["cc"] = "COD"
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7))
    seq = {}
    for c, frame in [(c, mw[mw.cc == c]) for c in codes] + [("COD", md)]:
        g = frame.dropna(subset=["date"]).sort_values("date")
        wk = (g.date - g.date.min()).dt.days / 7.0
        axes[0].plot(wk, np.arange(1, len(g) + 1), color=COL[c], lw=1.3,
                     ls="--" if c == "COD" else "-", label=CC[c])
        seq[c] = int(len(g))
    axes[0].set_xlabel("weeks since that country's first genome")
    axes[0].set_ylabel("cumulative genomes"); axes[0].legend(frameon=False, fontsize=7.2, loc="lower right")
    axes[0].set_title("(a) Sequencing over time", fontweight="bold")
    cases = {c: float(inc[:, cc == c].sum()) for c in codes}; cases["COD"] = float(dr[0]["inc"].sum())
    frac = [100 * seq[c] / cases[c] for c in codes + ["COD"]]
    axes[1].bar(range(4), frac, color=[COL[c] for c in codes + ["COD"]])
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels([CC[c] for c in codes + ["COD"]], rotation=25, ha="right")
    axes[1].set_ylabel("% of cases sequenced"); axes[1].set_title("(b) Sequencing intensity", fontweight="bold")
    for i, v in enumerate(frac): axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    covw = {c: (np.array([(G[i] > 0).any() for i in np.where(cc == c)[0]]).mean()) for c in codes}
    Gd = np.load(ROOT / "data" / "DRC1820_genomic.npz")[f"fold{len(FOLDS['DRC1820'][0])-1}"]
    covw["COD"] = float((Gd.sum(1) > 0).mean())
    axes[2].bar(range(4), [100 * covw[c] for c in codes + ["COD"]], color=[COL[c] for c in codes + ["COD"]])
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels([CC[c] for c in codes + ["COD"]], rotation=25, ha="right")
    axes[2].set_ylabel("% of regions with genomes"); axes[2].set_title("(c) Spatial genomic coverage", fontweight="bold")
    for i, c in enumerate(codes + ["COD"]): axes[2].text(i, 100 * covw[c], f"{100*covw[c]:.0f}", ha="center", va="bottom", fontsize=6)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "sequencing.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["n_hospitals_SLE"] = int(len(hosp))
    out["sequencing"] = {"genomes": seq, "cases": cases, "pct_sequenced": dict(zip(codes + ["COD"], frac)),
                         "pct_regions_covered": {k: float(100 * v) for k, v in covw.items()}}

    json.dump(out, open(OUT / "paper_numbers_nexus.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
