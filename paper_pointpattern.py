"""paper_pointpattern.py - descriptive decompositions of the two outbreaks.

(1) spatial point-pattern: kernel density, Ripley's L against a CSR envelope, nearest-neighbour distance
(2) temporal: weekly reporting, cumulative share by country, duration of local activity
(3) spatio-temporal: where the outbreak had reached in three successive periods
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import gaussian_kde
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load, ROOT

OUT = PAPER; FIG = OUT / "figs"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})
CCOL = {"GIN": "#1c7ed6", "LBR": "#2f9e44", "SLE": "#862e9c", "COD": "#d9480f"}
CNAME = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone", "COD": "DR Congo"}


def to_km(coords):
    """Local equirectangular projection (km) about the panel centroid."""
    lat0 = np.radians(coords[:, 0].mean())
    x = 111.32 * np.cos(lat0) * (coords[:, 1] - coords[:, 1].mean())
    y = 110.57 * (coords[:, 0] - coords[:, 0].mean())
    return np.column_stack([x, y])


def ripley_L(P, r, area, hull_path):
    """Ripley's L(r) with a simple border correction: only points whose r-disc lies inside the window."""
    n = len(P); tree = cKDTree(P)
    out = []
    for rr in r:
        inner = np.array([hull_path.contains_point(p) for p in P])
        cnt = np.array([len(tree.query_ball_point(p, rr)) - 1 for p in P])
        K = area * cnt[inner].mean() / max(n - 1, 1) if inner.any() else np.nan
        out.append(np.sqrt(max(K, 0) / np.pi))
    return np.array(out)


def case_points(P, counts, scale, rng, jitter=8.0):
    """One point per `scale` cases, placed at its region seat with a small jitter."""
    reps = np.maximum(np.round(counts / scale).astype(int), 0)
    pts = np.repeat(P, reps, axis=0)
    return pts + rng.normal(0, jitter, size=pts.shape)


def null_envelope(P, counts, scale, r, area, hull_path, nsim=99, seed=0):
    """Null: the same number of cases redistributed at random over the same region seats,
    so the envelope holds the administrative layout fixed and tests only the concentration of cases."""
    rng = np.random.default_rng(seed); n_pts = int(np.round(counts.sum() / scale)); sims = []
    for _ in range(nsim):
        idx = rng.integers(0, len(P), size=n_pts)
        pts = P[idx] + rng.normal(0, 8.0, size=(n_pts, 2))
        sims.append(ripley_L(pts, r, area, hull_path) - r)
    S = np.array(sims)
    return np.nanpercentile(S, 2.5, axis=0), np.nanpercentile(S, 97.5, axis=0)


def main():
    data = {}
    for dn in ("WA2014", "DRC1820", "GIN2021"):
        z, meta = load(dn)
        inc = z["inc"]; coords = z["coords"]; weeks = pd.to_datetime(meta["weeks"])
        cc = np.array([n.split(":")[0] if ":" in n else "COD" for n in meta["nodes"]])
        data[dn] = dict(inc=inc, coords=coords, weeks=weeks, cc=cc, nodes=meta["nodes"])

    pp_data = {k: v for k, v in data.items() if k != "GIN2021"}

    # ================= (1) spatial point-pattern =================
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.4))
    stats = {}
    for row, (dn, D) in enumerate(pp_data.items()):
        inc, coords, cc = D["inc"], D["coords"], D["cc"]
        active = inc.sum(0) > 0
        P = to_km(coords)[active]; tot = inc.sum(0)[active]
        hull = ConvexHull(P); hp = MplPath(P[hull.vertices]); area = hull.volume
        # (a) kernel density of reported cases
        ax = axes[row, 0]
        kde = gaussian_kde(P.T, weights=tot, bw_method=0.35)
        gx, gy = np.mgrid[P[:, 0].min() - 20:P[:, 0].max() + 20:160j,
                          P[:, 1].min() - 20:P[:, 1].max() + 20:160j]
        zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        inside = np.array([hp.contains_point(p) for p in np.column_stack([gx.ravel(), gy.ravel()])]).reshape(gx.shape)
        zz = np.where(inside, zz, np.nan); zz = zz / np.nanmax(zz)
        im = ax.pcolormesh(gx, gy, zz, cmap="YlOrRd", shading="auto")
        ax.scatter(P[:, 0], P[:, 1], s=4, c="k", zorder=3)
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=6.5)
        cb.set_label("relative density", fontsize=6.8)
        ax.set_title("(a) Case-weighted density" if row == 0 else "", fontsize=8.5)
        ax.set_xlabel("km east"); ax.set_ylabel(f"{dn}\nkm north")
        # (b) Ripley's L
        ax = axes[row, 1]
        rmax = 0.25 * np.sqrt(area); r = np.linspace(5, rmax, 20)
        rng = np.random.default_rng(0)
        scale = max(1.0, tot.sum() / 400)
        CP = case_points(P, tot, scale, rng)
        Lobs = ripley_L(CP, r, area, hp) - r
        lo, hi = null_envelope(P, tot, scale, r, area, hp)
        ax.fill_between(r, lo, hi, color="#adb5bd", alpha=0.55,
                        label="cases reshuffled over regions\n(99 sims)")
        ax.plot(r, Lobs, "k", lw=1.5, label="observed cases")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("$r$ (km)"); ax.set_ylabel("$L(r)-r$")
        ax.set_title("(b) Case clustering vs region layout" if row == 0 else "", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.4, loc="lower left")
        clustered = float(np.mean(Lobs > hi))
        # (c) nearest-neighbour distance by country
        ax = axes[row, 2]
        tree = cKDTree(P); nnd = tree.query(P, k=2)[0][:, 1]
        for c in np.unique(cc[active]):
            m = cc[active] == c
            ax.hist(nnd[m], bins=np.linspace(0, np.percentile(nnd, 99) * 1.1, 16), histtype="step",
                    lw=1.3, color=CCOL.get(c, "#495057"), label=CNAME.get(c, c))
        ax.set_xlabel("distance to nearest affected region (km)"); ax.set_ylabel("count")
        ax.set_title("(c) Nearest-neighbour distance" if row == 0 else "", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.4)
        stats[dn] = {"affected_regions": int(active.sum()), "window_area_km2": float(area),
                     "median_nnd_km": float(np.median(nnd)),
                     "frac_r_clustered": clustered, "max_L_minus_r": float(np.nanmax(Lobs))}
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "pointpattern.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ================= (2) temporal decomposition =================
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    D = data["WA2014"]; inc, cc, weeks = D["inc"], D["cc"], D["weeks"]
    ax = axes[0]; ax2 = ax.twinx()
    for c in ["GIN", "LBR", "SLE"]:
        y = inc[:, cc == c].sum(1)
        ax.plot(weeks, y, color=CCOL[c], lw=1.3, label=CNAME[c])
        ax2.plot(weeks, np.cumsum(y) / y.sum(), ls="--", lw=1.0, color=CCOL[c], alpha=0.7)
    ax.set_ylabel("weekly cases"); ax2.set_ylabel("cumulative share", fontsize=8)
    ax2.tick_params(labelsize=7); ax.tick_params(axis="x", rotation=35)
    ax.set_title("(a) Weekly cases and cumulative share", fontsize=8.5); ax.legend(frameon=False, fontsize=6.6)
    # (b) regions newly affected per week
    ax = axes[1]
    for dn, D2 in pp_data.items():
        first = np.where((D2["inc"] > 0).any(0), (D2["inc"] > 0).argmax(0), -1)
        wk = np.arange(D2["inc"].shape[0])
        newly = np.array([(first == t).sum() for t in wk])
        ax.step(wk, np.cumsum(newly) / (first >= 0).sum(), where="post",
                color=CCOL["SLE"] if dn == "WA2014" else CCOL["COD"], lw=1.4, label=dn)
    ax.set_xlabel("weeks since panel start"); ax.set_ylabel("cumulative share of regions affected")
    ax.set_title("(b) Spatial spread over time", fontsize=8.5); ax.legend(frameon=False)
    # (c) duration of local activity
    ax = axes[2]
    for dn, D2 in pp_data.items():
        dur = (D2["inc"] > 0).sum(0); dur = dur[dur > 0]
        ax.hist(dur, bins=np.arange(0, dur.max() + 3, 3), histtype="step", lw=1.4,
                color=CCOL["SLE"] if dn == "WA2014" else CCOL["COD"], label=dn)
    ax.set_xlabel("weeks with at least one case"); ax.set_ylabel("regions")
    ax.set_title("(c) Duration of local transmission", fontsize=8.5); ax.legend(frameon=False)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "temporal.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ================= (3) spatio-temporal spread in three periods =================
    fig, axes = plt.subplots(2, 3, figsize=(8.43, 5.41), gridspec_kw={"hspace": 0.72, "wspace": 0.28})
    MK = ["o", "s", "^", "D", "v"]
    PAL = ["#1c7ed6", "#2f9e44", "#862e9c", "#f59f00", "#e03131"]
    for row, (dn, D2) in enumerate(pp_data.items()):
        inc, coords, weeks, nodes = D2["inc"], D2["coords"], D2["weeks"], D2["nodes"]
        T = inc.shape[0]; cuts = [0, T // 3, 2 * T // 3, T]
        hot = list(np.argsort(-inc.sum(0))[:min(5, inc.shape[1])])   # largest cumulative hot spots
        for k in range(3):
            ax = axes[row, k]
            prev = inc[:cuts[k]].sum(0); cur = inc[cuts[k]:cuts[k + 1]].sum(0)
            ax.scatter(coords[:, 1], coords[:, 0], s=5, c="#cfe3e8", zorder=1)
            m = (prev > 0) & ~np.isin(np.arange(len(coords)), hot)
            ax.scatter(coords[m, 1], coords[m, 0], s=6 + 50 * prev[m] / max(prev.max(), 1),
                       facecolors="none", edgecolors="#5aa9b8", lw=0.8, zorder=2)
            m2 = (cur > 0) & ~np.isin(np.arange(len(coords)), hot)
            ax.scatter(coords[m2, 1], coords[m2, 0], s=6 + 50 * cur[m2] / max(cur.max(), 1),
                       c="#0b7285", alpha=0.75, edgecolors="none", zorder=3)
            for q, i in enumerate(hot):                         # hot spots keep their identity across periods
                s_cur = cur[i]; s_prev = prev[i]
                if s_prev > 0:
                    ax.scatter(coords[i, 1], coords[i, 0], s=16, marker=MK[q], facecolors="none",
                               edgecolors=PAL[q], lw=0.9, zorder=4)
                if s_cur > 0:
                    ax.scatter(coords[i, 1], coords[i, 0], s=16 + 52 * s_cur / max(cur.max(), 1),
                               marker=MK[q], c=PAL[q], edgecolors="k", lw=0.4, zorder=5, alpha=0.9,
                               label=nodes[i].split(":")[-1] if k == 0 else None)
            ax.set_title(f"{weeks[cuts[k]].date()} to {weeks[cuts[k+1]-1].date()}\n"
                         f"{int(cur.sum())} cases in {int((cur>0).sum())} regions", fontsize=8.0)
            ax.tick_params(labelsize=7.2)
            if row == len(pp_data) - 1: ax.set_xlabel("Longitude", fontsize=8.5)
            if k == 0: ax.set_ylabel(f"{dn}\nLatitude", fontsize=8.5)
        # one legend per row: the five hot spots, drawn from whichever period first shows them
        h, l = [], []
        for q, i in enumerate(hot):
            h.append(plt.Line2D([], [], marker=MK[q], color=PAL[q], ls="none", ms=5,
                                markeredgecolor="k", markeredgewidth=0.4))
            l.append(nodes[i].split(":")[-1])
        h += [plt.Line2D([], [], marker="o", color="#0b7285", ls="none", ms=4),
              plt.Line2D([], [], marker="o", color="none", markerfacecolor="none",
                         markeredgecolor="#5aa9b8", markeredgewidth=0.9, ls="none", ms=5),
              plt.Line2D([], [], marker="o", color="#cfe3e8", ls="none", ms=3)]
        l += ["other, this period", "earlier periods", "never affected"]
        axes[row, 1].legend(h, l, frameon=False, fontsize=6.8, ncol=4, loc="lower center",
                            bbox_to_anchor=(0.5, 1.20), handletextpad=0.25, columnspacing=0.8)
    label_axes(fig, axes)
    fig.subplots_adjust(hspace=1.30)        # room for the row-2 legend under the row-1 letters
    fig.savefig(FIG / "spread.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    json.dump(stats, open(OUT / "paper_numbers_pp.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
